"""
将 LaSOT / COCO / GOT-10k / TrackingNet 数据集转换为 LMDB 格式（多线程并行读取）。

架构：多线程读文件（瓶颈在磁盘IO） → 队列 → 单线程写 LMDB（LMDB 只支持单写事务）

用法：
    # LaSOT（压缩包 ~227GB，解压后 ~350~400GB）
    python liu_tools/lmdb_tools/convert_to_lmdb.py lasot \
        --src /data/LaSOT \
        --dst /data/lasot_lmdb \
        --map-size 1099511627776

    # COCO 2017
    python liu_tools/lmdb_tools/convert_to_lmdb.py coco \
        --src /data/coco \
        --dst /data/coco_lmdb \
        --version 2017

    # GOT-10k（压缩包 ~73GB，解压后 ~140GB）
    # --src 指向包含 list.txt 的目录（train 目录本身或其父目录均可，脚本自动检测）
    python liu_tools/lmdb_tools/convert_to_lmdb.py got10k \
        --src /data/got10k \
        --dst /data/got10k_lmdb

    # TrackingNet（压缩包 ~1051GB，解压后 ~1.5TB，共 12 个 set）
    # --src 指向包含 TRAIN_0 ~ TRAIN_11 的父目录
    # --dst 是输出根目录，脚本会在下面创建 TRAIN_0_lmdb ~ TRAIN_11_lmdb
    python liu_tools/lmdb_tools/convert_to_lmdb.py trackingnet \
        --src /data/trackingnet \
        --dst /data/trackingnet_lmdb \
        --map-size 2748779069440

    # 自定义线程数（HDD 建议 4~8，SSD 建议 16~32）
    python liu_tools/lmdb_tools/convert_to_lmdb.py got10k \
        --src /data/got10k \
        --dst /data/got10k_lmdb \
        --workers 32

生成的 LMDB 目录可直接用于：
    env_settings().lasot_lmdb_dir / got10k_lmdb_dir / coco_lmdb_dir / trackingnet_lmdb_dir
"""

import argparse
import os
import queue
import json
import threading
import lmdb
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────
#  并行写入核心：多线程读 + 单线程写
# ─────────────────────────────────────────────────────────────────────

_SENTINEL = None   # 写入线程的终止信号


def parallel_write_lmdb(env, entries, workers=16, commit_every=2000):
    """
    并行将 entries 写入 LMDB。

    参数：
        env          : lmdb.Environment（已 open）
        entries      : list of (key_str, file_path)  或  list of (key_str, bytes)
        workers      : 读取线程数
        commit_every : 每写入多少条提交一次事务
    """
    task_queue = queue.Queue(maxsize=workers * 4)  # 有界队列，控制内存

    # ── 写入线程（单线程，消费队列） ──────────────────────────────
    write_error = [None]

    def writer():
        txn = env.begin(write=True)
        count = 0
        try:
            while True:
                item = task_queue.get()
                if item is _SENTINEL:
                    break
                key, data = item
                txn.put(key.encode(), data)
                count += 1
                if count % commit_every == 0:
                    txn.commit()
                    txn = env.begin(write=True)
            txn.commit()
        except Exception as e:
            write_error[0] = e
            try:
                txn.abort()
            except:
                pass

    writer_thread = threading.Thread(target=writer, daemon=True)
    writer_thread.start()

    # ── 读取函数（线程池里执行） ─────────────────────────────────
    def read_entry(entry):
        key, fpath = entry
        with open(fpath, "rb") as f:
            data = f.read()
        return key, data

    # ── 生产者：线程池并行读文件，顺序送入队列 ──────────────────────
    #    用 ThreadPoolExecutor.map 保证结果按原始顺序产出
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for key, data in tqdm(
            pool.map(read_entry, entries, chunksize=32),
            total=len(entries),
            desc="读取并写入 LMDB",
            unit="file",
        ):
            if write_error[0] is not None:
                break
            task_queue.put((key, data))

    # 发送终止信号，等待写入线程结束
    task_queue.put(_SENTINEL)
    writer_thread.join()

    if write_error[0] is not None:
        raise RuntimeError(f"LMDB 写入失败：{write_error[0]}")

    return len(entries)


# ─────────────────────────────────────────────────────────────────────
#  LaSOT
# ─────────────────────────────────────────────────────────────────────

def convert_lasot(src, dst, map_size, workers):
    """
    LaSOT 目录结构（src 下）：
        airplane/
            airplane-1/
                groundtruth.txt
                full_occlusion.txt
                out_of_view.txt
                img/
                    00000001.jpg
                    ...
    """
    src = Path(src)
    dst = str(dst)

    # ── 第一遍：扫描文件路径（纯路径，不读文件） ────────────────────
    print("扫描 LaSOT 文件...")
    all_entries = []  # (lmdb_key, file_path)
    class_dirs = sorted([d for d in src.iterdir() if d.is_dir()])
    for class_dir in tqdm(class_dirs, desc="扫描类"):
        for seq_dir in sorted(class_dir.iterdir()):
            if not seq_dir.is_dir():
                continue
            # 标注文件
            for ann_file in ["groundtruth.txt", "full_occlusion.txt",
                             "out_of_view.txt", "nlp.txt"]:
                ann_path = seq_dir / ann_file
                if ann_path.exists():
                    key = str(ann_path.relative_to(src))
                    all_entries.append((key, str(ann_path)))
            # 图片
            img_dir = seq_dir / "img"
            if img_dir.is_dir():
                for img_file in sorted(img_dir.iterdir()):
                    if img_file.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                        key = str(img_file.relative_to(src))
                        all_entries.append((key, str(img_file)))

    print(f"共 {len(all_entries)} 个条目，使用 {workers} 线程读取")

    # LaSOT 解压后 ~350~400GB，map_size 设 1TB（虚拟地址空间，不占磁盘）
    if map_size is None:
        map_size = 1 * 1024 ** 4  # 1TB

    env = lmdb.open(dst, map_size=map_size)
    n = parallel_write_lmdb(env, all_entries, workers=workers)
    env.close()
    print(f"LaSOT LMDB 写入完成：{dst}，共 {n} 条")


# ─────────────────────────────────────────────────────────────────────
#  COCO
# ─────────────────────────────────────────────────────────────────────

def convert_coco(src, dst, version, map_size, workers):
    """
    COCO 目录结构（src 下）：
        annotations/
            instances_train2017.json
        images/
            train2017/
                000000000009.jpg
                ...
    """
    src = Path(src)
    dst = str(dst)
    split = "train"

    anno_json_path = src / "annotations" / f"instances_{split}{version}.json"
    if not anno_json_path.exists():
        print(f"找不到标注文件：{anno_json_path}")
        sys.exit(1)

    img_dir = src / "images" / f"{split}{version}"
    if not img_dir.is_dir():
        print(f"找不到图片目录：{img_dir}")
        sys.exit(1)

    all_imgs = sorted([f for f in img_dir.iterdir()
                       if f.suffix.lower() in ('.jpg', '.jpeg', '.png')])
    print(f"COCO {split}{version}：{len(all_imgs)} 张图片 + 1 个 JSON 标注")

    if map_size is None:
        map_size = max((len(all_imgs) + 1) * 300 * 1024, 50 * 1024 ** 3)  # ≥50GB

    env = lmdb.open(dst, map_size=map_size)

    # 先写 JSON 标注
    print("写入标注 JSON...")
    txn = env.begin(write=True)
    anno_key = f"annotations/instances_{split}{version}.json"
    with open(anno_json_path, "r") as f:
        anno_str = f.read()
    txn.put(anno_key.encode(), anno_str.encode("utf-8"))
    txn.commit()

    # 并行写入图片
    entries = [
        (f"images/{split}{version}/{img_path.name}", str(img_path))
        for img_path in all_imgs
    ]
    print(f"使用 {workers} 线程并行读取图片...")
    n = parallel_write_lmdb(env, entries, workers=workers)
    env.close()
    print(f"COCO LMDB 写入完成：{dst}，共 {n} 张图片")


# ─────────────────────────────────────────────────────────────────────
#  GOT-10k
#  LMDB key 结构（对应 got10k_lmdb.py 期望的格式）：
#      train/list.txt
#      train/{seq_name}/meta_info.ini
#      train/{seq_name}/groundtruth.txt
#      train/{seq_name}/absence.label
#      train/{seq_name}/cover.label
#      train/{seq_name}/{00000001}.jpg
# ─────────────────────────────────────────────────────────────────────

def convert_got10k(src, dst, map_size, workers):
    """
    GOT-10k 目录结构（src 或 src/train 下）：
        list.txt
        GOT-10k-Train_000001/
            meta_info.ini
            groundtruth.txt
            absence.label
            cover.label
            00000001.jpg
            ...

    --src 可以指向 train 目录本身或其父目录，脚本自动检测。
    """
    src = Path(src)
    dst = str(dst)

    # 自动检测 list.txt 位置（可能在 src 或 src/train 下）
    if (src / "list.txt").exists():
        train_dir = src
    elif (src / "train" / "list.txt").exists():
        train_dir = src / "train"
    else:
        print(f"在 {src} 或 {src / 'train'} 下找不到 list.txt，请检查 --src 路径")
        sys.exit(1)

    print(f"GOT-10k train 目录：{train_dir}")

    # 读取 list.txt 获取序列名
    with open(train_dir / "list.txt") as f:
        seq_names = [line.strip() for line in f if line.strip()]
    print(f"共 {len(seq_names)} 个序列")

    # 收集所有条目：(lmdb_key, disk_path)
    all_entries = []

    # list.txt 本身
    all_entries.append(("train/list.txt", str(train_dir / "list.txt")))

    for seq_name in tqdm(seq_names, desc="扫描 GOT-10k 序列"):
        seq_dir = train_dir / seq_name
        if not seq_dir.is_dir():
            print(f"警告：序列目录不存在，跳过 {seq_dir}")
            continue
        seq_prefix = f"train/{seq_name}"

        # 标注文件（按 got10k_lmdb.py 中实际使用的顺序）
        for ann_file in ["meta_info.ini", "groundtruth.txt", "absence.label", "cover.label"]:
            ann_path = seq_dir / ann_file
            if ann_path.exists():
                all_entries.append((f"{seq_prefix}/{ann_file}", str(ann_path)))

        # 图片
        for img_file in sorted(seq_dir.iterdir()):
            if img_file.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                all_entries.append((f"{seq_prefix}/{img_file.name}", str(img_file)))

    print(f"共 {len(all_entries)} 个条目，使用 {workers} 线程读取")

    # GOT-10k 解压后 ~140GB，map_size 设 300GB（虚拟地址空间，不占磁盘）
    if map_size is None:
        map_size = 300 * 1024 ** 3  # 300GB

    env = lmdb.open(dst, map_size=map_size)
    n = parallel_write_lmdb(env, all_entries, workers=workers)
    env.close()
    print(f"GOT-10k LMDB 写入完成：{dst}，共 {n} 条")


# ─────────────────────────────────────────────────────────────────────
#  TrackingNet（每个 set 独立一个 LMDB）
#  共 12 个 set (TRAIN_0 ~ TRAIN_11)，每个 set 约 88GB（压缩），解压后 ~125GB
#  总计约 1.5TB，需分别创建 12 个 LMDB 目录
#
#  LMDB key 结构（对应 tracking_net_lmdb.py）：
#      anno/{vid_name}.txt
#      frames/{vid_name}/{frame_id}.jpg    （frame_id 从 0 开始）
#
#  额外文件写入 dst 根目录：
#      seq_list.json   （序列列表，供 list_sequences() 读取）
# ─────────────────────────────────────────────────────────────────────

def _get_trackingnet_sequences(src_dir, set_id):
    """读取一个 TrackingNet set 的所有序列名（从 anno 目录读取）"""
    anno_dir = src_dir / f"TRAIN_{set_id}" / "anno"
    if not anno_dir.is_dir():
        return []
    seq_names = []
    for f in sorted(anno_dir.iterdir()):
        if f.suffix == ".txt":
            seq_names.append(f.stem)  # vid_name 不含 .txt 后缀
    return seq_names


def convert_trackingnet(src, dst, map_size, workers):
    """
    TrackingNet 目录结构（src 下）：
        TRAIN_0/
            anno/
                {vid_name}.txt       ← groundtruth (x,y,w,h) per line
            frames/
                {vid_name}/
                    0.jpg
                    1.jpg
                    ...
        TRAIN_1/
            ...
        TRAIN_11/
            ...

    输出：
        dst/
            seq_list.json                        ← 训练代码用此文件获取序列列表
            TRAIN_0_lmdb/                        ← 每个 set 一个 LMDB
                anno/{vid_name}.txt
                frames/{vid_name}/0.jpg
                ...
            TRAIN_1_lmdb/
                ...
            TRAIN_11_lmdb/
                ...
    """
    src = Path(src)
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)

    # ── 收集所有 set 的序列信息，写 seq_list.json ────────────────────
    print("扫描 TrackingNet 序列...")
    all_seq_list = []   # [(set_id, vid_name), ...]

    for set_id in range(12):
        seq_names = _get_trackingnet_sequences(src, set_id)
        if not seq_names:
            print(f"警告：TRAIN_{set_id}/anno 不存在或为空，跳过")
            continue
        print(f"  TRAIN_{set_id}：{len(seq_names)} 个序列")
        for vid_name in seq_names:
            all_seq_list.append([set_id, vid_name])

    # 写 seq_list.json（训练代码 list_sequences() 依赖此文件）
    seq_list_path = dst / "seq_list.json"
    with open(seq_list_path, "w") as f:
        json.dump(all_seq_list, f)
    print(f"写入 {seq_list_path}，共 {len(all_seq_list)} 个序列")

    # ── 每个 set 独立转 LMDB ────────────────────────────────────────
    # 每个 set 约 6~7M 张图，解压后 ~125GB/set
    if map_size is None:
        map_size = 200 * 1024 ** 3  # 每个 set 200GB，足够

    total_count = 0

    for set_id in range(12):
        set_dir = src / f"TRAIN_{set_id}"
        if not set_dir.is_dir():
            continue

        seq_names = [s[1] for s in all_seq_list if s[0] == set_id]
        lmdb_dir = str(dst / f"TRAIN_{set_id}_lmdb")

        print(f"\n{'='*60}")
        print(f"处理 TRAIN_{set_id}（{len(seq_names)} 个序列）→ {lmdb_dir}")
        print(f"{'='*60}")

        entries = []
        for vid_name in seq_names:
            # 标注文件
            ann_path = set_dir / "anno" / f"{vid_name}.txt"
            if ann_path.exists():
                entries.append((f"anno/{vid_name}.txt", str(ann_path)))

            # 图片帧
            frames_dir = set_dir / "frames" / vid_name
            if frames_dir.is_dir():
                for img_file in sorted(frames_dir.iterdir()):
                    if img_file.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                        frame_id = img_file.stem  # "0", "1", ...
                        entries.append(
                            (f"frames/{vid_name}/{frame_id}.jpg", str(img_file)))

        if not entries:
            continue

        print(f"共 {len(entries)} 个条目，使用 {workers} 线程读取")

        env = lmdb.open(lmdb_dir, map_size=map_size)
        n = parallel_write_lmdb(env, entries, workers=workers)
        env.close()
        total_count += n
        print(f"TRAIN_{set_id} LMDB 写入完成：{n} 条")

    print(f"\nTrackingNet 全部完成，共 {total_count} 条")


# ─────────────────────────────────────────────────────────────────────
#  验证
# ─────────────────────────────────────────────────────────────────────

def verify_lasot(lmdb_path):
    from lib.utils.lmdb_utils import decode_img, decode_str
    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    with env.begin() as txn:
        cursor = txn.cursor()
        cursor.first()
        first_key = cursor.key().decode()
        print(f"第一个 key：{first_key}")
        if first_key.endswith(('.jpg', '.jpeg', '.png')):
            img = decode_img(lmdb_path, first_key)
            print(f"图片 shape：{img.shape}")
        else:
            txt = decode_str(lmdb_path, first_key)
            print(f"文本前 200 字符：\n{txt[:200]}")
    env.close()
    print("验证通过")


def verify_coco(lmdb_path):
    from lib.utils.lmdb_utils import decode_img, decode_json
    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    with env.begin() as txn:
        anno_key = None
        for key, _ in txn.cursor():
            k = key.decode()
            if k.endswith(".json"):
                anno_key = k
                break
        if anno_key:
            coco_json = decode_json(lmdb_path, anno_key)
            print(f"JSON 标注 key：{anno_key}，含 {len(coco_json.get('images', []))} 张图片信息")
        for key, _ in txn.cursor():
            k = key.decode()
            if k.endswith(('.jpg', '.jpeg', '.png')):
                img = decode_img(lmdb_path, k)
                print(f"图片 key：{k}，shape：{img.shape}")
                break
    env.close()
    print("验证通过")


def verify_got10k(lmdb_path):
    from lib.utils.lmdb_utils import decode_img, decode_str
    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    with env.begin() as txn:
        # 验证 list.txt
        txt = decode_str(lmdb_path, "train/list.txt")
        print(f"list.txt 前 200 字符：\n{txt[:200]}")

        # 找第一张图
        for key, _ in txn.cursor():
            k = key.decode()
            if k.endswith(('.jpg', '.jpeg', '.png')):
                img = decode_img(lmdb_path, k)
                print(f"图片 key：{k}，shape：{img.shape}")
                break
    env.close()
    print("验证通过")


def verify_trackingnet(lmdb_root):
    from lib.utils.lmdb_utils import decode_img, decode_str
    lmdb_root = Path(lmdb_root)

    # 验证 seq_list.json
    seq_list_path = lmdb_root / "seq_list.json"
    with open(seq_list_path) as f:
        seq_list = json.load(f)
    print(f"seq_list.json：共 {len(seq_list)} 个序列")

    # 验证第一个 LMDB set
    lmdb_dir = lmdb_root / "TRAIN_0_lmdb"
    if lmdb_dir.is_dir():
        env = lmdb.open(str(lmdb_dir), readonly=True, lock=False)
        with env.begin() as txn:
            # 找第一张图
            for key, _ in txn.cursor():
                k = key.decode()
                if k.endswith(('.jpg', '.jpeg', '.png')):
                    img = decode_img(str(lmdb_dir), k)
                    print(f"TRAIN_0_lmdb 图片 key：{k}，shape：{img.shape}")
                    break
            # 验证第一个标注
            for key, _ in txn.cursor():
                k = key.decode()
                if k.startswith("anno/") and k.endswith(".txt"):
                    txt = decode_str(str(lmdb_dir), k)
                    print(f"TRAIN_0_lmdb 标注 key：{k}，前 2 行：\n{'  '.join(txt.split(chr(10))[:2])}")
                    break
        env.close()
    print("验证通过")


# ─────────────────────────────────────────────────────────────────────
#  主入口
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="将数据集转换为 LMDB 格式（多线程并行读取）")
    parser.add_argument("dataset", choices=["lasot", "coco", "got10k", "trackingnet"],
                        help="数据集类型")
    parser.add_argument("--src", required=True, help="源数据集根目录")
    parser.add_argument("--dst", required=True, help="LMDB 输出目录")
    parser.add_argument("--version", default="2017", help="COCO 版本（2014 或 2017）")
    parser.add_argument("--map-size", type=int, default=None,
                        help="LMDB map_size（字节），默认按数据集自动估算")
    parser.add_argument("--workers", type=int, default=16,
                        help="并行读取线程数（默认 16，SSD 可调至 32+，HDD 建议 4~8）")
    parser.add_argument("--verify", action="store_true", help="转换后立即验证")
    args = parser.parse_args()

    if args.dataset == "lasot":
        convert_lasot(args.src, args.dst, args.map_size, args.workers)
        if args.verify:
            verify_lasot(args.dst)
    elif args.dataset == "coco":
        convert_coco(args.src, args.dst, args.version, args.map_size, args.workers)
        if args.verify:
            verify_coco(args.dst)
    elif args.dataset == "got10k":
        convert_got10k(args.src, args.dst, args.map_size, args.workers)
        if args.verify:
            verify_got10k(args.dst)
    elif args.dataset == "trackingnet":
        convert_trackingnet(args.src, args.dst, args.map_size, args.workers)
        if args.verify:
            verify_trackingnet(args.dst)
