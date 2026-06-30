import os
import subprocess

# --- 请根据你的实际情况修改以下配置 ---

# 1. 存放视频文件的根目录
ROOT_VIDEO_DIR = "/data2/lqh/data_/CUADC2025_VIDEO_3090/"

# 2. 你的 Python 脚本的路径
PYTHON_SCRIPT_PATH = "test_video.py"  # 假设与此脚本在同一目录

# 3. 固定的跟踪器参数
TRACKER_NAME = "odonet_v1"
TRACKER_PARAM = "odonet_b224"

# --- 配置结束 ---


def process_all_videos():
    """
    遍历目录并处理所有视频文件。
    """
    # 遍历根目录下的所有文件和文件夹
    for root, dirs, files in os.walk(ROOT_VIDEO_DIR):
        for filename in files:
            # 检查文件扩展名是否为 .mp4 (不区分大小写)
            if filename.lower().endswith('.mp4'):
                # 构建视频文件的完整路径
                video_path = os.path.join(root, filename)

                print("=" * 60)
                print(f">>> 正在处理视频: {video_path}")
                print("=" * 60)

                # 构建要执行的命令列表
                command = [
                    'python',
                    PYTHON_SCRIPT_PATH,
                    TRACKER_NAME,
                    TRACKER_PARAM,
                    video_path,
                    '--save_results'
                ]

                try:
                    # 执行命令
                    # check=True 会在命令返回非零退出码时抛出异常
                    subprocess.run(command, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"--- 警告: 处理视频 {video_path} 时出错: {e} ---")
                except FileNotFoundError:
                    print(f"--- 错误: 找不到 'python' 命令或脚本 '{PYTHON_SCRIPT_PATH}' ---")
                    return # 严重错误，直接退出

if __name__ == '__main__':
    process_all_videos()
    print("\n>>> 所有视频处理完成！")