import os
import mmap
import numpy as np
import cv2
from tqdm import  tqdm

class FileListMemoryMapper:
    def __init__(self, file_path_list):
        """
        初始化内存映射器
        :param file_path_list: 图片路径序列的列表，每个元素是一个路径序列
        """
        self.file_path_list = file_path_list
        self.memory_map = {}
        #self.load_files_from_list()
    def map_image_file(self, file_path):
        """将图片文件映射到内存"""
        try:
            with open(file_path, 'rb') as f:
                # 映射文件到内存
                return mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ)
        except Exception as e:
            print(f"Error mapping image file {file_path}: {e}")
            return None

    def load_files_from_list(self, max_seqs=None):
        """
        加载指定路径序列的文件
        :param sequence_index: 指定列表中的第几个路径序列
        :param max_files: 限制加载的文件数量（可选）
        """
        # file_count = 0
        # if sequence_index < 0 or sequence_index >= len(self.file_path_list):
        #     raise IndexError("Sequence index out of range.")
        for sequence_index in range(0,len(self.file_path_list)):
            # 获取路径序列
            file_paths = self.file_path_list[sequence_index].frames
            for file_path in tqdm(file_paths):
                if os.path.isfile(file_path) and file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    # 映射图片文件
                    self.memory_map[file_path] = self.map_image_file(file_path)
                    self.get_image(file_path)
                    # file_count += 1
                    # if max_files and file_count >= max_files:
                    #     break

    def get_image(self, file_path):
        """将映射的图片数据解析为OpenCV图像对象"""
        mmap_obj = self.memory_map.get(file_path)
        if mmap_obj and isinstance(mmap_obj, mmap.mmap):
            mmap_obj.seek(0)
            img_array = np.frombuffer(mmap_obj, dtype=np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            return image
        return None

    def cleanup(self):
        """释放所有映射的内存资源"""
        for mmap_obj in self.memory_map.values():
            if mmap_obj:
                mmap_obj.close()
        self.memory_map.clear()


# 使用示例
# file_path_list = [
#     ["path/to/sequence1/image1.jpg", "path/to/sequence1/image2.jpg"],
#     ["path/to/sequence2/image1.jpg", "path/to/sequence2/image2.jpg"]
# ]
#
# mapper = FileListMemoryMapper(file_path_list)
#
# try:
#     # 加载第一个路径序列中的前两个文件
#     mapper.load_files_from_list(sequence_index=0, max_files=2)
#
#     # 获取并处理图片
#     for file_path in file_path_list[0]:
#         image_data = mapper.get_image(file_path)
#         if image_data is not None:
#             print(f"Image loaded successfully for {file_path}")
#             cv2.imshow("Image", image_data)  # 显示图片
#             cv2.waitKey(0)
#             cv2.destroyAllWindows()
#         else:
#             print(f"Failed to load image for {file_path}")
# finally:
#     # 清理资源
#     mapper.cleanup()
