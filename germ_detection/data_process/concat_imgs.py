import os
import shutil

# 设置原始图片文件夹和目标文件夹路径
source_folder = '../resources/dataset/quiron_full'
target_folder = '../resources/dataset/quiron_imgs'

# 创建目标文件夹（如果不存在）
os.makedirs(target_folder, exist_ok=True)

# 遍历原始图片文件夹中的文件夹
for folder in os.listdir(source_folder):
    # 获取文件夹路径
    folder_path = os.path.join(source_folder, folder)
    # 遍历文件夹中的图片文件
    for file in os.listdir(folder_path):
        # 获取图片文件路径
        file_path = os.path.join(folder_path, file)
        # 拼接新文件名
        new_filename = folder + '.' + file
        # 设置目标文件路径
        target_file_path = os.path.join(target_folder, new_filename)
        # 复制文件到目标文件夹
        shutil.copy(file_path, target_file_path)

print("拼接并复制完成！")
