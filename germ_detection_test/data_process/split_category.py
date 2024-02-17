import os
import csv
import shutil

# 定义图像文件夹和CSV文件的路径
image_folder = "../resources/dataset/quiron_imgs"
csv_file = "../resources/form/window_metadata.csv"

# 定义输出文件夹
output_folder = "../resources/dataset/quiron_category"
os.makedirs(output_folder, exist_ok=True)

# 定义类别标签映射
category_labels = {
    -1: "negative",
    0: "unconfirmed",
    1: "positive"
}

# 创建类别文件夹
for label in category_labels.values():
    category_folder = os.path.join(output_folder, label)
    os.makedirs(category_folder, exist_ok=True)

# 读取CSV文件并将图像移动到相应的文件夹中
with open(csv_file, 'r') as file:
    reader = csv.reader(file)
    next(reader)  # 跳过标题行
    for row in reader:
        image_name, category = row
        category_label = int(category)
        dest_folder = os.path.join(output_folder, category_labels[category_label])

        # 源图像路径和目标路径
        src_image_path = os.path.join(image_folder, image_name + '.png')
        dest_image_path = os.path.join(dest_folder, image_name + '.png')

        # 移动图像文件
        shutil.copy(src_image_path, dest_image_path)

print("图像分类完成！")
