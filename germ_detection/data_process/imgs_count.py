import os

def imgs_count(image_folder):
    # 设置图片文件夹路径
    image_folder = 'path/to/image_folder'

    # 获取图片文件夹中的所有文件
    all_files = os.listdir(image_folder)

    # 初始化图片计数器
    image_count = 0

    # 遍历所有文件，统计图片文件数量
    for file in all_files:
        # 获取文件的扩展名
        _, ext = os.path.splitext(file)
        # 如果文件是图片文件（扩展名为 '.jpg'、'.jpeg'、'.png' 等）
        if ext.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']:
            # 将图片计数器加一
            image_count += 1

    return image_count
