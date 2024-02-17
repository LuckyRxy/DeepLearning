import os
import cv2
import pandas as pd
import numpy as np
import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from torchvision import models
from torchvision import transforms
import torch.nn.functional as F

# 有 GPU 就用 GPU，没有就用 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device', device)

# 载入预训练图像分类模型
model = models.resnet18(weights=True)
# model = models.resnet152(pretrained=True)

model = model.eval()
model = model.to(device)

# 测试集图像预处理-RCTN：缩放、裁剪、转Tensor、归一化（RGB）
test_transform = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                                    ])

img_path = '../../dataset/test_image/cat.jpg'

# 用 pillow 载入
from PIL import Image
img_pil = Image.open(img_path)
print(img_pil)

input_img = test_transform(img_pil) # 预处理

input_img = input_img.unsqueeze(0).to(device)
print(input_img.shape)

# 执行前向预测，得到所有类别的 logit 预测分数
pred_logits = model(input_img)

# 对 logit 分数做 softmax 运算(置信度)
pred_softmax = F.softmax(pred_logits, dim=1)

# 绘制置信度柱状图
plt.figure(figsize=(8,4))
x = range(1000)
y = pred_softmax.cpu().detach().numpy()[0]

ax = plt.bar(x, y, alpha=0.5, width=0.3, color='yellow', edgecolor='red', lw=3)
plt.ylim([0, 1.0]) # y轴取值范围
# plt.bar_label(ax, fmt='%.2f', fontsize=15) # 置信度数值

plt.xlabel('Class', fontsize=20)
plt.ylabel('Confidence', fontsize=20)
plt.tick_params(labelsize=16) # 坐标文字大小
plt.title(img_path, fontsize=25)

plt.show()

# 去置信度最大的n个结果
n = 10
top_n = torch.topk(pred_softmax, n)

# 解析出类别
pred_ids = top_n[1].cpu().detach().numpy().squeeze()

# 解析出置信度
confs = top_n[0].cpu().detach().numpy().squeeze()


# 载入ImageNet 1000图像分类标签
df = pd.read_csv('../../dataset/imagenet_class_index.csv')
idx_to_labels = {}
for idx, row in df.iterrows():
    idx_to_labels[row['ID']] = [row['wordnet'], row['class']]

# 将图像分类结果写在原图上
# 用 opencv 载入原图
img_bgr = cv2.imread(img_path)

for i in range(n):
    class_name = idx_to_labels[pred_ids[i]][1]  # 获取类别名称
    confidence = confs[i] * 100  # 获取置信度
    text = '{:<15} {:>.4f}'.format(class_name, confidence)
    print(text)

    # !图片，添加的文字，左上角坐标，字体，字号，bgr颜色，线宽
    img_bgr = cv2.putText(img_bgr, text, (25, 50 + 40 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 0, 255), 3)

# 保存图像
cv2.imwrite('../../dataset/img_pred.jpg', img_bgr)