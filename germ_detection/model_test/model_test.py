import os
from tqdm import tqdm

import numpy as np
import pandas as pd

from PIL import Image
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Simhei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False  # 显示负号
import torch
import torch.nn.functional as F
from torchvision import transforms

# 有 GPU 就用 GPU，没有就用 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device', device)

# 测试集图像预处理-RCTN：缩放、裁剪、转 Tensor、归一化
test_transform = transforms.Compose([  # transforms.Resize(256),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
])

# 载入测试集
# 数据集文件夹路径
dataset_dir = '../resources/dataset/quiron_split'
test_path = os.path.join(dataset_dir, 'test')
from torchvision import datasets

# 载入测试集
test_dataset = datasets.ImageFolder(test_path, test_transform)
print('测试集图像数量', len(test_dataset))
print('类别个数', len(test_dataset.classes))
print('各类别名称', test_dataset.classes)
# 载入类别名称 和 ID索引号 的映射字典
idx_to_labels = np.load('../resources/form/idx_to_labels.npy', allow_pickle=True).item()
# 获得类别名称
classes = list(idx_to_labels.values())
print(classes)

# 载入训练好的模型
model = torch.load('../resources/model/best-0.958.pth')
model = model.eval().to(device)

# 表格A-测试集图像路径及标注
# print(test_dataset.imgs[:10])
img_paths = [each[0] for each in test_dataset.imgs]
df = pd.DataFrame()
df['图像路径'] = img_paths
df['标注类别ID'] = test_dataset.targets
df['标注类别名称'] = [idx_to_labels[ID] for ID in test_dataset.targets]
# print(df)

# 表格B-测试集每张图像的图像分类预测结果，以及各类别置信度
# 记录 top-n 预测结果
n = 3

df_pred = pd.DataFrame()
for idx, row in tqdm(df.iterrows()):
    img_path = row['图像路径']
    img_pil = Image.open(img_path).convert('RGB')
    input_img = test_transform(img_pil).unsqueeze(0).to(device)  # 预处理
    pred_logits = model(input_img)  # 执行前向预测，得到所有类别的 logit 预测分数
    pred_softmax = F.softmax(pred_logits, dim=1)  # 对 logit 分数做 softmax 运算

    pred_dict = {}

    top_n = torch.topk(pred_softmax, n)  # 取置信度最大的 n 个结果
    pred_ids = top_n[1].cpu().detach().numpy().squeeze()  # 解析出类别

    # top-n 预测结果
    for i in range(1, n + 1):
        pred_dict['top-{}-预测ID'.format(i)] = pred_ids[i - 1]
        pred_dict['top-{}-预测名称'.format(i)] = idx_to_labels[pred_ids[i - 1]]
    pred_dict['top-n预测正确'] = row['标注类别ID'] in pred_ids
    # 每个类别的预测置信度
    for idx, each in enumerate(classes):
        pred_dict['{}-预测置信度'.format(each)] = pred_softmax[0][idx].cpu().detach().numpy()

    df_pred = df_pred._append(pred_dict, ignore_index=True)

# print(df_pred)

# 拼接A-B表，并导出
df = pd.concat([df, df_pred], axis=1)
df.to_csv('../resources/form/测试集预测结果.csv', index=False)

'''
测试集总体指标评估
'''
import pandas as pd
import numpy as np
from tqdm import tqdm

idx_to_labels = np.load('../resources/form/idx_to_labels.npy', allow_pickle=True).item()
# 获得类别名称
classes = list(idx_to_labels.values())
print(classes)

df = pd.read_csv('../resources/form/测试集预测结果.csv')

# 准确率
sum(df['标注类别名称'] == df['top-1-预测名称']) / len(df)

# top-n准确率
sum(df['top-n预测正确']) / len(df)
from sklearn.metrics import classification_report

print(classification_report(df['标注类别名称'], df['top-1-预测名称'], target_names=classes))
report = classification_report(df['标注类别名称'], df['top-1-预测名称'], target_names=classes, output_dict=True)
del report['accuracy']
df_report = pd.DataFrame(report).transpose()

accuracy_list = []
for fruit in tqdm(classes):
    df_temp = df[df['标注类别名称'] == fruit]
    accuracy = sum(df_temp['标注类别名称'] == df_temp['top-1-预测名称']) / len(df_temp)
    accuracy_list.append(accuracy)

# 计算 宏平均准确率 和 加权平均准确率
acc_macro = np.mean(accuracy_list)
acc_weighted = sum(accuracy_list * df_report.iloc[:-2]['support'] / len(df))

accuracy_list.append(acc_macro)
accuracy_list.append(acc_weighted)

df_report['accuracy'] = accuracy_list

df_report.to_csv('../resources/form/各类别准确率评估指标.csv', index_label='类别')
