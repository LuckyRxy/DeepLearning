import time
import os

from torchvision.models import ResNet18_Weights
from tqdm import tqdm

import pandas as pd
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Simhei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False  # 显示负号

from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import models
import torch.optim as optim

import warnings
warnings.filterwarnings("ignore")


from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_auc_score

# # 存放结果文件
# os.mkdir('../../resources/output')
#
# # 存放训练得到的模型权重
# os.mkdir('../../resources/checkpoint')
# # 特别注意，不要把文件夹命名成 checkpoints （不要在最后加 s） ，否则在 Jupyter 中打不开，只能压缩后下载压缩包到本地才能打开
#
# # 存放生成的图表
# os.mkdir('../../resources/chart')

# 获取计算硬件
# 有 GPU 就用 GPU，没有就用 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device', device)

from torchvision import transforms
from torch.optim import lr_scheduler

# 训练集图像预处理：缩放裁剪、图像增强、转 Tensor、归一化
train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                      ])

# 测试集图像预处理-RCTN：缩放、裁剪、转 Tensor、归一化
test_transform = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                                     ])

# 数据集文件夹路径
dataset_dir = '../../dataset/fruit30_split'

train_path = os.path.join(dataset_dir, 'train')
test_path = os.path.join(dataset_dir, 'val')
print('训练集路径', train_path)
print('测试集路径', test_path)

# 载入训练集
train_dataset = datasets.ImageFolder(train_path, train_transform)
# 载入测试集
test_dataset = datasets.ImageFolder(test_path, test_transform)
print('训练集图像数量', len(train_dataset))
print('类别个数', len(train_dataset.classes))
print('各类别名称', train_dataset.classes)
print('测试集图像数量', len(test_dataset))
print('类别个数', len(test_dataset.classes))
print('各类别名称', test_dataset.classes)

# 各类别名称(映射关系)
class_names = train_dataset.classes
n_class = len(class_names)
# 映射关系：类别 到 索引号
print(train_dataset.class_to_idx)
# 映射关系：索引号 到 类别
idx_to_labels = {y: x for x, y in train_dataset.class_to_idx.items()}
# 保存为本地的 npy 文件
np.save('../../resources/output/idx_to_labels.npy', idx_to_labels)
np.save('../../resources/output/labels_to_idx.npy', train_dataset.class_to_idx)

# 定义数据加载器DataLoader
BATCH_SIZE = 32
# 训练集的数据加载器
train_loader = DataLoader(train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=0
                          )
# 测试集的数据加载器
test_loader = DataLoader(test_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=False,
                         num_workers=0
                         )
# DataLoader 是 python生成器，每次调用返回一个 batch 的数据
images, labels = next(iter(train_loader))
print(images.shape)
print(labels)

# 可视化一个batch的图像和标注
# 将数据集中的Tensor张量转为numpy的array数据类型
images = images.numpy()
print(images[5].shape)
plt.hist(images[5].flatten(), bins=50)
plt.show()

# 训练
# 选择一：只微调训练模型最后一层（全连接分类层）
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)  # 载入预训练模型
# 修改全连接层，使得全连接层的输出与当前数据集类别数对应
# 新建的层默认 requires_grad=True
model.fc = nn.Linear(model.fc.in_features, n_class)
print(model.fc)
# 只微调训练最后一层全连接层的参数，其它层冻结
optimizer = optim.Adam(model.fc.parameters())

# 选择二：微调训练所有层
# model = models.resnet18(pretrained=True) # 载入预训练模型
# model.fc = nn.Linear(model.fc.in_features, n_class)
# optimizer = optim.Adam(model.parameters())

# 选择三：随机初始化模型全部权重，从头训练所有层
# model = models.resnet18(pretrained=False) # 只载入模型结构，不载入预训练权重参数
# model.fc = nn.Linear(model.fc.in_features, n_class)
# optimizer = optim.Adam(model.parameters())

# 训练配置
model = model.to(device)
# 交叉熵损失函数
criterion = nn.CrossEntropyLoss()
# 训练轮次 Epoch
EPOCHS = 20

lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

def train_one_batch(images, labels):
    '''
    运行一个 batch 的训练，返回当前 batch 的训练日志
    '''

    # 获得一个 batch 的数据和标注
    images = images.to(device)
    labels = labels.to(device)

    outputs = model(images)  # 输入模型，执行前向预测
    loss = criterion(outputs, labels)  # 计算当前 batch 中，每个样本的平均交叉熵损失函数值

    # 优化更新权重
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 获取当前 batch 的标签类别和预测类别
    _, preds = torch.max(outputs, 1)  # 获得当前 batch 所有图像的预测类别
    preds = preds.cpu().numpy()
    loss = loss.detach().cpu().numpy()
    outputs = outputs.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    log_train = {}
    log_train['epoch'] = epoch
    log_train['batch'] = batch_idx
    # 计算分类评估指标
    log_train['train_loss'] = loss
    log_train['train_accuracy'] = accuracy_score(labels, preds)
    # log_train['train_precision'] = precision_score(labels, preds, average='macro')
    # log_train['train_recall'] = recall_score(labels, preds, average='macro')
    # log_train['train_f1-score'] = f1_score(labels, preds, average='macro')

    return log_train

# 在整个测试集上进行评估
def evaluate_testset():
    '''
    在整个测试集上评估，返回分类评估指标日志
    '''

    loss_list = []
    labels_list = []
    preds_list = []

    with torch.no_grad():
        for images, labels in test_loader:  # 生成一个 batch 的数据和标注
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)  # 输入模型，执行前向预测

            # 获取整个测试集的标签类别和预测类别
            _, preds = torch.max(outputs, 1)  # 获得当前 batch 所有图像的预测类别
            preds = preds.cpu().numpy()
            loss = criterion(outputs, labels)  # 由 logit，计算当前 batch 中，每个样本的平均交叉熵损失函数值
            loss = loss.detach().cpu().numpy()
            outputs = outputs.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()

            loss_list.append(loss)
            labels_list.extend(labels)
            preds_list.extend(preds)

    log_test = {}
    log_test['epoch'] = epoch

    # 计算分类评估指标
    log_test['test_loss'] = np.mean(loss_list)
    log_test['test_accuracy'] = accuracy_score(labels_list, preds_list)
    log_test['test_precision'] = precision_score(labels_list, preds_list, average='macro')
    log_test['test_recall'] = recall_score(labels_list, preds_list, average='macro')
    log_test['test_f1-score'] = f1_score(labels_list, preds_list, average='macro')

    return log_test

# 日志记录
epoch = 0
batch_idx = 0
best_test_accuracy = 0

# 训练日志-训练集
df_train_log = pd.DataFrame()
log_train = {}
log_train['epoch'] = 0
log_train['batch'] = 0
images, labels = next(iter(train_loader))
log_train.update(train_one_batch(images, labels))
df_train_log = df_train_log._append(log_train, ignore_index=True)

# 训练日志-测试集
df_test_log = pd.DataFrame()
log_test = {}
log_test['epoch'] = 0
log_test.update(evaluate_testset())
df_test_log = df_test_log._append(log_test, ignore_index=True)

# 创建wandb可视化项目
import wandb
wandb.init(project='fruit30', name=time.strftime('%m%d%H%M%S'))
for epoch in range(1, EPOCHS + 1):

    print(f'Epoch {epoch}/{EPOCHS}')

    ## 训练阶段
    model.train()
    for images, labels in tqdm(train_loader):  # 获得一个 batch 的数据和标注
        batch_idx += 1
        log_train = train_one_batch(images, labels)
        df_train_log = df_train_log._append(log_train, ignore_index=True)
        wandb.log(log_train)

    lr_scheduler.step()

    ## 测试阶段
    model.eval()
    log_test = evaluate_testset()
    df_test_log = df_test_log._append(log_test, ignore_index=True)
    wandb.log(log_test)

    # 保存最新的最佳模型文件
    if log_test['test_accuracy'] > best_test_accuracy:
        # 删除旧的最佳模型文件(如有)
        old_best_checkpoint_path = '../../resources/checkpoint/best-{:.3f}.pth'.format(best_test_accuracy)
        if os.path.exists(old_best_checkpoint_path):
            os.remove(old_best_checkpoint_path)
        # 保存新的最佳模型文件
        best_test_accuracy = log_test['test_accuracy']
        new_best_checkpoint_path = '../../resources/checkpoint/best-{:.3f}.pth'.format(log_test['test_accuracy'])
        torch.save(model, new_best_checkpoint_path)
        print('保存新的最佳模型', '../../resources/checkpoint/best-{:.3f}.pth'.format(best_test_accuracy))
        # best_test_accuracy = log_test['test_accuracy']

df_train_log.to_csv('训练日志-训练集.csv', index=False)
df_test_log.to_csv('训练日志-测试集.csv', index=False)

# 载入最佳模型作为当前模型
model = torch.load('../../resources/checkpoint/best-{:.3f}.pth'.format(best_test_accuracy))
model.eval()
print(evaluate_testset())
