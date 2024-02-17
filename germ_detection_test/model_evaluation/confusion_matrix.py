import cv2
import pandas as pd
import numpy as np
import itertools
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
plt.rcParams['font.sans-serif'] = ['Simhei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False  # 显示负号

# pd.set_option('display.max_columns', None)  # 显示完整的列
# pd.set_option('display.max_rows', None)  # 显示完整的行
# np.set_printoptions(threshold=np.inf)

'''
confusion_matrix interface
'''
def generate_confusion_matrix(pred_of_test: pd.DataFrame, label_category, pred_category):
    return confusion_matrix(pred_of_test[label_category], pred_of_test[pred_category])

'''
Visualizing the confusion matrix
传入混淆矩阵和标签名称列表，绘制混淆矩阵
'''
def cnf_matrix_plotter(cm, classes, cmap=plt.cm.Blues):
    plt.figure(figsize=(10, 10))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.colorbar() # 色条
    tick_marks = np.arange(len(classes))

    plt.title('confusion matrix', fontsize=30)
    plt.xlabel('pred category', fontsize=25, c='r')
    plt.ylabel('true category', fontsize=25, c='r')
    plt.tick_params(labelsize=16)  # 设置类别文字大小
    plt.xticks(tick_marks, classes)  # 横轴文字旋转
    plt.yticks(tick_marks, classes)

    # 写数字
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > threshold else "black",
                 fontsize=12)

    plt.tight_layout()

    # plt.savefig('混淆矩阵.pdf', dpi=300)  # 保存图像
    plt.show()

'''
筛选测试集中被误判的图片
参数:测试集预测结果、正确的标签名和误判的标签名
'''
def misjudged_images_filter(pred_of_test: pd.DataFrame,true_category, pred_category):
    wrong_df = pred_of_test[(pred_of_test['标注类别名称'] == true_category) & (pred_of_test['top-1-预测名称'] == pred_category)]
    # 可视化误判图像
    for idx, row in wrong_df.iterrows():
        img_path = row['图像路径']
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        title_str = img_path + '\nTrue:' + row['标注类别名称'] + ' Pred:' + row['top-1-预测名称']
        plt.title(title_str)
        plt.show()

if __name__ == '__main__':
    idx_to_labels = np.load('../resources/form/idx_to_labels.npy', allow_pickle=True).item()
    # 获得类别名称
    classes = list(idx_to_labels.values())
    # 载入测试集预测结果表格
    df = pd.read_csv('../resources/form/测试集预测结果.csv')
    confusion_matrix = generate_confusion_matrix(df,'标注类别名称','top-1-预测名称')
    cnf_matrix_plotter(confusion_matrix,classes,cmap='Blues')
    # misjudged_images_filter(df,'unconfirmed','negative')

