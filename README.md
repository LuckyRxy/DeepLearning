项目在germ_detection_test里，
使用的是 Annotated Patches 里的图片，

对数据进行预处理，根据标注文件window_metadata.csv将图片分为三个类别negative、positive、unconfirmed存储在 quiron_category中，
之后将该数据分为测试集（20%）和训练集（80%），存储在quiron_split中。

之后选择resnet18模型进行训练和测试

绘制ROC、PR、confused_matrix
