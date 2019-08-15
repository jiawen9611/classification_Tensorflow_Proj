# Classification Project based on Tensorflow
#### 相关信息
    作者：DUT-jiawen
    开始日期：2019.6.9

#### 使用方法
    运行方法是先在exp_configs里做好实验配置，再运行指定好的train.py文件。
    
#### 实现功能
    1.使用captcha生成简单的分类数据集，标签在图片名中
    2.训练，生成ckpt
    3.使用ckpt对单张图片进行测试
    4.训练结束直接保存为pb模型
    5.可以使用训练时直接保存的pb模型对单张图片进行测试
    6.把ckpt转化为pb模型,并对该pb做测试
    7.添加resnet系列模型，可以使用预训练模型
    8.record数据集的制作与读取
    
#### 相关说明
    1.这份代码的数据预处理过程是放在网络结构中的，和pt的版本有所不同