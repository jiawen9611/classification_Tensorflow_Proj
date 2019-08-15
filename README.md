# Classification Project based on Tensorflow
这是一份基于Tensorlflow的分类代码，用于deep-learning、Tensorflow的学习。
#### 相关信息
    开始日期：2019.6.9
#### 环境要求
    Tensoflow1.9.0+
    Tensorboard
    yaml
    easydict
    Some other libraries (find what you miss when running the code. hhhhh~)
#### 实现模型
    ResNet
#### 数据准备
    1.captcha_images:
    在captcha_images内新建captcha_images/images/train和val,分别用来放置训练和验证图片；
    运行datasets下的create_classification_data.py即可生成数据集，当然要对代码做简单修改；
    2.cifar10:
    直接运行训练代码即可，可以自行下载解压；
#### 预训练模型
    1.在classification_Tensorflow_Proj内新建pretrained_models放置需要的预训练模型,
    若找不到预训练模型，则下载下来放入指定的位置，模型保存在谷歌的models开源代码models/research/slim中；
#### 使用方法
    实验名在exp_configs文件夹下以文件夹名体现；
    模型输出在exp_output，在classification_Tensorflow_Proj路径下新建一个exp_output，内新建对应的实验名文件夹;
    运行方法是先在exp_configs里做好实验配置，再运行指定好的train.py文件;
#### 实现功能
    1.使用captcha生成简单的分类数据集，标签在图片名中
    2.训练模型，生成ckpt
    3.使用ckpt对单张图片进行测试
    4.训练结束直接保存为pb模型
    5.可以使用训练时直接保存的pb模型对单张图片进行测试
    6.把ckpt转化为pb模型,并对该pb做测试
    7.添加resnet系列模型，可以使用预训练模型
    8.record数据集的制作与读取
#### 有待完成
    模型训练的同时进行验证；
#### 相关说明
    1.这份代码的数据预处理过程是放在网络结构中的，和pt的版本有所不同
