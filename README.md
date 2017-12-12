# TrafficSignRecongnition-LeNet5
This is a method about traffic sign recognition based LeNet-5 and Tensorflow and Python3

there are two files in in the GTSRB-lenet5. One is about direct recognition method. The other is about two level lenet-5 recognition method.
These two method use Germany traffic sign dataset.

directly-recognition:
                1. train.py is the code of training and testing.
                2. showpic.py is the user interface which use the tarined model.
                3. tmp folder contains the trained model. Accuracy is about 93%. Train iteraton is 1200 .
                4. If you want to use the code and model, you need to download the dataset, and put it in this folder.
                
two-level-recognition:
                1. Coarse_classification.py is the code of Coarse_classification training and tesing.
                2. fine_classification.py is the code of fine_classification training and tesing .
                3. comnine.py is to use the trained Coarse_classification and fine_classification models to test.
                4. showpic.py is the user interface which use the tarined model.
                5. tmp folder contains the Coarse_classification trained model. Accuracy is about 98%. Train iteraton is 1200.
                6. 00001-00006 contains the fine_classification trained model. All accuracy are about 100%. Train iteraton is 1200.
                7. If you want to use the code and model, you need to download the dataset, and put it in this folder.
                
                
有两个文件夹，一个放的是直接分类的方法，一个放的是二级lenet-5的方法

直接分类文件夹：
                这个方法使用一级（一层）lenet-5
                1. train.py是training以及testing的代码，训练次数1200
                2. showpic.py是直接使用训练好的模型设计的ui
                3. tmp文件夹中放的是训练好的模型，使用德国数据集gtsrb，准确率为93%左右
                4. 使用代码与模型需要下载数据集，并放到这个目录下
二级lenet5: 
                这个方法使用二级（二层）lemet-5
                1. Coarse_classification.py是粗识别training以及testing的代码，使用德国数据集gtsrb，准确率为98%左右，训练次数1200
                2. fine_classification.py是细识别training以及testing的代码，使用德国数据集gtsrb，准确率均为100%，训练次数1200
                3. combine.py是使用粗识别以及细识别训练好的模型在gtsrdb上做测试
                4. showpic.py是直接使用训练好的模型设计的ui
                5. tmp文件夹放的是粗识别的训练好的模型，00001-00006放的是细识别训练好的模型
                6.使用代码与模型需要下载数据集，并放到这个目录下
