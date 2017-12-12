import os
import random
import skimage.data
import skimage.transform
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time
from datetime import timedelta
import math


'''
    http://www.jianshu.com/p/3c7f329b29ee
'''

#images_ph = tf.placeholder(tf.float32, [None, 32, 32, 1]) channel 3 改成 1
n_classes = 8

# 运行图形嵌入到notebook中
#%matplotlib inline

#Training目录下包含了名字从00000到00061连续编号的子目录。这些名字代表了标签是从0到61编号，每个目录下的交通标志图片就是属于该标签的样本。这些图片是用一种古老的格式.ppm来存储的，幸运的是，Scikit Image库支持这个格式。

def load_data(data_dir):
    """Loads a data set and returns two lists:
        
        images: a list of Numpy arrays, each representing an image.
        labels: a list of numbers that represent the images labels.
        """
    #获取data_dir的所有子目录。 每个代表一个标签。
    directories = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    #循环标签目录并收集数据
    #两个列表 标签和图像。
    labels = []#labels列表是标签，值为0到6的整数。
    images = []#images列表包含了一组图像，每个图像都转换为numpy数组。
    address = []#返回图片路径
    coarse_class = []# 标签 但label是1-6， 这个是00001-00006，这个的作用是加载不同的大类对应的训练后的模型
    
    for d in directories:
        directories2 = [d2 for d2 in os.listdir(os.path.join(data_dir, d)) if os.path.isdir(os.path.join(data_dir, d, d2))]
        #print()
        #print("d is",d)
        #print()
        #print("directories2 is",directories2)
        for d2 in directories2:
            label_dir = os.path.join(data_dir, d, d2)
            file_names = [os.path.join(label_dir, f)
                          for f in os.listdir(label_dir) if f.endswith(".ppm")]
            #对于每个标签，加载它的图像并将它们添加到图像列表中。
            #将标签号（即目录名）添加到标签列表中。
            for f in file_names:
                images.append(skimage.data.imread(f))
                labels.append(int(d))#int(d)
                coarse_class.append(d)# d
                address.append(f)
    return images, labels, address, coarse_class


#Load training and testing datasets.
test_data_dir = os.path.join("./Test_sorted/Images")


#Model Architecture
### Define your architecture here.
### Feel free to use as many code cells as needed.
from tensorflow.contrib.layers import flatten
global_conv2 = tf.zeros((1,5,5,16))
def LeNet(x):
    global global_conv2
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    # SOLUTION: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
    
    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1)
    
    # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    # SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    
    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2)
    
    # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    global_conv2 = conv2
    
    # SOLUTION: Flatten. Input = 5x5x16. Output = 400.
    fc0   = flatten(conv2)
    
    # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    
    # SOLUTION: Activation.
    fc1    = tf.nn.relu(fc1)
    
    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(84))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    
    # SOLUTION: Activation.
    fc2    = tf.nn.relu(fc2)
    
    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = n_classes （43 here, number of classes).
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, n_classes), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(n_classes))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
   
    return logits



#LeNet-5 网络结构
def fine_LeNet(x, n_classes):
    global global_conv2
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    # SOLUTION: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
    
    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1)
    
    # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    # SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    
    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2)
    
    # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    global_conv2 = conv2
    
    # SOLUTION: Flatten. Input = 5x5x16. Output = 400.
    fc0   = flatten(conv2)
    
    # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    
    # SOLUTION: Activation.
    fc1    = tf.nn.relu(fc1)
    
    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(84))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    
    # SOLUTION: Activation.
    fc2    = tf.nn.relu(fc2)
    
    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = n_classes （43 here, number of classes).
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, n_classes), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(n_classes))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
  
    return logits


#网络构建以及训练， 测试， d代表大类的数字，00001-00006
def Fine_classification_Test(fine_labels, fine_address, coarse_predicted, truth_predict):
    # Load the test dataset.
    #变量赋值，懒得去一个一个修改变量名
    d = coarse_predicted
    
    #图片的类别的数量
    n_classes = 43
    
    
    # Create a graph to hold the model. 创建Graphic
    #首先，我先创建一个Graph对象。TensorFlow有一个默认的全局图，但是我不建议使用它。设置全局变量通常是一个很坏的习惯，因为它太容易引入错误了。我更倾向于自己明确地创建一个图。
    graph = tf.Graph()
    
    # Create model in the graph.
    with graph.as_default():
        
        #images_ph的占位符形式大概是[None, 32, 32, 3]，这个分别表示[批次，高，宽，通道]
        #批次是None表示批次是灵活的，这就意味着我们可以在不改变代码的情况下修改批次。
        #请注意参数的顺序，因为在像NCHW这样的模型中，参数的顺序是不同的。
        #接下来，我定义了全连接层。与往常不同，我没有实现y = xW + b等式，我使用了一个简单的非线性函数来实现激活函数的功能。我期望输入是个1维的数组，所以首先我将图片平整化。
        #ReLU 作为激活函数。
        #所有负数的函数值都是0，这样对于分类任务和训练速度来说都会强于sigmoid和tanh函数
        
        # 设置占位符用来放置图片和标签，占位符石tensorflow从主程序中接受输入的方式
        # 在graph.as_default() 中创建的占位符（和其他所有操作）， 这样的好处是他们成为了创建图的一部分，而不是在全局图中
        #参数 images_ph 的维度是 [None, 32, 32, 3]，这四个参数分别表示 [批量大小，高度，宽度，通道] （通常缩写为 NHWC）。批处理大小用 None 表示，意味着批处理大小是灵活的，也就是说，我们可以向模型中导入任意批量大小的数据，而不用去修改代码。注意你输入数据的顺序，因为在一些模型和框架下面可以使用不同的排序，比如 NCHW。
        # Placeholders for inputs and labels.
        images_ph = tf.placeholder(tf.float32, [None, 32, 32, 3])
        labels_ph = tf.placeholder(tf.int32, [None])
        
        #接下来，我定义一个全连接层，而不是实现原始方程 y = xW + b。在这一行中，我使用一个方便的函数，并且使用激活函数。模型的输入时一个一维向量，所以我先要压平图片。
        # Flatten input from: [None, height, width, channels]
        # To: [None, height * width * channels] == [None, 3072]
        images_flat = tf.contrib.layers.flatten(images_ph)
        
        #全连接层的输出是一个长度是62的对数矢量（从技术上分析，它的输出维度应该是 [None, 62]，因为我们是一批一批处理的）。
        #输出的数据可能看起来是这样的：[0.3, 0, 0, 1.2, 2.1, 0.01, 0.4, ... ..., 0, 0]。值越高，图片越可能表示该标签。
        # Fully connected layer.
        # Generates logits of size [None, 62]
        # logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)
        logits = fine_LeNet(images_ph, n_classes)
        
        #在这个项目中，我们只需要知道最大值所对应的索引就行了，因为这个索引代表着图片的分类标签，这个求解最大操作可以如下表示：
        #argmax 函数的输出结果将是一个整数，范围是 [0, 61]。
        # Convert logits to label indexes (int).
        # Shape [None], which is a 1D vector of length == batch_size.
        predicted_labels = tf.argmax(logits, 1)
        
        #我们需要将标签和神经网络的输出转换成概率向量。TensorFlow中有一个 sparse_softmax_cross_entropy_with_logits 函数可以实现这个操作。这个函数将标签和神经网络的输出作为输入参数，并且做三件事：第一，将标签的维度转换为 [None, 62]（这是一个0-1向量）；第二，利用softmax函数将标签数据和神经网络输出结果转换成概率值；第三，计算两者之间的交叉熵。这个函数将会返回一个维度是 [None] 的向量（向量长度是批处理大小），然后我们通过 reduce_mean 函数来获得一个值，表示最终的损失值。
        # Define the loss function.
        # Cross-entropy is a good choice for classification.
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels_ph))
        
        #下一个需要处理的就是选择一个合适的优化算法。我一般都是使用 ADAM 优化算法，因为它的收敛速度比一般的梯度下降法更快。如果你想知道不同优化器之间的比较结果，
        # Create training op.
        train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
        
        #图中的最后一个节点是初始化所有的操作，它简单的将所有变量的值设置为零（或随机值）。
        # And, finally, an initialization op to execute before training.
        # TODO: rename to tf.global_variables_initializer() on TF 0.12.
        init = tf.initialize_all_variables()
        
        #模型保存加载工具
        saver = tf.train.Saver()
        
        #判断模型保存路径是否存在，不存在就创建
        if not os.path.exists(d+'/'):
            print("model file does not exist", d)
            os.mkdir(d+'/')
            print()
            print("file created", d)
            print()
        
        #这是我们迭代训练模型的地方。在我们开始训练之前，我们需要先创建一个会话（Session）对象。
        
        #初始化
        session = tf.Session(graph=graph)
        if os.path.exists(d+'/checkpoint'): #判断模型是否存在
            saver.restore(session, d+'/model.ckpt') #存在就从模型中恢复变量
            #print("model haved been loaded", d)
        else:
            _ = session.run([init])
    
        #评估
        
        fine_images = skimage.data.imread(fine_address)

        # Transform the images, just like we did with the training set.
        fine_images32 = [skimage.transform.resize(fine_images, (32, 32))]
        
        # Run predictions against the full test set.
        predicted = session.run([predicted_labels],
                                 feed_dict={images_ph: fine_images32})[0]
         
        truth = fine_labels
        prediction = predicted
        if truth != prediction:
            print("address is  ",fine_address)
        else:
            #len(truth_predict) = 0, truth_predict[0]
            #用于统计预测正确的图片的数量
            truth_predict.append(truth)
        
            print()
            print("truth_predict len is ", len(truth_predict))
            print()

        # Close the session. This will destroy the trained model.
        session.close()
    return truth_predict


# Create a graph to hold the model. 创建Graphic
#首先，我先创建一个Graph对象。TensorFlow有一个默认的全局图，但是我不建议使用它。设置全局变量通常是一个很坏的习惯，因为它太容易引入错误了。我更倾向于自己明确地创建一个图。
graph = tf.Graph()

# Create model in the graph.
with graph.as_default():
    
#images_ph的占位符形式大概是[None, 32, 32, 3]，这个分别表示[批次，高，宽，通道]
#批次是None表示批次是灵活的，这就意味着我们可以在不改变代码的情况下修改批次。
#请注意参数的顺序，因为在像NCHW这样的模型中，参数的顺序是不同的。
#接下来，我定义了全连接层。与往常不同，我没有实现y = xW + b等式，我使用了一个简单的非线性函数来实现激活函数的功能。我期望输入是个1维的数组，所以首先我将图片平整化。
#ReLU 作为激活函数。
#所有负数的函数值都是0，这样对于分类任务和训练速度来说都会强于sigmoid和tanh函数

    # 设置占位符用来放置图片和标签，占位符石tensorflow从主程序中接受输入的方式
    # 在graph.as_default() 中创建的占位符（和其他所有操作）， 这样的好处是他们成为了创建图的一部分，而不是在全局图中
    #参数 images_ph 的维度是 [None, 32, 32, 3]，这四个参数分别表示 [批量大小，高度，宽度，通道] （通常缩写为 NHWC）。批处理大小用 None 表示，意味着批处理大小是灵活的，也就是说，我们可以向模型中导入任意批量大小的数据，而不用去修改代码。注意你输入数据的顺序，因为在一些模型和框架下面可以使用不同的排序，比如 NCHW。
    # Placeholders for inputs and labels.
    images_ph = tf.placeholder(tf.float32, [None, 32, 32, 3])
    labels_ph = tf.placeholder(tf.int32, [None])
    
    #接下来，我定义一个全连接层，而不是实现原始方程 y = xW + b。在这一行中，我使用一个方便的函数，并且使用激活函数。模型的输入时一个一维向量，所以我先要压平图片。
    # Flatten input from: [None, height, width, channels]
    # To: [None, height * width * channels] == [None, 3072]
    images_flat = tf.contrib.layers.flatten(images_ph)
    
    #全连接层的输出是一个长度是62的对数矢量（从技术上分析，它的输出维度应该是 [None, 62]，因为我们是一批一批处理的）。
    #输出的数据可能看起来是这样的：[0.3, 0, 0, 1.2, 2.1, 0.01, 0.4, ... ..., 0, 0]。值越高，图片越可能表示该标签。
    # Fully connected layer.
    # Generates logits of size [None, 62]
    # logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)
    logits = LeNet(images_ph)
    
    #在这个项目中，我们只需要知道最大值所对应的索引就行了，因为这个索引代表着图片的分类标签，这个求解最大操作可以如下表示：
    #argmax 函数的输出结果将是一个整数，范围是 [0, 61]。
    # Convert logits to label indexes (int).
    # Shape [None], which is a 1D vector of length == batch_size.
    predicted_labels = tf.argmax(logits, 1)
    
    #我们需要将标签和神经网络的输出转换成概率向量。TensorFlow中有一个 sparse_softmax_cross_entropy_with_logits 函数可以实现这个操作。这个函数将标签和神经网络的输出作为输入参数，并且做三件事：第一，将标签的维度转换为 [None, 62]（这是一个0-1向量）；第二，利用softmax函数将标签数据和神经网络输出结果转换成概率值；第三，计算两者之间的交叉熵。这个函数将会返回一个维度是 [None] 的向量（向量长度是批处理大小），然后我们通过 reduce_mean 函数来获得一个值，表示最终的损失值。
    # Define the loss function.
    # Cross-entropy is a good choice for classification.
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels_ph))
    
    #下一个需要处理的就是选择一个合适的优化算法。我一般都是使用 ADAM 优化算法，因为它的收敛速度比一般的梯度下降法更快。如果你想知道不同优化器之间的比较结果，
    # Create training op.
    train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    
    #图中的最后一个节点是初始化所有的操作，它简单的将所有变量的值设置为零（或随机值）。
    # And, finally, an initialization op to execute before training.
    # TODO: rename to tf.global_variables_initializer() on TF 0.12.
    init = tf.initialize_all_variables()
    
    #模型保存加载工具
    saver = tf.train.Saver()

#请注意，上面的代码还没有执行任何操作。它只是构建图，并且描述了输入。在上面我们定义的变量，比如，init，loss和predicted_labels，它们都不包含具体的数值。它们是我们接下来要执行的操作的引用。


#判断模型保存路径是否存在，不存在就创建
if not os.path.exists('tmp/'):
    os.mkdir('tmp/')

#这是我们迭代训练模型的地方。在我们开始训练之前，我们需要先创建一个会话（Session）对象。
#开始训练


#初始化
session = tf.Session(graph=graph)
if os.path.exists('tmp/checkpoint'): #判断模型是否存在
    saver.restore(session, 'tmp/model.ckpt') #存在就从模型中恢复变量
    print("model haved been loaded")
else:
    _ = session.run([init])


#评估
# Load the test dataset.
test_images, test_labels, test_address, coarse_class = load_data(test_data_dir)

# Transform the images, just like we did with the training set.
test_images32 = [skimage.transform.resize(image, (32, 32)) for image in test_images]

# Run predictions against the full test set.
predicted = session.run([predicted_labels], feed_dict={images_ph: test_images32})[0]

a=1#记录错误的图片数量
truth_predict = []

for i in range(len(test_images32)):
    truth = test_labels[i]
    prediction = predicted[i]
    if truth != prediction:
        print("address is  ",test_address[i])
        print("a is ",a)
        a=a+1
    else:
        #粗分类识别正确的图片，其信息才会被送到细分类网络中
        #输入的是单个的图片信息而不是数组，因为不同的图片需要加载不同的模型
        truth_predict = Fine_classification_Test(test_labels[i],test_address[i],coarse_class[i], truth_predict)


# Calculate how many matches we got.
match_count = sum([int(y == y_) for y, y_ in zip(test_labels, predicted)])
coarse_accuracy = match_count / len(test_labels)
print("Coarse_Accuracy: {:.3f}".format(coarse_accuracy))

fine_accuracy = len(truth_predict) / len(test_labels)

print("Fine_Accuracy: {:.3f}".format(fine_accuracy))
# Close the session. This will destroy the trained model.
session.close()
