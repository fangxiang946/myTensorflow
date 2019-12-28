import tensorflow as tf
import input_data
import os
import numpy as np


def fx():
    '''
        全连接对手写数字进行识别
        1.准备数据
        2.构造模型
        3.构建损失函数
        4.优化损失
    '''
    # 1.准备数据
    with tf.variable_scope("prepare_data"):
        mnist = input_data.read_data_sets('../MNIST_data', one_hot=True)
        # 计算一共有多少个批次
        n_batch = mnist.train.num_examples

        x_data = tf.placeholder(name="x_data",dtype=tf.float32, shape=(None, 784))
        y_data = tf.placeholder(name="y_data",dtype=tf.float32, shape=(None, 10))

    # 2.构造模型
    y_predict = createCnn_Model(x_data)

    # 3.构建损失函数
    with tf.variable_scope("loss_function"):
        #loss = -tf.reduce_sum(y_data * tf.log(y_predict))
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_data, logits=y_predict))
    # 4.优化损失
    with tf.variable_scope("optimaizer_train"):
        optimaizer = tf.train.AdamOptimizer(0.01).minimize(loss)

    #5.计算准确率
    with tf.variable_scope("accuracy_train"):
        accuracy = getAccuracy(y_data,y_predict)

    # # #00_1 收集变量
    s1 = tf.summary.scalar("loss", loss)
    s2 = tf.summary.scalar("accuracy", accuracy)
    # # #00_2 合并变量
    merged = tf.summary.merge_all()

    # 创建saver对象
    saver = tf.train.Saver()

    # 显示初始化全局变量
    init = tf.global_variables_initializer()
    # 开启会话
    with tf.Session() as sess:
        sess.run(init)

        # 00_3 创建事件文件
        filewriter = tf.summary.FileWriter("../tmp/summary", graph=sess.graph)

        '''
           1.下面是训练+保存模型
        '''

        for i in range(2000):
            x, y = mnist.train.next_batch(500)
            mydict = {x_data: x, y_data: y}
            _, loss_value, accuracy_value, merged_value = sess.run([optimaizer, loss, accuracy, merged],
                                                                   feed_dict=mydict)
            if i % 200 == 0:
                print("训练第%d次,当前损失值为%f" % (i+1, loss_value))
                print("训练第%d次,准确率为%f" % (i+1,accuracy_value))
                #00_4 每次迭代都将此写入事件文件中
                filewriter.add_summary(merged_value, i)
                #保存模型 每隔100保存一次模型 防止中途断电了..
                saver.save(sess, "../tmp/model/mycnn_model.ckpt")


        '''
            2.下面是预测
        '''
        acc = sess.run(accuracy, feed_dict={x_data: mnist.test.images[0:500],y_data: mnist.test.labels[0:500]})
        print("我的预测,准确率为%s" % acc)


        '''
            3.下面是读取模型+预测
        '''

        # mytestdict = {x_data: mnist.test.images[2000:3000],y_data: mnist.test.labels[2000:3000]}
        # # loss_value, accuracy_value = sess.run([loss,accuracy], feed_dict=mytestdict)
        # # print("未载入模型-损失值为%f" % loss_value)
        # # print("未载入模型-准确率为%f" % accuracy_value)
        # path = "../tmp/model/mycnn_model.ckpt"
        # saver.restore(sess, path)
        # loss_value, accuracy_value,mypredict = sess.run([loss, accuracy, y_predict], feed_dict=mytestdict)
        # print("载入模型后-损失值为%f" % loss_value)
        # print("载入模型后-准确率为%f" % accuracy_value)
        # num = 8 #想对比第几个数字
        # print("我的预测值", np.argmax(mypredict[num]))
        # print("真实值", np.argmax(mnist.test.labels[2000 + num]))




    return None

def getAccuracy(y_true,y_predict):
    '''
        计算准确率
        1.比较输出结果所在位置与真实值所在位置是否一致
        2.求平均值
    '''
    # 拿到true、false列表
    equal_list = tf.equal(tf.argmax(y_true,1),tf.argmax(y_predict, 1))
    #将bool列表转成0/1列表
    accuracy = tf.reduce_mean(tf.cast(equal_list,tf.float32))
    return accuracy

def createRandomByShape(shape):
    return tf.Variable(initial_value=tf.random_normal(shape=shape))

def createCnn_Model(x):
    '''
        构建我自己的卷积神经网络
        结构 ：
            第一个卷积大层
                卷积层  filiter=5*5  步长=1   零填充=same  个数=32
                激活层  Relu
                池化层  filiter=2*2  步长=2
            第二个卷积大层：
                卷积层  filiter=5*5  步长=1   零填充=same  个数=64
                激活层  Relu
                池化层  filiter=2*2  步长=2
            全连接层
    '''
    y_predict = 0
    #1) 第一个卷积大层
    with tf.variable_scope("conv1"):
        #卷积层
        #将x[None,784]转成[None,长,宽,通道]
        input_x = tf.reshape(x, shape=[-1, 28, 28, 1])
        conv1_w = createRandomByShape(shape=[5,5,1,32])
        conv1_b = createRandomByShape(shape=[32])
        conv1_x = tf.nn.conv2d(input_x, filter=conv1_w, strides=[1, 1, 1, 1], padding="SAME") + conv1_b

        #激活层
        relu1_x = tf.nn.relu(conv1_x)

        #池化层
        pool1_x = tf.nn.max_pool(value=relu1_x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # 2) 第二个卷积大层
    with tf.variable_scope("conv2"):
        # 卷积层
        conv2_w = createRandomByShape(shape=[5, 5, 32, 64])
        conv2_b = createRandomByShape(shape=[64])
        conv2_x = tf.nn.conv2d(pool1_x, filter=conv2_w, strides=[1, 1, 1, 1], padding="SAME") + conv2_b

        # 激活层
        relu2_x = tf.nn.relu(conv2_x)

        # 池化层
        pool2_x = tf.nn.max_pool(value=relu2_x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # 3) 全连接层
    with tf.variable_scope("full_connection"):
        # [None,7,7,64]->[None,7*7*64]
        # [None,7*7*64]*[?,?]=[None,10]
        # 得出[7*7*64,10]
        x_3 = tf.reshape(pool2_x, shape=[-1, 7 * 7 * 64])
        weights_3 = createRandomByShape(shape=[7 * 7 * 64, 10])
        bais_3 = createRandomByShape(shape=[10])
        y_predict = tf.matmul(x_3, weights_3) + bais_3

    return y_predict

if __name__ == "__main__":
    fx()
