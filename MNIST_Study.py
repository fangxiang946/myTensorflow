import tensorflow as tf
import input_data
import os
import numpy as np


def fx():
    '''
        全连接对手写数字进行识别    最终训练集-准确率：90%    测试机-准确率：78%
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
    weigths = tf.Variable(initial_value=tf.random_normal(shape=[784, 10]), name="weigths")
    bais = tf.Variable(initial_value=tf.random_normal(shape=[10]), name="bais")
    y_predict = createfullmatmul_Model(x_data,weigths,bais)

    # 3.构建损失函数
    with tf.variable_scope("loss_function"):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_data, logits=y_predict))
    # 4.优化损失
    with tf.variable_scope("optimaizer_train"):
        optimaizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

    #5.计算准确率
    with tf.variable_scope("accuracy_train"):
        accuracy = getAccuracy(y_data,y_predict)

    # #00_1 收集变量
    s1 = tf.summary.scalar("loss", loss)
    s2 = tf.summary.scalar("accuracy", accuracy)
    s3 = tf.summary.histogram("weigths", weigths)
    s4 = tf.summary.histogram("bais", bais)
    # #00_2 合并变量
    merged = tf.summary.merge_all()

    # 创建saver对象
    saver = tf.train.Saver()

    # 显示初始化全局变量
    init = tf.global_variables_initializer()
    # 开启会话
    with tf.Session() as sess:
        sess.run(init)
        x, y = mnist.train.next_batch(10)
        print("y\n",y)
        # 00_3 创建事件文件
        #filewriter = tf.summary.FileWriter("../tmp/summary", graph=sess.graph)

        '''
           1.下面是训练+保存模型
        '''

        # for i in range(5):
        #     x, y = mnist.train.next_batch(500)
        #     mydict = {x_data: x, y_data: y}
        #     for k in range(6000):
        #         _, merged_value, loss_value, accuracy_value = sess.run([optimaizer, merged, loss, accuracy],
        #                                                                feed_dict=mydict)
        #
        #     print("训练第%d次,当前损失值为%f" % (i, loss_value))
        #     print("训练第%d次,准确率为%f" % (i,accuracy_value))
        #     # 00_4 每次迭代都将此写入事件文件中
        #     filewriter.add_summary(merged_value, i)
        #     # 保存模型 每隔100保存一次模型 防止中途断电了..
        #     saver.save(sess, "../tmp/model/myfull_model.ckpt")
        #
        #
        # '''
        #     2.下面是预测
        # '''
        # acc = sess.run(accuracy, feed_dict={x_data: mnist.test.images[0:500],y_data: mnist.test.labels[0:500]})
        # print("我的预测,准确率为%s" % acc)


        '''
            3.下面是读取模型+预测
        '''

        # mytestdict = {x_data: mnist.test.images[3000:4000],y_data: mnist.test.labels[3000:4000]}
        # # loss_value, accuracy_value = sess.run([loss,accuracy], feed_dict=mytestdict)
        # # print("未载入模型-损失值为%f" % loss_value)
        # # print("未载入模型-准确率为%f" % accuracy_value)
        # path = "../tmp/model/myfull_model.ckpt"
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

def createfullmatmul_Model(x,w,b):
    '''
        创建全连接模型
    '''
    with tf.variable_scope("create_model"):
        y_predict = tf.matmul(x, w) + b
    return y_predict

def createsoftmax_Model(x,w,b):
    '''
        创建卷积模型
    '''
    with tf.variable_scope("create_model"):
        y_predict = tf.nn.softmax(tf.matmul(x, w) + b)
    return y_predict




if __name__ == "__main__":
    fx()
