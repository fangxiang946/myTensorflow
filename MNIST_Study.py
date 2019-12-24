import tensorflow as tf
import input_data
import os


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
        x_data = tf.placeholder(name="x_data",dtype=tf.float32, shape=(None, 784))
        y_data = tf.placeholder(name="y_data",dtype=tf.float32, shape=(None, 10))

    # 2.构造模型
    with tf.variable_scope("create_model"):
        weigths = tf.Variable(initial_value=tf.random_normal(shape=[784, 10]),name="weigths")
        bais = tf.Variable(initial_value=tf.random_normal(shape=[10]),name="bais")
        y_predict = tf.matmul(x_data, weigths) + bais

    # 3.构建损失函数
    with tf.variable_scope("loss_function"):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_data, logits=y_predict))
    # 4.优化损失
    with tf.variable_scope("optimaizer_train"):
        optimaizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

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
        x, y = mnist.train.next_batch(100)
        print("x的形状\n",x.shape)
        print("y的形状\n", y.shape)
        mydict = {x_data: x, y_data: y}
        loss_value = sess.run(loss, feed_dict=mydict)
        print("训练前损失值为%f" % loss_value)

        # 00_3 创建事件文件
        filewriter = tf.summary.FileWriter("../tmp/summary", graph=sess.graph)

        #开始训练
        # for i in range(1000):
        #     _,loss_value,accuracy_value,merged_value = sess.run([optimaizer,loss,accuracy,merged],feed_dict=mydict)
        #
        #     #00_4 每次迭代都将此写入事件文件中
        #     filewriter.add_summary(merged_value,i)
        #
        #     if i%100 == 0:
        #         print("训练第%d次,当前损失值为%f" % (i+1, loss_value))
        #         print("准确率为%f" % accuracy_value)
        #         #保存模型 每隔100保存一次模型 防止中途断电了..
        #         saver.save(sess,"../tmp/model/my_Classfilier.ckpt")
        path = "./tmp/model/my_Classfilier.ckpt"
        print(os.path.exists(path))
        if os.path.exists(path):
            print("准确率为%f" % weigths.eval())

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

def fang():
    # 创建saver对象
    saver = tf.train.Saver()
    with tf.Session() as sess:
        path = "../tmp/model/my_Classfilier.ckpt"
        print(os.path.exists(path))
        if os.path.exists(path):
            saver.restore(sess, path)
            print("准确率为%f" % loss.eval())

if __name__ == "__main__":
    fx()
