import tensorflow as tf
import pandas as pd
import glob

'''
    使用面向对象实现了对cifar数据集的多分类问题 
    数据源下载地址：http://www.cs.toronto.edu/~kriz/cifar.html
    分别写了全连接和构建自己简单卷积神经网络层
    各参数都还没有细调，6W张图片，使得当前模型准确率较低：40%不到
'''

class Cifar(object):
    def __init__(self):
        self.height = 32
        self.width = 32
        self.channels = 3

        self.image_bytes = self.height * self.width * self.channels
        self.label_bytes = 1
        self.all_bytes = self.image_bytes + self.label_bytes
    '''
        全连接模式
    '''
    def fullconnection_train(self,filenames):
        '''
            queue runner
            1.构建文件名队列
            2.读取+解码
            3.批处理
        '''
        filenames_queue = tf.train.string_input_producer(filenames)
        print("filenames_queue\n", filenames_queue)
        reader = tf.FixedLengthRecordReader(self.all_bytes)#二进制读取器一定要指定长度
        key, value = reader.read(filenames_queue)

        decoded = tf.decode_raw(value,tf.uint8)
        print("decoded:\n", decoded)
        #数据处理一下：将目标值和特征值拆分
        label = tf.slice(decoded, [0], [self.label_bytes])
        image = tf.slice(decoded, [self.label_bytes], [self.image_bytes])
        print("label:\n",label)
        print("image:\n", image)
        # #调整图片大小
        image_reshape = tf.reshape(image, shape=[self.channels, self.height, self.width])
        print("image_reshape:\n", image_reshape[0])
        # 转置一下
        image_T = tf.transpose(image_reshape, [1, 2, 0])
        print("image_T:\n", image_T)
        # 调整图片类型
        image_cast = tf.cast(image_T,tf.float32)
        print("image_cast:\n", image_cast)

        label_batch, image_batch = tf.train.batch([label, image_cast], batch_size=100, num_threads=1, capacity=200)

        #注意!!这里需要将真实值转化成one-hot编码进行下面步骤
        x=tf.reshape(image_batch,shape=[-1,self.image_bytes])

        y_data =tf.reshape( tf.one_hot(label_batch,depth=10),shape=[-1,10])
        print("y_data\n",y_data)
        # A.构造模型
        weigths = tf.Variable(initial_value=tf.random_normal(shape=[self.image_bytes, 10]), name="weigths")
        bais = tf.Variable(initial_value=tf.random_normal(shape=[10]), name="bais")
        y_predict = tf.matmul(x, weigths) + bais
        print("y_predict\n", y_predict)
        # B.构建损失函数
        with tf.variable_scope("loss_function"):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_data, logits=y_predict))
        # C.优化损失
        with tf.variable_scope("optimaizer_train"):
            optimaizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

        # D.模型评估
        with tf.variable_scope("model_doctor"):
            accuracy = getAccuracy(y_data,y_predict)


        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            coordinator = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess,coord=coordinator)

            #label_new,image_new,image_reshape_new = sess.run([label,image,image_reshape])
            # print("label_new\n", label_new)
            # print("image_new\n", image_new)
            # print("image_reshape_new\n", image_reshape_new)

            # label_batch_new, y_data_new,image_batch_new = sess.run([label_batch,y_data, image_batch])
            # print("label_batch_new\n", label_batch_new)
            # print("y_data_new\n", y_data_new)

            for i in range(6000):
                _, loss_new, accuracy_new = sess.run([optimaizer, loss, accuracy])
                if i % 600==0:
                    print("第%d次训练后,损失值为%f,准确率为%f"% (i,loss_new,accuracy_new))

            coordinator.request_stop()
            coordinator.join(threads)

    '''
       卷积神经 
    '''
    def cnn_train(self,filenames):
        '''
            queue runner
            1.构建文件名队列
            2.读取+解码
            3.批处理
        '''
        filenames_queue = tf.train.string_input_producer(filenames)
        print("filenames_queue\n", filenames_queue)
        reader = tf.FixedLengthRecordReader(self.all_bytes)#二进制读取器一定要指定长度
        key, value = reader.read(filenames_queue)

        decoded = tf.decode_raw(value,tf.uint8)
        print("decoded:\n", decoded)
        #数据处理一下：将目标值和特征值拆分
        label = tf.slice(decoded, [0], [self.label_bytes])
        image = tf.slice(decoded, [self.label_bytes], [self.image_bytes])
        print("label:\n",label)
        print("image:\n", image)
        # #调整图片大小
        image_reshape = tf.reshape(image, shape=[self.channels, self.height, self.width])
        print("image_reshape:\n", image_reshape[0])
        # 转置一下
        image_T = tf.transpose(image_reshape, [1, 2, 0])
        print("image_T:\n", image_T)
        # 调整图片类型
        image_cast = tf.cast(image_T,tf.float32)
        print("image_cast:\n", image_cast)

        label_batch, image_batch = tf.train.batch([label, image_cast], batch_size=100, num_threads=1, capacity=200)

        #注意!!这里需要将真实值转化成one-hot编码进行下面步骤
        x=tf.reshape(image_batch,shape=[-1,self.image_bytes])
        print("xn", x)
        y_data =tf.reshape( tf.one_hot(label_batch,depth=10),shape=[-1,10])
        print("y_data\n",y_data)
        # A.构造模型
        y_predict = createCnn_Model(x)
        print("y_predict\n", y_predict)
        # B.构建损失函数
        with tf.variable_scope("loss_function"):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_data, logits=y_predict))
        # C.优化损失
        with tf.variable_scope("optimaizer_train"):
            optimaizer = tf.train.AdamOptimizer(0.01).minimize(loss)

        # D.模型评估
        with tf.variable_scope("model_doctor"):
            accuracy = getAccuracy(y_data,y_predict)


        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            coordinator = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess,coord=coordinator)

            #label_new,image_new,image_reshape_new = sess.run([label,image,image_reshape])
            # print("label_new\n", label_new)
            # print("image_new\n", image_new)
            # print("image_reshape_new\n", image_reshape_new)

            # label_batch_new, y_data_new,image_batch_new = sess.run([label_batch,y_data, image_batch])
            # print("label_batch_new\n", label_batch_new)
            # print("y_data_new\n", y_data_new)

            for i in range(6000):
                _, loss_new, accuracy_new = sess.run([optimaizer, loss, accuracy])
                if i % 600==0:
                    print("第%d次训练后,损失值为%f,准确率为%f"% (i,loss_new,accuracy_new))

            coordinator.request_stop()
            coordinator.join(threads)

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
        input_x = tf.reshape(x, shape=[-1, 32, 32, 3])
        conv1_w = createRandomByShape(shape=[5,5,3,32])
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
        # [None,32,32,3]->[None,16*16*32]
        # [None,8*8*64]*[?,?]=[None,10]
        # 得出[8*8*64,10]
        x_3 = tf.reshape(pool2_x, shape=[-1, 8*8*64])
        weights_3 = createRandomByShape(shape=[8*8*64, 10])
        bais_3 = createRandomByShape(shape=[10])
        y_predict = tf.matmul(x_3, weights_3) + bais_3

    return y_predict

def createRandomByShape(shape):
    return tf.Variable(initial_value=tf.random_normal(shape=shape))

if __name__ == "__main__":
    path = "../cifar-10-batches-bin/*.bin"
    filenames = glob.glob(path)
    print(filenames[:1])

    cifar = Cifar()
    cifar.fullconnection_train(filenames=filenames[:1])



