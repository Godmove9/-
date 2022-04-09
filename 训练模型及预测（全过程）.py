import matplotlib.pyplot as plt
import numpy as np
import struct
def data_preprocessing(batch_size):
    #打开数据集
    train_image = open('train-images.idx3-ubyte', 'rb')
    train_label = open('train-labels.idx1-ubyte', 'rb')
    test_image = open('t10k-images.idx3-ubyte', 'rb')
    test_label = open('t10k-labels.idx1-ubyte', 'rb')

    # 读取前八个字节
    magic, n = struct.unpack('>II',train_label.read(8))  # 读取文件的前8字节
    #读取训练数据中的六万个标签
    y_train_label = np.array(np.fromfile(train_label, dtype=np.uint8), ndmin=1)
    # 将读取的标签转化为结果矩阵
    y_train = np.ones((10, 60000)) * 0.01
    for i in range(60000):
        y_train[y_train_label[i]][i] = 1
    magic_t, n_t = struct.unpack('>II',test_label.read(8))
    #读取测试数据中的标签
    y_test_label = np.array(np.fromfile(test_label,dtype=np.uint8), ndmin=1)
    # 将读取的标签转化为结果矩阵
    y_test = np.ones((10, 10000)) * 0.01
    for j in range(10000):
        y_test[y_test_label[j]][j] = 1
    magic, num, rows, cols = struct.unpack('>IIII', train_image.read(16))
    # 读取训练数据中的图片信息（矩阵形式）
    x_train = np.fromfile(train_image, dtype=np.uint8).reshape(len(y_train_label), 784).T

    magic_2, num_2, rows_2, cols_2 = struct.unpack('>IIII', test_image.read(16))
    # 读取测试数据中的图片信息（矩阵形式）
    x_test = np.fromfile(test_image, dtype=np.uint8).reshape(len(y_test_label), 784).T
    #归一化处理
    x_train = x_train / 255 * 0.99 + 0.01
    x_test = x_test / 255 * 0.99 + 0.01

    #将训练集分批，SGD预处理
    xtrain_batches = np.array([x_train[:, i:i + batch_size] for i in range(0, x_train.shape[1], batch_size)])
    ytrain_batches = np.array([y_train[:, i:i + batch_size] for i in range(0, y_train.shape[1], batch_size)])
    ytrain_label_batches = np.array([y_train_label[i:(i + batch_size)] for i in range(0, y_train.shape[1], batch_size)])
    #计算批次数
    batches_num = xtrain_batches.shape[0]

    # 关闭打开的文件
    train_image.close()
    train_label.close()
    test_image.close()
    test_label.close()

    return x_train, y_train, x_test, y_test,y_test_label, y_train_label, xtrain_batches,ytrain_batches,ytrain_label_batches,batches_num

class Nerual_Network(object):
    # 定义各参数
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate, decayrate, regularization,batch_size):
        """
        :param inputnodes: 输入层结点数
        :param hiddennodes: 隐藏层结点数
        :param outputnodes: 输出层结点数
        :param learningrate: 学习率
        :param decayrate: 学习衰减率
        :param regularization: 正则化强度
        """
        # 设定结点数和学习率
        self.inputnodes = inputnodes
        self.hiddennodes = hiddennodes
        self.outputnodes = outputnodes
        self.learningrate = learningrate
        self.decayrate = decayrate
        self.regularization = regularization
        self.batch_size=batch_size
        # 设定权重和常数初始值
        self.w1 = np.random.randn(self.hiddennodes, self.inputnodes) * 0.01
        self.w2 = np.random.randn(self.outputnodes, self.hiddennodes) * 0.01
        self.b1 = np.zeros((self.hiddennodes, 100))
        self.b2 = np.zeros((10, 100))
        self.b1_train = np.zeros((self.hiddennodes, 60000))
        self.b2_train = np.zeros((10, 60000))
        self.b1_t = np.zeros((self.hiddennodes, 10000))
        self.b2_t = np.zeros((10, 10000))

        # 设定训练次数
        self.epoch = 50
        #设置训练和测试loss，测试accurancy列表
        self.v_accurancy = []
        self.v_loss = []
        self.t_loss =[]

    # 激活函数
    def softmax(self, z):
        a=1/(1+np.exp(-z))
        return a

    # 前向传播
    def forward_propagation(self, input_data, weight_matrix, b):
        z = np.add(np.dot(weight_matrix, input_data), b)
        return z, self.softmax(z)

    # 定义训练集损失函数
    def loss_function(self,X, Y, w1,b1, w2,b2):
        z1, a1 = self.forward_propagation(X, w1, b1)
        z2, a2 = self.forward_propagation(a1, w2, b2)
        loss1 = -np.sum(a2 * np.log(Y + 1e-7) + (1 - a2) * np.log(1 - Y + 1e-7)) / 60000
        # L2正则化
        loss2 = self.regularization * (np.sum(np.square(w1)) + np.sum(np.square(w2))) / 60000 / 2
        return loss1 + loss2

    #定义测试集损失函数
    def loss_test(self, X, Y,w1,b1, w2,b2):
        z1, a1 = self.forward_propagation(X, w1, b1)
        z2, a2 = self.forward_propagation(a1, w2, b2)
        loss1 = -np.sum(a2 * np.log(Y + 1e-7) + (1 - a2) * np.log(1 - Y + 1e-7)) / 10000
        # L2正则化
        loss2 = self.regularization * (np.sum(np.square(w1)) + np.sum(np.square(w2))) / 10000 / 2
        return loss1 + loss2

    #计算测试集准确率
    def accurancy_test(self,input_data, label,w1,b1,w2,b2):
        z1, a1 = self.forward_propagation(input_data, w1, b1)
        z2, a2 = self.forward_propagation(a1, w2, b2)
        correct_number = 0
        for item in range(10000):
            if np.argmax(a2[:, item]) == label[item]:
                correct_number += 1
        print('该两层神经网络分类器的准确率：{0}%'.format(correct_number * 100 / 10000))
        return correct_number/10000

    # 模型训练部分
    def train(self, x_batches, y_batches):
        # 将权重回归初始值,列表清零
        w2 = self.w2
        b2 = self.b2
        w1 = self.w1
        b1 = self.b1
        print('开始训练')
        for i in range(self.epoch):
            print('正在进行第%d轮训练' % i)
            learningrate = self.learningrate * self.decayrate ** ((i + 1) / self.epoch)
            for j in range(600):
                print('正在进行第%d个批次' % (j+1))
                z1, a1 = self.forward_propagation(x_batches[j], w1, b1)
                z2, a2 = self.forward_propagation(a1, w2, b2)
                # 反向传播
                dz2 = a2 - y_batches[j]
                dz1 = np.dot(w2.T, dz2) * a1 * (1 - a1)
                #计算梯度
                w2_gradient = np.add(np.dot(dz2, a1.T), self.regularization * w2)/self.batch_size
                b2_gra = np.sum(dz2, axis=1, keepdims=True)/self.batch_size
                b2_gradient = np.repeat(b2_gra,self.batch_size,axis=1)
                w1_gradient = np.add(np.dot(dz1, x_batches[j].T), self.regularization * w1)/self.batch_size
                b1_gra = np.sum(dz1, axis=1, keepdims=True)/self.batch_size
                b1_gradient = np.repeat(b1_gra,self.batch_size,axis=1)
                #更新参数
                w2 -= learningrate * w2_gradient
                w1 -= learningrate * w1_gradient
                b2 -= learningrate * b2_gradient
                b1 -= learningrate * b1_gradient
            accurancy = self.accurancy_test(x_test,y_test_label,w1,np.repeat(b1,100,axis=1),w2,np.repeat(b2,100,axis=1))
            loss_v = self.loss_function(x_train, y_train, w1,np.repeat(dl.b1,600,axis=1), w2,np.repeat(dl.b2,600,axis=1))
            loss_t = self.loss_test(x_test, y_test,w1,np.repeat(dl.b1,100,axis=1), w2,np.repeat(dl.b2,100,axis=1))
            self.v_accurancy.append(accurancy)
            self.v_loss.append(loss_v)
            self.t_loss.append(loss_t)
        self.w2 = w2
        self.b2 = b2
        self.w1 = w1
        self.b1 = b1


if __name__ == '__main__':
    # 输入层数据维度784，隐藏层150，输出层10,初始学习率0.2,学习率衰减率0.1，正则化强度0.01,每批次大小为100,进行参数查找
    #dl = Nerual_Network(784, 150, 10, 0.1,0.1,0.01,100)
    #dl = Nerual_Network(784, 200, 10, 0.15, 0.1, 0.05, 100)
    #dl = Nerual_Network(784, 100, 10, 0.2, 0.1, 0.01, 100)
    #dl = Nerual_Network(784, 100, 10, 0.2, 0.1, 0.05, 100)
    dl = Nerual_Network(784, 150, 10, 0.2, 0.1, 0.01, 100)
    #dl = Nerual_Network(784, 200, 10, 0.1, 0.1, 0.01, 100)
    #设定每批大小并预处理
    x_train, y_train, x_test, y_test,y_test_label, y_train_label, xtrain_batches,ytrain_batches,ytrain_label_batches,batches_num\
        = data_preprocessing(100)
    dl.train(xtrain_batches, ytrain_batches)
    #保存模型
    #np.save('w1.npy',dl.w1)
    #np.save('w2.npy', dl.w2)
    #np.save('b1.npy', dl.b1)
    #np.save('b2.npy', dl.b2)

    # 用训练好的模型进行预测
    dl.accurancy_test(x_test,y_test_label,dl.w1,np.repeat(dl.b1,100,axis=1),dl.w2,np.repeat(dl.b2,100,axis=1))

    #绘制测试集accurancy曲线
    # 解决中文显示问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(1)
    x = range(1,len(dl.v_accurancy)+1)
    plt.plot(x,dl.v_accurancy,'-b',label="accuracy")
    plt.legend(loc='upper left')
    plt.title("测试的accurancy曲线")
    plt.show()

    #绘制训练和测试的loss曲线
    plt.figure(2)
    x = range(1, len(dl.v_loss) + 1)
    plt.plot(x,dl.v_loss,'-b',label="训练的loss")
    plt.plot(x,dl.t_loss, '-r', label="测试的loss")
    plt.legend(loc='upper left')
    plt.title("训练和测试的loss曲线")
    plt.show()

    #可视化每层的网络参数
    plt.figure(3)
    plt.matshow(dl.w1, cmap=plt.cm.gray)
    plt.show()
    plt.figure(4)
    plt.matshow(dl.b1, cmap=plt.cm.gray)
    plt.show()
    plt.figure(5)
    plt.matshow(dl.w2, cmap=plt.cm.gray)
    plt.show()
    plt.figure(6)
    plt.matshow(dl.b2, cmap=plt.cm.gray)
    plt.show()

