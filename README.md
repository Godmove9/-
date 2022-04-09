下载MINIST数据集，将四个数据文件放入项目中
打开训练模型及预测（全过程）.py
直接运行整个文件，其中前面为预处理函数及模型的定义，主程序部分dl = Nerual_Network()为导入模型，data_preprocessing(100)为数据预处理，dl.train(xtrain_batches, ytrain_batches)为最主要的训练函数，dl.accurancy_test()为输出分类精度的函数，之后均为绘制曲线图及可视化参数的代码
或者利用已经训练好的模型进行测试，百度云与github上均有保存好的模型的压缩包，里面有4个npy文件，分别对应w1，w2,b1,b2
打开读取保存的模型参数进行测试（无需训练）.py，直接运行整个文件，np.load读取npy文件获得训练好的权重，省去了最耗时的dl.train(xtrain_batches, ytrain_batches)训练过程
