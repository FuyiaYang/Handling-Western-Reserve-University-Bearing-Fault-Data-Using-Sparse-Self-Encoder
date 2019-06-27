import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.io as scio

#引入文件
def input_Data():
    Data='../Data/combileData/non_rate/5_error/combile_5_fault_1000_number_EEMD_DE-FE-BA-MFE.mat'
    Data = scio.loadmat(Data)
    Data = Data['output']
    print("Data:", Data.shape)
    col_number=len(Data[0])
    Data, Table = np.split(Data, (col_number-1,), axis=1)
    # 加上波形熵
    # MFE_number=MFE(Data)
    # Data=insertFactor(Data,MFE_number)
    return Data,Table

#计算波形熵
def MFE(oriFile):
    Oristaticnumber = len(oriFile)
    # RMS 均方根 ARV 整流平均值
    tempSUM = 0
    tempARV = 0
    for i in range(0, Oristaticnumber):
        tempSUM = tempSUM + np.power(oriFile[i], 2)
        tempARV = tempARV + abs(oriFile[i])
    RMS = np.sqrt(tempSUM / Oristaticnumber)
    ARV = np.sqrt(tempARV / Oristaticnumber)
    tempSUM = 0
    tempARV = 0
    # waveFactor波形因素
    waveFactor = RMS / ARV
    # WFE 波形熵
    WFE = waveFactor * math.log(waveFactor)
    print("WFE:", WFE[0])
    return WFE

#向数据File中插入新的一个定量维度number
def insertFactor(File,number):
    Oristaticnumber = len(File[0])
    File_RPM = []
    for i in range(Oristaticnumber):
        File_RPM.append(number)
    File = np.vstack((File, File_RPM))
    return File

def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2

def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2

if __name__ == '__main__':
    Data, Table=input_Data()
    learning_rate = 0.001
    training_epochs = 500
    batch_size = 5
    display_step = 1
    feature_size = len(Data[0])
    LAN = 0.001
    BETA=0.01
    p=0.1
    loss_list=[]

    X = tf.placeholder("float", [None, feature_size])

    n_hidden_1 = 15
    n_hidden_2 = 5

    weights = {
        'encoder_h1': tf.Variable(tf.truncated_normal([feature_size, n_hidden_1], )),
        'encoder_h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2], )),
        'decoder_h1': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_1], )),
        'decoder_h2': tf.Variable(tf.truncated_normal([n_hidden_1, feature_size], )),
    }

    biases = {
        'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'decoder_b2': tf.Variable(tf.random_normal([feature_size])),
    }

    encoder_op = encoder(X)
    decoder_op = decoder(encoder_op)

    y_pred = decoder_op
    y_true = X

    reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(0.001), tf.trainable_variables())
    pj = tf.reduce_mean(encoder_op, 1)
    sparse_cost = tf.reduce_sum(p*tf.log(p/pj)+(1-p)*tf.log((1-p)/(1-pj)))
    cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2)) + LAN * reg + BETA * sparse_cost
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)



    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for epoch in range(training_epochs):
            for i in range(batch_size):
                _, c = sess.run([optimizer, cost], feed_dict={X: Data})
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c))
            loss_list.append(c)
            File_hidden,reg_temp,sparse_cost_temp=sess.run([encoder_op,reg,sparse_cost], feed_dict={X: Data})
            print("      reg:",reg_temp,"sparse_cost:",sparse_cost_temp)
        print("File_hidden:",File_hidden)
        print("Optimization Finished!")
    print("Data:", Data)
    File_hidden = np.hstack((File_hidden, Table))
    #scio.savemat('../Data/combileData/non_rate/5_error/combile_5_fault_1000_number_EEMD_DE-FE-BA-MFE.mat', {'output': File_hidden})
    scio.savemat('../Data/combile_double_hidden_Data/non_rate/25_10_5fault_1000_number_EEMD_DE-FE-BA-MFE.mat.mat',{'output': File_hidden})
    plt.figure(figsize=(18, 6))
    plt.plot(loss_list, "-", color="r", label="loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("The graph of loss value varing with the number of iterations")
    plt.legend()
    plt.show()