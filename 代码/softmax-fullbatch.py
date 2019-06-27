import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.io as scio
from sklearn.model_selection import train_test_split

def relu(inX):
    return np.maximum(0,inX)

def change_y(Table):
    #print("label:",Table)

    true = np.zeros((len(Table), classes), dtype=np.float64)
    for i in range(len(Table)):
        true[i][int(Table[i]-1)] = 1.0
    return true

def input_Data():
    Data='../Data/combile_fourth_hidden_Data/non_rate/25_20_15_10fault_1000_number_EEMD_DE-FE-BA-MFE.mat.mat'

    #Data='../Data/combile_double_hidden_Data/non_rate/30_50_5fault_1000_number.mat'
    #Data='../Data/combileData/non_rate/5_error/combile_5_fault_1000_number_EEMD_DE-FE-BA-MFE.mat'
    #Data='../Data/combile_double_hidden_Data/non_rate/30_50_5fault_1000_number_EEMD_DE-FE-BA-MFE.mat.mat'
    Data = scio.loadmat(Data)
    Data = Data['output']
    col_number=len(Data[0])
    Train, Test = train_test_split(Data, test_size=0.2,random_state=2)

    train_Data, train_Table = np.split(Train, (col_number-1,), axis=1)
    test_Data, test_Table = np.split(Test, (col_number - 1,), axis=1)
    print("train_Data.SHAPE:",train_Data.shape)
    return np.float64(train_Data),np.float64(train_Table),np.float64(test_Data),np.float64(test_Table)

if __name__ == '__main__':
    train_Data, train_Lable, test_Data, test_Lable = input_Data()


    classes = 5
    epoch_number = 12000
    batch_number=100
    each_number=400
    learning_rate=5.5
    LAN = 0.0000001
    features=len(train_Data[0])

    loss_list=[]
    accuracy_list=[]
    train_accuracy_list=[]
    Y_train = change_y(train_Lable)
    Y_test = change_y(test_Lable)

    x = tf.placeholder(tf.float64, [None, features])
    y_label = tf.placeholder(tf.float64, [None, classes])

    # w: (5, 7) b: (5,)
    w=tf.Variable(tf.zeros([features, classes],dtype=tf.float64))
    b=tf.Variable(tf.zeros([classes],dtype=tf.float64))
    y=tf.matmul(x,w)+b
    y_pred = tf.nn.softmax(logits=y)

    reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(0.1),
                                                 tf.trainable_variables())
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y,labels=y_label,
                                                            name='cross_entropy')
    loss = tf.reduce_mean(cross_entropy, name='loss')+LAN*reg

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))
    yy=0

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for epoch in range(epoch_number):
            for i in range(batch_number):
                train_xs = train_Data
                train_ys = Y_train
                _, loss_batch,train_accuracy,y_pred_temp = sess.run(
                    [optimizer, loss,accuracy,y_pred],
                    feed_dict={x: train_xs, y_label: train_ys})
            test_xs = test_Data
            test_ys = Y_test
            y_temp, w_temp, accuracy_temp, loss_temp,y_pred_temp,reg_temp = sess.run([y_pred, w, accuracy, loss,y_pred,reg],feed_dict={x: test_xs, y_label: test_ys})
            yy=y_temp
            #for i in range(10):
            #    print("第",i,"y:", yy[i + 55])
            #    print("lable:", test_Lable[i + 55])
            print(accuracy_temp)
            loss_list.append(loss_temp)
            accuracy_list.append(accuracy_temp)
    #print("y:",yy)
    #print("lable:", test_Lable)




    print(accuracy_list[-1])
    plt.figure(figsize=(14, 6))
    #plt.plot(loss_list, "-", color="r", label="loss")
    plt.plot(accuracy_list, "-", color="b", label="accuracy_list")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("The graph of loss value varing with the number of iterations")
    plt.legend()
    plt.show()

