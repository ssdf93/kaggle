# encoding=utf-8
# This Program is written by Victor Zhang at 2016-07-18 15:53:25
#
#
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score

isSample = False


def to_categorical(y, nb_classes=None):
    '''Convert class vector (integers from 0 to nb_classes)
    to binary class matrix, for use with categorical_crossentropy.
    '''
    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.
    return Y


def load_data():
    if isSample:
        train = pd.read_csv('data/train_min.csv', header=0)
        test = pd.read_csv('data/test_min.csv', header=0)
    else:
        train = pd.read_csv('data/train.csv', header=0)
        test = pd.read_csv('data/test.csv', header=0)
    print(test.head())
    return train, test


def build_model(train, test):
    input_size = 28*28
    print(input_size)
    hidden_size = 200
    output_size = 10
    train_x=train.ix[:,1:].values.astype(np.float32)
    train_y=train.ix[:,0].values
    train_y=to_categorical(train_y,10).astype(np.float32)
    print(train_x.dtype)
    print(train_x.shape)
    print(train_y.shape)

    inputs_data = tf.placeholder(tf.float32,[None,input_size],name='x_input')
    y_data = tf.placeholder(tf.float32,[None,10],name='y_label')

    with tf.name_scope('layer1'):
        Weights = tf.Variable(tf.truncated_normal([input_size, hidden_size]),name='W1')
        biases = tf.Variable(tf.zeros([hidden_size])+0.5,name='b1')
        hidden1 = tf.nn.relu(tf.matmul(inputs_data,Weights))
        # hidden1=tf.nn.dropout(hidden1,keep_prob=0.6)


    with tf.name_scope('layer2'):
        Weights2 = tf.Variable(tf.truncated_normal([hidden_size, output_size]),name='W2')
        biases2 = tf.Variable(tf.zeros([output_size])+0.1,name='b2')
        tmp=tf.matmul(hidden1,Weights2)+biases2
        prediction = tf.nn.softmax(tmp)



    with tf.name_scope('loss_fun'):
        # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y_data))
        lg=tf.log(prediction)
        loss = -tf.reduce_sum(y_data*lg)


        # correct_prediction = tf.equal(tf.argmax(prediction,1), y_data)
        # y_pred=tf.argmax(prediction,1)
        # accuracy=tf.reduce_mean(correct_prediction, tf.float32)


    train_epoch = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    n_sample=train_x.shape[0]

    s=tf.nn.softmax(tf.constant([[1.,2.,3.],[4.,5.,5.]]))
    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)
        print(sess.run(s))
        for i in range(100):
            sess.run(train_epoch,feed_dict={inputs_data:train_x,y_data:train_y})
            # for j in range(1000):
            #     mini_batch=np.random.randint(0,n_sample,100)
            #     sess.run(train_epoch,feed_dict={inputs_data:train_x[mini_batch,:],y_data:train_y[mini_batch]})
            if i%10==0:
                train_loss=sess.run(loss,feed_dict={inputs_data:train_x,y_data:train_y})
                # accuracy_s=sess.run(accuracy,feed_dict={inputs_data:train_x,y_data:train_y})
                pred_y_1=sess.run(tmp,feed_dict={inputs_data:train_x,y_data:train_y})
                pred_y_2=sess.run(prediction,feed_dict={inputs_data:train_x,y_data:train_y})
                # pred_y_2=sess.run(tmp,feed_dict={inputs_data:train_x,y_data:train_y})
                print(pred_y_1[:5,],pred_y_2[:5,])
                # print(pred_y_1,train_y)
                # ans=accuracy_score(train_y,pred_y_1)
                print(train_loss)


if __name__ == '__main__':
    train,test=load_data()
    build_model(train,test)
