import os
import sys
import math
import time

import tensorflow as tf
import numpy as np
from scipy import ndimage
import random
from pandas import DataFrame
os.environ["CUDA_VISIBLE_DEVICES"]="4"

BATCH_SIZE = 10
IMAGE_SIZE = 256
NUM_CHANNEL = 1
NUM_LABELS = 2
NUM_ITER = 200000
NUM_SHOWTRAIN = 400 #show result eveary epoch 
NUM_SHOWTEST = 20000


LEARNING_RATE =0.001
LEARNING_RATE_DECAY = 0.1
MOMENTUM = 0.9
decay_step = 20000
is_train = tf.placeholder(tf.bool,name='is_train')

path1 = '/data1/liugq/BOSSbase/BOSS_base_256x256/train'
path2 = '/data1/liugq/BOSSbase/BOSS_base_256x256/test'
path3 = '/data1/liugq/BOSSbase/BOSS_base_suniward40_256x256/train'
path4 = '/data1/liugq/BOSSbase/BOSS_base_suniward40_256x256/test'

fileList1 = []
for (dirpath1,dirnames1,filenames1) in os.walk(path1):  #0~5000 for training  5001~10000 for testing
    fileList1 = filenames1

fileList2 = []
for (dirpath2,dirnames2,filenames2) in os.walk(path2):  #0~5000 for training  5001~10000 for testing
    fileList2 = filenames2

def weight_variable(shape,n_layer):
    w_name = 'w%s' %n_layer
    initial = tf.random_normal(shape,mean=0.0,stddev=0.01)
    return tf.Variable(initial, name=w_name)

def batch_norm(input,n_out,n_layer):
    bn_name = 'bn%s' % n_layer
    #    moments_name = 'moments%s' % n_layer
    #    beta_name = 'bata%s' % n_layer
    #    gamma_name = 'gamma%s' % n_layer
    with tf.name_scope(bn_name):
    #        with tf.name_scope(moments_name):
    #            batch_mean,batch_var = tf.nn.moments(input,[0,1,2],name=moments_name)
    #        with tf.name_scope(beta_name):
    #            beta = tf.Variable(tf.zeros([n_out]),name=beta_name)
    #        with tf.name_scope(gamma_name):
    #            gamma = tf.Variable(tf.ones([n_out]),name=gamma_name)
    #        output = tf.nn.batch_normalization(input,batch_mean,batch_var,beta,gamma,1e-3)
        output = tf.contrib.slim.layers.batch_norm(input,is_training=is_train,updates_collections=None,decay=0.05)
    return output


def conv2d(input,w,n_layer):
    conv_name = 'conv%s' % n_layer
    with tf.name_scope(conv_name):
        conv = tf.nn.conv2d(input,w,strides=[1,1,1,1],padding='SAME',name=conv_name)
    return conv

def relu(input,n_layer):
    relu_name = 'relu%s' % n_layer
    with tf.name_scope(relu_name):
        output = tf.nn.relu(input,name=relu_name)
    return output

def tanh(input,n_layer):
    tanh_name = 'tanh%s' % n_layer
    with tf.name_scope(tanh_name):
        output = tf.nn.tanh(input,name=tanh_name)
    return output

def abs(input,n_layer):
    abs_name = 'abs%s' % n_layer
    with tf.name_scope(abs_name):
        output = tf.abs(input,name=abs_name)
    return output

def avg_pool(input,k,s,pad,n_layer):
    pool_name = 'avgpool%s' % n_layer
    with tf.name_scope(pool_name):
        output = tf.nn.avg_pool(input,ksize=[1,k,k,1],strides=[1,s,s,1],padding=pad,name=pool_name)
    return output

def max_pool(input,k,s,pad):
    output = tf.nn.max_pool(input,ksize=[1,k,k,1],strides=[1,s,s,1],padding=pad)
    return output

def conv_relu_pool_bn(input,shape,out_channel,n_layer):
    conv_name = 'conv_%s'%n_layer
    with tf.name_scope(conv_name):
        w =  weight_variable(shape,n_layer)
        conv = tf.nn.relu(conv2d(input,w,n_layer))
        pool = max_pool(conv,3,1,'SAME')
        bn = batch_norm(pool,out_channel,n_layer)
    return bn

def non_bottleneck(input,in_channel,n_layer):
    nonbottleneck_name = 'nonbottleneck%s' % n_layer
    with tf.name_scope(nonbottleneck_name):
        shortcut = input
        input = conv_relu_pool_bn(input,[3,3,in_channel,in_channel],in_channel,1)
        input = tf.nn.relu(input)
        input = conv_relu_pool_bn(input,[3,3,in_channel,in_channel],in_channel,2)
    return tf.nn.relu(shortcut + input)
        
def bottleneck(input,in_channel,n_layer):
    bottleneck_name = 'bottleneck%s' % n_layer
    with tf.name_scope(bottleneck_name):
        shortcut = input
        input = conv_relu_pool_bn(input,[1,1,in_channel,in_channel],in_channel,1)
        input = tf.nn.relu(input)
        input = conv_relu_pool_bn(input,[3,3,in_channel,in_channel],in_channel,2)
        input = tf.nn.relu(input)
        input = conv_relu_pool_bn(input,[1,1,in_channel,in_channel],in_channel,3)
    return tf.nn.relu(shortcut + input)    

def dimension_increase(input,shape,n_layer):
    dimensionincrease_name = 'dimension_increase%s'%n_layer
    with tf.name_scope(dimensionincrease_name):
        w =  weight_variable(shape,n_layer)
        conv = conv2d(input,w,n_layer)
        maxpool = max_pool(conv,5,4,'SAME')
    return maxpool

def model(x):
    with tf.variable_scope("conv0") as scope:
        hpf = np.zeros([5,5,1,1],dtype=np.float32)
        hpf[:,:,0,0] = np.array([[-1,2,-2,2,-1],[2,-6,8,-6,2],[-2,8,-12,8,-2],[2,-6,8,-6,2],[-1,2,-2,2,-1]],dtype=np.float32)/(12*255)
        w0 = tf.Variable(hpf,name="w0")
        conv0 = conv2d(x,w0,0)

    with tf.variable_scope("conv1") as scope:
        conv1 = conv_relu_pool_bn(conv0,[7,7,1,64],64,1)
        maxpool1 = max_pool(conv1,3,2,'SAME')
        
    with tf.variable_scope("n1") as scope:
        n1 =  non_bottleneck(maxpool1,64,1)
        d1 = dimension_increase(n1,[3,3,64,128],1)

    #with tf.variable_scope("n2") as scope:
    #    n2 = non_bottleneck(d1,128,1)
    #    n2 = non_bottleneck(n2,128,2)
    #    d2 = dimension_increase(n2,[3,3,128,256],1)
 
    #with tf.variable_scope("n3") as scope:
    #    n3 =  non_bottleneck(d1,128,1)
    #    d3 = dimension_increase(n3,[3,3,128,256],1)

    with tf.variable_scope("n4") as scope:
        n4 =  non_bottleneck(d1,128,1)
        d4 = dimension_increase(n4,[3,3,128,256],1)
        avgpool = avg_pool(d4,7,4,'SAME',1)
            
    with tf.variable_scope('fully_connecting_1') as scope:
        w6 = weight_variable([1024,1000],6)
        bias1 = tf.Variable(tf.random_normal([1000],mean=0.0,stddev=0.01),name="bias" )
        pool_shape = avgpool.get_shape().as_list()
        pool_reshape = tf.reshape( avgpool, [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
        y1 = tf.matmul(pool_reshape, w6) + bias1 
    
    with tf.variable_scope('fully_connecting_2') as scope:
        w7 = weight_variable([1000,2],7)
        bias2 = tf.Variable(tf.random_normal([2],mean=0.0,stddev=0.01),name="bias" )
        y_ = tf.matmul(y1, w7) + bias2 

    vars = tf.trainable_variables()
    params = [v for v in vars if ( v.name.startswith('conv1/') or  v.name.startswith('n1/') or  v.name.startswith('n2/') or  v.name.startswith('n3/') or  v.name.startswith('n4/') or  v.name.startswith('fully_connecting_1/') or  v.name.startswith('fully_connecting_2/') ) ]
        
    return y_,params

def process():

    with tf.name_scope('input') as scope:
        x = tf.placeholder(tf.float32,shape=[BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNEL])
        y = tf.placeholder(tf.float32,shape=[BATCH_SIZE,NUM_LABELS])
   
    y_,params = model(x)
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

    with tf.name_scope('acc'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('acc',accuracy)
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y_))
        tf.summary.scalar('loss',loss)

    
    opt = tf.train.GradientDescentOptimizer(0.001).minimize(loss,var_list=params)

    data_x = np.zeros([BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNEL])
    data_y = np.zeros([BATCH_SIZE,NUM_LABELS])
    #merged = tf.merge_all_summaries()
    saver = tf.train.Saver()
    Data=DataFrame(np.zeros([40000,4]),index=[np.arange(0,40000,1)],columns=['predict1','predict2','label1','lable2'])
    with tf.Session() as sess:
        saver.restore(sess,"/home/liugq/Workspace/ResNet_BN/data/models/2018.01/DRN_resnet_3_180000.ckpt")
        test_writer = tf.summary.FileWriter("/home/liugq/Workspace/ResNet_BN/data/logs/2018.01/drn_3/test_vary")        
        cover_count = 0
        stego_count = 0
        
        result2 = np.array([]) #accuracy for testing set
        test_count = 0
        i = 180000
        print 'DRN result test:'
        print 'epoch:', i
        ############################# BATCH_SIZE=1 ######################################
        #        while test_count<40000:
        #            if test_count%2==0:
        #                imc=ndimage.imread(path2+'/'+fileList2[test_count%20000])
        #                data_y[0,0] = 0
        #                data_y[0,1] = 1
        #            else:
        #                imc=ndimage.imread(path4+'/'+fileList2[test_count%20000])
        #                data_y[0,0] = 0
        #                data_y[0,1] = 1
        #            test_count=test_count+1
        #            data_x[0,:,:,0] = imc.astype(np.float32)
        ##################################################################################
        while test_count<20000:
            for j in range(BATCH_SIZE):
                if j%2==0:
                    imc=ndimage.imread(path2+'/'+fileList2[test_count])
                    data_y[j,0] = 0
                    data_y[j,1] = 1
                else:
                    imc=ndimage.imread(path4+'/'+fileList2[test_count])
                    data_y[j,0] = 1
                    data_y[j,1] = 0
                    test_count=test_count+1
                data_x[j,:,:,0] = imc.astype(np.float32)

            c2,temp2 = sess.run([loss,accuracy],feed_dict={x:data_x,y:data_y,is_train:True})
            result2 = np.insert(result2,0,temp2)
            if test_count % 1000==0:
                print temp2
        print 'test:',np.mean(result2)
        summary = tf.Summary(value=[tf.Summary.Value(tag="test_acc", simple_value=np.mean(result2)),])
        test_writer.add_summary(summary,i)


if __name__ == '__main__':
    process()
        
