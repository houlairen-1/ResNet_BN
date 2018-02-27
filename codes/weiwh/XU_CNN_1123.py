import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import os
import random
import time
import datetime
from scipy import ndimage
from pandas import DataFrame
os.environ["CUDA_VISIBLE_DEVICES"]="2"

BATCH_SIZE = 10
bs = 5
IMAGE_SIZE = 512
NUM_CHANNEL = 1
NUM_LABELS = 2
NUM_ITER =200000
NUM_SHOWTRAIN = 200 #show result eveary epoch 
NUM_SHOWTEST = 5000
BN_DECAY = 0.95
UPDATE_OPS_COLLECTION = 'Discriminative_update_ops'


LEARNING_RATE =0.001
LEARNING_RATE_DECAY = 0.9
MOMENTUM = 0.9
decay_step = 5000
is_train = True
path1 = './2classdf_mv/double/'
path2 = './2classdf_mv/forge/'
fileList1 = []
for (dirpath,dirnames,filenames) in os.walk(path1):  
    fileList1 = filenames
fileList2 = []
for (dirpath,dirnames,filenames) in os.walk(path2):  
    fileList2 = filenames


np.set_printoptions(threshold='nan')
random.seed(1234)

#random.shuffle(fileList)

x = tf.placeholder(tf.float32,shape=[BATCH_SIZE, 720, 1280, NUM_CHANNEL])
y = tf.placeholder(tf.float32,shape=[BATCH_SIZE,NUM_LABELS])
is_train = tf.placeholder(tf.bool,name='is_train')
#select channel
#pro =  tf.placeholder(tf.float32,shape=[BATCH_SIZE,IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNEL])
#global_step = tf.Variable(0,trainable = False)

hpf = np.zeros([5,5,1,1],dtype=np.float32 ) #[height,width,input,output]
hpf[:,:,0,0] = np.array([[-1,2,-2,2,-1],[2,-6,8,-6,2],[-2,8,-12,8,-2],[2,-6,8,-6,2],[-1,2,-2,2,-1]],dtype=np.float32)/(12*255)

kernel0 = tf.Variable(hpf,name="kernel0")
conv0 = tf.nn.conv2d(x,kernel0,[1,1,1,1],'SAME',name="conv0")



with tf.variable_scope("Group1") as scope:
     kernel1 = tf.Variable( tf.random_normal( [5,5,1,8],mean=0.0,stddev=0.01 ),name="kernel1" )  
     conv1 = tf.nn.conv2d(conv0, kernel1, [1,2,2,1], padding='SAME',name="conv1"  )
     abs1 = tf.abs(conv1,name="abs1")

     bn1 = slim.layers.batch_norm(abs1,is_training=is_train,updates_collections=None,decay=0.05)
     tanh1 = tf.nn.tanh(bn1,name="tanh1")
     pool1 = tf.nn.avg_pool(tanh1, ksize=[1,5,5,1], strides=[1,2,2,1], padding='SAME',name="pool1" )



with tf.variable_scope("Group2") as scope:
     kernel2_1 = tf.Variable( tf.random_normal( [5,5,8,16],mean=0.0,stddev=0.01 ),name="kernel2_1")
     conv2_1 = tf.nn.conv2d( pool1, kernel2_1, [1,2,2,1], padding="SAME",name="conv2_1"  )
         
     bn2_1 = slim.layers.batch_norm(conv2_1,is_training=is_train,updates_collections=None,decay=0.05) 
     tanh2_1 = tf.nn.tanh(bn2_1,name="tanh2_1")
     pool2 = tf.nn.avg_pool(tanh2_1, ksize=[1,5,5,1], strides=[1,2,2,1], padding='SAME',name="pool2_1" )

with tf.variable_scope("Group3") as scope:
     kernel3 = tf.Variable( tf.random_normal( [1,1,16,32],mean=0.0,stddev=0.01 ),name="kernel3" )
     conv3 = tf.nn.conv2d( pool2, kernel3, [1,1,1,1], padding="SAME",name="conv3"  )
     
     bn3 = slim.layers.batch_norm(conv3,is_training=is_train,updates_collections=None,decay=0.05)  

     relu3 = tf.nn.relu(bn3,name="bn3")
     pool3 = tf.nn.avg_pool(relu3, ksize=[1,5,5,1], strides=[1,2,2,1], padding="SAME",name="pool3" ) 

with tf.variable_scope("Group4") as scope:
     kernel4 = tf.Variable( tf.random_normal( [1,1,32,64],mean=0.0,stddev=0.01 ),name="kernel4_1" )
     conv4 = tf.nn.conv2d( pool3, kernel4, [1,1,1,1], padding="SAME",name="conv4_1"  )
     
     bn4 = slim.layers.batch_norm(conv4,is_training=is_train,updates_collections=None,decay=0.05)       
     relu4 = tf.nn.relu(bn4,name="relu4_1")
     pool4 = tf.nn.avg_pool(relu4, ksize=[1,7,7,1], strides=[1,2,2,1], padding="SAME",name="pool4_1" ) 

with tf.variable_scope("Group5") as scope:
     kernel5 = tf.Variable( tf.random_normal( [1,1,64,128],mean=0.0,stddev=0.01 ),name="kernel5" )
     conv5 = tf.nn.conv2d( pool4, kernel5, [1,1,1,1], padding="SAME",name="conv5"  )
     
     bn5 = slim.layers.batch_norm(conv5,is_training=is_train,updates_collections=None,decay=0.05)      
     relu5 = tf.nn.relu(bn5,name="relu5")
     pool5 = tf.nn.avg_pool(relu5, ksize=[1,7,7,1], strides=[1,2,2,1], padding="SAME",name="pool5_1" ) 

with tf.variable_scope("Group6") as scope:
     kernel6 = tf.Variable( tf.random_normal( [1,1,128,256],mean=0.0,stddev=0.01 ),name="kernel5" )
     conv6 = tf.nn.conv2d( pool5, kernel6, [1,1,1,1], padding="SAME",name="conv5"  )
     
     bn6 = slim.layers.batch_norm(conv6,is_training=is_train,updates_collections=None,decay=0.05)      
     relu6 = tf.nn.relu(bn6,name="relu6")
     pool6 = tf.nn.avg_pool(relu6, ksize=[1,6,10,1], strides=[1,1,1,1], padding="VALID",name="pool5" ) 


with tf.variable_scope('Group7') as scope:
     pool_shape = pool6.get_shape().as_list()
     pool_reshape = tf.reshape( pool6, [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
     weights = tf.Variable( tf.random_normal( [256,2],mean=0.0,stddev=0.01 ),name="weights" )
     bias = tf.Variable( tf.random_normal([2],mean=0.0,stddev=0.01),name="bias" )
     y_ = tf.matmul(pool_reshape, weights) + bias 


vars = tf.trainable_variables()
params = [v for v in vars if ( v.name.startswith('Group1/') or  v.name.startswith('Group2/') or  v.name.startswith('Group3/') or  v.name.startswith('Group4/') or  v.name.startswith('Group5/') or  v.name.startswith('Group6/') or  v.name.startswith('Group7/')) ]

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('acc',accuracy)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits( labels=y,logits=y_))
tf.summary.scalar('loss',loss)
global_step = tf.Variable(0,trainable = False)
decayed_learning_rate=tf.train.exponential_decay(LEARNING_RATE, global_step, decay_step,LEARNING_RATE_DECAY, staircase=True)
opt = tf.train.MomentumOptimizer(decayed_learning_rate,MOMENTUM).minimize(loss,var_list=params)

variables_averages = tf.train.ExponentialMovingAverage(0.95)
variables_averages_op = variables_averages.apply(tf.trainable_variables())

data_x = np.zeros([BATCH_SIZE,720,1280,NUM_CHANNEL])
data_y = np.zeros([BATCH_SIZE,NUM_LABELS])
for i in range(0,bs):
    data_y[i,1] = 1
for i in range(bs,BATCH_SIZE):
    data_y[i,0] = 1


saver = tf.train.Saver()
merged = tf.summary.merge_all()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session() as sess:
     writer = tf.summary.FileWriter("./logs/XU_CNN_s0.4_binary_",sess.graph)
     tf.global_variables_initializer().run()
     
     summary = tf.Summary()	
     start = datetime.datetime.now()
     count = 0
     list = [h for h in range(4000)]
     for i in range(1,NUM_ITER+1):
         for j in range(bs):
             if count%5000==0:
                 count = count%5000
                 random.seed(i)
                 random.shuffle(list)
             cover=ndimage.imread(path1+'/'+fileList1[list[count]])   
             stego=ndimage.imread(path2+'/'+fileList2[list[count]])   
             data_x[j,:,:,0] = cover.astype(np.float32)
             data_x[j+bs,:,:,0] = stego.astype(np.float32)           
             count = count+1
             
         _,temp,l,lr = sess.run([opt,accuracy,loss,decayed_learning_rate],feed_dict={x:data_x,y:data_y,is_train:True})        
         if i%100==0:  
             summary.ParseFromString(sess.run(merged,feed_dict={x:data_x,y:data_y,is_train:True}))
             writer.add_summary(summary, i)
         if i%100 ==0: 
             end = datetime.datetime.now()
             print ('time:',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(int(time.time()))))
             print ('XU_CNN_111505_S0.4:  batch result')
             print ('XU_CNN_111505_S0.4:  trainning iterations:', i,'   trainning epoch',i ,'   learingrate',lr)
             
             print ('trainning loss:', l,'     trainning accuracy:', temp)
             print (' ')
         if i%(20000)==0:
            saver = tf.train.Saver()
            saver.save(sess,'./snapshot/XU_CNN_s0.4_binary_'+str(i)+'.ckpt')        
   
         if i%5000==0:
            result1 = np.array([]) #accuracy for training set
            num = i/NUM_SHOWTEST - 1
            val_count = 4000
            while val_count<5000:
                for j in range(bs):
                    cover=ndimage.imread(path1+'/'+fileList1[val_count])   
                    stego=ndimage.imread(path2+'/'+fileList2[val_count])   
                    data_x[j,:,:,0] = cover.astype(np.float32)
                    data_x[j+bs,:,:,0] = stego.astype(np.float32)           
                    val_count = val_count+1
                c1,temp1 = sess.run([loss,accuracy],feed_dict={x:data_x,y:data_y,is_train:False})
                result1 = np.insert(result1,0,temp1)
            summary.value.add(tag='val_acc', simple_value=np.mean(result1))
            writer.add_summary(summary, i)
            print ('time:',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(int(time.time()))))
            print ('val accuracy:', np.mean(result1))


            result2 = np.array([]) #accuracy for testing set
            test_count = 5000
            while test_count<10000:
                for j in range(bs):
                    cover=ndimage.imread(path1+'/'+fileList1[test_count])   
                    stego=ndimage.imread(path2+'/'+fileList2[test_count])   
                    data_x[j,:,:,0] = cover.astype(np.float32)
                    data_x[j+bs,:,:,0] = stego.astype(np.float32)           
                    test_count = test_count+1
                c2,temp2 = sess.run([loss,accuracy],feed_dict={x:data_x,y:data_y,is_train:False})
                result2= np.insert(result2,0,temp2)
            summary.value.add(tag='test_acc', simple_value=np.mean(result2))
            writer.add_summary(summary, i)
            print ('time:',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(int(time.time()))))
            print ('Testing :', np.mean(result2))
            print (' ')

