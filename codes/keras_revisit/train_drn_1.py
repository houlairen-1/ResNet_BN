import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import MaxPooling2D, AveragePooling2D, Input, Flatten
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2
from keras import initializers
from keras.models import Model
from keras.utils import plot_model

import numpy as np
from scipy import ndimage
import os
os.environ["CUDA_VISIBLE_DEVICES"]="7"

# DEFINE 
batch_size = 10
epochs = 10 # every epoch equal 2W iterations
IMAGE_SIZE = 256
NUM_CHANNEL = 1
NUM_LABELS = 2
NUM_IMAGES = 20000
LEARNING_RATE =0.001

# path of dataset
path_train_cover = '/data1/liugq/BOSSbase/BOSS_base_256x256/train'
path_test_cover = '/data1/liugq/BOSSbase/BOSS_base_256x256/test'
path_train_stego = '/data1/liugq/BOSSbase/BOSS_base_suniward40_256x256/train'
path_test_stego = '/data1/liugq/BOSSbase/BOSS_base_suniward40_256x256/test'

fileList_train = []
for (dirpath_train_cover,dirnames1,filenames1) in os.walk(path_train_cover):  #0~5000 for training
    fileList_train = filenames1

fileList_test = []
for (dirpath_test_cover,dirnames2,filenames2) in os.walk(path_test_cover):  #0~5000 for testing
    fileList_test = filenames2

#################################################### keras ##########################################################################
def non_bottleneck(inputs,
                   num_filters=64,
                   kernel_size=3,
                   strides=1,
                   activation='relu',
                   batch_normalization=True):
    conv_1 = Conv2D(num_filters,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding='same',
                    kernel_initializer='random_normal',
                    kernel_regularizer=l2(1e-4))
    conv_2 = Conv2D(num_filters,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding='same',
                    kernel_initializer='random_normal',
                    kernel_regularizer=l2(1e-4))
    
    x = inputs
    x = conv_1(x)
    x = Activation(activation)(x)
    x = MaxPooling2D(pool_size=3,strides=1,padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = conv_2(x)
    x = Activation(activation)(x)
    x = MaxPooling2D(pool_size=3,strides=1,padding='same')(x)
    x = BatchNormalization()(x)
    
    return keras.layers.add([inputs,x])


def dimension_increase(inputs,
                       num_filters=128,
                       kernel_size=3,
                       strides=1):
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='random_normal',
                  kernel_regularizer=l2(1e-4))
    x = inputs
    x = conv(x)
    x = MaxPooling2D(pool_size=3,strides=2,padding='same')(x)
    
    return x


def init_hpf(shape=[5,5,1,1], dtype=np.float32):
    hpf = np.zeros(shape=shape,dtype=dtype)
    hpf[:,:,0,0] = np.array([[-1,2,-2,2,-1],[2,-6,8,-6,2],[-2,8,-12,8,-2],[2,-6,8,-6,2],[-1,2,-2,2,-1]],dtype=np.float32)/(12*255)
    return hpf
    
###########################################################################################################################################
def resnet_l15(input_shape):
    inputs = Input(shape=input_shape)
    x = inputs
    ################## conv0 256x256 #################################
    x = Conv2D(filters=1, kernel_size=5, kernel_initializer=init_hpf)(x)
        
    ###################### conv1 256 to 128 ##########################
    conv = Conv2D(filters=64,
               kernel_size=7,
               strides=1,
               padding='same',
               kernel_initializer='random_normal',
               kernel_regularizer=l2(1e-4))
    x = conv(x)
    x = Activation(activation='relu')(x)
    x = MaxPooling2D(pool_size=3,strides=1,padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=3,strides=2,padding='same')(x)
       
    ###################### n1 128 to 64 #################################
    x =  non_bottleneck(x,64)
    x = dimension_increase(x,128)
    
    ###################### n2 64 to 32 ##################################
    x = non_bottleneck(x,128)
    x = non_bottleneck(x,128)
    x = dimension_increase(x,256)
 
    ###################### n3 32 to 16 #################################
    x =  non_bottleneck(x,256)
    x = dimension_increase(x,512)
    
    ###################### n4 16 to 4 ##################################
    x = non_bottleneck(x,512)
    x = AveragePooling2D(pool_size=7,strides=4,padding='same')(x)
            
    ###################### fully_connecting_1 8192[512x(4x4)] to 1000 #########
    y = Flatten()(x)
    y = Dense(1000,activation='softmax',kernel_initializer='random_normal')(y)
    
    ###################### fully_connecting_2 1000 to 2 #####################
    outputs = Dense(NUM_LABELS,activation='softmax',kernel_initializer='random_normal')(y)
    
    model = Model(inputs=inputs,outputs=outputs)        
    return model

def process():
    model = resnet_l15(input_shape=(IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNEL))
    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(lr=LEARNING_RATE),
                  metrics=['accuracy'])
    print(model.summary())
    plot_model(model, to_file = os.path.join(os.getcwd(), 'resnet_l15.png'))
    model_name = 'resnet_l15_model.{epoch}.h5'
    save_dir = '/home/liugq/Workspace/ResNet_BN/data/models/2018.02'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)
    
    # Prepare callbacks for model saving and for learning rate adjustment.
    checkpoint = ModelCheckpoint(filepath=filepath,
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True)
    
    callbacks = [checkpoint]
    ####################### data ##########################################
    x_train = np.zeros([NUM_IMAGES,IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNEL])
    y_train = np.zeros([NUM_IMAGES,NUM_LABELS])
    cover_count = 0
    stego_count = 0
    for i in range(NUM_IMAGES):
        cover_count = cover_count % NUM_IMAGES
        stego_count = stego_count % NUM_IMAGES
        if i%2==0:
            img=ndimage.imread(path_train_cover+'/'+fileList_train[cover_count])
            y_train[i,0] = 0
            y_train[i,1] = 1
            cover_count = cover_count+1
        else:
            img=ndimage.imread(path_train_stego+'/'+fileList_train[stego_count])
            y_train[i,0] = 1
            y_train[i,1] = 0
            stego_count=stego_count+1
        x_train[i,:,:,0] = img.astype(np.float32)
    
    x_test = np.zeros([NUM_IMAGES,IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNEL])
    y_test = np.zeros([NUM_IMAGES,NUM_LABELS])
    cover_count = 0
    stego_count = 0
    for i in range(NUM_IMAGES):
        cover_count = cover_count % NUM_IMAGES
        stego_count = stego_count % NUM_IMAGES
        if i%2==0:
            img=ndimage.imread(path_test_cover+'/'+fileList_test[cover_count])
            y_test[i,0] = 0
            y_test[i,1] = 1
            cover_count = cover_count+1
        else:
            img=ndimage.imread(path_test_stego+'/'+fileList_test[stego_count])
            y_test[i,0] = 1
            y_test[i,1] = 0
            stego_count=stego_count+1
        x_test[i,:,:,0] = img.astype(np.float32)
    ##########################################################################
    # Run training, with or without data augmentation.
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True,
              callbacks=callbacks)
    
    # Score trained model.
    scores = model.evaluate(x_test,y_test,verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:',scores[1])


if __name__ == '__main__':
    process()
        
