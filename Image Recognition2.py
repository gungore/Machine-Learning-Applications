
c# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from collections import Counter
 
FLAGS = None

seed = 128
rng = np.random.RandomState(seed)
 
imageSize = 2304 # 48 x 48 pixels

def getData():
    # images are 48x48 = 2304 size vectors
    YTrain = []
    XTrain = []
    YTestPrivate = []
    XTestPrivate = []
    YTestPublic = []
    XTestPublic = []
    
    first = True
    for line in open('images.csv'): 
        if first:
            first = False
        else:
            row = line.split(',')
            if str(row[2]).strip() == 'Training':
                YTrain.append(int(row[0]))
                XTrain.append([int(p) for p in row[1].split()])
            elif str(row[2]).strip() == 'PrivateTest':
                YTestPrivate.append(int(row[0]))
                XTestPrivate.append([int(p) for p in row[1].split()])
            elif str(row[2]).strip()== 'PublicTest':
                YTestPublic.append(int(row[0]))
                XTestPublic.append([int(p) for p in row[1].split()])
            else:
                print(str(row[2]))
                print("Unknown Usage Type!")
    return XTrain,YTrain,XTestPrivate, YTestPrivate,XTestPublic, YTestPublic

def resampling(XTrain, YTrain, oversampling=1, num_samples=2000):
    min_count = 100000000
    for i in sorted(Counter(YTrain).items()):
        if i[1] < min_count:
            min_count = i[1]

    index_0 = [i for i,x in enumerate(YTrain) if x == 0]
    index_1 = [i for i,x in enumerate(YTrain) if x == 1]
    index_2 = [i for i,x in enumerate(YTrain) if x == 2]
    index_3 = [i for i,x in enumerate(YTrain) if x == 3]
    index_4 = [i for i,x in enumerate(YTrain) if x == 4]
    index_5 = [i for i,x in enumerate(YTrain) if x == 5]
    index_6 = [i for i,x in enumerate(YTrain) if x == 6]
    
    mask_0 = rng.choice(index_0, num_samples, replace=False)
    mask_1 = rng.choice(index_1, num_samples)
    mask_2 = rng.choice(index_2, num_samples, replace=False)
    mask_3 = rng.choice(index_3, num_samples, replace=False)
    mask_4 = rng.choice(index_4, num_samples, replace=False)
    mask_5 = rng.choice(index_5, num_samples, replace=False)
    mask_6 = rng.choice(index_6, num_samples, replace=False)
    
    if oversampling != 1:
        num_samples=min_count
        mask_1=index_1
        
    XTrain_resampled = []
    for i,x in enumerate(XTrain):
        if i in mask_0:
            XTrain_resampled.append(x)
        elif i in mask_1:
            XTrain_resampled.append(x)
        elif i in mask_2:
            XTrain_resampled.append(x)
        elif i in mask_3:
            XTrain_resampled.append(x)
        elif i in mask_4:
            XTrain_resampled.append(x)
        elif i in mask_5:
            XTrain_resampled.append(x)
        elif i in mask_6:
            XTrain_resampled.append(x)
    
    YTrain_resampled = []
    for i,x in enumerate(YTrain):
        if i in mask_0:
            YTrain_resampled.append(x)
        elif i in index_1:
            YTrain_resampled.append(x)
        elif i in mask_2:
            YTrain_resampled.append(x)
        elif i in mask_3:
            YTrain_resampled.append(x)
        elif i in mask_4:
            YTrain_resampled.append(x)
        elif i in mask_5:
            YTrain_resampled.append(x)
        elif i in mask_6:
            YTrain_resampled.append(x)
    
    return XTrain_resampled, YTrain_resampled
    
def normalization(XTrain,YTrain,XTestPrivate, YTestPrivate,XTestPublic, YTestPublic):    
    # normalize X
    XTrain, YTrain = np.array(XTrain) / 255.0, np.array(YTrain)
    XTestPrivate, YTestPrivate = np.array(XTestPrivate) / 255.0, np.array(YTestPrivate)
    XTestPublic, YTestPublic = np.array(XTestPublic) / 255.0, np.array(YTestPublic)
    return XTrain, YTrain, XTestPrivate, YTestPrivate, XTestPublic, YTestPublic

def one_hot_vector(Y1):
    Y = []
    for j in Y1:
        a = [0, 0, 0, 0, 0, 0, 0]
        a[j] = 1
        Y.append(a)
    Y = np.array(Y)
    return Y

def deepnn(x):
    x_image = tf.reshape(x, [-1, 48, 48, 1])
    
    # First convolutional layer - maps one grayscale image to 32 feature maps.
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    
    # Pooling layer - downsamples by 2X.
    h_pool1 = max_pool_2x2(h_conv1)
    
    # Second convolutional layer -- maps 32 feature maps to 64.
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    
    # Second pooling layer.
    h_pool2 = max_pool_2x2(h_conv2)
    
    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.
    W_fc1 = weight_variable([12 * 12 * 64, 1024])
    b_fc1 = bias_variable([1024])
    
    h_pool2_flat = tf.reshape(h_pool2, [-1, 12*12*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    
    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    # Map the 1024 features to 10 classes, one for each digit
    W_fc2 = weight_variable([1024, 7])
    b_fc2 = bias_variable([7])
    
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return y_conv, keep_prob


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)



# Import data
XTrain1, YTrain1, XTestPrivate, YTestPrivate1, XTestPublic, YTestPublic1 = getData()
#XTrain_resampled, YTrain_resampled = resampling(XTrain1, YTrain1, oversampling=0)
#XTrain, YTrain, XTestPrivate, YTestPrivate1, XTestPublic, YTestPublic1 = normalization(XTrain_resampled, YTrain_resampled, XTestPrivate, YTestPrivate1, XTestPublic, YTestPublic1)
XTrain, YTrain, XTestPrivate, YTestPrivate1, XTestPublic, YTestPublic1 = normalization(XTrain1, YTrain1, XTestPrivate, YTestPrivate1, XTestPublic, YTestPublic1)

#YTrain = one_hot_vector(YTrain1)
YTestPublic = one_hot_vector(YTestPublic1)
YTestPrivate = one_hot_vector(YTestPrivate1)

#XTrain_tensor = tf.stack(XTrain)
#YTrain_tensor = tf.stack(XTrain)
#XTestPublic_tensor = tf.stack(XTestPublic)
#YTestPublic_tensor = tf.stack(YTestPublic)

# Create the model
x = tf.placeholder(tf.float32, [None, 2304])

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 7])

# Build the graph for the deep net
y_conv, keep_prob = deepnn(x)

cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
        batch_mask = rng.choice(XTrain.shape[0], 100, replace=False)
        batch_x = XTrain[[batch_mask]]
        batch_y1 = YTrain[[batch_mask]]
        batch_y = []
        for j in batch_y1:
            a = [0, 0, 0, 0, 0, 0, 0]
            a[j] = 1
            batch_y.append(a)
        batch_y = np.array(batch_y)
        if i %  1000 == 0:
            
            train_accuracy = accuracy.eval(feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))
        train_step.run(feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})
    print('test accuracy - private test dataset %g' % accuracy.eval(feed_dict={
          x: XTestPrivate, y_: YTestPrivate, keep_prob: 1.0}))
