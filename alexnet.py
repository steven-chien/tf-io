import os
import csv
#from hostlist import expand_hostlist
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelEncoder
import time

tf.app.flags.DEFINE_string('sample_list', './sample_list.csv', 'List of samples.')
tf.app.flags.DEFINE_integer('batch_size', 32, 'Number of images per batch.')
tf.app.flags.DEFINE_integer('num_batches', 10, 'Number of batches.')
tf.app.flags.DEFINE_integer('prefetch', 1, 'Number of batches to pre-fetch.')
tf.app.flags.DEFINE_integer('parallel_calls', 16, 'Number of threads.')
tf.app.flags.DEFINE_integer('num_epoch', 1, 'Number of epoch.')
tf.app.flags.DEFINE_float('learning_rate', 0.001, 'Learning rate.')
tf.app.flags.DEFINE_float('dropout', 0.5, 'Dropout rate.')
FLAGS = tf.app.flags.FLAGS

def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

def get_samples():
    label_encoder = LabelEncoder()
    samples = []
    labels = []

    with open(FLAGS.sample_list, 'r') as f:
        reader = csv.reader(f)
        for i, val in enumerate(reader):
            file_name, label  = val
            samples.append(str(file_name))
            labels.append(int(label))
            if i == FLAGS.batch_size * FLAGS.num_batches:
                break
    labels = np.array(label_encoder.fit_transform(labels))
    labels = np.array(labels)
    num_classes = np.size(np.unique(labels))
    print('num classes: '+str(num_classes))
    
    labels = dense_to_one_hot(labels, num_classes=num_classes)
    return samples, labels, num_classes

def parse_function(filename, label):
    file_path = filename
    #tf.Print(filename, [filename], message='filename')
    
    #file_path = tf.Print(file_path, [file_path], message='path')
    image_string = tf.read_file(file_path)
    
    # Don't use tf.image.decode_image, or the output shape will be undefined
    #image = tf.image.decode_jpeg(image_string, channels=3, try_recover_truncated=True)
    image = tf.image.decode_png(image_string, channels=3)
    
    # This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)
    
    image = tf.image.resize_images(image, [224, 224])
    return image, label
    #return image, filename

def build_dataset(task_index, num_workers):
    with tf.device('/cpu:0'):
        filenames, labels, num_classes = get_samples()
        print('num classes: '+str(num_classes))
        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
        dataset = dataset.shuffle(len(filenames))
        dataset = dataset.map(parse_function, num_parallel_calls=FLAGS.parallel_calls)
        dataset = dataset.apply(tf.contrib.data.ignore_errors())
        #dataset = dataset.repeat(100)
        dataset = dataset.batch(FLAGS.batch_size)
        dataset = dataset.prefetch(FLAGS.prefetch)
        iterator = dataset.make_initializable_iterator()
    return iterator, num_classes

################ TensorFlow standard operations wrappers #####################
def weight(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.01)
    w = tf.Variable(initial, name=name)
    tf.add_to_collection('weights', w)
    return w

def bias(value, shape, name):
    initial = tf.constant(value, shape=shape)
    return tf.Variable(initial, name=name)

def conv2d(x, W, stride, padding):
    return tf.nn.conv2d(x, W, strides=[1, stride[0], stride[1], 1], padding=padding)

def max_pool2d(x, kernel, stride, padding):
    return tf.nn.max_pool(x, ksize=kernel, strides=stride, padding=padding)

def lrn(x, depth_radius, bias, alpha, beta):
    return tf.nn.local_response_normalization(x, depth_radius, bias, alpha, beta)

def relu(x):
    return tf.nn.relu(x)

def alex_net(x, dropout, num_classes):
    with tf.name_scope('alexnet_cnn') as scope:
        with tf.name_scope('alexnet_cnn_conv1') as inner_scope:
            wcnn1 = weight([11, 11, 3, 96], name='wcnn1')
            bcnn1 = bias(0.0, [96], name='bcnn1')
            conv1 = tf.add(conv2d(x, wcnn1, stride=(4, 4), padding='SAME'), bcnn1)
            conv1 = relu(conv1)
            norm1 = lrn(conv1, depth_radius=2, bias=1.0, alpha=2e-05, beta=0.75)
            pool1 = max_pool2d(norm1, kernel=[1, 3, 3, 1], stride=[1, 2, 2, 1], padding='VALID')

        with tf.name_scope('alexnet_cnn_conv2') as inner_scope:
            wcnn2 = weight([5, 5, 96, 256], name='wcnn2')
            bcnn2 = bias(1.0, [256], name='bcnn2')
            conv2 = tf.add(conv2d(pool1, wcnn2, stride=(1, 1), padding='SAME'), bcnn2)
            conv2 = relu(conv2)
            norm2 = lrn(conv2, depth_radius=2, bias=1.0, alpha=2e-05, beta=0.75)
            pool2 = max_pool2d(norm2, kernel=[1, 3, 3, 1], stride=[1, 2, 2, 1], padding='VALID')

        with tf.name_scope('alexnet_cnn_conv3') as inner_scope:
            wcnn3 = weight([3, 3, 256, 384], name='wcnn3')
            bcnn3 = bias(0.0, [384], name='bcnn3')
            conv3 = tf.add(conv2d(pool2, wcnn3, stride=(1, 1), padding='SAME'), bcnn3)
            conv3 = relu(conv3)

        with tf.name_scope('alexnet_cnn_conv4') as inner_scope:
            wcnn4 = weight([3, 3, 384, 384], name='wcnn4')
            bcnn4 = bias(1.0, [384], name='bcnn4')
            conv4 = tf.add(conv2d(conv3, wcnn4, stride=(1, 1), padding='SAME'), bcnn4)
            conv4 = relu(conv4)

        with tf.name_scope('alexnet_cnn_conv5') as inner_scope:
            wcnn5 = weight([3, 3, 384, 256], name='wcnn5')
            bcnn5 = bias(1.0, [256], name='bcnn5')
            conv5 = tf.add(conv2d(conv4, wcnn5, stride=(1, 1), padding='SAME'), bcnn5)
            conv5 = relu(conv5)
            pool5 = max_pool2d(conv5, kernel=[1, 3, 3, 1], stride=[1, 2, 2, 1], padding='VALID')

    dim = pool5.get_shape().as_list()
    flat_dim = dim[1] * dim[2] * dim[3] # 6 * 6 * 256
    flat = tf.reshape(pool5, [-1, flat_dim])

    with tf.name_scope('alexnet_classifier') as scope:
        with tf.name_scope('alexnet_classifier_fc1') as inner_scope:
            wfc1 = weight([flat_dim, 4096], name='wfc1')
            bfc1 = bias(0.0, [4096], name='bfc1')
            fc1 = tf.add(tf.matmul(flat, wfc1), bfc1)
            fc1 = relu(fc1)
            fc1 = tf.nn.dropout(fc1, tf.constant(dropout))

        with tf.name_scope('alexnet_classifier_fc2') as inner_scope:
            wfc2 = weight([4096, 4096], name='wfc2')
            bfc2 = bias(0.0, [4096], name='bfc2')
            fc2 = tf.add(tf.matmul(fc1, wfc2), bfc2)
            fc2 = relu(fc2)
            fc2 = tf.nn.dropout(fc2, dropout)

        with tf.name_scope('alexnet_classifier_output') as inner_scope:
            wfc3 = weight([4096, num_classes], name='wfc3')
            bfc3 = bias(0.0, [num_classes], name='bfc3')
            fc3 = tf.add(tf.matmul(fc2, wfc3), bfc3)

    return fc3

def build_worker(task_index, num_workers):
    iterator, num_classes = build_dataset(task_index, num_workers)
    next_batch, next_labels = iterator.get_next()

#    l = np.zeros([64, 102]).astype(np.float32)
#    b = np.zeros([64, 224, 224, 3]).astype(np.float32)
    logits = alex_net(next_batch, FLAGS.dropout, num_classes)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=next_labels))
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
    correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(next_labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    # return conv1, next_labels
    return optimizer, cost, accuracy, iterator

def training():
    task_index = 0
    num_workers = 1

    with tf.device('/cpu:0'):
        global_step = tf.Variable(0, name='global_step',trainable=False)
    with tf.device('/gpu:0'):
        optimizer, cost, accuracy, iterator = build_worker(task_index, num_workers)
        train_op = optimizer.minimize(cost, global_step=global_step)
    init = tf.initialize_all_variables()
    init_iterator = iterator.initializer

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
        sess.run(init)
        for i in range(FLAGS.num_epoch):
            t0 = time.time()
            print('worker 0: epoch '+str(i)+' start: '+str(t0))
            sess.run(init_iterator)
            for j in range(FLAGS.num_batches):
                s, _, c, a = sess.run([global_step, train_op, cost, accuracy])
                print("Epoch="+str(i)+" Iter=" + str(j) + " Step=" + str(s) + " Loss=" + "{:.6f}".format(c) + " Accuracy=" + "{:.5f}".format(a))
            t1 = time.time()
            print('worker 0: epoch '+str(i)+' end: '+str(t1))
            print('worker 0: epoch '+str(i)+' runtime: '+str(t1-t0))
        exit(1)

def main(argv):
    print('num threads: '+str(FLAGS.parallel_calls))
    tf.logging.set_verbosity(tf.logging.INFO)
    training()

if __name__ == "__main__":
    tf.app.run(main=main, argv=None)
