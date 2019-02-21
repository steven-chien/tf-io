import os
import csv
import glob
import time
#from hostlist import expand_hostlist
import tensorflow as tf

tf.app.flags.DEFINE_string('sample_list', './sample_list.csv', 'List of samples.')
tf.app.flags.DEFINE_integer('batch_size', 32, 'Number of images per batch.')
tf.app.flags.DEFINE_integer('num_batches', 10, 'Number of batches drawn by each worker.')
tf.app.flags.DEFINE_integer('prefetch', 0, 'Number of batches to pre-fetch.')
tf.app.flags.DEFINE_integer('num_gpus', 1, 'Number of GPUs per node.')
tf.app.flags.DEFINE_integer('parallel_calls', 1, 'Number of parallel calls.')
FLAGS = tf.app.flags.FLAGS

def get_samples():
    samples = []
    labels = []

    with open(FLAGS.sample_list, 'r') as f:
        reader = csv.reader(f)
        for file_name, label in reader:
            samples.append(str(file_name))
            labels.append(str(label))

    return samples, labels

def parse_function(filename, label):
    file_path = filename
    
    image_string = tf.read_file(file_path)
#    image_string = tf.Print(image_string, [file_path], message='path')
 
    # Don't use tf.image.decode_image, or the output shape will be undefined
    image = tf.image.decode_png(image_string, channels=3)
    
    # This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)
    
    image = tf.image.resize_images(image, [64, 64])
    return image, filename
    return image_string, filename

def build_dataset(task_index, num_workers):
    filenames, labels = get_samples()
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    #dataset = dataset.shard(num_workers, task_index)
    dataset = dataset.shuffle(FLAGS.batch_size*FLAGS.num_batches)
    dataset = dataset.map(parse_function, num_parallel_calls=FLAGS.parallel_calls)
    dataset = dataset.apply(tf.contrib.data.ignore_errors())
    dataset = dataset.batch(FLAGS.batch_size)
    #dataset = dataset.prefetch(FLAGS.prefetch)
    iterator = dataset.make_one_shot_iterator()
    next_batch, next_labels = iterator.get_next()
    return next_batch, next_labels

def build_worker(task_index, num_workers):
    next_batch, next_labels = build_dataset(task_index, num_workers)
    #with tf.device('/gpu:0'):
    #    wcnn1 = tf.truncated_normal([11, 11, 3, 96], stddev=0.01)
    #    bcnn1 = tf.constant(0.0, shape=[96])
    #    conv1 = tf.add(tf.nn.conv2d(next_batch, wcnn1, strides=[1, 4, 4, 1], padding='SAME'), bcnn1)

    #return conv1, next_labels
    return next_batch, next_labels

def training():
    tf.logging.set_verbosity(tf.logging.INFO)
    task_index = 0
    num_workers = 1
    train_op, labels = build_worker(task_index, num_workers)
    
    with tf.Session() as sess:
        print('worker 0: No. images: '+str(FLAGS.num_batches*FLAGS.batch_size))
        t0 = time.time()
        print('worker 0: start: '+str(t0))
        for i in range(FLAGS.num_batches):
            try:
                sess.run([train_op, labels])
            except tf.errors.OutOfRangeError:
                print('Dataset out of range at batch %d' % i)
                exit(1)
            #print(l)
        t1 = time.time()
    print('worker 0: end: '+str(t1))
    print('worker 0: runtime: '+str(t1-t0))
    print('worker 0: image/second: '+str((FLAGS.num_batches*FLAGS.batch_size)/(t1-t0)))
    exit(0)

def main(argv):
    training()
    
if __name__ == "__main__":
    tf.app.run(main=main, argv=None)
