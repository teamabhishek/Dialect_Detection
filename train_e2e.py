#!/data/sls/u/swshon/tools/pytf/bin/python
import os,sys
import tensorflow as tf
import numpy as np
sys.path.insert(0, './scripts')
sys.path.insert(0, './models')
 
sys.path.append('models')
# from tensorflow.contrib.learn.python.learn.datasets import base



# def accuracy(predictions, labels):
#     pred_class = np.argmax(predictions, 1)
#     true_class = np.argmax(labels, 1)
# #     print pred_class
# #     print true_class
#     return (100.0 * np.sum(pred_class == true_class) / predictions.shape[0])


# modified one in tf 2.7 and python 3.10

def accuracy(predictions, labels):
    pred_class = np.argmax(predictions, axis=1)
    true_class = np.argmax(labels, axis=1)
    return (100.0 * np.sum(pred_class == true_class) / predictions.shape[0])

# def txtwrite(filename, dict):
#     with open(filename, "w") as text_file:
#         for key, vec in dict.iteritems():
#             text_file.write('%s [' % key)
#             for i, ele in enumerate(vec):
#                 text_file.write(' %f' % ele)
#             text_file.write(' ]\n')
            
            
            # modified one in tf 2.7 and python 3.10
            
def txtwrite(filename, dictionary):
    with open(filename, "w") as text_file:
        for key, vec in dictionary.items():
            text_file.write('%s [' % key)
            for i, ele in enumerate(vec):
                text_file.write(' %f' % ele)
            text_file.write(' ]\n')

            
# def variable_summaries(var):
#     with tf.name_scope('summaries'):
#         mean = tf.reduce_mean(var)
#         tf.summary.scalar('mean', mean)
#         with tf.name_scope('stddev'):
#             stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
#         tf.summary.scalar('stddev', stddev)
#         tf.summary.scalar('max', tf.reduce_max(var))
#         tf.summary.scalar('min', tf.reduce_min(var))
#         tf.summary.histogram('histogram', var)
        
        
         # modified one in tf 2.7 and python 3.10
        
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)
          
        
#### function for read tfrecords
# def read_and_decode_emnet_mfcc(filename):
#     # first construct a queue containing a list of filenames.
#     # this lets a user split up there dataset in multiple files to keep
#     # size down
#     filename_queue = tf.train.string_input_producer(filename, name = 'queue')
#     # Unlike the TFRecordWriter, the TFRecordReader is symbolic
#     reader = tf.TFRecordReader()
#     # One can read a single serialized example from a filename
#     # serialized_example is a Tensor of type string.
#     _, serialized_example = reader.read(filename_queue)
#     # The serialized example is converted back to actual values.
#     # One needs to describe the format of the objects to be returned
#     features = tf.parse_single_example(
#         serialized_example,
#         features={
#             # We know the length of both fields. If not the
#             # tf.VarLenFeature could be used
#             'labels': tf.FixedLenFeature([], tf.int64),
#             'shapes': tf.FixedLenFeature([2], tf.int64),
#             'features': tf.VarLenFeature( tf.float32)
#         })
#     # now return the converted data
#     labels = features['labels']
#     shapes = features['shapes']
#     feats = features['features']
#     shapes = tf.cast(shapes, tf.int32)
#     feats2d = tf.reshape(feats.values, shapes)
#     feats1d = feats.values
#     return labels, shapes, feats2d


    # modified one in tf 2.7 and python 3.10
import tensorflow as tf

def read_and_decode_emnet_mfcc(filename):
    # First, construct a queue containing a list of filenames.
    # This allows splitting the dataset into multiple files.
    filename_queue = tf.data.Dataset.from_tensor_slices(filename)

    # Create a TFRecordDataset and initialize a TFRecordReader.
    dataset = tf.data.TFRecordDataset(filename_queue)

    # Define the feature description for parsing the serialized example.
    feature_description = {
        'labels': tf.io.FixedLenFeature([], tf.int64),
        'shapes': tf.io.FixedLenFeature([2], tf.int64),
        'features': tf.io.VarLenFeature(tf.float32)
    }

    # Parse the serialized examples.
    def _parse_function(serialized_example):
        features = tf.io.parse_single_example(serialized_example, feature_description)
        labels = features['labels']
        shapes = features['shapes']
        feats = features['features']
        shapes = tf.cast(shapes, tf.int32)
        feats2d = tf.reshape(feats.values, shapes)
    
        feats1d = feats.values
        
        return labels, shapes, feats2d

    parsed_dataset = dataset.map(_parse_function)
    return parsed_dataset

#     # Iterate over the dataset and collect the labels, shapes, and feats2d.
#     labels_list, shapes_list, feats2d_list = [], [], []
#     for labels, shapes, feats2d in parsed_dataset:
#         labels_list.append(labels)
#         shapes_list.append(shapes)
#         feats2d_list.append(feats2d)

#     return tf.concat(labels_list, axis=0), tf.concat(shapes_list, axis=0), tf.concat(feats2d_list, axis=0)


# def average_gradients(tower_grads):
#     average_grads = []
#     for grad_and_vars in zip(*tower_grads):
        
#         # Note that each grad_and_vars looks like the following:
#         #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
#         grads = []
#         for g, _ in grad_and_vars:
#             # Add 0 dimension to the gradients to represent the tower.
#             expanded_g = tf.expand_dims(g, 0)

#             # Append on a 'tower' dimension which we will average over below.
#             grads.append(expanded_g)

#         # Average over the 'tower' dimension.
#         grad = tf.concat(axis=0, values=grads)
#         grad = tf.reduce_mean(grad, 0)

#         v = grad_and_vars[0][1]
#         grad_and_var = (grad, v)
#         average_grads.append(grad_and_var)
#     return average_grads


    # modified one in tf 2.7 and python 3.10

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, axis=0)
        grad = tf.reduce_mean(grad, axis=0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


### Variable Initialization    
NUMGPUS = 1
BATCHSIZE = 4
ITERATION = 4000000
SAVE_INTERVAL = 4000
LOSS_INTERVAL = 100
TESTSET_INTERVAL = 2000
MAX_SAVEFILE_LIMIT = 1000
# DATASET_NAME = 'post_pooled/train_gmm.h5'
# DURATION_LIMIT = 1000 #(utterance below DURATION_LIMIT/100 seconds will be mismissed :default=1000)
# SPKCOUNT_LIMIT = 3 #(speaker with equal or less than this number will be dismissed :default=3)
# MIXTURE = 2048  # = number of softmax layer
# filelist = ['../post_pooled/swb_gmm','../post_pooled/sre_gmm'] # dataset for training
# post_mean = np.empty((0,MIXTURE),dtype='float32')
# post_std = np.empty((0,MIXTURE),dtype='float32')
utt_label = []
duration = np.empty(0,dtype='int')
spklab = []
TFRECORDS_FOLDER = 'data_add/tfrecords/'
SAVER_FOLDERNAME = 'saver'

if len(sys.argv)< 7:
    print ("not enough arguments")
    print ("command : ./new_training.py [nn_model_name] [learning rate] [input_dim(feat dim)] [is_batch_norm] [feature_filename]")
    print ("(example) ./new_training.py e2e_model 0.001 40 True aug_mfcc_fft512_hop160_vad_cmn")

is_batchnorm = False
NN_MODEL = sys.argv[1]
LEARNING_RATE = np.float64(sys.argv[2])
INPUT_DIM = int(sys.argv[3])
IS_BATCHNORM = sys.argv[4]
FEAT_TYPE = sys.argv[5]
ITERATION = int(sys.argv[6])
BATCHSIZE = int(sys.argv[7])
#NN_MODEL = 'new_nn_model'
#LEARNING_RATE = 0.001
#INPUT_DIM = 40
#IS_BATCHNORM = True
#FEAT_TYPE = 'mfcc_fft512_hop160_vad_cmn'

SAVER_FOLDERNAME = 'saver/'+NN_MODEL+'_'+FEAT_TYPE
if IS_BATCHNORM=='True':
    SAVER_FOLDERNAME = SAVER_FOLDERNAME + '_BN'
    is_batchnorm = True
nn_model = __import__(NN_MODEL)

# records_list = []
# for i in range(0,1):
#     records_list.append(TFRECORDS_FOLDER+'mgb3_train.'+str(i)+'.tfrecords')
records_shuffle_list = []
for i in range(0,1):
    records_shuffle_list.append(TFRECORDS_FOLDER+'mgb3_'+FEAT_TYPE+'_train_shuffle.'+str(i)+'.tfrecords')
records_dev_shuffle_list = []
for i in range(0,1):
    records_shuffle_list.append(TFRECORDS_FOLDER+'mgb3_'+FEAT_TYPE+'_dev_shuffle.'+str(i)+'.tfrecords')
    records_shuffle_list.append(TFRECORDS_FOLDER+'mgb3_'+FEAT_TYPE+'_dev_shuffle.'+str(i)+'.tfrecords')
    records_shuffle_list.append(TFRECORDS_FOLDER+'mgb3_'+FEAT_TYPE+'_dev_shuffle.'+str(i)+'.tfrecords')
    records_shuffle_list.append(TFRECORDS_FOLDER+'mgb3_'+FEAT_TYPE+'_dev_shuffle.'+str(i)+'.tfrecords')
    records_shuffle_list.append(TFRECORDS_FOLDER+'mgb3_'+FEAT_TYPE+'_dev_shuffle.'+str(i)+'.tfrecords')
    records_shuffle_list.append(TFRECORDS_FOLDER+'mgb3_'+FEAT_TYPE+'_dev_shuffle.'+str(i)+'.tfrecords')
    records_shuffle_list.append(TFRECORDS_FOLDER+'mgb3_'+FEAT_TYPE+'_dev_shuffle.'+str(i)+'.tfrecords')
    records_shuffle_list.append(TFRECORDS_FOLDER+'mgb3_'+FEAT_TYPE+'_dev_shuffle.'+str(i)+'.tfrecords')
    records_shuffle_list.append(TFRECORDS_FOLDER+'mgb3_'+FEAT_TYPE+'_dev_shuffle.'+str(i)+'.tfrecords')


dataset = read_and_decode_emnet_mfcc(records_shuffle_list)
 
dataset = dataset.padded_batch(
    BATCHSIZE,
    padded_shapes=(
        tf.TensorShape([]),      # labels
        tf.TensorShape([None]),  # feats
        tf.TensorShape([None, None])  # shapes
    ),
    drop_remainder=True
)
 

# Prefetch the next batch for performance optimization
# dataset = dataset.prefetch(1)
 
# Obtain an iterator from the dataset
# iterator = iter(dataset)
 
# Get the next batch of data
# labels_batch, shapes_batch, feats_batch = next(iterator)
 

 
   
#FEAT_TYPE = 'mfcc_fft512_hop160_vad_cmn'
FEAT_TYPE = FEAT_TYPE.split('_exshort')[0]
FEAT_TYPE = FEAT_TYPE.split('aug_')[-1]
FEAT_TYPE = FEAT_TYPE.split('vol_')[-1]
FEAT_TYPE = FEAT_TYPE.split('speed_')[-1]
records_test_list = []
for i in range(0,1):
    records_test_list.append(TFRECORDS_FOLDER+'mgb3_'+FEAT_TYPE+'_test_shuffle.'+str(i)+'.tfrecords')


#data for validation


vali_dataset = read_and_decode_emnet_mfcc(records_test_list)
 
vali_dataset = vali_dataset.padded_batch(
    BATCHSIZE,
    padded_shapes=(
        tf.TensorShape([]),      # labels
        tf.TensorShape([None]),  # feats
        tf.TensorShape([None, None])  # shapes
    ),
    drop_remainder=True
)
# vali_dataset = vali_dataset.prefetch(1)

# vali_labels_batch, vali_shapes_batch, vali_feats_batch = next(iter(vali_dataset))
model=nn_model.B_model() 
    
softmax_num = 5
 
learning_rate_fn = tf.keras.optimizers.schedules.ExponentialDecay(LEARNING_RATE, decay_steps=50000, decay_rate=0.98, staircase=True)
 
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate_fn)#ok
 

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

valid_loss = tf.keras.metrics.Mean(name='valid_loss')
valid_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')


@tf.function(experimental_relax_shapes=True)

def train_step(model, feats_batch, labels_batch,shapes_batch, softmax_num, INPUT_DIM, IS_BATCHNORM):

    with tf.GradientTape() as tape:
        predictions = model(feats_batch, shapes_batch, softmax_num, True, INPUT_DIM, IS_BATCHNORM)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_batch, logits= predictions))
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss.update_state(loss)
    train_metric.update_state(labels_batch, predictions)


                              
@tf.function(experimental_relax_shapes=True)
def valid_step(model, vali_feats_batch,vali_labels_batch, vali_shapes_batch,softmax_num, INPUT_DIM, IS_BATCHNORM):
    predictions = model(vali_feats_batch, vali_shapes_batch,softmax_num,False, INPUT_DIM, IS_BATCHNORM)
    batch_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=vali_labels_batch, logits= predictions))
    valid_loss.update_state(batch_loss)
    valid_metric.update_state(vali_labels_batch, predictions)
    
import time
def train_model(model,dataset,vali_dataset,epochs):

    for epoch in tf.range(1,epochs+1):
        start_time = time.time()
        for labels_batch, shapes_batch, feats_batch in dataset:
            train_step(model,feats_batch, labels_batch,shapes_batch, softmax_num, INPUT_DIM, IS_BATCHNORM)
#             print('current learning rate')#new line added
#             print(optimizer._decayed_lr(tf.float32))#newline added

        for vali_labels_batch, vali_shapes_batch, vali_feats_batch in vali_dataset:
            valid_step(model,vali_feats_batch,vali_labels_batch, vali_shapes_batch,softmax_num, INPUT_DIM, IS_BATCHNORM)

        logs = 'Epoch={},Loss:{},Accuracy:{},Valid Loss:{},Valid Accuracy:{}'

        if epoch%1 ==0:
#             printbar()
            tf.print(tf.strings.format(logs,
            (epoch,train_loss.result(),train_metric.result(),valid_loss.result(),valid_metric.result())))
            tf.print("")
            print("Time taken: %.2fs" % (time.time() - start_time))
#             print("Time taken: %.2fs" % (time.time()))

        train_loss.reset_states()
        valid_loss.reset_states()
        train_metric.reset_states()
        valid_metric.reset_states()
        

train_model(model,dataset,vali_dataset,60) 

# for labels_batch, shapes_batch, feats_batch in dataset:
#             valid_step(model,feats_batch, labels_batch,shapes_batch, softmax_num, INPUT_DIM, IS_BATCHNORM)

# for vali_labels_batch, vali_shapes_batch, vali_feats_batch in vali_dataset:
#             valid_step(model,vali_feats_batch,vali_labels_batch, vali_shapes_batch,softmax_num, INPUT_DIM, IS_BATCHNORM)

# logs = 'Loss:{},Accuracy:{},Valid Loss:{},Valid Accuracy:{}'
# tf.print(tf.strings.format(logs,
#             (train_loss.result(),train_metric.result(),valid_loss.result(),valid_metric.result())))


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
   

                                    
                                   