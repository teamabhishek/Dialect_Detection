import tensorflow as tf 
from tensorflow import keras
import numpy as np
#import tensorflow_quantum as tfq

import os
import io
import csv
import random
import numpy as np
import pandas as pd
import tensorflow as tf
#import tensorflow_hub as hub
#import tensorflow_quantum as tfq
#import cirq
#import sympy
#from cirq.contrib.svg import SVGCircuit
#import tensorflow_io as tfio
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from IPython.display import Audio

class B_batch_norm_wrapper_1dcnn(keras.layers.Layer):
    def __init__(self,name):
        super().__init__()
#         self.name=name
        
    def build(self,input_shape):
        self.scale = tf.Variable(       
            initial_value=tf.ones_initializer()(shape=(input_shape[-1],), dtype=tf.float32),
            trainable=True,           
            name=self.name + 'scale'
        )
        self.beta =  tf.Variable(            
           initial_value=tf.zeros_initializer()(shape=(input_shape[-1],), dtype=tf.float32), 
            trainable=True,
            name=self.name + 'beta'
            
        )
        self.pop_mean = tf.Variable(         
            initial_value=tf.zeros_initializer()(shape=(input_shape[-1],), dtype=tf.float32), 
            trainable=False,
            name=self.name + 'pop_mean'
            )
           
        self.pop_var = tf.Variable(       
            initial_value=tf.ones_initializer()(shape=(input_shape[-1],), dtype=tf.float32),
            trainable=False,
           name=self.name + 'pop_var')

    def call(self,inputs, shape_list,is_training,is_batchnorm):#(conv1, is_training,'bn1',shape_list,is_batchnorm)
        if is_batchnorm:
            shape_list = tf.cast(shape_list, tf.float32)
#             print(shape_list)
            decay=0.999
            epsilon = 1e-3
            if is_training:
                batch_mean = tf.reduce_sum(inputs, axis=[0, 1]) / tf.reduce_sum(shape_list)  # for variable length input
                batch_var = tf.reduce_sum(tf.square(inputs - batch_mean), axis=[0, 1]) / tf.reduce_sum(shape_list)  # for variable length input
                train_mean = self.pop_mean.assign(self.pop_mean * decay + batch_mean * (1 - decay))
                train_var = self.pop_var.assign(self.pop_var * decay + batch_var * (1 - decay))

                with tf.control_dependencies([train_mean, train_var]):
                    return tf.nn.batch_normalization(inputs, batch_mean, batch_var, self.beta, self.scale, epsilon)
            else:
                return tf.nn.batch_normalization(inputs, self.pop_mean, self.pop_var, self.beta, self.scale, epsilon)
        else:
            return inputs
        

class B_batch_norm_wrapper_fc(keras.layers.Layer):
    def __init__(self,name):
        super().__init__()
#         self.name=name
        
    def build(self,input_shape):
        self.scale = self.add_weight(name=self.name + 'scale', shape=(input_shape[-1],), initializer=tf.ones_initializer(), trainable=True, dtype=tf.float32)
        self.beta = self.add_weight(name=self.name + 'beta', shape=(input_shape[-1],), initializer=tf.zeros_initializer(), trainable=True, dtype=tf.float32)
        self.pop_mean = self.add_weight(name=self.name + 'pop_mean', shape=(input_shape[-1],), initializer=tf.zeros_initializer(), trainable=False, dtype=tf.float32)
        self.pop_var = self.add_weight(name=self.name+ 'pop_var', shape=(input_shape[-1],), initializer=tf.ones_initializer(), trainable=False, dtype=tf.float32)
        
    def call(self,inputs,is_training,is_batchnorm):#(fc1, is_training,'bn5',is_batchnorm)
        decay=0.999
        epsilon = 1e-3
        if is_batchnorm:
            if is_training:
                batch_mean, batch_var = tf.nn.moments(inputs,axes=[0])
                train_mean = self.pop_mean.assign(self.pop_mean * decay + batch_mean * (1 - decay))
                train_var = self.pop_var.assign(self.pop_var * decay + batch_var * (1 - decay))

#                 train_mean = tf.assign(self.pop_mean,self.pop_mean * decay + batch_mean * (1 - decay))
#                 train_var = tf.assign(self.pop_var, self.pop_var * decay + batch_var * (1 - decay))
                with tf.control_dependencies([train_mean, train_var]):
                    return tf.nn.batch_normalization(inputs,batch_mean, batch_var, self.beta, self.scale, epsilon)
            else:
                return tf.nn.batch_normalization(inputs, self.pop_mean, self.pop_var, self.beta,self.scale, epsilon)
        else:
            return inputs

class B_fc_layer(keras.layers.Layer):
        def __init__(self,n_prev_weight,n_weight,name=None): 
            super().__init__()    
            self.W =  tf.Variable(
               
                initial_value=self.xavier_init(n_prev_weight,n_weight)(shape=[n_prev_weight, n_weight],dtype=tf.float32),
                trainable=True,
                name=name + 'W'
            )
            self.b =  tf.Variable(
                
                initial_value=tf.keras.initializers.Constant(value=0.001)(shape=[n_weight],dtype=tf.float32),
                trainable=True,
                name=name + 'b'
            )
        def call(self, inputs):  #ok             
            fc = tf.nn.bias_add(tf.matmul(inputs, self.W), self.b)
            return fc


        def xavier_init(self,n_inputs, n_outputs, uniform=True):
            if uniform:
                init_range = np.sqrt(6.0 / (n_inputs + n_outputs))
                return tf.random_uniform_initializer(-init_range, init_range)
            else:
                stddev = np.sqrt(3.0 / (n_inputs + n_outputs))
                return tf.truncated_normal_initializer(stddev=stddev)


class B_conv_layer(keras.layers.Layer):
        def __init__(self,num_channels, kernel_size, depth,name=None): 
            super().__init__()
            self.W = tf.Variable(
                shape=[kernel_size, num_channels, depth],
                initial_value=tf.keras.initializers.GlorotUniform()(shape=[kernel_size, num_channels, depth],dtype=tf.float32),
                trainable=True,
                name=name + 'W' 
            )
            self.b = tf.Variable(
                
                initial_value=tf.keras.initializers.Constant(value=0.001)(shape=[depth],dtype=tf.float32),
                trainable=True,
                name=name + 'b' 
            )


        def call(self, inputs,stride,shape_list):#(x,kernel_size,featdim,stride,depth,'conv1',shape_list)
            inputlayer = inputs
            conv =  ( tf.nn.bias_add( tf.nn.conv1d(inputlayer, self.W, stride, padding='SAME'), self.b))
            
            mask = tf.sequence_mask(shape_list,tf.shape(conv)[1]) # make mask with batch x frame size
            mask = tf.where(mask, tf.ones_like(mask,dtype=tf.float32), tf.zeros_like(mask,dtype=tf.float32))
            mask=tf.tile(mask, tf.stack([tf.shape(conv)[2],1])) #replicate make with depth size
            mask=tf.reshape(mask,[tf.shape(conv)[2], tf.shape(conv)[0], -1])
            mask = tf.transpose(mask,[1, 2, 0])
#             print (mask)
#            
            conv=tf.multiply(conv,mask)
#             print(conv)
            return conv



# class B_PQC_layer(keras.layers.Layer):
#     def __init__(self, upstream_symbols, managed_symbols):
#         super().__init__()
#         self.all_symbols = upstream_symbols + managed_symbols
#         self.upstream_symbols = upstream_symbols
#         self.managed_symbols = managed_symbols

  

#     def build(self, input_shape):
#         self.managed_weights = tf.Variable(
#             initial_value=tf.keras.initializers.RandomUniform(0, 2 * np.pi)(shape=(1, len(self.managed_symbols)),dtype=tf.float32),
#             trainable=True
#             )
        


      
#     def call(self, inputs):
#         # inputs are: circuit tensor, upstream values
#         upstream_shape = tf.gather(tf.shape(inputs[0]), 0)
#         tiled_up_weights = tf.tile(self.managed_weights, [upstream_shape, 1])
#         joined_params = tf.concat([inputs[1], tiled_up_weights], 1)
#         output=tfq.layers.Expectation()(inputs[0],
#                                         operators=inputs[2],
#                                         symbol_names=self.all_symbols,
#                                         symbol_values=joined_params)
#         return output
    

        
class B_model(keras.Model):
    """Combines the encoder and decoder into an end-to-end model for training."""
    def __init__(self):#(self, x1, y_, y_string, shapes_batch, softmax_num,is_training,input_dim, is_batchnorm):
        super().__init__()
        
        
        self.conv1 = B_conv_layer(40,5,500,"conv1")#it can be 40 0r 201(spec)
        assert self.conv1.weights == [self.conv1.W, self.conv1.b]
        self.conv2 = B_conv_layer(500,7,500,"conv2") 
        assert self.conv2.weights == [self.conv2.W, self.conv2.b]
        self.conv3 = B_conv_layer(500,1,500,"conv3")
        assert self.conv3.weights == [self.conv3.W, self.conv3.b]
        self.conv4 = B_conv_layer(500,1,3000,"conv4")
        assert self.conv4.weights == [self.conv4.W, self.conv4.b]
        
       
        
        self.fc1=B_fc_layer(3000,1500,"fc1")
        assert self.fc1.weights == [self.fc1.W, self.fc1.b]
        self.fc2=B_fc_layer(1500,600,"fc2")
        assert self.fc2.weights == [self.fc2.W, self.fc2.b]
        self.fc3=B_fc_layer(600,5,"fc3")#here change 600 to 40
        assert self.fc3.weights == [self.fc3.W, self.fc3.b]

        
        
        self.conv1_bn=B_batch_norm_wrapper_1dcnn("conv1_bn")
#         assert self.conv1_bn.weights == [self.conv1_bn.scale,self.conv1_bn.beta,self.conv1_bn.pop_mean,self.conv1_bn.pop_var]
        self.conv2_bn=B_batch_norm_wrapper_1dcnn("conv2_bn")
        self.conv3_bn=B_batch_norm_wrapper_1dcnn("conv3_bn")
        self.conv4_bn=B_batch_norm_wrapper_1dcnn("conv4_bn")
        
        self.fc1_bn=B_batch_norm_wrapper_fc("fc1_bn")
        self.fc2_bn=B_batch_norm_wrapper_fc("fc2_bn") 
#         self.PQC1=B_PQC_layer(['u0','u1','u2','u3','u4'],['m0','m1','m2','m3','m4','m5','m6','m7','m8','m9'])
        
        
             
    def call(self,x, shapes_batch,softmax_num,is_training, input_dim, is_batchnorm):
        shape_list = shapes_batch[:,0]
        is_exclude_short = False
        if is_exclude_short:
            #randomly select start of sequences
            sequence_limit = tf.reduce_min(shape_list)/2
#            sequence_limit = tf.cond(sequence_limit<=200, lambda: sequence_limit, lambda: tf.subtract(sequence_limit,200))
#             random_start_pt = tf.random_uniform([1],minval=0,maxval=sequence_limit,dtype=tf.int32)
            random_start_pt = tf.random.uniform(shape=[1], minval=0, maxval=sequence_limit, dtype=tf.int32)

            end_pt = tf.reduce_max(shape_list)
            x = tf.gather(x,tf.range(tf.squeeze(random_start_pt),end_pt),axis=1)
            shape_list = shape_list-random_start_pt

            #randomly chunk sequences
            batch_quantity = tf.size(shape_list)
            aug_list = tf.constant([200, 300, 400], dtype=tf.float32)
            aug_quantity = tf.size(aug_list)
#             rand_index = tf.random_uniform([batch_quantity],minval=0,maxval=aug_quantity-1,dtype=tf.int32)
            rand_index = tf.random.uniform(shape=[batch_quantity], minval=0, maxval=aug_quantity-1, dtype=tf.int32)

            rand_aug_list = tf.gather(aug_list,rand_index)

            shape_list_f = tf.cast(shape_list, tf.float32)
            temp = tf.multiply(shape_list_f, rand_aug_list/shape_list_f)
            aug_shape_list = tf.cast(temp, tf.int32)
            shape_list = tf.minimum(shape_list,aug_shape_list)
                    
#         featdim = input_dim #channel
#         weights = []
#         kernel_size =5
        stride = 1
#         depth = 500
                
        shape_list = shape_list/stride
#         print(shape_list)
        conv1 = self.conv1(x,stride,shape_list)
        conv1_bn = self.conv1_bn(conv1,shape_list,is_training,is_batchnorm)
        conv1r= tf.nn.relu(conv1_bn)
        
        
#         featdim = depth #channel
#         weights = []
#         kernel_size =7
        stride = 2
#         depth = 500
                
        shape_list = shape_list/stride
        conv2 = self.conv2(conv1r, stride,shape_list)
        conv2_bn = self.conv2_bn(conv2, shape_list,is_training,is_batchnorm)
        conv2r= tf.nn.relu(conv2_bn)
        
#         featdim = depth #channel
#         weights = []
#         kernel_size =1
        stride = 1
#         depth = 500
                
        shape_list = shape_list/stride
        conv3 = self.conv3(conv2r,stride,shape_list)
        conv3_bn = self.conv3_bn(conv3,shape_list,is_training,is_batchnorm)
        conv3r= tf.nn.relu(conv3_bn)
       
    #         featdim = depth #channel
#         weights = []
#         kernel_size =1
        stride = 1
#         depth = 3000
                
        shape_list = shape_list/stride
        conv4 = self.conv4(conv3r, stride,shape_list)
        conv4_bn = self.conv4_bn(conv4,shape_list,is_training,is_batchnorm)
        conv4r= tf.nn.relu(conv4_bn)
        
#         print (conv1)
        

        
        shape_list = tf.cast(shape_list, tf.float32)
        shape_list = tf.reshape(shape_list,[-1,1,1])
        mean = tf.reduce_sum(conv4r, axis=1, keepdims=True) / shape_list

#         mean = tf.reduce_sum(conv4r,1,keep_dims=True)/shape_list
        res1=tf.squeeze(mean,axis=1)
        

        fc1 = self.fc1(res1)
        fc1_bn = self.fc1_bn(fc1, is_training,is_batchnorm)
        ac1 = tf.nn.relu(fc1_bn)
        
        fc2 = self.fc2(ac1)
        fc2_bn = self.fc2_bn(fc2,is_training,is_batchnorm)
        ac2 = tf.nn.relu(fc2_bn)

        
        fc3 = self.fc3(ac2)#changed ac2 to ac5 
        
#         qubits = cirq.LineQubit.range(5)
#         params = sympy.symbols('u0,u1,u2,u3,u4,m0,m1,m2,m3,m4,m5,m6,m7,m8,m9')

#         circuit = cirq.Circuit()   
#         circuit += cirq.rx(params[0] * np.pi).on(qubits[0])
#         circuit += cirq.rx(params[1] * np.pi).on(qubits[1])
#         circuit += cirq.rx(params[2] * np.pi).on(qubits[2]) 
#         circuit += cirq.rx(params[3] * np.pi).on(qubits[3])
#         circuit += cirq.rx(params[4] * np.pi).on(qubits[4])    
        
        
#         circuit += cirq.ry(params[0] * np.pi).on(qubits[0])
#         circuit += cirq.ry(params[1] * np.pi).on(qubits[1])
#         circuit += cirq.ry(params[2] * np.pi).on(qubits[2]) 
#         circuit += cirq.ry(params[3] * np.pi).on(qubits[3])
#         circuit += cirq.ry(params[4] * np.pi).on(qubits[4]) 
#         for i in range(4):
#             circuit +=cirq.CNOT(qubits[4-i],qubits[4-i-1])
# #        
#         circuit += cirq.rx(params[0] * np.pi).on(qubits[0])
#         circuit += cirq.rx(params[1] * np.pi).on(qubits[1])
#         circuit += cirq.rx(params[2] * np.pi).on(qubits[2]) 
#         circuit += cirq.rx(params[3] * np.pi).on(qubits[3])
#         circuit += cirq.rx(params[4] * np.pi).on(qubits[4])    
       
        
#         circuit += cirq.ry(params[5] * np.pi).on(qubits[0])
#         circuit += cirq.ry(params[6] * np.pi).on(qubits[1])
#         circuit += cirq.ry(params[7] * np.pi).on(qubits[2]) 
#         circuit += cirq.ry(params[8] * np.pi).on(qubits[3])
#         circuit += cirq.ry(params[9] * np.pi).on(qubits[4]) 
#         for i in range(4):
#             circuit +=cirq.CNOT(qubits[4-i],qubits[4-i-1])


        
#         Moperators = [cirq.Z(qubits[0]),cirq.Z(qubits[1]),cirq.Z(qubits[2]),cirq.Z(qubits[3]),cirq.Z(qubits[4])]
#         tinput=tfq.convert_to_tensor([circuit for _ in range(4)])
#         PQC1=self.PQC1([tinput ,fc3,Moperators])
# #         
#         return PQC1
        return fc3
# 
