#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 18:41:22 2018

@author: andrey
"""
#основа кода взята с гита(keras_examples)
from __future__ import print_function

import numpy as np
import time
from keras import backend as K
from keras.models import load_model

MATRIX = 'matrix'
VECTOR = 'vector'

# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def get_model(model_name):
#    model = load_model('/home/andrey/datasetsNN/landScapes/landScape_3000_32/model_AlexNet_fullConv_30_07_2018.hdf5')
    model = load_model(model_name)
    print('Model loaded.')
    model.summary()
    return model

def get_kept_filters(model,count_filters,input_img_width,input_img_height,input_count_filters,type_out=MATRIX):
    input_img = model.input
    layer_dict = dict([(layer.name, layer) for layer in model.layers[0:]])# было model.layers[1:]
    def normalize(x):
        # utility function to normalize a tensor by its L2 norm
        return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())
    
    
    kept_filters = []
    for filter_index in range(count_filters): # было 200
        # we only scan through the first 200 filters,
        # but there are actually 512 of them
#        print('Processing filter %d' % filter_index)
#        start_time = time.time()
    
        # we build a loss function that maximizes the activation
        # of the nth filter of the layer considered 
        #(мы создаем функцию потерь, которая максимизирует n-й фильтр рассматриваемого слоя)
        layer_output = layer_dict[layer_name].output
        if K.image_data_format() == 'channels_first':
            if type_out == MATRIX:
                loss = K.mean(layer_output[:, filter_index, :, :])
            if type_out == VECTOR:
                loss = K.mean(layer_output[:, filter_index, :])
                
        else:
            if type_out == MATRIX:
                loss = K.mean(layer_output[:, :, :, filter_index])
            if type_out == VECTOR:
                loss = K.mean(layer_output[:, :, filter_index])
                    
        # we compute the gradient of the input picture wrt this loss
        # мы вычисляем градиент входного изображения по этой потере
        grads = K.gradients(loss, input_img)[0]
    
        # normalization trick: we normalize the gradient
        grads = normalize(grads)
    
        # this function returns the loss and grads given the input picture
        iterate = K.function([input_img], [loss, grads])
    
        # step size for gradient ascent
#        step = 0.1
        step = 1.
    
        # we start from a gray image with some random noise
    #    if K.image_data_format() == 'channels_first':
    #        input_img_data = np.random.random((1, 3, img_width, img_height))
    #    else:
        input_img_data = np.random.random((1, input_img_width, input_img_height, input_count_filters))
        input_img_data = (input_img_data - 0.5) * 20 + 128
        
    
        # we run gradient ascent for 20 steps
        for i in range(20):
            loss_value, grads_value = iterate([input_img_data])
            input_img_data += grads_value * step
    
    #        print('Current loss value:', loss_value)
    #        if loss_value <= 0.:
    #            # some filters get stuck to 0, we can skip them
    #            break
    
        # decode the resulting input image
    #    if loss_value > -10000: # было ноль
        img = deprocess_image(input_img_data[0])
        kept_filters.append((img, loss_value))
#        end_time = time.time()
#        print('Filter %d processed in %ds' % (filter_index, end_time - start_time))
    return kept_filters
    
    
def get_stitched_filters(n1,n2,kept_filters,input_img_width,input_img_height):   #n1;n2 - кол-во филтров по горизонтали и вертикали 
    # we will stich the best 64 filters on a 8 x 8 grid.
#    n1 = 4
#    n2 = 8
    
    # the filters that have the highest loss are assumed to be better-looking.
    # we will only keep the top 64 filters.
    #kept_filters.sort(key=lambda x: x[1], reverse=True)
    #kept_filters = kept_filters[:n1 * n2]
    
    # build a black picture with enough space for
    # our 8 x 8 filters of size 128 x 128, with a 5px margin in between
    margin = 5
    width = n1 * input_img_width + (n1-1) * margin
    height = n2 * input_img_height + (n2-1) * margin
    stitched_filters = np.zeros((width, height, 2))
    
    # fill the picture with our saved filters
    for i in range(n1):
        for j in range(n2):
            img, loss = kept_filters[i * n1 + j]
            stitched_filters[(input_img_width + margin) * i: (input_img_width + margin) * i + input_img_width,
                             (input_img_height + margin) * j: (input_img_height + margin) * j + input_img_height, :] = img
    return stitched_filters                         

def save_image(path_name, image):        
    import imageio 
#    imageio.imwrite('cnn_visualization_pictures/stitched32_conv_2d_3_1.jpg', stitched_filters[:, :, 1].astype(np.uint8))
    imageio.imwrite(path_name, image)

#%% for conv2d
import os    
if __name__ == "__main__":   
    model  = get_model('/home/andrey/datasetsNN/landScapes/landScape_3000_32/model_AlexNet_fullConv550_01_08_2018.hdf5')
    count_2d_conv = 3
    count_filters =np.array([64,64,32])
    n1 = np.array([8,8,4])
    n2 = np.array([8,8,8])
    input_img_width = 32
    input_img_height = 32
    input_count_filters = 2
    path_save = 'dissection/report/'
    for i in range(2,3):
        print('startconv number = ',i)
        layer_name = 'conv2d_' + str(i+1)
        kept_filters = get_kept_filters(model,count_filters[i],input_img_width,input_img_height,input_count_filters)
        stitched_filters = get_stitched_filters(n1[i],n2[i],kept_filters,input_img_width,input_img_height)
#        for j in range(input_count_filters):
#            save_image( os.path.join(path_save,'stitched'+str(count_filters[i])+'_conv_2d_'+str(i+1)+'_'+str(j) + '.jpg') ,
#                       stitched_filters[:, :, j].astype(np.uint8)  )
#%% conv1d 

import os    
if __name__ == "__main__":   
    model  = get_model('/home/andrey/datasetsNN/landScapes/landScape_3000_32/model_AlexNet_fullConv550_01_08_2018.hdf5')
    count_1d_conv = 3
    count_filters =np.array([32,8,1])
    n1 = np.array([4,2,1])
    n2 = np.array([8,4,1])#n1*n2 = count_filters(нужны для зрительного представления)
#    layer_name = 'conv2d_1'
    input_img_width = 32
    input_img_height = 32
    input_count_filters = 2
    path_save = 'dissection/report/'
    for i in range(count_1d_conv):
        print('startconv number = ',i)
        layer_name = 'conv1d_' + str(i+1)
        kept_filters = get_kept_filters(model,count_filters[i],input_img_width,input_img_height,input_count_filters,VECTOR)
        stitched_filters = get_stitched_filters(n1[i],n2[i],kept_filters,input_img_width,input_img_height)
        for j in range(input_count_filters):
            save_image( os.path.join(path_save,'stitched'+str(count_filters[i])+'_conv_1d_'+str(i+1)+'_'+str(j) + '.jpg') ,
                       stitched_filters[:, :, j].astype(np.uint8)  )
    
#%%
print(len(kept_filters))
print(len(kept_filters[0]))
print(kept_filters[0][0].shape)
#%%
print(stitched_filters.shape)        