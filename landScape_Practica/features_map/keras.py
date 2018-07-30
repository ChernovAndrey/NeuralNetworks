#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 18:41:22 2018

@author: andrey
"""

'''Visualization of the filters of VGG16, via gradient ascent in input space.
This script can run on CPU in a few minutes.
Results example: http://i.imgur.com/4nj4KjN.jpg
'''
#%%
from __future__ import print_function

import numpy as np
import time
#from keras.preprocessing.image import save_img
#from keras.applications import vgg16
from keras import backend as K
from keras.models import load_model
# dimensions of the generated pictures for each filter.
img_width = 32
img_height = 32
count_filters = 32

# the name of the layer we want to visualize
# (see model definition at keras/applications/vgg16.py)
layer_name = 'conv2d_1'

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


# build the VGG16 network with ImageNet weights
#model = vgg16.VGG16(weights='imagenet', include_top=False)
model = load_model('/home/andrey/datasetsNN/landScapes/landScape_3000_32/AlexNet/model_AlexNet_second.hdf5')
print('Model loaded.')

model.summary()
model.layers
#%%
# this is the placeholder for the input images
input_img = model.input

# get the symbolic outputs of each "key" layer (we gave them unique names).
layer_dict = dict([(layer.name, layer) for layer in model.layers[0:]])# было model.layers[1:]

print(layer_dict)
#%%
#%%
def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())


kept_filters = []
for filter_index in range(count_filters): # было 200
    # we only scan through the first 200 filters,
    # but there are actually 512 of them
    print('Processing filter %d' % filter_index)
    start_time = time.time()

    # we build a loss function that maximizes the activation
    # of the nth filter of the layer considered
    layer_output = layer_dict[layer_name].output
    if K.image_data_format() == 'channels_first':
        loss = K.mean(layer_output[:, filter_index, :, :])
    else:
        loss = K.mean(layer_output[:, :, :, filter_index])

    # we compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, input_img)[0]

    # normalization trick: we normalize the gradient
    grads = normalize(grads)

    # this function returns the loss and grads given the input picture
    iterate = K.function([input_img], [loss, grads])

    # step size for gradient ascent
    step = 1.

    # we start from a gray image with some random noise
#    if K.image_data_format() == 'channels_first':
#        input_img_data = np.random.random((1, 3, img_width, img_height))
#    else:
    input_img_data = np.random.random((1, img_width, img_height, 2))
    input_img_data = (input_img_data - 0.5) * 20 + 128
    

    # we run gradient ascent for 20 steps
    for i in range(20):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

        print('Current loss value:', loss_value)
#        if loss_value <= 0.:
#            # some filters get stuck to 0, we can skip them
#            break

    # decode the resulting input image
    if loss_value > -10000: # было ноль
        img = deprocess_image(input_img_data[0])
        kept_filters.append((img, loss_value))
    end_time = time.time()
    print('Filter %d processed in %ds' % (filter_index, end_time - start_time))
#%%
# we will stich the best 64 filters on a 8 x 8 grid.
n1 = 4
n2 = 8

# the filters that have the highest loss are assumed to be better-looking.
# we will only keep the top 64 filters.
#kept_filters.sort(key=lambda x: x[1], reverse=True)
#kept_filters = kept_filters[:n1 * n2]

# build a black picture with enough space for
# our 8 x 8 filters of size 128 x 128, with a 5px margin in between
margin = 5
width = n1 * img_width + (n1 - 1) * margin
height = n2 * img_height + (n2 - 1) * margin
stitched_filters = np.zeros((width, height, 2))

# fill the picture with our saved filters
for i in range(n1):
    for j in range(n2):
        img, loss = kept_filters[i * n1 + j]
        stitched_filters[(img_width + margin) * i: (img_width + margin) * i + img_width,
                         (img_height + margin) * j: (img_height + margin) * j + img_height, :] = img

#%%
import imageio 
import numpy as np
imageio.imwrite('stitched32_conv_2d_1_0.jpg', stitched_filters[:, :, 0].astype(np.uint8))