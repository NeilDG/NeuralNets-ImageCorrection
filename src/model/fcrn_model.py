import tensorflow as tf
import numpy as np

#IN_H = 304
#IN_W = 228
weights = np.load('D:/Users/delgallegon/Documents/GithubProjects/NeuralNets-ImageDepthExperiment/inference_code/NYU_ResNet-UpProj.npy', encoding='latin1', allow_pickle = True).item()

#taken from Deeper Depth Prediction with Fully Convolutional Residual Networks
def sample_build_model(input):

    shape = input.get_shape()
    print(shape)

    conv1 = conv(input=input,name='conv1',stride=2,kernel_size=(7,7),num_filters=64)
    bn_conv1 = batch_norm(input=conv1,name='bn_conv1',relu=True)
    pool1 = tf.nn.max_pool(bn_conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME',name='pool1')

    shape = pool1.get_shape()
    print(shape)
    
    #d1 refers to lower branch num_filters, d2 refers to upper branch num_filters
    res2a_relu = build_res_block(input=pool1,block_name='2a',d1=64,d2=256,projection=True,down_size=False)
    res2b_relu = build_res_block(input=res2a_relu,block_name='2b',d1=64,d2=256)
    res2c_relu = build_res_block(input=res2b_relu,block_name='2c',d1=64,d2=256)

    shape = res2c_relu.get_shape()
    print(shape)

    res3a_relu = build_res_block(input=res2c_relu,block_name='3a',d1=128,d2=512,projection=True)
    res3b_relu = build_res_block(input=res3a_relu,block_name='3b',d1=128,d2=512)
    res3c_relu = build_res_block(input=res3b_relu,block_name='3c',d1=128,d2=512)
    res3d_relu = build_res_block(input=res3c_relu,block_name='3d',d1=128,d2=512)

    shape = res3d_relu.get_shape()
    print(shape)

    res4a_relu = build_res_block(input=res3d_relu,block_name='4a',d1=256,d2=1024,projection=True)
    res4b_relu = build_res_block(input=res4a_relu,block_name='4b',d1=256,d2=1024)
    res4c_relu = build_res_block(input=res4b_relu,block_name='4c',d1=256,d2=1024)
    res4d_relu = build_res_block(input=res4c_relu,block_name='4d',d1=256,d2=1024)
    res4e_relu = build_res_block(input=res4d_relu,block_name='4e',d1=256,d2=1024)
    res4f_relu = build_res_block(input=res4e_relu,block_name='4f',d1=256,d2=1024)

    shape = res4f_relu.get_shape()
    print(shape)

    res5a_relu = build_res_block(input=res4f_relu,block_name='5a',d1=512,d2=2048,projection=True)
    res5b_relu = build_res_block(input=res5a_relu,block_name='5b',d1=512,d2=2048)
    res5c_relu = build_res_block(input=res5b_relu,block_name='5c',d1=512,d2=2048)

    shape = res5c_relu.get_shape()
    print(shape)

    layer1 = conv(input=res5c_relu,name='layer1',stride=1,kernel_size=(1,1),num_filters=1024)
    layer1_BN = batch_norm(input=layer1,name='layer1_BN',relu=False)

    shape = layer1_BN.get_shape()
    print(shape)
    # UP-CONV

    up_2x = build_up_conv_block(input=layer1_BN,block_name='2x',num_filters=512)
    up_4x = build_up_conv_block(input=up_2x, block_name='4x', num_filters=256)
    up_8x = build_up_conv_block(input=up_4x, block_name='8x', num_filters=128)
    up_16x = build_up_conv_block(input=up_8x, block_name='16x', num_filters=64)

    drop = tf.nn.dropout(up_16x, keep_prob = 1., name='drop')
    pred = conv(input=drop,name='ConvPred',stride=1,kernel_size=(3,3),num_filters=1)

    return pred


def build_up_conv_block(input,block_name,num_filters, trainable = False):

    # Branch 1
    br1_name = "%s_br1" % (block_name)
    branch1 = build_up_conv_branch(input=input,branch_name=br1_name,relu=True,num_filters=num_filters, trainable = trainable)

    layerName = "layer%s_Conv" % (block_name)
    layer_br1_conv = conv(input=branch1, name=layerName, stride=1, kernel_size=(3, 3), num_filters=num_filters,
                  trainable = trainable,use_bias=True)

    layerName = "layer%s_BN" % (block_name)
    layer_br1_bn = batch_norm(input=layer_br1_conv, name=layerName)

    # Branch 2
    br2_name = "%s_br2" % (block_name)
    branch2 = build_up_conv_branch(input=input, branch_name=br2_name,num_filters=num_filters, trainable = trainable)

    layerName = "layer%s_Sum" % (block_name)
    branches_sum = tf.add(layer_br1_bn,branch2, name=layerName)

    layerName = "layer%s_ReLU" % (block_name)
    block = tf.nn.relu(branches_sum, name=layerName)

    return block

def build_up_conv_branch(input,branch_name,num_filters,relu = False, trainable = False):

    layerName = "layer%s_ConvA" % (branch_name)
    conv_A = conv(input=input, name=layerName, stride=1, kernel_size=(3,3), num_filters=num_filters,
                          trainable = trainable, use_bias=True)

    layerName = "layer%s_ConvB" % (branch_name)
    padded_input_B = tf.pad(input, [[0, 0], [1, 0], [1, 1], [0, 0]], "CONSTANT")
    conv_B = conv(input=padded_input_B, name=layerName, stride=1, kernel_size=(2, 3), num_filters=num_filters,
                  trainable = trainable, use_bias=True,padding='VALID')

    layerName = "layer%s_ConvC" % (branch_name)
    padded_input_C = tf.pad(input, [[0, 0], [1, 1], [1, 0], [0, 0]], "CONSTANT")
    conv_C = conv(input=padded_input_C, name=layerName, stride=1, kernel_size=(3, 2), num_filters=num_filters,
                  trainable = trainable, use_bias=True,padding='VALID')

    layerName = "layer%s_ConvD" % (branch_name)
    padded_input_D = tf.pad(input, [[0, 0], [1, 0], [1, 0], [0, 0]], "CONSTANT")
    conv_D = conv(input=padded_input_D, name=layerName, stride=1, kernel_size=(2, 2), num_filters=num_filters,
                  trainable = trainable, use_bias=True,padding='VALID')

    left = interleave([conv_A, conv_B], axis=1)
    right = interleave([conv_C, conv_D], axis=1)
    I = interleave([left, right], axis=2)

    layerName = "layer%s_BN" % (branch_name)
    branch = batch_norm(input=I,name=layerName)

    layerName = "layer%s_ReLU" % (branch_name)
    if relu:
        branch = tf.nn.relu(branch,name=layerName)

    return branch

def interleave(tensors, axis):
    old_shape = get_incoming_shape(tensors[0])[1:]
    new_shape = [-1] + old_shape
    new_shape[axis] *= len(tensors)
    return tf.reshape(tf.stack(tensors, axis + 1), new_shape)

def get_incoming_shape(incoming):
    """ Returns the incoming data shape """
    if isinstance(incoming, tf.Tensor):
        return incoming.get_shape().as_list()
    elif type(incoming) in [np.array, list, tuple]:
        return np.shape(incoming)
    else:
        raise Exception("Invalid incoming layer.")

def build_res_block(input,block_name,d1,d2,projection = False,down_size = True, trainable = False):
    # Projection Block
    #                  Upper Branch
    # res[conv(res)-Bn(bn)]2[Box_no]a[Block_char]_branch2[Lower(1)-Upper(2)]a[order @ Branch(a -> b -> c -> d)]
    upper_branch = build_upper_branch(input=input,block_name=block_name,in_depth=d1,out_depth=d2,projection=projection,down_size=down_size,
                                      trainable = trainable)
    lower_branch = build_lower_branch(input=input,block_name=block_name,out_depth=d2,projection = projection,down_size=down_size,
                                      trainable = trainable)
    res =  tf.add(upper_branch,lower_branch,name='res' + block_name)
    res_relu = tf.nn.relu(res,name='res2a_relu')
    return res_relu

def build_lower_branch(input,block_name,out_depth,projection=False,down_size = True, trainable = False):
    padding = 'SAME'
    stride = 1
    if down_size:
        padding = 'VALID'
        stride = 2
    if projection:
        res_branch1 = conv(input=input,name='res'+block_name+'_branch1',stride=stride,kernel_size=(1,1),num_filters=out_depth,
                           trainable = trainable,use_bias=False,padding=padding)
        bn_branch1 = batch_norm(input=res_branch1,name='bn'+block_name+'_branch1')
        return bn_branch1
    return input

def build_upper_branch(input,block_name,in_depth,out_depth,projection = False,down_size = True, trainable = False):
    stride = 1
    padding = 'SAME'
    if projection and down_size:
        stride = 2
        padding = 'VALID'
    res_branch2a = conv(input=input, name='res'+block_name+'_branch2a', stride=stride, kernel_size=(1,1), num_filters=in_depth,
                        trainable = trainable, use_bias=False,padding=padding)
    bn_branch2a = batch_norm(input=res_branch2a, name='bn'+ block_name +'_branch2a', relu=True)
    res_branch2b = conv(input=bn_branch2a, name='res'+block_name+'_branch2b', stride=1, kernel_size=(3,3), num_filters=in_depth,
                          trainable = trainable, use_bias=False)
    bn_branch2b = batch_norm(input=res_branch2b, name='bn'+block_name+'_branch2b', relu=True)
    res_branch2c = conv(input=bn_branch2b, name='res'+block_name+'_branch2c', stride=1, kernel_size=(1,1), num_filters=out_depth,
                          trainable = trainable, use_bias=False)
    bn_branch2c = batch_norm(input=res_branch2c, name='bn'+block_name+'_branch2c')

    return bn_branch2c

"""Arguments
   tensor: Tensor input.
   binary_mask: Tensor, a mask with the same size as tensor, channel size = 1
   filters: Integer, the dimensionality of the output space (i.e. the number
      of filters in the convolution).
   kernel_size: An integer or tuple/list of 2 integers, specifying the
      height and width of the 2D convolution window.
   strides: An integer or tuple/list of 2 integers,
      specifying the strides of the convolution along the height and width.
   l2_scale: float, A scalar multiplier Tensor. 0.0 disables the regularizer.
   
 Returns:
   Output tensor, binary mask.
 """
#Taken from Sparsity Invariant CNNs
def sparse_conv(tensor,binary_mask = None,filters=32,kernel_size=3,strides=2,l2_scale=0.0):

    if binary_mask == None: #first layer has no binary mask
        b,h,w,c = tensor.get_shape()
        channels=tf.split(tensor,c,axis=3)
        #assume that if one channel has no information, all channels have no information
        binary_mask = tf.where(tf.equal(channels[0], 0), tf.zeros_like(channels[0]), tf.ones_like(channels[0])) #mask should only have the size of (B,H,W,1)
    
    features = tf.multiply(tensor,binary_mask)
    features = tf.layers.conv2d(features, filters=filters, kernel_size=kernel_size, strides=(strides, strides), trainable=True, use_bias=False, padding="same",kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_scale))

    norm = tf.layers.conv2d(binary_mask, filters=filters,kernel_size=kernel_size,strides=(strides, strides),kernel_initializer=tf.ones_initializer(),trainable=False,use_bias=False,padding="same")
    norm = tf.where(tf.equal(norm,0),tf.zeros_like(norm),tf.reciprocal(norm))
    _,_,_,bias_size = norm.get_shape()

    #b = tf.Variable(tf.constant(0.0, shape=[bias_size]),trainable=True)
    feature = tf.multiply(features,norm) #+ b
    mask = tf.layers.max_pooling2d(binary_mask,strides = strides,pool_size=kernel_size,padding="same")

    return feature,mask

def conv_new(input,name,stride,kernel_size,num_filters,trainable = False , use_bias = True,padding = 'SAME'):
    if use_bias:
        layer = tf.layers.conv2d(inputs=input,
                                 strides=(stride, stride),
                                 filters=num_filters,
                                 kernel_size=kernel_size,
                                 padding=padding,
                                 name=name,
                                 trainable=trainable,
                                 kernel_initializer = tf.contrib.layers.xavier_initializer(),
                                 bias_initializer = tf.contrib.layers.xavier_initializer(),
                                 #kernel_initializer=tf.constant_initializer(weights[name]['weights'], dtype=tf.float32),
                                 #bias_initializer=tf.constant_initializer(weights[name]['biases'], dtype=tf.float32),
                                 use_bias=use_bias)
    else :
        layer = tf.layers.conv2d(inputs=input,
                                 strides=(stride, stride),
                                 filters=num_filters,
                                 kernel_size=kernel_size,
                                 padding=padding,
                                 name=name,
                                 trainable=trainable,
                                 kernel_initializer = tf.contrib.layers.xavier_initializer(),
                                 #kernel_initializer=tf.constant_initializer(weights[name]['weights'], dtype=tf.float32),
                                 use_bias=use_bias)

    return layer

def conv(input,name,stride,kernel_size,num_filters,trainable = False , use_bias = True,padding = 'SAME'):
    if use_bias:
        layer = tf.layers.conv2d(inputs=input,
                                 strides=(stride, stride),
                                 filters=num_filters,
                                 kernel_size=kernel_size,
                                 padding=padding,
                                 name=name,
                                 trainable=trainable,
                                 #kernel_initializer = tf.keras.initializers.glorot_normal(),
                                 #bias_initializer = tf.keras.initializers.glorot_normal(),
                                 kernel_initializer=tf.constant_initializer(weights[name]['weights'], dtype=tf.float32),
                                 bias_initializer=tf.constant_initializer(weights[name]['biases'], dtype=tf.float32),
                                 use_bias=use_bias)
    else :
        layer = tf.layers.conv2d(inputs=input,
                                 strides=(stride, stride),
                                 filters=num_filters,
                                 kernel_size=kernel_size,
                                 padding=padding,
                                 name=name,
                                 trainable=trainable,
                                 #kernel_initializer = tf.contrib.layers.xavier_initializer(),
                                 kernel_initializer=tf.constant_initializer(weights[name]['weights'], dtype=tf.float32),
                                 use_bias=use_bias)

    return layer


def batch_norm(input,name,relu = False):
    '''
    layer = tf.layers.batch_normalization(input=input ,
                                          moving_variance_initializer=tf.constant_initializer(weights[name]['variance'], dtype=tf.float32),
                                          moving_mean_initializer=tf.constant_initializer(weights[name]['mean'], dtype=tf.float32),
                                          training=isTraining)
    '''
    layer = tf.nn.batch_normalization(x=input,
                                      mean=weights[name]['mean'],
                                      variance=weights[name]['variance'],
                                      offset=weights[name]['offset'],
                                      scale=weights[name]['scale'],
                                      #mean = 0.0,
                                      #variance = tf.random.normal(shape = [1]),
                                      #offset = tf.random.normal(shape = [1]),
                                      #scale = tf.random.normal(shape = [1]),
                                      variance_epsilon=1e-4,
                                      name=name)
    if relu :
        layer = tf.nn.relu(layer)
    return layer

#def batch_norm_default(input,name,relu = False,isTraining = False):
#    '''
#    layer = tf.layers.batch_normalization(input=input ,
#                                          moving_variance_initializer=tf.constant_initializer(weights[name]['variance'], dtype=tf.float32),
#                                          moving_mean_initializer=tf.constant_initializer(weights[name]['mean'], dtype=tf.float32),
#                                          training=isTraining)
#    '''
#    layer = tf.nn.batch_normalization(x = input,
#                                      mean = 0.0,
#                                      variance = tf.random.normal(shape = [1]),
#                                      offset = tf.random.normal(shape = [1]),
#                                      scale = tf.random.normal(shape = [1]),
#                                      variance_epsilon = 1e-4,
#                                      name = name)
#    if relu :
#        layer = tf.nn.relu(layer)
#    return layer

def weights_init(shape,layer_name,trainable = True):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.001),name=layer_name+"_B",trainable=trainable)
