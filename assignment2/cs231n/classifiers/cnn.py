import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *

def conv_size(H,W,filter_size,stride,poolHW,pStride=2):
    """
    To compute window size
    """
    pad = (filter_size-1)/2
    FH,FW = filter_size,filter_size
    oH = 1 + (H-FH + 2*pad)/stride 
    oW = 1 + (W-FW + 2*pad)/stride
    poolH = poolW = poolHW
    pH = 1 + (oH - poolH)/pStride
    pW = 1 + (oW - poolW)/pStride
    return pH,pW


class ThreeLayerConvNet(object):
  """
  Convolutional network with the following architecture:  
  [conv - relu - 2x2 max pool]xL-1 - [affine - relu]xM-1 - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """ 
  def __init__(self, input_dim=(3, 32, 32), num_filters=[32,32], filter_size=3,hidden_dim=[100,100], num_classes=10, weight_scale=1e-3,reg=0.0,dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
       ####################      Note        #########################################################
    # MaxPool-2x2 of stride = 2 (To downsample the image to halve of it's  original size)            #  
    # Filter_Crosssection (H,W) is set to be same throughout the network FH, FW = 3x3                #
    # Not included spatial batchNorm  but can be extended as I have written it's layers in layers.py #
    ##################################################################################################
    
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    C = input_dim[0]
    H,W = input_dim[1],input_dim[2]
    h_out_dim = hidden_dim + [num_classes]
    self.total_layers = len(num_filters)+len(h_out_dim)
    
    #Filter
    self.num_filters = num_filters    
    self.filter_size = filter_size
    FW = FH = filter_size
    stride = 1
    
    #Maxpool attributes
    pstride = 2
    poolHW = 2 #maxpool = 2x2
    ############################################################################
        # TODO: Initialize weights and biases for the convolutional#
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    for i in range(len(num_filters)): #init W among conv layers
        idx = i+1
        if idx==1:
            self.params['W%d'%(idx)] = weight_scale*np.random.randn(num_filters[i],C,FH,FW)
            self.params['b%d'%(idx)] = np.zeros(num_filters[i])
        else:
            self.params['W%d'%(idx)] = weight_scale*np.random.randn(num_filters[i],num_filters[i-1],FH,FW)
            self.params['b%d'%(idx)] = np.zeros(num_filters[i])
            
    for _ in range(len(num_filters)): #To find the final conv_relu_pool_layer size (i.e., (L-1)th layer size).
        
        H,W = conv_size(H,W,filter_size,stride,poolHW,pstride) #Maxpool (2x2 with stride=2)

    j = 0  #index of hidden_dim    
    for i in range(len(num_filters),self.total_layers):
        idx = i+1
        if i == len(num_filters): #W to connect (L-1)th conv_relu_pool_layer to FC  
            self.params['W%d'%(idx)] = weight_scale*np.random.randn(num_filters[i-1]*H*W, h_out_dim[j]) #H,W from above for loop
            self.params['b%d'%(idx)] = np.zeros(h_out_dim[j])
        else: # W's among FC layers
            self.params['W%d'%(idx)] = weight_scale*np.random.randn(h_out_dim[j-1], h_out_dim[j])
            self.params['b%d'%(idx)] = np.zeros(h_out_dim[j])
        j += 1

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
      #print k  
 
  def loss(self, X, y=None):
        
    X = X.astype(self.dtype)
    
    filter_size = self.filter_size
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
                    # TODO: Implement the forward pass   #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    num_filters = self.num_filters
    dout = X
    cache = []
    for i in range(len(num_filters)):
        idx = i+1
        W = self.params['W%d'%(idx)]
        b = self.params['b%d'%(idx)]
        dout,conv_cache = conv_relu_pool_forward(dout, W, b, conv_param, pool_param)
        cache.append(conv_cache)
        
    #total_layers = L-1 layers + M-layers + affine_layer
    total_layers = self.total_layers            
    for i in range(len(num_filters),total_layers-1): # perform aff_Relu_fwd only on M-layers 
        idx = i+1
        W = self.params['W%d'%(idx)]
        b = self.params['b%d'%(idx)]
        dout,relu_cache = affine_relu_forward(dout, W, b)
        cache.append(relu_cache)
        
    idx = total_layers           #perform affine_fwd on the last layer 
    W = self.params['W%d'%(idx)]
    b = self.params['b%d'%(idx)]    
    scores,aff_cache = affine_forward(dout, W, b)
    cache.append(aff_cache)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
            # TODO: Implement the backward pass #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, dout = softmax_loss(scores,y)

    for i in range(total_layers): #To update Reg loss
        idx = i+1
        W = self.params['W%d'%(idx)]
        reg_loss = 0.5*self.reg*(np.sum(np.square(W)))
        loss += reg_loss
    
    # Perform backropogate on affine to conv layers

    for i in range(total_layers,0,-1): 
        w = 'W%d'%(i);  b = 'b%d'%(i) 
        if i == total_layers:
            dout,grads[w],grads[b]=affine_backward(dout, cache.pop())
        elif i >len(num_filters):#  and i < h_conv_layers:
            dout,grads[w],grads[b] = affine_relu_backward(dout,cache.pop())
        elif i <= len(num_filters):
            dout,grads[w],grads[b] = conv_relu_pool_backward(dout,cache.pop())
        grads[w] += self.reg*self.params[w] 
        
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  


