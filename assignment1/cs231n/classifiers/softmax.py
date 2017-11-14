import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """   

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  loss = 0.0
  dW = np.zeros_like(W)
  num_train,num_class = X.shape[0],W.shape[1]
    
  probs = np.zeros((num_train,num_class))
  data_loss = 0.0
  for i in range(num_train):     
        scores = np.dot(X[i,:],W)
        scores = scores - np.max(scores)
        probs[i,:] = np.exp(scores)/np.sum(np.exp(scores))
        log_probs = -np.log(probs[i,:])
        correct_probs = log_probs[y[i]]
        probs[i,y[i]] -= 1.00
        data_loss += correct_probs
        
  data_loss /= num_train
  reg_loss = 0.5*reg*np.sum(W*W)
  loss = data_loss + reg_loss
  dW = (np.dot(X.T,probs))/num_train
  dW += reg*W #dReg = reg*W
    
    
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  # Initialize the loss and gradient to zero.
  # W DxC
  #X = X  #NxD
  num_train = X.shape[0]
  num_class = W.shape[1]
  W = W.astype('float64')  
  loss = 0.0    
  scores = np.dot(X,W)   #NxC
  scores -= np.max(scores,axis=1,keepdims=True) # or scores = (scores.T - np.max(scores,axis=1)).T or scores = scores - np.max(scores,axis=1)[:,None] or scores = scores - np.max(scores,axis=1).reshape(num_train,1)
     
  probs = np.exp(scores)/np.sum(np.exp(scores),axis=1,keepdims=True)
  logprobs = -np.log(probs)
  correct_probs = logprobs[range(num_train),y]
    
  data_loss = np.sum(correct_probs)/num_train
  reg_loss = 0.5*reg*np.sum(W*W)
  
  loss = data_loss+reg_loss  
  #loss_1 = np.sum(np.log(np.sum(np.exp(scores),axis=1))-scores[range(num_train),y])/num_train +reg_loss    
  dscores = probs #NxC
  dscores[range(num_train),y] += -1.0
  dW = np.zeros_like(W)                              #DxC
  dW = np.dot(X.T,dscores)/num_train
  dReg = reg*W
  dW += dReg
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

