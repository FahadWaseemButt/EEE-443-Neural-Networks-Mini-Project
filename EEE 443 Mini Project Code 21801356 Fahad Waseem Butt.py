# -*- coding: utf-8 -*-
"""
EEE 443 Mini Project Code 21801356 Fahad Waseem Butt.ipynb
"""

"""# Question 1"""

# Importing Libraries
import numpy as np
import h5py
import matplotlib.pyplot as plt
import random
from time import sleep
import time

# Extracting Data
df = h5py.File('data1.h5', 'r')
data = np.array(df['data'])

# Data Preprocessing 
reshaped_data = data.reshape(10240, 3, 16*16)
# Converting to Grayscale with Luminosity Model
Y = reshaped_data[:, 0, :]*0.2126 + reshaped_data[:, 1, :]*0.7152 + reshaped_data[:, 2, :]*0.0722
# Removing mean pixel intensity from each image
Y = Y - np.mean(Y, axis=1).reshape(-1, 1)
# Clipping by 3 std. dev. and setting data to range [0.1,0.90]
Y = np.clip(Y, -3*np.std(Y), 3*np.std(Y))
Y = np.interp(Y, (np.min(Y), np.max(Y)), (0.1, 0.9))

Y_im = Y.reshape(10240, 16, 16)

# Plotting Patches as RGB and Grayscale
data_list = []
for i in range(len(data)):
  data_list.append(data[i, :, :, :].T)

plot1, ax1 = plt.subplots(10, 20, figsize=(20, 10))
plot2, ax2 = plt.subplots(10, 20, figsize=(20, 10), dpi=200, facecolor='w', edgecolor='k')

for i in range(10):
    for j in range(20):
        k = np.random.randint(69, data.shape[0])

        ax1[i, j].imshow(data_list[k].astype('float'))
        ax1[i, j].axis("off")

        ax2[i, j].imshow(Y_im[k], cmap='gray')
        ax2[i, j].axis("off")

plot1.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
plot2.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

def sigmoid(X):
  # Sigmoid Function
  S = 1/(1 + np.exp(-X))
  # Sigmoid Derivative
  dS = S * (1 - S)
  return S, dS

def aeCost(We, data, params):
  W1, W2, b1, b2 = We
  Lin, Lhid, Lout, lmda, beta, rho = params
  h1, dh1 = sigmoid(np.matmul(data, W1) + b1)
  h2, dh2 = sigmoid(np.matmul(h1, W2) + b2)
  rho_b = h1.mean(axis=0, keepdims=True)

  # Mean Squared Error
  MSE = (1/(2*data.shape[0])) * np.sum(np.square(np.linalg.norm(data - h2, axis=1)))
  dMSE = (-1/(data.shape[0])) * (data - h2)

  # Tykhonov Regularization
  Tykhonov = (1/2) * lmda * (np.sum(W1 ** 2) + np.sum(W2 ** 2))
  dTykhonov1 = lmda * W1 
  dTykhonov2 =  lmda * W2

  # Kullback-Leibler Divergence
  KL_Div = beta * np.sum(rho * np.log(rho/rho_b) + (1 - rho)*np.log((1-rho) / (1 - rho_b)))
  dKL_Div = beta * (- rho/rho_b + (1 - rho)/(1 - rho_b))/data.shape[0]

  # Full Loss Function
  J = MSE + Tykhonov + KL_Div

  cache = (data, h1, dh1, dh2)
  dJ = (dMSE, dTykhonov1, dTykhonov2, dKL_Div)
  
  return  J, dJ, cache

def gradSolver(We, mWe, lr, alpha, dJ, cache):
  W1, W2, b1, b2 = We
  data, h1, dh1, dh2 = cache
  dMSE, dTy1, dTy2, dSp = dJ
  mW1, mW2, mb1, mb2 = mWe
  dW1, dW2, db1, db2 = 0, 0, 0, 0

  # Findindg Derivates by Backpropagation 
  dW2 = np.matmul(h1.T, (dMSE*dh2)) + dTy2
  db2 = (dMSE*dh2).sum(axis = 0, keepdims=True)

  dW1 = np.matmul((dMSE*dh2), W2.T)*dh1
  dW1 = np.matmul(data.T, (dh1 * (np.matmul((dMSE*dh2), W2.T) + dSp))) + dTy1
  db1 = (dh1 * (np.matmul((dMSE*dh2), W2.T) + dSp)).sum(axis = 0, keepdims=True)

  dW2 = (dW1.T + dW2)/2
  dW1 = dW2.T

  # Momentum Update of Variables
  mW1 = lr*dW1 + alpha*mW1
  mW2 = lr*dW2 + alpha*mW2
  mb1 = lr*db1 + alpha*mb1
  mb2 = lr*db2 + alpha*mb2
  mWe = (mW1, mW2, mb1, mb2)

  W1 = W1 - mW1
  W2 = W2 - mW2
  b1 = b1 - mb1
  b2 = b2 - mb2
  We = (W1, W2, b1, b2)

  return We, mWe

# Training Model
def aetrain(momWe, We, data, params, lr, alpha, batch, epoch):
  # Initializing
  loss_epoch = []
  Lin, Lhid, Lout, lmbda, beta, rho = params
  W1, W2, b1, b2 = We
  mW1, mW2, mb1, mb2 = momWe
  
  count_batch = 0
  
  # Running for chosen epoch count
  for i in range(epoch):
    iter_loss = 0
    momWe = [0, 0, 0, 0]
    count_batch = 0

    for j in range(50):
      batch_data = data[count_batch:count_batch+64]
      J, dJ, cache = aeCost(We, batch_data, params)
      We, momWe = gradSolver(We, momWe, lr, alpha, dJ, cache)
      count_batch += 64
      iter_loss += J

    print(("Loss: {:.2f} [Epoch {}]".format(iter_loss/50, i+1, epoch)))
    loss_epoch.append(iter_loss)

  return We

# Hyperparameters are changed for each run in part d, currently optimal hyperparameters from part c
lr = 0.05
alpha = 0.99
epoch = 200
batch = 64
lmbda = 5e-4
beta = 0.1
rho = 0.001

Lin = 256
Lout = 256
Lhid = 64

wi = np.sqrt(6/(Lin + Lhid))
W1 = np.random.uniform(-wi, wi, size=(Lin, Lhid))
b1 = np.random.uniform(-wi, wi, size=(1, Lhid))

wo = np.sqrt(6/(Lhid + Lout))
W2 = W1.T
b2 = np.random.uniform(-wo, wo, size=(1, Lout))

We = [W1, W2, b1, b2]
momWe = [0, 0, 0, 0]
params = [Lin, Lhid, Lout, 5e-4, beta, rho]

We = aetrain(momWe, We, Y, params, lr, alpha, batch, epoch)

temp = We[0].T.reshape(Lhid, 16, 16)
dim = np.sqrt(Lhid).astype(int)

fig, ax = plt.subplots(dim, dim, figsize=(dim, dim), dpi=320, facecolor='w', edgecolor='k')
k = 0
for i in range(dim):
    for j in range(dim):
        ax[i, j].imshow(temp[k], cmap='gray')
        ax[i, j].axis("off")
        k += 1

fig.subplots_adjust(wspace=0.1, hspace=0.1, left=0, right=1, bottom=0, top=1)

"""# Question 2"""

# Importing Libraries
import numpy as np
import h5py
import matplotlib.pyplot as plt
import random
import time

# Extractin Data
df = h5py.File('data2.h5', 'r')
list(df.keys())

testd = np.array(df['testd'])
testx = np.array(df['testx'])
traind = np.array(df['traind'])
trainx = np.array(df['trainx'])
vald = np.array(df['vald'])
valx = np.array(df['valx'])
words = np.array(df['words'])

# One-Hot Encoding
train_Y = np.zeros((traind.shape[0], 250))
for i in range(traind.shape[0]):
  train_Y[i, traind[i] - 1] = 1

val_Y = np.zeros((vald.shape[0], 250))
for i in range(vald.shape[0]):
  val_Y[i, vald[i] - 1] = 1

def sigmoid(X):
  # Sigmoid Function
  S = 1/(1 + np.exp(-X))
  # Sigmoid Derivative
  dS = S * (1 - S)
  return S, dS

def softmax(x):
    S = np.exp(x)/(np.sum(np.exp(x), axis=1)[:,None])
    return S

def NNtrain(trainx, traind, valx, vald, word_embed_weights, embed_to_hidden_weights, hid_bias, hid_to_out_weights, output_bias, lr, epoch, batch, alpha):
  # Initialization
  loss_epoch = []
  flag_count = 0

  # Run for chosen epoch count
  for i in range(epoch):
      sample = np.random.permutation(len(trainx))
      data = trainx[sample, :]
      label = traind[sample, :]
      count = 0
      train_epoch_loss = 0
      mdW2 = np.zeros((P, 250))
      mdW3 = np.zeros((P, D*3))

      for j in range(1862):
        if j == 1862:
          batch = 100
        X  = data[count: count + batch]
        Y  = label[count: count + batch]

        # Forward Pass
        # Weights for the 3 Input Words
        word_embedding_1 = word_embed_weights[X[:, 0] - np.min(trainx), :]
        word_embedding_2 = word_embed_weights[X[:, 1] - np.min(trainx), :]
        word_embedding_3 = word_embed_weights[X[:, 2] - np.min(trainx), :]
        h1 = np.hstack((word_embedding_1, word_embedding_2, word_embedding_3))
        
        h2 = np.matmul(h1, embed_to_hidden_weights.T) + hid_bias
        v2, dv2 = sigmoid(h2) 

        o2 = np.matmul(v2, hid_to_out_weights) + output_bias
        v3 = softmax(o2)
        
        # Backpropagation Gradient Calculation
        dW3 = np.matmul(v2.T, v3 - Y)

        dW2 = np.matmul((np.matmul(v3 - Y, hid_to_out_weights.T) * dv2).T, h1)
        
        dW1 = np.matmul((np.matmul(v3 - Y, hid_to_out_weights.T) * dv2), embed_to_hidden_weights)

        # Momentum Update for Weights
        mdW2 = alpha*mdW2 + lr*(dW3/batch)
        mdW3 = alpha*mdW3 + lr*(dW2/batch)
        
        hid_to_out_weights -= mdW2
        embed_to_hidden_weights -= mdW3
        word_embed_weights[X[:,0]-1, :] -= lr * (dW1[:,:D]/batch)
        word_embed_weights[X[:,1]-1, :] -= lr * (dW1[:,D:D*2]/batch)
        word_embed_weights[X[:,2]-1, :] -= lr * (dW1[:,D*2:D*3]/batch)

        output_bias -= lr*np.mean((v3 - Y), axis=0)
        hid_bias -= lr*np.mean((np.matmul(v3 - Y, hid_to_out_weights.T)), axis=0)
  
      loss = val_loss(valx, vald, word_embed_weights, embed_to_hidden_weights, hid_to_out_weights, hid_bias, output_bias)
      loss_epoch.append(loss)
      if loss_epoch[i] > loss_epoch[i - 1]:
        flag_count += 1
      else:
        flag_count = 0
      
      if flag_count == 3:
        break
      print(i + 1, loss)

  plt.plot(loss_epoch)
  plt.xlabel('epochs')
  plt.ylabel('loss')
  plt.title('Cross Entropy Validation Loss for (D=' + str(D) + ', P=' + str(P))
  plt.show()
  return loss_epoch, word_embed_weights, embed_to_hidden_weights, hid_to_out_weights, hid_bias, output_bias

def val_loss(valx, vald, word_embed_weights, embed_to_hidden_weights, hid_to_out_weights, hid_bias, output_bias):
  # Weights for the 3 Input Words
  word_embedding_1 = word_embed_weights[valx[:, 0] - np.min(valx), :]
  word_embedding_2 = word_embed_weights[valx[:, 1] - np.min(valx), :]
  word_embedding_3 = word_embed_weights[valx[:, 2] - np.min(valx), :]
  h1 = np.hstack((word_embedding_1, word_embedding_2, word_embedding_3))

  h2 = np.matmul(h1, embed_to_hidden_weights.T) + hid_bias
  v2, dv2 = sigmoid(h2) 

  o2 = np.matmul(v2, hid_to_out_weights) + output_bias
  v3 = softmax(o2)
  
  loss = -(1/46500) * np.sum(np.sum(vald * np.log(v3 + 0.001), axis=0, keepdims=True), axis=1)

  return loss

"""(D,P) = (32, 256)"""

batch = 200
lr = 0.00001
alpha = 0.85
epoch = 50
D = 32
P = 256

# Embedded layer
word_embeded_weights = np.random.normal(0.0, 0.1, (250, D))
# Hidden layer
embed_to_hidden_weights = np.random.normal(0.0, 0.1, (P, D*3))
hid_bias = np.random.normal(0.0, 0.1, (P))
# Output layer
hid_to_out_weights = np.random.normal(0.0, 0.1, (P, 250))
output_bias = np.random.normal(0.0, 0.1, (250))

loss, word_embeded_weights, embed_to_hidden_weights, hid_to_out_weights, hid_bias, output_bias = NNtrain(trainx, train_Y, valx, val_Y, word_embeded_weights, embed_to_hidden_weights, hid_bias, hid_to_out_weights, output_bias, lr, epoch, batch, alpha)

"""(D,P) = (16, 128)"""

batch = 200
lr = 0.00001
alpha = 0.85
epoch = 30
D = 16
P = 128

# Embedded layer
word_embeded_weights = np.random.normal(0.0, 0.1, (250, D))
# Hidden layer
embed_to_hidden_weights = np.random.normal(0.0, 0.1, (P, D*3))
hid_bias = np.random.normal(0.0, 0.1, (P))
# Output layer
hid_to_out_weights = np.random.normal(0.0, 0.1, (P, 250))
output_bias = np.random.normal(0.0, 0.1, (250))

loss, word_embeded_weights, embed_to_hidden_weights, hid_to_out_weights, hid_bias, output_bias = NNtrain(trainx, train_Y, valx, val_Y, word_embeded_weights, embed_to_hidden_weights, hid_bias, hid_to_out_weights, output_bias, lr, epoch, batch, alpha)

"""(D, P) = (8, 64)"""

batch = 200
lr = 0.00001
alpha = 0.85
epoch = 50
D = 8
P = 64

# Embedded layer
word_embeded_weights = np.random.normal(0.0, 0.1, (250, D))
# Hidden layer
embed_to_hidden_weights = np.random.normal(0.0, 0.1, (P, D*3))
hid_bias = np.random.normal(0.0, 0.1, (P))
# Output layer
hid_to_out_weights = np.random.normal(0.0, 0.1, (P, 250))
output_bias = np.random.normal(0.0, 0.1, (250))

loss, word_embeded_weights, embed_to_hidden_weights, hid_to_out_weights, hid_bias, output_bias = NNtrain(trainx, train_Y, valx, val_Y, word_embeded_weights, embed_to_hidden_weights, hid_bias, hid_to_out_weights, output_bias, lr, epoch, batch, alpha)

# For Checking How Model Performs with Trigram Input
 
batch = 200
lr = 0.00001
alpha = 0.85
epoch = 50
D = 32
P = 256

# Initializnig Random Permutations to Choose 5 Sample Trigrams
sample = np.random.permutation(len(testx))
data = testx[sample, :]

data = data[:5, :]

# Weights for the 3 Input Words
word_embedding_1 = word_embeded_weights[data[:, 0] - np.min(testx), :]
word_embedding_2 = word_embeded_weights[data[:, 1] - np.min(testx), :]
word_embedding_3 = word_embeded_weights[data[:, 2] - np.min(testx), :]
h1 = np.hstack((word_embedding_1, word_embedding_2, word_embedding_3))

h2 = np.matmul(h1, embed_to_hidden_weights.T) + hid_bias
v2, dv2 = sigmoid(h2) 

o2 = np.matmul(v2, hid_to_out_weights) + output_bias
v3 = softmax(o2)

pred = (np.argsort(v3))

# Returning Output
for i in range(5):
  temp = pred[i, 0:10]
  print(words[data[i]])
  print(words[temp])


"""# Question 3"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
import random
import pandas as pd 
import seaborn as sns
import math

with h5py.File('/content/data3.h5','r') as F:
# Names variable contains the names of training and testing file 
    names = list(F.keys())
    X_train = np.array(F[names[0]][()])
    y_train = np.array(F[names[1]][()])
    X_test = np.array(F[names[2]][()])
    y_test = np.array(F[names[3]][()])

def sigmoid(x):
    c = np.clip(x,-700,700)
    return 1 / (1 + np.exp(-c))
def dsigmoid(y):
    return y * (1 - y)
def tanh(x):
    return np.tanh(x)
def dtanh(y):
    return 1 - y * y

class Metrics: 
    """
    To evaluate the model.

    """ 
    def confusion_matrix(self,labels,preds):
        label = pd.Series(labels,name='Actual')
        pred = pd.Series(preds,name='Predicted')
        return pd.crosstab(label,pred)

    def accuracy_score(self,labels,preds):      
        count = 0
        size = labels.shape[0]
        for i in range(size):
            if preds[i] == labels[i]:
                count +=1
        return  100 * (count/size)

    def accuracy(self,labels,preds):
        return 100 * (labels == preds).mean()

class Activations:
    """
    Activation functions
    """
    def relu_alternative(self,X):
        return np.maximum(X, 0)

    def ReLU(self,X):
        return (abs(X) + X) / 2

    def relu_another(self,X):
        return X * (X > 0)

    def tanh(self,X):
        return np.tanh(X)

    def tanh_manuel(self,X):  
        return (np.exp(X) - np.exp(-X))/(np.exp(X) + np.exp(-X))

    def sigmoid(self,X):
        c = np.clip(X,-700,700)
        return 1/(1 + np.exp(-c))

    def softmax(self,X):
        e_x = np.exp(X - np.max(X, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)
        
    def softmax_stable(self,X):
        e_x = np.exp(X - np.max(X))
        return e_x / np.sum(e_x)

    def ReLUDerivative(self,X): 
        return 1 * (X > 0)

    def ReLU_grad(self,X):
        X[X<=0] = 0
        X[X>1] = 1
        return X

    def dReLU(self,X):       
        return np.where(X <= 0, 0, 1)

    def dtanh(self,X):     
        return  1-(np.tanh(X)**2)

    def dsigmoid(self,X):
        return self.sigmoid(X) * (1-self.sigmoid(X))    
    
    def softmax_stable_gradient(self,soft_out):           
        return soft_out * (1 - soft_out)

    def softmax_grad(self,softmax):        
        s = softmax.reshape(-1,1)
        return np.diagflat(s) - np.dot(s, s.T)

    def softmax_gradient(self,Sz):
        D = -np.outer(Sz, Sz) + np.diag(Sz.flatten())
        return D

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(y):
    return y * (1 - y)

def tanh(x):
    return np.tanh(x)

def dtanh(y):
    return 1 - y * y

activations = Activations()
metrics = Metrics()

"""Part a: MLP RNN"""

class Multi_Layer_RNN(object):

    def __init__(self,input_dim = 3,hidden_dim_1 = 128, hidden_dim_2 = 64, seq_len = 150, learning_rate = 1e-1, mom_coeff = 0.85, batch_size = 32, output_class = 6):
        """
        Initialization
        """
        np.random.seed(150)
        self.input_dim = input_dim
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        # Unfold case T = 150 :
        self.seq_len = seq_len
        self.output_class = output_class
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.mom_coeff = mom_coeff

        # Xavier uniform scaler :
        Xavier = lambda fan_in,fan_out : math.sqrt(6/(fan_in + fan_out))

        lim_inp2hid = Xavier(self.input_dim,self.hidden_dim_1)
        self.W1 = np.random.uniform(-lim_inp2hid,lim_inp2hid,(self.input_dim,self.hidden_dim_1))
        self.B1 = np.random.uniform(-lim_inp2hid,lim_inp2hid,(1,self.hidden_dim_1))

        lim_hid2hid = Xavier(self.hidden_dim_1,self.hidden_dim_1)
        self.W1_rec= np.random.uniform(-lim_hid2hid,lim_hid2hid,(self.hidden_dim_1,self.hidden_dim_1))


        lim_hid2hid2 = Xavier(self.hidden_dim_1,self.hidden_dim_2)
        self.W2 = np.random.uniform(-lim_hid2hid2,lim_hid2hid2,(self.hidden_dim_1,self.hidden_dim_2))
        self.B2 = np.random.uniform(-lim_hid2hid2,lim_hid2hid2,(1,self.hidden_dim_2))

        lim_hid2out = Xavier(self.hidden_dim_2,self.output_class)
        self.W3 = np.random.uniform(-lim_hid2out,lim_hid2out,(self.hidden_dim_2,self.output_class))
        self.B3 = np.random.uniform(-lim_inp2hid,lim_inp2hid,(1,self.output_class))

        # Track loss and accuracy score :     
        self.train_loss,self.test_loss,self.train_acc,self.test_acc = [],[],[],[]
        
        # Storing previous momentum updates :
        self.prev_updates = {'W1'       : 0,
                             'B1'       : 0,
                             'W1_rec'   : 0,
                             'W2'       : 0,
                             'B2'       : 0,
                             'W3'       : 0,
                             'B3'       : 0}


    def forward(self,X) -> tuple:
        """
        Forward propagation
        """ 
        X_state = dict()
        hidden_state_1 = dict()
        hidden_state_mlp = dict()
        output_state = dict()
        probs = dict()
        mlp_linear = dict()
        
        self.h_prev_state = np.zeros((1,self.hidden_dim_1))
        hidden_state_1[-1] = np.copy(self.h_prev_state)

        # Loop over time T = 150 :
        for t in range(self.seq_len):

            # Selecting first record with 3 inputs, dimension = (batch_size,input_size)
            X_state[t] = X[:,t]

            # Recurrent hidden layer :
            hidden_state_1[t] = np.tanh(np.dot(X_state[t],self.W1) + np.dot(hidden_state_1[t-1],self.W1_rec) + self.B1)
            mlp_linear[t] = np.dot(hidden_state_1[t],self.W2) + self.B2
            hidden_state_mlp[t] = activations.ReLU(mlp_linear[t])
            output_state[t] = np.dot(hidden_state_mlp[t],self.W3) + self.B3

            # Per class probabilites :
            probs[t] = activations.softmax(output_state[t])

        return (X_state,hidden_state_1,mlp_linear,hidden_state_mlp,probs)
        

    def BPTT(self,cache,Y):
        """
        Back propagation

        """

        X_state,hidden_state_1,mlp_linear,hidden_state_mlp,probs = cache

        # backward pass: compute gradients going backwards
        dW1, dW1_rec, dW2, dW3 = np.zeros_like(self.W1), np.zeros_like(self.W1_rec), np.zeros_like(self.W2),np.zeros_like(self.W3)

        dB1, dB2,dB3 = np.zeros_like(self.B1), np.zeros_like(self.B2),np.zeros_like(self.B3)

        dhnext = np.zeros_like(hidden_state_1[0])

        dy = np.copy(probs[149])        
        dy[np.arange(len(Y)),np.argmax(Y,1)] -= 1
        #dy = probs[0] - Y[0]

        dW3 += np.dot(hidden_state_mlp[149].T,dy)
        dB3 += np.sum(dy,axis = 0, keepdims = True)

        dy1 = np.dot(dy,self.W3.T) * activations.ReLU_grad(mlp_linear[149])

        dB2 += np.sum(dy1,axis = 0, keepdims = True)
        dW2 += np.dot(hidden_state_1[149].T,dy1)


        for t in reversed(range(1,self.seq_len)):
    
            dh = np.dot(dy1,self.W2.T) + dhnext        
            dhrec = (1 - (hidden_state_1[t] * hidden_state_1[t])) * dh

            dB1 += np.sum(dhrec,axis = 0, keepdims = True)            
            dW1 += np.dot(X_state[t].T,dhrec)
            
            dW1_rec += np.dot(hidden_state_1[t-1].T,dhrec)

            dhnext = np.dot(dhrec,self.W1_rec.T)

               
        for grad in [dW1,dB1,dW1_rec,dW2,dB2,dW3,dB3]:
            np.clip(grad, -10, 10, out = grad)


        return [dW1,dB1,dW1_rec,dW2,dB2,dW3,dB3]    

    def CategoricalCrossEntropy(self,labels,preds):
        predictions = np.clip(preds, 1e-12, 1. - 1e-12)
        N = predictions.shape[0]         
        return -np.sum(labels * np.log(predictions + 1e-9)) / N

    def step(self,grads,momentum = True):

        if momentum:
            
            delta_W1 = -self.learning_rate * grads[0] -  self.mom_coeff * self.prev_updates['W1']
            delta_B1 = -self.learning_rate * grads[1] -  self.mom_coeff * self.prev_updates['B1']  
            delta_W1_rec = -self.learning_rate * grads[2] -  self.mom_coeff * self.prev_updates['W1_rec']
            delta_W2 = -self.learning_rate * grads[3] - self.mom_coeff * self.prev_updates['W2']              
            delta_B2 = -self.learning_rate * grads[4] -  self.mom_coeff * self.prev_updates['B2']
            delta_W3 = -self.learning_rate * grads[5] -  self.mom_coeff * self.prev_updates['W3']
            delta_B3 = -self.learning_rate * grads[6] -  self.mom_coeff * self.prev_updates['B3']
                       
            self.W1 += delta_W1
            self.W1_rec += delta_W1_rec
            self.W2 += delta_W2
            self.B1 += delta_B1
            self.B2 += delta_B2 
            self.W3 += delta_W3
            self.B3 += delta_B3   
            
            self.prev_updates['W1'] = delta_W1
            self.prev_updates['W1_rec'] = delta_W1_rec
            self.prev_updates['W2'] = delta_W2
            self.prev_updates['B1'] = delta_B1
            self.prev_updates['B2'] = delta_B2
            self.prev_updates['W3'] = delta_W3
            self.prev_updates['B3'] = delta_B3
            
            self.learning_rate *= 0.9999

    def fit(self,X,Y,X_val,y_val,epochs = 50 ,verbose = True, crossVal = False):
        """
        Fitting the model and measure the performance
        """
                
        for epoch in range(epochs):
            
            print(f'Epoch : {epoch + 1}')

            perm = np.random.permutation(3000)           
            
            for i in range(round(X.shape[0]/self.batch_size)): 

                batch_start  =  i * self.batch_size
                batch_finish = (i+1) * self.batch_size
                index = perm[batch_start:batch_finish]
                
                X_feed = X[index]    
                y_feed = Y[index]
                
                cache_train = self.forward(X_feed)                                                          
                grads = self.BPTT(cache_train,y_feed)                
                self.step(grads)
      
                if crossVal:
                    stop = self.cross_validation(X,X_val,Y,y_val,threshold = 5)
                    if stop: 
                        break
            
            cross_loss_train = self.CategoricalCrossEntropy(y_feed,cache_train[4][149])
            predictions_train = self.predict(X)
            acc_train = metrics.accuracy(np.argmax(Y,1),predictions_train)

            _,__,___,____, probs_test = self.forward(X_val)
            cross_loss_val = self.CategoricalCrossEntropy(y_val,probs_test[149])
            predictions_val = np.argmax(probs_test[149],1)
            acc_val = metrics.accuracy(np.argmax(y_val,1),predictions_val)

            if verbose:

                print(f"[{epoch + 1}/{epochs}] ------> Training :  Accuracy : {acc_train}")
                print(f"[{epoch + 1}/{epochs}] ------> Training :  Loss     : {cross_loss_train}")
                print('______________________________________________________________________________________\n')                         
                print(f"[{epoch + 1}/{epochs}] ------> Testing  :  Accuracy : {acc_val}")                                        
                print(f"[{epoch + 1}/{epochs}] ------> Testing  :  Loss     : {cross_loss_val}")
                print('______________________________________________________________________________________\n')
                
            self.train_loss.append(cross_loss_train)              
            self.test_loss.append(cross_loss_val) 
            self.train_acc.append(acc_train)              
            self.test_acc.append(acc_val)

    def predict(self,X):
        _,__,___,____,probs = self.forward(X)
        return np.argmax(probs[149],axis=1)

    def history(self):
        return {'TrainLoss' : self.train_loss,
                'TrainAcc'  : self.train_acc,
                'TestLoss'  : self.test_loss,
                'TestAcc'   : self.test_acc}

multilayer_rnn = Multi_Layer_RNN(learning_rate=1e-4,mom_coeff=0.0,hidden_dim_1 = 128, hidden_dim_2 = 64)

multilayer_rnn.fit(X_train,y_train,X_test,y_test,epochs = 35)

multilayer_rnn_history = multilayer_rnn.history()

plt.figure()
plt.plot(multilayer_rnn_history['TestAcc'],'-o')
plt.plot(multilayer_rnn_history['TrainAcc'],'-x')
plt.xlabel('# of epochs')
plt.ylabel('Accuracy')
plt.title('MLP RNN Accuracy over epochs')
plt.legend(['Test','Validation'])
plt.show()

plt.figure()
plt.plot(multilayer_rnn_history['TestLoss'],'-o')
plt.plot(multilayer_rnn_history['TrainLoss'],'-x')
plt.xlabel('# of epochs')
plt.ylabel('Loss')
plt.title('MLP RNN Loss over epochs')
plt.legend(['Test','Validation'])
plt.show()

train_preds_multilayer_rnn = multilayer_rnn.predict(X_train)
test_preds_multilayer_rnn = multilayer_rnn.predict(X_test)
confusion_mat_train_multilayer_rnn = metrics.confusion_matrix(np.argmax(y_train,1),train_preds_multilayer_rnn)
confusion_mat_test_multilayer_rnn = metrics.confusion_matrix(np.argmax(y_test,1),test_preds_multilayer_rnn)

body_movements = ['downstairs','jogging','sitting','standing','upstairs','walking']
confusion_mat_train_multilayer_rnn.columns = body_movements
confusion_mat_train_multilayer_rnn.index = body_movements
confusion_mat_test_multilayer_rnn.columns = body_movements
confusion_mat_test_multilayer_rnn.index = body_movements

plt.figure()
sns.heatmap(confusion_mat_train_multilayer_rnn/np.sum(confusion_mat_train_multilayer_rnn), annot=True, fmt='.1%',cmap = 'Greens')
plt.title('MLP RNN Training Confusion Matrix')
plt.show()

plt.figure()
sns.heatmap(confusion_mat_test_multilayer_rnn/np.sum(confusion_mat_test_multilayer_rnn), annot=True, fmt='.1%',cmap = 'Greens')
plt.title('MLP Testing Confusion Matrix')
plt.show()

"""Part b: LSTM"""

class Multi_Layer_LSTM(object):

    def __init__(self,input_dim = 3,hidden_dim_1 = 128,hidden_dim_2 =64,output_class = 6,seq_len = 150,batch_size = 30,learning_rate = 1e-1,mom_coeff = 0.85):
        """
        Initialization 
        """
        np.random.seed(150)
        self.input_dim = input_dim
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2

        # Unfold case T = 150 :
        self.seq_len = seq_len
        self.output_class = output_class
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.mom_coeff = mom_coeff

        self.input_stack_dim = self.input_dim + self.hidden_dim_1
        
        # Xavier uniform scaler :
        Xavier = lambda fan_in,fan_out : math.sqrt(6/(fan_in + fan_out))

        lim1 = Xavier(self.input_dim,self.hidden_dim_1)
        self.W_f = np.random.uniform(-lim1,lim1,(self.input_stack_dim,self.hidden_dim_1))
        self.B_f = np.random.uniform(-lim1,lim1,(1,self.hidden_dim_1))

        self.W_i = np.random.uniform(-lim1,lim1,(self.input_stack_dim,self.hidden_dim_1))
        self.B_i = np.random.uniform(-lim1,lim1,(1,self.hidden_dim_1))

        self.W_c = np.random.uniform(-lim1,lim1,(self.input_stack_dim,self.hidden_dim_1))
        self.B_c = np.random.uniform(-lim1,lim1,(1,self.hidden_dim_1))

        self.W_o = np.random.uniform(-lim1,lim1,(self.input_stack_dim,self.hidden_dim_1))
        self.B_o = np.random.uniform(-lim1,lim1,(1,self.hidden_dim_1))
        
        lim2 = Xavier(self.hidden_dim_1,self.hidden_dim_2)
        self.W_hid = np.random.uniform(-lim2,lim2,(self.hidden_dim_1,self.hidden_dim_2))
        self.B_hid = np.random.uniform(-lim2,lim2,(1,self.hidden_dim_2))

        lim3 = Xavier(self.hidden_dim_2,self.output_class)
        self.W = np.random.uniform(-lim3,lim3,(self.hidden_dim_2,self.output_class))
        self.B = np.random.uniform(-lim3,lim3,(1,self.output_class))

        # To keep track loss and accuracy score :     
        self.train_loss,self.test_loss,self.train_acc,self.test_acc = [],[],[],[]
        
        # To keep previous updates in momentum :
        self.previous_updates = [0] * 13
        
        # For AdaGrad:
        self.cache = [0] * 13     
        self.cache_rmsprop = [0] * 13
        self.m = [0] * 13
        self.v = [0] * 13
        self.t = 1

    def cell_forward(self,X,h_prev,C_prev):

        # Stacking previous hidden state vector with inputs:
        stack = np.column_stack([X,h_prev])

        # Forget gate:
        forget_gate = activations.sigmoid(np.dot(stack,self.W_f) + self.B_f)
       
        # İnput gate:
        input_gate = activations.sigmoid(np.dot(stack,self.W_i) + self.B_i)

        # New candidate:
        cell_bar = np.tanh(np.dot(stack,self.W_c) + self.B_c)

        # New Cell state:
        cell_state = forget_gate * C_prev + input_gate * cell_bar

        # Output fate:
        output_gate = activations.sigmoid(np.dot(stack,self.W_o) + self.B_o)

        # Hidden state:
        hidden_state = output_gate * np.tanh(cell_state)

        # Classifiers (Softmax) :
        dense_hid = np.dot(hidden_state,self.W_hid) + self.B_hid
        act = activations.ReLU(dense_hid)

        dense = np.dot(act,self.W) + self.B
        probs = activations.softmax(dense)

        return (stack,forget_gate,input_gate,cell_bar,cell_state,output_gate,hidden_state,dense,probs,dense_hid,act)
        

    def forward(self,X,h_prev,C_prev):
        x_s,z_s,f_s,i_s = {},{},{},{}
        C_bar_s,C_s,o_s,h_s = {},{},{},{}
        v_s,y_s,v_1s,y_1s = {},{},{},{}


        h_s[-1] = np.copy(h_prev)
        C_s[-1] = np.copy(C_prev)

        for t in range(self.seq_len):
            x_s[t] = X[:,t,:]
            z_s[t], f_s[t], i_s[t], C_bar_s[t], C_s[t], o_s[t], h_s[t],v_s[t], y_s[t],v_1s[t],y_1s[t] = self.cell_forward(x_s[t],h_s[t-1],C_s[t-1])

        return (z_s, f_s, i_s, C_bar_s, C_s, o_s, h_s,v_s, y_s,v_1s,y_1s)
    
    def BPTT(self,outs,Y):

        z_s, f_s, i_s, C_bar_s, C_s, o_s, h_s,v_s, y_s,v_1s,y_1s = outs

        dW_f, dW_i,dW_c, dW_o,dW,dW_hid = np.zeros_like(self.W_f), np.zeros_like(self.W_i), np.zeros_like(self.W_c),np.zeros_like(self.W_o),np.zeros_like(self.W),np.zeros_like(self.W_hid)

        dB_f, dB_i,dB_c,dB_o,dB,dB_hid  = np.zeros_like(self.B_f), np.zeros_like(self.B_i),np.zeros_like(self.B_c),np.zeros_like(self.B_o),np.zeros_like(self.B),np.zeros_like(self.B_hid)

        dh_next = np.zeros_like(h_s[0]) 
        dC_next = np.zeros_like(C_s[0])   

        # w.r.t. softmax input
        ddense = np.copy(y_s[149])
        ddense[np.arange(len(Y)),np.argmax(Y,1)] -= 1

        # Softmax classifier's :
        dW = np.dot(v_1s[149].T,ddense)
        dB = np.sum(ddense,axis = 0, keepdims = True)

        ddense_hid = np.dot(ddense,self.W.T) * activations.dReLU(v_1s[149])
        dW_hid = np.dot(h_s[149].T,ddense_hid)
        dB_hid = np.sum(ddense_hid,axis = 0, keepdims = True)


        # Backprop
        for t in reversed(range(1,self.seq_len)):           
            
            # equating to more meaningful names
            stack,forget_gate,input_gate,cell_bar,cell_state,output_gate,hidden_state,dense,probs = z_s[t], f_s[t], i_s[t], C_bar_s[t], C_s[t], o_s[t], h_s[t],v_s[t], y_s[t]
            C_prev = C_s[t-1]
            
            # Output gate :
            dh = np.dot(ddense_hid,self.W_hid.T) + dh_next            
            do = dh * np.tanh(cell_state)
            do = do * dsigmoid(output_gate)
            dW_o += np.dot(stack.T,do)
            dB_o += np.sum(do,axis = 0, keepdims = True)

            # Cell state:
            dC = np.copy(dC_next)
            dC += dh * output_gate * activations.dtanh(cell_state)
            dC_bar = dC * input_gate
            dC_bar = dC_bar * dtanh(cell_bar) 
            dW_c += np.dot(stack.T,dC_bar)
            dB_c += np.sum(dC_bar,axis = 0, keepdims = True)
            
            # Input gate:
            di = dC * cell_bar
            di = dsigmoid(input_gate) * di
            dW_i += np.dot(stack.T,di)
            dB_i += np.sum(di,axis = 0,keepdims = True)

            # Forget gate:
            df = dC * C_prev
            df = df * dsigmoid(forget_gate) 
            dW_f += np.dot(stack.T,df)
            dB_f += np.sum(df,axis = 0, keepdims = True)

            dz = np.dot(df,self.W_f.T) + np.dot(di,self.W_i.T) + np.dot(dC_bar,self.W_c.T) + np.dot(do,self.W_o.T)

            dh_next = dz[:,-self.hidden_dim_1:]
            dC_next = forget_gate * dC
        
        # List of gradients :
        grads = [dW,dB,dW_hid,dB_hid,dW_o,dB_o,dW_c,dB_c,dW_i,dB_i,dW_f,dB_f]

        # Clipping gradients anyway
        for grad in grads:
            np.clip(grad, -15, 15, out = grad)

        return h_s[self.seq_len - 1],C_s[self.seq_len -1 ],grads
    


    def fit(self,X,Y,X_val,y_val,epochs = 50 ,optimizer = 'SGD',verbose = True, crossVal = False):

        for epoch in range(epochs):
            
            print(f'Epoch : {epoch + 1}')

            perm = np.random.permutation(3000)           
            h_prev,C_prev = np.zeros((self.batch_size,self.hidden_dim_1)),np.zeros((self.batch_size,self.hidden_dim_1))
            for i in range(round(X.shape[0]/self.batch_size) - 1): 
               
                batch_start  =  i * self.batch_size
                batch_finish = (i+1) * self.batch_size                
                index = perm[batch_start:batch_finish]
                
                # Feeding random indexes:
                X_feed = X[index]    
                y_feed = Y[index]
               
                # Forward + BPTT + SGD:
                cache_train = self.forward(X_feed,h_prev,C_prev)
                h,c,grads = self.BPTT(cache_train,y_feed)

                if optimizer == 'SGD':                                                           
                  self.SGD(grads)

                elif optimizer == 'AdaGrad' :
                  self.AdaGrad(grads)

                elif optimizer == 'RMSprop':
                  self.RMSprop(grads)
                
                elif optimizer == 'VanillaAdam':
                  self.VanillaAdam(grads)
                else:
                  self.Adam(grads)

                # Hidden state -------> Previous hidden state
                # Cell state ---------> Previous cell state
                h_prev,C_prev = h,c

            # Training metrics calculations:
            cross_loss_train = self.CategoricalCrossEntropy(y_feed,cache_train[8][149])
            predictions_train = self.predict(X)
            acc_train = metrics.accuracy(np.argmax(Y,1),predictions_train)

            # Validation metrics calculations:
            test_prevs = np.zeros((X_val.shape[0],self.hidden_dim_1))
            _,__,___,____,_____,______,_______,________,probs_test,a,b = self.forward(X_val,test_prevs,test_prevs)
            cross_loss_val = self.CategoricalCrossEntropy(y_val,probs_test[149])
            predictions_val = np.argmax(probs_test[149],1)
            acc_val = metrics.accuracy(np.argmax(y_val,1),predictions_val)

            if verbose:

                print(f"[{epoch + 1}/{epochs}] ------> Training :  Accuracy : {acc_train}")
                print(f"[{epoch + 1}/{epochs}] ------> Training :  Loss     : {cross_loss_train}")
                print('______________________________________________________________________________________\n')                         
                print(f"[{epoch + 1}/{epochs}] ------> Testing  :  Accuracy : {acc_val}")                                        
                print(f"[{epoch + 1}/{epochs}] ------> Testing  :  Loss     : {cross_loss_val}")
                print('______________________________________________________________________________________\n')
                
            self.train_loss.append(cross_loss_train)              
            self.test_loss.append(cross_loss_val) 
            self.train_acc.append(acc_train)              
            self.test_acc.append(acc_val)
      
    
    def params(self):
        """
        Return all weights/biases in sequential order starting from end in list form.

        """        
        return [self.W,self.B,self.W_hid,self.B_hid,self.W_o,self.B_o,self.W_c,self.B_c,self.W_i,self.B_i,self.W_f,self.B_f]


    def SGD(self,grads):
      """

      Stochastic gradient descent with momentum on mini-batches.
      """
      prevs = []
     
      for param,grad,prev_update in zip(self.params(),grads,self.previous_updates):            
          delta = self.learning_rate * grad - self.mom_coeff * prev_update
          param -= delta 
          prevs.append(delta)
     
         

      self.previous_updates = prevs     

      self.learning_rate *= 0.99999   

    
    def AdaGrad(self,grads):
      """
      AdaGrad adaptive optimization algorithm.
      """         
      i = 0
      for param,grad in zip(self.params(),grads):
        self.cache[i] += grad **2
        param += -self.learning_rate * grad / (np.sqrt(self.cache[i]) + 1e-6)
        i += 1


    def RMSprop(self,grads,decay_rate = 0.9):
      """
      RMSprop adaptive optimization algorithm
      """

      i = 0
      for param,grad in zip(self.params(),grads):
        self.cache_rmsprop[i] = decay_rate * self.cache_rmsprop[i] + (1-decay_rate) * grad **2
        param += - self.learning_rate * grad / (np.sqrt(self.cache_rmsprop[i])+ 1e-6)
        i += 1


    def VanillaAdam(self,grads,beta1 = 0.9,beta2 = 0.999):
        """
        Adam optimizer, but bias correction is not implemented
        """
        i = 0

        for param,grad  in zip(self.params(),grads):

          self.m[i] = beta1 * self.m[i] + (1-beta1) * grad          
          self.v[i] = beta2 * self.v[i] + (1-beta2) * grad **2  
          param += -self.learning_rate * self.m[i] / (np.sqrt(self.v[i]) + 1e-8)
          i += 1


    def Adam(self,grads,beta1 = 0.9,beta2 = 0.999):
        """

        Adam optimizer, bias correction is implemented.
        """
      
        i = 0

        for param,grad  in zip(self.params(),grads):
          
          self.m[i] = beta1 * self.m[i] + (1-beta1) * grad          
          self.v[i] = beta2 * self.v[i] + (1-beta2) * grad **2
          m_corrected = self.m[i] / (1-beta1**self.t)
          v_corrected = self.v[i] / (1-beta2**self.t)
          param += -self.learning_rate * m_corrected / (np.sqrt(v_corrected) + 1e-8)
          i += 1
          
        self.t +=1
    
    
    def CategoricalCrossEntropy(self,labels,preds):

        predictions = np.clip(preds, 1e-12, 1. - 1e-12)
        N = predictions.shape[0]         
        return -np.sum(labels * np.log(predictions + 1e-9)) / N
    
    def predict(self,X):

        # Give zeros to hidden/cell states:
        pasts = np.zeros((X.shape[0],self.hidden_dim_1))
        _,__,___,____,_____,______,_______,_______,probs,a,b = self.forward(X,pasts,pasts)
        return np.argmax(probs[149],axis=1)

    def history(self):
        return {'TrainLoss' : self.train_loss,
                'TrainAcc'  : self.train_acc,
                'TestLoss'  : self.test_loss,
                'TestAcc'   : self.test_acc}

mutl_layer_lstm = Multi_Layer_LSTM(learning_rate=1e-3,batch_size=32,hidden_dim_1 = 128,hidden_dim_2=64,mom_coeff=0.0)

mutl_layer_lstm.fit(X_train,y_train,X_test,y_test,epochs=15,optimizer='Adam')

mutl_layer_lstm_history = mutl_layer_lstm.history()

plt.figure()
plt.plot(mutl_layer_lstm_history['TestAcc'],'-o')
plt.plot(mutl_layer_lstm_history['TrainAcc'],'-x')
plt.xlabel('# of epochs')
plt.ylabel('Accuracy')
plt.title('LSTM Accuracy over epochs')
plt.legend(['Test','Validation'])
plt.show()

plt.figure()
plt.plot(mutl_layer_lstm_history['TestLoss'],'-o')
plt.plot(mutl_layer_lstm_history['TrainLoss'],'-x')
plt.xlabel('# of epochs')
plt.ylabel('Loss')
plt.title('LSTM Loss over epochs')
plt.legend(['Test','Validation'])
plt.show()

train_preds_lstm = mutl_layer_lstm.predict(X_train)
test_preds_lstm = mutl_layer_lstm.predict(X_test)
confusion_mat_train_lstm = metrics.confusion_matrix(np.argmax(y_train,1),train_preds_lstm)
confusion_mat_test_lstm = metrics.confusion_matrix(np.argmax(y_test,1),test_preds_lstm)

body_movements = ['downstairs','jogging','sitting','standing','upstairs','walking']
confusion_mat_train_lstm.columns = body_movements
confusion_mat_train_lstm.index = body_movements
confusion_mat_test_lstm.columns = body_movements
confusion_mat_test_lstm.index = body_movements

plt.figure()
sns.heatmap(confusion_mat_train_lstm/np.sum(confusion_mat_train_lstm), annot=True, fmt='.1%',cmap = 'Greens')
plt.title('LSTM Training Confusion Matrix')
plt.show()

plt.figure()
sns.heatmap(confusion_mat_test_lstm/np.sum(confusion_mat_test_lstm), annot=True, fmt='.1%',cmap = 'Greens')
plt.title('LSTM Testing Confusion Matrix')
plt.show()

"""Part c: GRU"""

class Multi_layer_GRU(object):

    def __init__(self,input_dim = 3,hidden_dim_1 = 128,hidden_dim_2 = 64,output_class = 6,seq_len = 150,batch_size = 32,learning_rate = 1e-1,mom_coeff = 0.85):
        """
        Initialization
        """
        np.random.seed(150)
        self.input_dim = input_dim
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2

        # Unfold case T = 150 :
        self.seq_len = seq_len
        self.output_class = output_class
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.mom_coeff = mom_coeff
        
        # Xavier uniform scaler :
        Xavier = lambda fan_in,fan_out : math.sqrt(6/(fan_in + fan_out))

        lim1 = Xavier(self.input_dim,self.hidden_dim_1)
        lim1_hid = Xavier(self.hidden_dim_1,self.hidden_dim_1)
        self.W_z = np.random.uniform(-lim1,lim1,(self.input_dim,self.hidden_dim_1))
        self.U_z = np.random.uniform(-lim1_hid,lim1_hid,(self.hidden_dim_1,self.hidden_dim_1))
        self.B_z = np.random.uniform(-lim1,lim1,(1,self.hidden_dim_1))

        self.W_r = np.random.uniform(-lim1,lim1,(self.input_dim,self.hidden_dim_1))
        self.U_r = np.random.uniform(-lim1_hid,lim1_hid,(self.hidden_dim_1,self.hidden_dim_1))
        self.B_r = np.random.uniform(-lim1,lim1,(1,self.hidden_dim_1))

        self.W_h = np.random.uniform(-lim1,lim1,(self.input_dim,self.hidden_dim_1))
        self.U_h = np.random.uniform(-lim1_hid,lim1_hid,(self.hidden_dim_1,self.hidden_dim_1))
        self.B_h = np.random.uniform(-lim1,lim1,(1,self.hidden_dim_1))

        lim2_hid = Xavier(self.hidden_dim_1,self.hidden_dim_2)
        self.W_hid = np.random.uniform(-lim2_hid,lim2_hid,(self.hidden_dim_1,self.hidden_dim_2))
        self.B_hid = np.random.uniform(-lim2_hid,lim2_hid,(1,self.hidden_dim_2))
        
        lim2 = Xavier(self.hidden_dim_2,self.output_class)
        self.W = np.random.uniform(-lim2,lim2,(self.hidden_dim_2,self.output_class))
        self.B = np.random.uniform(-lim2,lim2,(1,self.output_class))

        # To keep track loss and accuracy score :     
        self.train_loss,self.test_loss,self.train_acc,self.test_acc = [],[],[],[]
        
        # To keep previous updates in momentum :
        self.previous_updates = [0] * 13
        
        # For AdaGrad:
        self.cache = [0] * 13  
        self.cache_rmsprop = [0] * 13
        self.m = [0] * 13
        self.v = [0] * 13
        self.t = 1

    def cell_forward(self,X,h_prev):     

        # Update gate:
        update_gate = activations.sigmoid(np.dot(X,self.W_z) + np.dot(h_prev,self.U_z) + self.B_z)
       
        # Reset gate:
        reset_gate = activations.sigmoid(np.dot(X,self.W_r) + np.dot(h_prev,self.U_r) + self.B_r)

        # Current memory content:
        h_hat = np.tanh(np.dot(X,self.W_h) + np.dot(np.multiply(reset_gate,h_prev),self.U_h) + self.B_h)

        # Hidden state:
        hidden_state = np.multiply(update_gate,h_prev) + np.multiply((1-update_gate),h_hat)

        # Hidden MLP:
        hid_dense = np.dot(hidden_state,self.W_hid) + self.B_hid
        relu = activations.ReLU(hid_dense)

        # Classifiers (Softmax) :
        dense = np.dot(relu,self.W) + self.B
        probs = activations.softmax(dense)

        return (update_gate,reset_gate,h_hat,hidden_state,hid_dense,relu,dense,probs)        

    def forward(self,X,h_prev):
        x_s,z_s,r_s,h_hat = {},{},{},{}
        h_s = {}
        hd_s,relu_s = {},{}
        y_s,p_s = {},{}        

        h_s[-1] = np.copy(h_prev)
        

        for t in range(self.seq_len):
            x_s[t] = X[:,t,:]
            z_s[t], r_s[t], h_hat[t], h_s[t],hd_s[t],relu_s[t], y_s[t], p_s[t] = self.cell_forward(x_s[t],h_s[t-1])

        return (x_s,z_s, r_s, h_hat, h_s, hd_s,relu_s, y_s, p_s)
    
    def BPTT(self,outs,Y):

        x_s,z_s, r_s, h_hat, h_s, hd_s,relu_s, y_s, p_s = outs

        dW_z, dW_r,dW_h, dW = np.zeros_like(self.W_z), np.zeros_like(self.W_r), np.zeros_like(self.W_h),np.zeros_like(self.W)
        dW_hid = np.zeros_like(self.W_hid)
        dU_z, dU_r,dU_h = np.zeros_like(self.U_z), np.zeros_like(self.U_r), np.zeros_like(self.U_h)


        dB_z, dB_r,dB_h,dB = np.zeros_like(self.B_z), np.zeros_like(self.B_r),np.zeros_like(self.B_h),np.zeros_like(self.B)
        dB_hid = np.zeros_like(self.B_hid)
        dh_next = np.zeros_like(h_s[0]) 
           
        # w.r.t. softmax input
        ddense = np.copy(p_s[149])
        ddense[np.arange(len(Y)),np.argmax(Y,1)] -= 1
        # Softmax classifier's :
        dW = np.dot(relu_s[149].T,ddense)
        dB = np.sum(ddense,axis = 0, keepdims = True)

        ddense_hid = np.dot(ddense,self.W.T) * activations.dReLU(hd_s[149])
        dW_hid = np.dot(h_s[149].T,ddense_hid)
        dB_hid = np.sum(ddense_hid,axis = 0, keepdims = True)

   
        # Backprop
        for t in reversed(range(1,self.seq_len)):           

            # Curernt memory state :
            dh = np.dot(ddense_hid,self.W_hid.T) + dh_next            
            dh_hat = dh * (1-z_s[t])
            dh_hat = dh_hat * dtanh(h_hat[t])
            dW_h += np.dot(x_s[t].T,dh_hat)
            dU_h += np.dot((r_s[t] * h_s[t-1]).T,dh_hat)
            dB_h += np.sum(dh_hat,axis = 0, keepdims = True)

            # Reset gate:
            dr_1 = np.dot(dh_hat,self.U_h.T)
            dr = dr_1  * h_s[t-1]
            dr = dr * dsigmoid(r_s[t])
            dW_r += np.dot(x_s[t].T,dr)
            dU_r += np.dot(h_s[t-1].T,dr)
            dB_r += np.sum(dr,axis = 0, keepdims = True)

            # Forget gate:
            dz = dh * (h_s[t-1] - h_hat[t])
            dz = dz * dsigmoid(z_s[t])
            dW_z += np.dot(x_s[t].T,dz)
            dU_z += np.dot(h_s[t-1].T,dz)
            dB_z += np.sum(dz,axis = 0, keepdims = True)

            # Nexts:
            dh_next = np.dot(dz,self.U_z.T) + (dh * z_s[t]) + (dr_1 * r_s[t]) + np.dot(dr,self.U_r.T)


        # List of gradients :
        grads = [dW,dB,dW_hid,dB_hid,dW_z,dU_z,dB_z,dW_r,dU_r,dB_r,dW_h,dU_h,dB_h]
              
        # Clipping gradients anyway
        for grad in grads:
            np.clip(grad, -15, 15, out = grad)

        return h_s[self.seq_len - 1],grads
    

    def fit(self,X,Y,X_val,y_val,epochs = 50 ,optimizer = 'SGD',verbose = True, crossVal = False):  
        
        for epoch in range(epochs):
            
            print(f'Epoch : {epoch + 1}')

            perm = np.random.permutation(3000)

            # Equate 0 in every epoch:           
            h_prev = np.zeros((self.batch_size,self.hidden_dim_1))

            for i in range(round(X.shape[0]/self.batch_size) - 1): 
               
                batch_start  =  i * self.batch_size
                batch_finish = (i+1) * self.batch_size                
                index = perm[batch_start:batch_finish]
                
                # Feeding random indexes:
                X_feed = X[index]    
                y_feed = Y[index]
               
                # Forward + BPTT + Optimization:
                cache_train = self.forward(X_feed,h_prev)
                h,grads = self.BPTT(cache_train,y_feed)

                if optimizer == 'SGD':                                                                
                  self.SGD(grads)

                elif optimizer == 'AdaGrad' :
                  self.AdaGrad(grads)

                elif optimizer == 'RMSprop':
                  self.RMSprop(grads)
                
                elif optimizer == 'VanillaAdam':
                  self.VanillaAdam(grads)
                else:
                  self.Adam(grads)

                # Hidden state -------> Previous hidden state
                h_prev = h

            # Training metrics calculations:
            cross_loss_train = self.CategoricalCrossEntropy(y_feed,cache_train[8][149])
            predictions_train = self.predict(X)
            acc_train = metrics.accuracy(np.argmax(Y,1),predictions_train)

            # Validation metrics calculations:
            test_prevs = np.zeros((X_val.shape[0],self.hidden_dim_1))
            _,__,___,____,_____,______,_______,________,probs_test = self.forward(X_val,test_prevs)
            cross_loss_val = self.CategoricalCrossEntropy(y_val,probs_test[149])
            predictions_val = np.argmax(probs_test[149],1)
            acc_val = metrics.accuracy(np.argmax(y_val,1),predictions_val)

            if verbose:

                print(f"[{epoch + 1}/{epochs}] ------> Training :  Accuracy : {acc_train}")
                print(f"[{epoch + 1}/{epochs}] ------> Training :  Loss     : {cross_loss_train}")
                print('______________________________________________________________________________________\n')                         
                print(f"[{epoch + 1}/{epochs}] ------> Testing  :  Accuracy : {acc_val}")                                        
                print(f"[{epoch + 1}/{epochs}] ------> Testing  :  Loss     : {cross_loss_val}")
                print('______________________________________________________________________________________\n')
                
            self.train_loss.append(cross_loss_train)              
            self.test_loss.append(cross_loss_val) 
            self.train_acc.append(acc_train)              
            self.test_acc.append(acc_val)
      
    
    def params(self):
        """
        Return all weights/biases in sequence

        """        
        return [self.W,self.B,self.W_hid,self.B_hid,self.W_z,self.U_z,self.B_z,self.W_r,self.U_r,self.B_r,self.W_h,self.U_h,self.B_h]

    def SGD(self,grads):

      prevs = []
      
      for param,grad,prev_update in zip(self.params(),grads,self.previous_updates): 
                     
          delta = self.learning_rate * grad + self.mom_coeff * prev_update
          param -= delta 
          prevs.append(delta)
        

      self.previous_updates = prevs     
      self.learning_rate *= 0.99999   

    
    def AdaGrad(self,grads):
      """
      AdaGrad adaptive optimization algorithm.
      """      

      i = 0
      for param,grad in zip(self.params(),grads):

        self.cache[i] += grad **2
        param += -self.learning_rate * grad / (np.sqrt(self.cache[i]) + 1e-6)

        i += 1


    def RMSprop(self,grads,decay_rate = 0.9):
      """
      RMSprop adaptive optimization algorithm
      """
      i = 0
      for param,grad in zip(self.params(),grads):
        self.cache_rmsprop[i] = decay_rate * self.cache_rmsprop[i] + (1-decay_rate) * grad **2
        param += - self.learning_rate * grad / (np.sqrt(self.cache_rmsprop[i])+ 1e-6)
        i += 1


    def VanillaAdam(self,grads,beta1 = 0.9,beta2 = 0.999):
        """
        Adam optimizer, but bias correction is not implemented
        """
        i = 0

        for param,grad  in zip(self.params(),grads):

          self.m[i] = beta1 * self.m[i] + (1-beta1) * grad          
          self.v[i] = beta2 * self.v[i] + (1-beta2) * grad **2  
          param += -self.learning_rate * self.m[i] / (np.sqrt(self.v[i]) + 1e-8)
          i += 1


    def Adam(self,grads,beta1 = 0.9,beta2 = 0.999):
        """

        Adam optimizer, bias correction is implemented.
        """
      
        i = 0

        for param,grad  in zip(self.params(),grads):
          
          self.m[i] = beta1 * self.m[i] + (1-beta1) * grad          
          self.v[i] = beta2 * self.v[i] + (1-beta2) * grad **2
          m_corrected = self.m[i] / (1-beta1**self.t)
          v_corrected = self.v[i] / (1-beta2**self.t)
          param += -self.learning_rate * m_corrected / (np.sqrt(v_corrected) + 1e-8)
          i += 1
          
        self.t +=1
    
    
    def CategoricalCrossEntropy(self,labels,preds):

        predictions = np.clip(preds, 1e-12, 1. - 1e-12)
        N = predictions.shape[0]         
        return -np.sum(labels * np.log(predictions + 1e-9)) / N
    
    def predict(self,X):
      
        # Give zeros to hidden states:
        pasts = np.zeros((X.shape[0],self.hidden_dim_1))
        _,__,___,____,_____,______,_______,________,probs = self.forward(X,pasts)
        return np.argmax(probs[149],axis=1)

    def history(self):
        return {'TrainLoss' : self.train_loss,
                'TrainAcc'  : self.train_acc,
                'TestLoss'  : self.test_loss,
                'TestAcc'   : self.test_acc}

multi_layer_gru = Multi_layer_GRU(hidden_dim_1=128,hidden_dim_2=64,learning_rate=1e-3,mom_coeff=0.0,batch_size=32)

multi_layer_gru.fit(X_train,y_train,X_test,y_test,epochs = 15,optimizer = 'RMSprop')

multi_layer_gru_history = multi_layer_gru.history()

plt.figure()
plt.plot(multi_layer_gru_history['TestAcc'],'-o')
plt.plot(multi_layer_gru_history['TrainAcc'],'-x')
plt.xlabel('# of epochs')
plt.ylabel('Accuracy')
plt.title('GRU Accuracy over epochs')
plt.legend(['Test','Validation'])
plt.show()


plt.figure()
plt.plot(multi_layer_gru_history['TestLoss'],'-o')
plt.plot(multi_layer_gru_history['TrainLoss'],'-x')
plt.xlabel('# of epochs')
plt.ylabel('Loss')
plt.title('GRU Loss over epochs')
plt.legend(['Test','Validation'])
plt.show()

train_preds_gru = multi_layer_gru.predict(X_train)
test_preds_gru = multi_layer_gru.predict(X_test)
confusion_mat_train_gru = metrics.confusion_matrix(np.argmax(y_train,1),train_preds_gru)
confusion_mat_test_gru = metrics.confusion_matrix(np.argmax(y_test,1),test_preds_gru)

body_movements = ['downstairs','jogging','sitting','standing','upstairs','walking']
confusion_mat_train_gru.columns = body_movements
confusion_mat_train_gru.index = body_movements
confusion_mat_test_gru.columns = body_movements
confusion_mat_test_gru.index = body_movements

plt.figure()
sns.heatmap(confusion_mat_train_gru/np.sum(confusion_mat_train_gru), annot=True, fmt='.1%',cmap = 'Greens')
plt.title('GRU Training Confusion Matrix')
plt.show()

plt.figure()
sns.heatmap(confusion_mat_test_gru/np.sum(confusion_mat_test_gru), annot=True, fmt='.1%',cmap = 'Greens')
plt.title('GRU Testing Confusion Matrix')
plt.show()