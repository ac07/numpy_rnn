from __future__ import division
import os
import gzip
import cPickle
import numpy as np


def load_data():
	f_path = os.getenv("HOME") + '/iEd/iWork/expt/data/mnist.pkl.gz'
	f = gzip.open(f_path,'rb')
	training_data, validation_data, test_data = cPickle.load(f)
	f.close()
	return(training_data, validation_data, test_data)

#PREPARE DATA IN BATCHES FOR APPLYING TO RNN

def prepare_training_data(train_x, train_y, params):
  train_batch = []
  resp_batch = []
  for batch_indx in range(0, len(train_x), params.get('N')):
    xt_batch = train_x[batch_indx:batch_indx+params.get('N')]
    xr_batch = train_y[batch_indx:batch_indx+params.get('N')]
    step_data = np.zeros((params.get('N'), params.get('D')))
    batch_data = []
    for indx in range(0,params.get('full_d'),params.get('D')): 
      step_data = xt_batch[:,indx:indx+params.get('D')]
      batch_data.append(step_data)
    train_batch.append(batch_data)
    resp_batch.append(xr_batch)
  return train_batch, resp_batch  

#Reshape Train
reshape_train = lambda x: x.reshape(28,28)

#Convert Response to vectors
response_vec = lambda sample : np.array([1 if i == sample else 0 for i in range(10)]).reshape(10,)

#LOAD DATA
train, valid, test = load_data()
train_x = np.array(train[0])
train_y = map(response_vec, train[1])
test_x = np.array(test[0])
test_y = test[1]

#CREATE BATCHES
# params = {'T':28, 'D':28, 'N':50, 'full_d':784, 'H':128, 'O':10}
# train_batch, resp_batch = prepare_training_data(train_x, train_y, params)
params = {'T':4, 'D':196, 'N':10, 'full_d':784, 'H':150, 'O':10}
train_batch, resp_batch = prepare_training_data(train_x, train_y, params)


############################################################################################
#RNN_STEPS


#OUTPUT ACTIVATION
softmax = lambda arr: np.array([elem/np.sum(arr) for elem in arr])
softmax_e = lambda x: np.exp(x) / np.sum(np.exp(x), axis=0)

#FORWARD_ON_ONE_TIMESTEP 
def rnn_step_forward(x, prev_h, Wx, Wh, bh):
  
  z_ix = np.dot(x, Wx)  #n*h
  z_hh = np.dot(prev_h, Wh)  #n*h
  next_h = np.tanh(np.add(z_ix, z_hh) + bh) #n*h
  cache = (next_h, x, prev_h, Wx, Wh, bh)
  return next_h, cache

#FORWARD_ACROSS_ENTIRE_TIMESTEPS_FOR_A_BATCH 
def rnn_forward(x, h0, Wx, Wh, Wy, bh):
  N = x[0].shape[0]
  D = x[0].shape[1]
  T = len(x)
  H = Wh.shape[0] 
  cache = []
  h = np.zeros([N,T,H])
  prev_h = h0 


  for time_step in range(len(x)):
    prev_h, f_cache =  rnn_step_forward(x[time_step], prev_h, Wx, Wh, bh)
    cache.append(f_cache)
    h[:,time_step,:] = prev_h


  hzo = h[:,-1,:]
  zo = np.dot(hzo, Wy)
  timestep_pred = np.array(map(softmax_e, zo))
  

  return timestep_pred, h, cache

#BACKPROP_OVER_ONE_TIMESTEP
def rnn_step_backward(dnext_h, cache):

  dx, dprev_h, dWx, dWh, db = None, None, None, None, None
  del_h = np.multiply(dnext_h,(1-np.power(cache[0],2))) #grad for nonlinearity
  dbh = np.sum(del_h, axis=0) #grad for bias units
  dWx = np.dot(cache[1].T, del_h) #grad for input weights
  dWh = np.dot(cache[2].T, del_h) #grad for hidden weights
  dprev_h = np.dot(del_h,cache[4].T) #hidden-state grad going to the prev step
  #dx = np.dot(del_h, cache[3].T) #grad for inputs - req for embeddings
  # return dx, dprev_h, dWx, dWh, db
  return dprev_h, dWx, dWh, dbh


#BACKPROP_OVER_ALL_TIMESTEPS_IN_A_BATCH
def rnn_backward(dh,cache):

  N, H = dh.shape
  T = len(cache)
  D = cache[0][1].shape[1]
  #tdprev_h = np.random.randn(dh.shape[0],dh.shape[1]) + dh
  tdprev_h = np.zeros((N,H)) + dh
  #dx  = np.zeros((N,T,D))
  dWx = np.zeros((D,H))
  dWh = np.zeros((H,H))
  dbh = np.zeros((H,))


  for time_step in reversed(range(len(cache))): 
    # tdx, tdprev_h, tdWx, tdWh, tdb = rnn_step_backward(dh[:,time_step,:]+tdprev_h, cache[time_step])
    #dx[:,time_step,:] = tdx
    tdprev_h, tdWx, tdWh, tdbh = rnn_step_backward(tdprev_h, cache[time_step])
    dWx += tdWx
    dWh += tdWh
    dbh += tdbh
    
  #return dx, dh0, dWx, dWh, db
  return dWx, dWh, dbh


#RNN_BACKPROP_CONTROL
def rnn_backprop(timestep_pred, hidden_states, cache, y, Wy):
  #What is the loss at the last layer?
  #Backprop through softmax_activation, output
  del_y = y - timestep_pred
  dby = np.sum(del_y, axis=0)
  dWy = np.dot(hidden_states[:,-1,:].T, del_y)
  dh  = np.dot(del_y, Wy.T)
  dWx, dWh, dbh  =  rnn_backward(dh, cache)
  return dWy, dWx, dWh, dby, dbh

translate = lambda x: np.argmax(x)
get_acc = lambda x,y: 1 if x==y else 0

def compute_acc(resp, y):
  pred = map(translate, resp)
  resp = map(translate, y)
  perf = map(get_acc, pred, resp)
  return sum(perf)*100/len(perf)


#INITIALIZE WEIGHTS
Wx = np.random.randn(params.get('D'), params.get('H'))
Wh = np.random.randn(params.get('H'), params.get('H'))
Wy = np.random.randn(params.get('H'), params.get('O'))
bh = np.random.randn(params.get('H'),)
by = np.random.randn(params.get('O'),)
h0 = np.random.randn(params.get('N'), params.get('H')) 

test = 1

#OPTIMIZER
epochs = 1
#START ITERATIONS
for epoch in range(epochs): 
  print '\nIteration: ', epoch+1
  #ITERATE OVER BATCHES
  for batch_indx in range(len(train_batch)):
    print '>', batch_indx
    x = train_batch[batch_indx]
    y = np.array(resp_batch[batch_indx])
    #FORWARD PROP
    timestep_pred, hidden_states, cache = rnn_forward(x, h0, Wx, Wh, Wy, bh)
    #UPDATE H0
    h0 = hidden_states[:,-1,:]
    #BACKPROP
    dWy, dWx, dWh, dby, dbh = rnn_backprop(timestep_pred, hidden_states, cache, y, Wy)
    #WEIGHT_UPDATE

    if test==1: 
      print '\n\n\tWEIGHTS: '
      for i,j in [(Wy,dWy),(Wh,dWh),(Wx,dWx),(by,dby), (bh,dbh)]: print '\t>', np.sum(i), np.sum(j)

      # print '\tGRadIENTS: '
      # for items in [dWy, dWx, dWh, dby, dbh]: print'\t>', np.sum(items)

    Wy += dWy
    Wh += dWh
    Wx += dWx
    by += dby
    bh += dbh    

    # if test==1: 
    #   print '\tUWEIGHTS: ',
    #   for items in [Wy, Wx, Wh, by, bh]: print '\t>', np.sum(items)

    if (batch_indx+1)%100 == 0: 
      print '>', batch_indx
      print '>Accuracy: ', compute_acc(timestep_pred, y)

  #REPEAT
