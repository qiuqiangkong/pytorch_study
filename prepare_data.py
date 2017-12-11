'''
SUMMARY:  Download mnist dataset
Ref:      http://deeplearning.net/tutorial/code/logistic_sgd.py
AUTHOR:   Qiuqiang Kong
Created:  2016.05.01
Modified: 2016.07.19 Modify dataset path
--------------------------------------
'''
import numpy as np
import cPickle
import gzip
import os

def load_data():
    dataset = 'mnist.pkl.gz'
    if not os.path.isfile(dataset):
        from six.moves import urllib
        print 'downloading data ... (16.2 Mb)'
        urllib.request.urlretrieve( 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz', dataset )
        
    f = gzip.open( dataset, 'rb' )
    train_set, valid_set, test_set = cPickle.load(f)
    [tr_X, tr_y] = train_set
    [va_X, va_y] = valid_set
    [te_X, te_y] = test_set
    f.close()
    return tr_X, tr_y, va_X, va_y, te_X, te_y
    
def sparse_to_categorical(x, n_out):
    x = x.astype(int)   # force type to int
    shape = x.shape
    x = x.flatten()
    N = len(x)
    x_categ = np.zeros((N,n_out))
    x_categ[np.arange(N), x] = 1
    return x_categ.reshape((shape)+(n_out,))
    
def categorical_error(p_y_pred, y_gt):
    assert len(p_y_pred)==len(y_gt), "Length of y_out and y_gt (ground true) is not equal!"
    N = len(p_y_pred)
    sp_y_pred = np.argmax(p_y_pred, axis=-1)
    sp_y_gt = np.argmax(y_gt, axis=-1)
    err = np.sum(np.not_equal(sp_y_pred, sp_y_gt)) / float(np.prod(sp_y_gt.shape))
    return err