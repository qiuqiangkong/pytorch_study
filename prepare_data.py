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

def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)

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



# Load cifar data. 
def load_cifar10():
    def _load_file(path):
        data_lb = cPickle.load(open(path, 'rb'))
        return data_lb['data'], data_lb['labels']
    
    # load train data
    dataset_dir = "/vol/vssp/msos/qk/Datasets/cifar10/cifar-10-batches-py"
    
    tr_x, tr_y = [], []
    for na in ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']:
        data_path = os.path.join(dataset_dir, na)
        (x, y) = _load_file(data_path)
        tr_x.append(x)
        tr_y.append(y)
    
    tr_x = np.concatenate(tr_x, axis=0)
    tr_y = np.concatenate(tr_y, axis=0)
    
    # load test data
    data_path = os.path.join(dataset_dir, 'test_batch')
    (te_x, te_y) = _load_file(data_path)
    te_y = np.array(te_y)
    
    
    # normalize data to [-1,1]
    tr_x = (tr_x.astype(np.float32) - 128.) / 128.
    te_x = (te_x.astype(np.float32) - 128.) / 128.
    print(tr_x.dtype)
    
    return tr_x, tr_y, te_x, te_y
    
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