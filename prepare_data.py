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
import matplotlib.pyplot as plt
import torch

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
    
    return tr_x, tr_y, te_x, te_y
    
def load_imdb(path="imdb.pkl", nb_words=None, skip_top=0,
              maxlen=None, test_split=0.2, seed=113,
              start_char=1, oov_char=2, index_from=3):
    """
    Load imdb data, from https://github.com/fchollet/keras/blob/master/keras/datasets/imdb.py
    oov: out-of vocabulary
    """

    if not os.path.isfile(path):
        from six.moves import urllib
        print 'downloading data ... (15.3 Mb)'
        urllib.request.urlretrieve( 'https://s3.amazonaws.com/text-datasets/imdb.pkl', path )

    if path.endswith(".gz"):
        f = gzip.open(path, 'rb')
    else:
        f = open(path, 'rb')

    X, labels = cPickle.load(f)
    f.close()

    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(labels)

    if start_char is not None:
        X = [[start_char] + [w + index_from for w in x] for x in X]
    elif index_from:
        X = [[w + index_from for w in x] for x in X]

    if maxlen:
        new_X = []
        new_labels = []
        for x, y in zip(X, labels):
            if len(x) < maxlen:
                new_X.append(x)
                new_labels.append(y)
        X = new_X
        labels = new_labels
    if not X:
        raise Exception('After filtering for sequences shorter than maxlen=' +
                        str(maxlen) + ', no sequence was kept. '
                        'Increase maxlen.')
    if not nb_words:
        nb_words = max([max(x) for x in X])

    # by convention, use 2 as OOV word
    # reserve 'index_from' (=3 by default) characters: 0 (padding), 1 (start), 2 (OOV)
    if oov_char is not None:
        X = [[oov_char if (w >= nb_words or w < skip_top) else w for w in x] for x in X]
    else:
        nX = []
        for x in X:
            nx = []
            for w in x:
                if (w >= nb_words or w < skip_top):
                    nx.append(w)
            nX.append(nx)
        X = nX

    X = np.array([np.array(e) for e in X])
    

    X_train = np.array(X[:int(len(X) * (1 - test_split))])
    y_train = np.array(labels[:int(len(X) * (1 - test_split))])

    X_test = np.array(X[int(len(X) * (1 - test_split)):])
    y_test = np.array(labels[int(len(X) * (1 - test_split)):])

    return X_train, y_train, X_test, y_test
    
    
class PtbDictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)
    
class PtbCorpus(object):
    """Copy from https://github.com/pytorch/examples/blob/master/word_language_model/data.py
    """
    def __init__(self, path):
        self.dictionary = PtbDictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids

###
    
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
    

def pad_trunc_seqs(x, max_len, pad_type='post'):
    """Pad or truncate sequences. 
    
    Args:
      x: ndarray | list of ndarray. Each element in x should be ndarray. Each
          element in x is padded with 0 or truncated to max_len. 
      max_len: int, length to be padded with 0 or truncated. 
      pad_type, string, 'pre' | 'post'. 
      
    Returns: 
      x_new: ndarray, (N, ndarray), padded or truncated sequences. 
      mask: ndarray, (N, max_len), mask of padding. 
    """
    list_x_new, list_mask = [], []
    for e in x:
        L = len(e)
        e_new, mask = pad_trunc_seq(e, max_len, pad_type)
        list_x_new.append(e_new)
        list_mask.append(mask)
    
    type_x = type(x)
    if type_x==list:
        return list_x_new, list_mask
    elif type_x==np.ndarray:
        return np.array(list_x_new), np.array(list_mask)
    else:
        raise Exception("Input should be list or ndarray!")
    

def pad_trunc_seq(x, max_len, pad_type='post'):
    """Pad or truncate ndarray. 
    
    Args:
      x: ndarray. 
      max_len: int, length to be padded with 0 or truncated. 
      pad_type, string, 'pre' | 'post'. 
      
    Returns:
      x_new: ndarray, padded or truncated ndarray. 
      mask: 1d-array, mask of padding. 
    """
    L = len(x)
    shape = x.shape
    data_type = x.dtype
    if L < max_len:
        pad_shape = (max_len-L,) + shape[1:]
        pad = np.zeros(pad_shape)
        if pad_type=='pre': 
            x_new = np.concatenate((pad, x), axis=0)
            mask = np.concatenate([np.zeros(max_len-L), np.ones(L)])
        elif pad_type=='post': 
            x_new = np.concatenate((x, pad), axis=0)
            mask = np.concatenate([np.ones(L), np.zeros(max_len-L)])
        else:
            raise Exception("pad_type should be 'post' | 'pre'!")
    else:
        if pad_type=='pre':
            x_new = x[L-max_len:]
            mask = np.ones(max_len)
        elif pad_type=='post':
            x_new = x[0:max_len]
            mask = np.ones(max_len)
        else:
            raise Exception("pad_type should be 'post' | 'pre'!")
    x_new = x_new.astype(data_type)
    return x_new, mask