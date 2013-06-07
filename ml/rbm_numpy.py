#!/usr/bin/python
#Retyped directly from cudamat as a learning excercise
import time
import numpy as np

def load_dat(fname, target_dict, verbose = True):
    import gzip
    import cPickle as pickle
    fo = gzip.GzipFile(fname, 'rb')
    var_list = pickle.load(fo)
    if verbose:
        print var_list
    for var in var_list:
        target_dict[var] = pickle.load(fo)
    fo.close()

def train_rbm(data, learning_rate=0.01, momentum=.9, num_epochs=1, batch_size=64, num_hid=1024):
    num_batches = data.shape[1]/batch_size
    num_vis = data.shape[0]

    w_vh = 0.1 * np.random.randn(num_vis, num_hid)
    w_v = np.zeros((num_vis, 1))
    w_h = np.zeros((num_hid, 1))

    wu_vh = np.zeros(w_vh.shape)
    wu_v = np.zeros(w_v.shape)
    wu_h = np.zeros(w_h.shape)

    start_time = time.time()
    err = []
    for epoch in range(num_epochs):
        print "Epoch " + str(epoch)
        for batch in range(num_batches):
            v_true = data[:, batch*batch_size:(batch + 1)*batch_size]
            v = v_true

            wu_vh *= momentum
            wu_v *= momentum
            wu_h *= momentum

            h = 1./ (1 + np.exp(-(np.dot(w_vh.T, v) + w_h)))

            wu_vh += np.dot(v, h.T)
            wu_v += v.sum(1)[:, np.newaxis]
            wu_h += h.sum(1)[:, np.newaxis]

            h = 1. * (h > np.random.rand(num_hid, batch_size))

            v = 1./ (1 + np.exp(-(np.dot(w_vh, h) + w_v)))
            h = 1. / (1 + np.exp(-(np.dot(w_vh.T, v) + w_h)))

            wu_vh -= np.dot(v, h.T)
            wu_v -= v.sum(1)[:, np.newaxis]
            wu_h -= h.sum(1)[:, np.newaxis]

            w_vh += learning_rate/batch_size * wu_vh
            w_v += learning_rate/batch_size * wu_v
            w_h += learning_rate/batch_size * wu_h

            err.append(np.mean((v - v_true)**2))
    print "Mean squared error " + str(np.mean(err))
    print "Time: " + str(time.time() - start_time)
    return err, w_vh

if __name__ == "__main__":
    load_dat('mnist.dat', globals())
    dat = dat/255.
    err, weights = train_rbm(dat)
    import matplotlib.pyplot as plot
    plot.plot(err)
    plot.show()

