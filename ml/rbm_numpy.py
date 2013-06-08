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
    b_v = np.zeros((num_vis, 1))
    b_h = np.zeros((num_hid, 1))

    wu_vh = np.zeros(w_vh.shape)
    bu_v = np.zeros(b_v.shape)
    bu_h = np.zeros(b_h.shape)

    start_time = time.time()
    err = []
    for epoch in range(num_epochs):
        print "Epoch " + str(epoch)
        for batch in range(num_batches):
            v_true = data[:, batch*batch_size:(batch + 1)*batch_size]
            v = v_true

            wu_vh *= momentum
            bu_v *= momentum
            bu_h *= momentum

            h = 1./ (1 + np.exp(-(np.dot(w_vh.T, v) + b_h)))

            wu_vh += np.dot(v, h.T)
            bu_v += v.sum(1)[:, np.newaxis]
            bu_h += h.sum(1)[:, np.newaxis]

            h = 1. * (h > np.random.rand(num_hid, batch_size))

            v = 1./ (1 + np.exp(-(np.dot(w_vh, h) + b_v)))
            h = 1. / (1 + np.exp(-(np.dot(w_vh.T, v) + b_h)))

            wu_vh -= np.dot(v, h.T)
            bu_v -= v.sum(1)[:, np.newaxis]
            bu_h -= h.sum(1)[:, np.newaxis]

            w_vh += learning_rate/batch_size * wu_vh
            b_v += learning_rate/batch_size * bu_v
            b_h += learning_rate/batch_size * bu_h

            err.append(np.mean((v - v_true)**2))
    print "Mean squared error " + str(np.mean(err))
    print "Time: " + str(time.time() - start_time)
    return err, w_vh, b_v, b_h

if __name__ == "__main__":
    load_dat('mnist.dat', globals())
    dat = dat/255.
    err, weights, visbias, hidbias = train_rbm(dat)
    import matplotlib.pyplot as plot
    plot.plot(err)
    plot.show()

