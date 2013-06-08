#!/usr/bin/python
import time
import numpy as np
import rbm_numpy
import random

rbm_numpy.load_dat('mnist.dat', globals())
dat = dat/255.
layer_sizes = [1024,512,256,128,64]
weights = []
err = []
for n,l in enumerate(layer_sizes):
    if n == 0:
        v = dat
    e, w, b_v, b_h = rbm_numpy.train_rbm(v,num_hid=l)
    p = 1./ (1 + np.exp(-(np.dot(w.T, v) + b_h)))
    h = (p > np.random.rand(*p.shape))
    v = h
    weights.append(w)
    err.append(e)

import matplotlib.pyplot as plot
for e in err:
    plot.plot(e)
plot.show()


