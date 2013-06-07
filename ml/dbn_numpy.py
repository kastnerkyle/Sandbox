#!/usr/bin/python
import time
import numpy as np
import rbm_numpy

rbm_numpy.load_dat('mnist.dat', globals())
dat = dat/255.
layer_sizes = [500]
weights = []
err = []
for n,l in enumerate(layer_sizes):
    if n == 0:
        v = dat
        e, w = rbm_numpy.train_rbm(v,num_hid=l)
    weights.append(w)
    err.append(e)

import matplotlib.pyplot as plot
for e in err:
    plot.plot(e)
plot.show()


