#!/usr/bin/python
import time
import numpy as np
import rbm_numpy

rbm_numpy.load_dat('mnist.dat', globals())
dat = dat/255.
num_layers = 3
layer_sizes = [dat.shape[0], 500, 250]
w = []
err, weights = rbm_numpy.train_rbm(dat)
w.append(weights)
import matplotlib.pyplot as plot
plot.plot(err)
plot.show()


