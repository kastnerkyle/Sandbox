#!/usr/bin/python
#Needs the following libs
#sudo apt-get install python-numpy python-scipy python-matplotlib

import numpy as np
import argparse
import sys
import matplotlib.pyplot as plot
import scipy.signal as sg
from scipy.io import loadmat
from scipy.io import wavfile
from numpy.lib.stride_tricks import as_strided as ast

#Grabbed samples from http://www.colorado.edu/physics/phys4830/phys4830_fa01/sounds/

class EndpointsAction(argparse.Action):
    def __call__(self, parser, args, values, option = None):
        setattr(args, self.dest, map(int,values))
        if len(args.endpoints) < 3:
            defaults = [0,None, 1]
            print "Wrong number of arguments, require 3 values, --endpoints start stop step"
            print "Using default endpoints of " + `args.endpoints`
            setattr(args, self.dest, defaults)

parser = argparse.ArgumentParser(description="Apply filter tutorial to input data")
parser.add_argument("-f", "--filename", dest="filename", default=".noexist", help="Optional WAV file to be processed, default generates a 1 sec full range complex chirp to filter")
parser.add_argument("-e", "--endpoints", dest="endpoints", default=[0,None, 1], action=EndpointsAction, nargs="*", help='Start and stop endpoints for data, default will try to process the whole file')
parser.add_argument("-v", "--verbose", dest="verbose", action="count", help='Verbosity, -v for verbose or -vv for very verbose')

try:
    args = parser.parse_args()
except SystemExit:
    parser.print_help()
    sys.exit()

if args.filename[-4:] == ".wav":
    sr, data = wavfile.read(args.filename)
    data = np.asarray(data)[::args.endpoints[2]]
    data = data[args.endpoints[0]:args.endpoints[1]]
elif args.filename[-4:] == ".mat":
    mat = loadmat(args.filename)
    data = mat[mat.keys()[0]].flatten()
elif args.filename == ".noexist":
    parser.print_help()
    sys.exit()

def overlap_data_stream(data, chunk=256, overlap_percentage=.75):
    chunk_count = len(data)/chunk
    overlap_samples = int(chunk*overlap_percentage)+1
    extended_length = (chunk_count+1)*(chunk-overlap_samples)
    data = np.hstack((np.asarray(data),np.asarray([0]*(extended_length-len(data)))))
    shape = (len(data)/(chunk-overlap_samples),chunk)
    strides = (data.itemsize*(chunk-overlap_samples), data.itemsize)
    return ast(data, shape=shape, strides=strides)

def autocorr(data):
    return [np.correlate(x,x, mode="full")[len(x):2*len(x)] for x in data]

def peak_search(data):
    data = data.flatten()
    data = [x - min(data) for x in data]
    norm_data = [float(x)/max(data) for x in data]
    all_max = [max(norm_data)]
    pos_max = []
    all_min = []
    pos_min = []
    delta = np.std(norm_data)
    find_max = False

    for n,i in enumerate(norm_data):
        if n == 0 or n == len(norm_data)-1:
            continue

        if i < norm_data[n+1] and i < norm_data[n-1] and not find_max and i < all_max[-1]-delta:
            all_min.append(i)
            pos_min.append(n)
            find_max = True

        if i > norm_data[n+1] and i > norm_data[n-1] and find_max and all_min[-1] < i-delta:
            all_max.append(i)
            pos_max.append(n)
            find_max = False

    all_max.pop(0)
    #return np.asarray(all_max),np.asarray(pos_max)
    return np.asarray(pos_max)

def force_1d(data):
    out = [0]*5*len(data)
    i = 0
    for d in data:
        for v in d:
            out[i] = v
            i += 1
            #Hack around to only do the first peak
            break

    #[[out.append(i) for i in x] for x in data]
    return filter(lambda x: x != 0, out)

d = overlap_data_stream(data)
d = overlap_data_stream(data, overlap_percentage=0)
#x = [x for x in autocorr(d)]
#plot.plot(x[50])
#plot.show()

v = [np.asarray(peak_search(x)) for x in autocorr(d)]
v = np.asarray(filter(lambda x: len(x) != 0, v))
plot.plot(force_1d(v))
plot.show()
