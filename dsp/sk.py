#!/usr/bin/python
#Needs the following libs
#sudo apt-get install python-numpy python-scipy python-matplotlib

import numpy as np
import argparse
import sys
import matplotlib.pyplot as plot
import scipy.signal as sg
from scipy.io import loadmat
from numpy.lib.stride_tricks import as_strided as ast

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

#out = np.fft.ifft(filtered_data_streams, n=decimate_by, axis=0)

try:
    args = parser.parse_args()
except SystemExit:
    parser.print_help()
    sys.exit()

if args.filename[-4:] == ".wav":
    sr, data = wavfile.read(args.filename)
    data = np.asarray(data, dtype=np.complex64)[::args.endpoints[2]]
elif args.filename[-4:] == ".mat":
    mat = loadmat(args.filename)
    data = mat[mat.keys()[0]].flatten()
if args.endpoints[1] == None:
    data = np.asarray(data, dtype=np.complex64)[0:4000]
else:
    data = data[args.endpoints[0]:args.endpoints[1]]

def overlap_data_stream(data, nfft=256, overlap_percentage=.75):
    nfft_count = len(data)/nfft
    overlap_samples = int(nfft*overlap_percentage)+1
    extended_length = (nfft_count+1)*(nfft-overlap_samples)
    data = np.hstack((np.asarray(data),np.asarray([0]*(extended_length-len(data)))))
    shape = (len(data)/(nfft-overlap_samples),nfft)
    strides = (data.itemsize*(nfft-overlap_samples), data.itemsize)
    return ast(data, shape=shape, strides=strides)

d = overlap_data_stream(data)
print d
print d.shape
plot.pcolor(np.fft.fft(d, n=256, axis=0))
plot.show()
