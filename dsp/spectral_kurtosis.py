#!/usr/bin/python
#Needs the following libs
#sudo apt-get install python-numpy python-scipy python-matplotlib

import argparse
import sys
import matplotlib.pyplot as plot
import matplotlib.colors as colors
from matplotlib import cm
from scipy.io import loadmat
from scipy.io import wavfile
import pandas as pd
import numpy as np
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

try:
    args = parser.parse_args()
except SystemExit:
    parser.print_help()
    sys.exit()

if args.filename[-4:] == ".wav":
    sr, data = wavfile.read(args.filename)
    data = np.asarray(data, dtype=np.complex64)[::args.endpoints[2]]
elif args.filename[-4:] == ".asc":
    all_sensor_data = pd.read_csv(args.filename, sep="\t", skiprows=2)
    data = all_sensor_data['Mic[Pa]']
    #data = all_sensor_data['accel_Y[g]']
    #data = all_sensor_data['P_waterjacket[bar]']

def overlap_data_stream(data, chunk=256, overlap_percentage=.75):
    chunk_count = len(data)/chunk
    overlap_samples = int(chunk*overlap_percentage)+1
    extended_length = (chunk_count+1)*(chunk-overlap_samples)
    data = np.hstack((np.asarray(data),np.asarray([0]*(extended_length-len(data)))))
    shape = (len(data)/(chunk-overlap_samples),chunk)
    strides = (data.itemsize*(chunk-overlap_samples), data.itemsize)
    return ast(data, shape=shape, strides=strides)

FFT_SIZE=256
TRANSIENT_TIME_S=.075
SAMPLES_PER_S=44100
overlapped = overlap_data_stream(data, chunk=FFT_SIZE, overlap_percentage=0).T
windowed_overlapped = np.apply_along_axis(lambda x:np.hanning(FFT_SIZE)*x,0,overlapped)
raw_spectrogram = np.fft.fftshift(np.fft.fft(windowed_overlapped, n=FFT_SIZE, axis=0), axes=0)
log_abs_spectrogram = np.log(np.abs(raw_spectrogram))
window_length = int(float((TRANSIENT_TIME_S)*SAMPLES_PER_S)/FFT_SIZE)
if window_length < 3:
    print "Unable to process, increase TRANSIENT_TIME_S to a larger value!"
    print "Setting to minimal transient time"
    window_length = 25
spec_dframe = pd.DataFrame(np.abs(raw_spectrogram))
#spec_dframe[0] is the same as np.abs(raw_spectrogram[:,0]), which means each row represents an FFT for a certain period of time
rolling_kurtosis = pd.rolling_kurt(spec_dframe, window_length, axis=1).fillna()
f, axarr = plot.subplots(3)
#colors.normalize takes in vmin and vmax to try to set the colormap - this should do the same as
#the default argument to norm but I wanted to document this for changing later
axarr[0].set_xlim([0,FFT_SIZE*rolling_kurtosis.shape[1]])
axarr[0].set_autoscalex_on(False)
axarr[0].plot(data)
axarr[1].imshow(log_abs_spectrogram,
        norm=colors.normalize(vmin=np.amin(log_abs_spectrogram),
                vmax=np.amax(log_abs_spectrogram),
                clip=False),
        cmap=cm.jet,
        aspect='normal')
axarr[2].imshow(rolling_kurtosis,
        norm=colors.normalize(vmin=rolling_kurtosis.min().min(),
            vmax=rolling_kurtosis.max().max(),
            clip=False),
        cmap=cm.jet,
        aspect='normal')
plot.show()
