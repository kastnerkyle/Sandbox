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
import filterbank
import scipy.signal as sg

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
    import wave, struct
    waveFile = wave.open(args.filename, 'r')
    length = waveFile.getnframes()
    data = np.zeros((length,))
    for i in range(0,length):
        try:
            waveData = waveFile.readframes(1)
            d = struct.unpack("<h", waveData)
            data[i] = int(d[0])
        except struct.error:
            data[i] = data[i-1]
    waveFile.close()

    #sr, data = wavfile.read(args.filename)
    #data = np.asarray(data, dtype=np.complex64)[::args.endpoints[2]]

elif args.filename[-4:] == ".asc":
    all_sensor_data = pd.read_csv(args.filename, sep="\t", skiprows=2)
    #data = all_sensor_data['Mic[Pa]']
    #data = all_sensor_data['accel_Y[g]']
    data = all_sensor_data['P_waterjacket[bar]']

def overlap_data_stream(data, chunk=256, overlap_percentage=.75):
    chunk_count = len(data)/chunk
    overlap_samples = int(chunk*overlap_percentage)+1
    extended_length = (chunk_count+1)*(chunk-overlap_samples)
    data = np.hstack((np.asarray(data),np.asarray([0]*(extended_length-len(data)))))
    shape = (len(data)/(chunk-overlap_samples),chunk)
    strides = (data.itemsize*(chunk-overlap_samples), data.itemsize)
    return ast(data, shape=shape, strides=strides)

def get_adjusted_lims(dframe, num_bins=100, lower_bound=.1, upper_bound=.9):
    #rmin = dframe.min().min()
    #rmax = dframe.max().max()
    #plot.figure()
    #hist,bins,_ = plot.hist(dframe.values.flatten(), num_bins, range=(rmin,rmax))
    dframe_vals = dframe.values.flatten()
    dframe_vals = dframe_vals[np.isfinite(dframe_vals)]
    dframe_vals = np.clip(dframe_vals, -2*np.std(dframe_vals), 2*np.std(dframe_vals))
    hist,bins = np.histogram(dframe_vals, num_bins)
    area = np.asarray(np.cumsum(hist),dtype=np.double)
    area /= np.max(area)
    hist_group = zip(area,bins)
    lower_bin = filter(lambda x: x[0] > lower_bound, hist_group)[0][1]
    upper_bin = filter(lambda x: x[0] > upper_bound, hist_group)[0][1]
    return lower_bin, upper_bin

FFT_SIZE=256
f, axarr = plot.subplots(2)
decimate_by = 4
data = filterbank.polyphase_single_filter(data, decimate_by, sg.firwin(200, 1./(decimate_by+1)))
overlapped = overlap_data_stream(data, chunk=FFT_SIZE, overlap_percentage=.5).T
windowed_overlapped = np.apply_along_axis(lambda x: np.hanning(len(x))*x,0,overlapped)
raw_spectrogram = np.fft.fftshift(np.fft.fft(windowed_overlapped, n=FFT_SIZE, axis=0), axes=0)
window_length = 25/decimate_by
#axarr[0].specgram(data,
#        cmap=cm.gray,
#        sides='onesided')
spec_dframe = pd.DataFrame(np.abs(raw_spectrogram[:raw_spectrogram.shape[0]/2,:]))
axarr[0].imshow(np.log(spec_dframe.values),
        cmap=cm.gray,
        aspect='normal')
#[pxx, freqs, bins, specax] = plot.specgram(data)
#spec_dframe = pd.DataFrame(np.abs(pxx[::-1]))#raw_spectrogram))
#spec_dframe[0] is the same as np.abs(raw_spectrogram[:,0]), which means each row represents an FFT for a certain period of time
rolling_kurtosis = pd.rolling_kurt(spec_dframe, window_length, axis=1).fillna()

#rolling_skewness = pd.rolling_skew(spec_dframe, window_length, axis=1).fillna()
#lower,upper = get_adjusted_lims(rolling_skewness, num_bins=10000)
#skewax = axarr[1].imshow(rolling_skewness,
#        vmin=lower,
#        vmax=upper,
#        cmap=cm.gray,
#        aspect='normal')

lower,upper = get_adjusted_lims(rolling_kurtosis, num_bins=10000)
kurtax = axarr[1].imshow(rolling_kurtosis.values,
        vmin=lower,
        vmax=upper,
        cmap=cm.gray,
        aspect='normal')
plot.show()
