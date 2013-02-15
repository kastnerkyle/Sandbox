#!/usr/bin/python
#Needs the following libs
#sudo apt-get install python-numpy python-scipy python-matplotlib
#Also using filterbank.py from www.github.com/kastnerkyle/dsp
#Automate this file with
#for i in `ls -d`; do ./spectral_kurtosis.py -f $i; done
#For example
#for i in `ls -d ~/engine_data/*/*`;do echo "Running $i"; ./spectral_kurtosis.py -f $i; done

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
    dframe_vals = np.clip(dframe_vals, -3*np.std(dframe_vals), 3*np.std(dframe_vals))
    hist,bins = np.histogram(dframe_vals, num_bins)
    area = np.asarray(np.cumsum(hist),dtype=np.double)
    area /= np.max(area)
    hist_group = zip(area,bins)
    lower_bin = filter(lambda x: x[0] > lower_bound, hist_group)[0][1]
    upper_bin = filter(lambda x: x[0] > upper_bound, hist_group)[0][1]
    return lower_bin, upper_bin

def run_kurtosis(data, nfft, decimate_by, overlap_fraction, info=""):
    #Heuristic window to get nice plots
    base_window_length = int(overlap_fraction*nfft)
    f, axarr = plot.subplots(2)
    if decimate_by > 1:
        data = filterbank.polyphase_single_filter(data, decimate_by, sg.firwin(200, 1./(decimate_by+.25)))
        window_length = base_window_length/decimate_by
    else:
        window_length = base_window_length
    overlapped = overlap_data_stream(data, chunk=nfft, overlap_percentage=overlap_fraction).T
    windowed_overlapped = np.apply_along_axis(lambda x: np.hanning(len(x))*x,0,overlapped)
    raw_spectrogram = np.fft.fftshift(np.fft.fft(windowed_overlapped, n=nfft, axis=0), axes=0)
    spec_dframe = pd.DataFrame(np.abs(raw_spectrogram[:raw_spectrogram.shape[0]/2,:]))
    #spec_dframe = pd.DataFrame(np.abs(raw_spectrogram))
    fulltitle = "Spectrogram and spectral kurtosis\n " + info + " $F_s=$" + `44100/decimate_by` + ", $O=$" + `overlap_fraction` + ", $NFFT=$" + `nfft/2` + ",  $NWND=$" + `base_window_length`
    f.suptitle(fulltitle)
    axarr[0].imshow(np.log(spec_dframe.values),
            cmap=cm.gray,
            #cmap=cm.spectral,
            #cmap=cm.gist_stern,
            interpolation='bicubic',
            origin='lower',
            aspect='normal')
    rolling_kurtosis = pd.rolling_kurt(spec_dframe, window_length, axis=1).fillna()
    lower,upper = get_adjusted_lims(rolling_kurtosis, num_bins=10000)
    kurtax = axarr[1].imshow(rolling_kurtosis.values,
            vmin=lower,
            vmax=upper,
            cmap=cm.gray,
            #cmap=cm.spectral,
            #cmap=cm.gist_stern,
            interpolation='bicubic',
            origin='lower',
            aspect='normal')
    plot.savefig("".join(fulltitle.split(" ")) + ".png")
    #plot.show()

class EndpointsAction(argparse.Action):
    def __call__(self, parser, args, values, option = None):
        setattr(args, self.dest, map(int,values))
        if len(args.endpoints) < 3:
            defaults = [0,None, 1]
            print "Wrong number of arguments, require 3 values, --endpoints start stop step"
            print "Using default endpoints of " + `args.endpoints`
            setattr(args, self.dest, defaults)

parser = argparse.ArgumentParser(description="Apply filter tutorial to input data")
parser.add_argument("-f", "--filename", dest="filename", default=".noexist", help="File to be processed (.wav or .asc)")
parser.add_argument("-n", "--nfft", dest="nfft", type=int, default=128, help="Number of FFT points to use for STFT processing")
parser.add_argument("-e", "--endpoints", dest="endpoints", default=[0,None, 1], action=EndpointsAction, nargs="*", help='Start and stop endpoints for data, default will try to process the whole file')
parser.add_argument("-v", "--verbose", dest="verbose", action="count", help='Verbosity, -v for verbose or -vv for very verbose')

try:
    args = parser.parse_args()
except SystemExit:
    parser.print_help()
    sys.exit()

nfft=args.nfft
decimate_by = 1
overlap_fraction = .85

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
    run_kurtosis(data, nfft, decimate_by, overlap_fraction, info=args.filename.split("/")[-1].split(".")[0])

elif args.filename[-4:] == ".asc":
    all_sensor_data = pd.read_csv(args.filename, sep="\t", skiprows=2)
    #Get information tag from string (ex Jan16_0003.asc -> 3)
    tag = int(args.filename.split("/")[-1].split("_")[-1].split(".")[0])

    #Correct for 1 based indexing on file tags
    tag -= 1

    #Map tag to rpm and load values"
    rpms = {0:" 1500 rpm ",
            1:" 2000 rpm ",
            2:" 2500 rpm ",
            3:" 3000 rpm "}
    loads = {0:" 25% load ",
            1:" 50% load ",
            2:" 75% load ",
            3:" 100% load "}

    #Skip time,rpm,and rev data
    for i in all_sensor_data.columns[2:]:
        run_kurtosis(all_sensor_data[i], nfft, decimate_by, overlap_fraction, info=i+rpms[tag/4]+loads[(tag)%4])

    #data = all_sensor_data['Mic[Pa]']
    #data = all_sensor_data['accel_X[g]']
    #data = all_sensor_data['accel_Y[g]']
    #data = all_sensor_data['accel_Z[g]']
    #data = all_sensor_data['P_waterjacket[bar]']

