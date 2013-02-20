#!/usr/bin/python
#Needs the following libs
#sudo apt-get install python-numpy python-scipy python-matplotlib
#Also using filterbank.py from www.github.com/kastnerkyle/dsp
#Automate this file with
#for i in `ls -d`; do ./spectral_kurtosis.py -f $i -s; done
#For example
#for i in `ls -d ~/engine_data/*/*`; do for v in 128 256; do echo "Running $i"; ./spectral_kurtosis.py -f $i -w -n $v -s; done; done

import argparse
import sys
import matplotlib.pyplot as plot
import matplotlib.colors as colors
from matplotlib import cm
from scipy.io import loadmat
from scipy.io import wavfile
import pandas as pd
import numpy as np
import scipy as sp
from numpy.lib.stride_tricks import as_strided as ast
import filterbank
import scipy.signal as sg
import copy
import time

def overlap_data_stream(data, chunk=256, overlap_percentage=.75):
    chunk_count = len(data)/chunk
    overlap_samples = int(chunk*overlap_percentage)
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
    if type(dframe) == pd.core.frame.DataFrame:
        dframe_vals = dframe.values.flatten()
        dframe_vals = dframe_vals[np.isfinite(dframe_vals)]
    else:
        dframe_vals = dframe[np.isfinite(dframe)]
    dframe_vals = np.clip(dframe_vals, -3*np.ma.std(dframe_vals), 3*np.ma.std(dframe_vals))
    hist,bins = np.histogram(dframe_vals, num_bins)
    area = np.asarray(np.cumsum(hist),dtype=np.double)
    area /= np.max(area)
    hist_group = zip(area,bins)
    lower_bin = filter(lambda x: x[0] > lower_bound, hist_group)[0][1]
    upper_bin = filter(lambda x: x[0] > upper_bound, hist_group)[0][1]
    return lower_bin, upper_bin

def run_kurtosis(data, nfft, decimate_by, overlap_fraction, info="", whiten=False, save_plot=False, twosided=False):
    if whiten:
        #Apply an lpc filter to perform "pre-whitening"
        #See "The Application of Spectral Kurtosis to Bearing Diagnostics", N. Sawalhi and R. Randall, ACOUSTICS 2004
        coeffs = 100
        data = data - np.mean(data)
        acorr_data = np.correlate(data, data, 'full')
        r = acorr_data[data.size-1:data.size+coeffs]
        phi = np.dot(sp.linalg.inv(sp.linalg.toeplitz(r[:-1])), -r[1:])
        lpfilt = np.concatenate(([1.], phi))
        data = sg.lfilter(lpfilt, 1, data)

        #Remove filter transient
        data = data[coeffs+1:]
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
    if twosided:
        spec_dframe = pd.DataFrame(np.abs(raw_spectrogram))
    else:
        spec_dframe = pd.DataFrame(np.abs(raw_spectrogram[:raw_spectrogram.shape[0]/2,:]))
    fulltitle = "Spectrogram and spectral kurtosis" + (", prewhitened" if whiten else "") + "\n" + info + " $F_s=$" + `44100/decimate_by` + ", $O=$" + `overlap_fraction` + ", $NFFT=$" + `nfft if twosided else nfft/2` + ",  $NWND=$" + `base_window_length`
    f.suptitle(fulltitle)
    #axarr[0].specgram(data,
    #        NFFT=nfft,
    #        noverlap=int(overlap_fraction*nfft),
    #        cmap=cm.gray,
    #        origin='lower',
    #        interpolation='bicubic',
    #        sides='onesided',
    #        aspect='normal')
    log_spec = copy.copy(spec_dframe.values.flatten())
    log_spec = np.ma.log(log_spec)
    log_spec = np.reshape(log_spec, spec_dframe.values.shape)
    lower, upper = get_adjusted_lims(log_spec, num_bins=10000)
    specax = axarr[0].imshow(log_spec,
            cmap=cm.gray,
            vmin=lower,
            vmax=upper,
    #        cmap=cm.spectral,
    #        cmap=cm.gist_stern,
            interpolation='bicubic',
            origin='lower',
            aspect='normal')
    xaxislabel="Time (Overlapped Samples)"
    yaxislabel="Frequency (FFT Bins)"
    axarr[0].set_xlabel(xaxislabel)
    axarr[0].set_ylabel(yaxislabel)
    rolling_kurtosis = pd.rolling_kurt(spec_dframe, window_length, axis=1).fillna()
    lower,upper = get_adjusted_lims(rolling_kurtosis, num_bins=10000)
    #Remove 0:nfft*overlap_fraction column values to adjust for plotting offest
    kurtax = axarr[1].imshow(rolling_kurtosis.values[:, int(nfft*overlap_fraction):],
            vmin=lower,
            vmax=upper,
            cmap=cm.gray,
            #cmap=cm.spectral,
            #cmap=cm.gist_stern,
            interpolation='bicubic',
            origin='lower',
            aspect='normal')
    axarr[1].set_xlabel(xaxislabel)
    axarr[1].set_ylabel(yaxislabel)
    speccblabel = "Amplitude (dB)"
    kurtcblabel = "Unbiased Kurtosis"
    f.subplots_adjust(right=0.8)
    speccbax = f.add_axes([.85,.53,.025,.35])
    kurtcbax = f.add_axes([.85,.1,.025,.35])
    speccb = f.colorbar(specax, cax=speccbax)
    speccb.set_label(speccblabel)
    kurtcb = f.colorbar(kurtax, cax=kurtcbax)
    kurtcb.set_label(kurtcblabel)

    if save_plot:
        plot.savefig("".join(fulltitle.split(" ")) + ".png")
        plot.close()
    else:
        plot.show()

class EndpointsAction(argparse.Action):
    def __call__(self, parser, args, values, option = None):
        setattr(args, self.dest, map(int,values))
        if len(args.endpoints) < 3:
            defaults = [0,None, 1]
            print "Wrong number of arguments, require 3 values, --endpoints start stop step"
            print "Using default endpoints of " + `args.endpoints`
            setattr(args, self.dest, defaults)

parser = argparse.ArgumentParser(description="Apply filter tutorial to input data")
parser.add_argument("-f", "--filename", default=".nofile", dest="filename", help="File to be processed (.wav or .asc)")
parser_nfft_default = 128
parser_decimation_default = 1
parser.add_argument("-n", "--nfft", dest="nfft", default=parser_nfft_default, type=int, help="Number of FFT points, default is " + `parser_nfft_default`)
parser.add_argument("-t", "--twosided", dest="twosided", action="store_true", help="Flag to enable two-sided graphs, default is one-sided")
parser.add_argument("-d", "--decimate", dest="decimate", default=parser_decimation_default, type=int, help="Value to decimate by, default is " + `parser_nfft_default`)
parser.add_argument("-w", "--whiten", dest="whiten", action="store_true", help="Flag to enable additive whitening which can help with visualization")
parser.add_argument("-s", "--save", dest="save", action="store_true", help="Flag to save data to .png instead of a plot view")
parser.add_argument("-e", "--endpoints", dest="endpoints", default=[0,None, 1], action=EndpointsAction, nargs="*", help='Start and stop endpoints for data, default will try to process the whole file')

try:
    args = parser.parse_args()
except SystemExit:
    parser.print_help()
    sys.exit()

nfft=args.nfft
decimate_by = args.decimate
overlap_fraction = .66
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
    run_kurtosis(data, nfft, decimate_by, overlap_fraction,
            info=args.filename.split("/")[-1].split(".")[0],
            save_plot=args.save,
            twosided=args.twosided)
    #sr, data = wavfile.read(args.filename)
    #data = np.asarray(data, dtype=np.complex64)[::args.endpoints[2]]

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
    print "Tag number is " + `tag`
    print "Title is calculated as " + `rpms[tag/4]` + `loads[tag%4]`
    for i in all_sensor_data.columns[2:]:
        run_kurtosis(all_sensor_data[i], nfft, decimate_by, overlap_fraction,
                info=i+rpms[tag/4]+loads[tag%4],
                save_plot=args.save,
                twosided=args.twosided)

    #data = all_sensor_data['Mic[Pa]']
    #data = all_sensor_data['accel_X[g]']
    #data = all_sensor_data['accel_Y[g]']
    #data = all_sensor_data['accel_Z[g]']
    #data = all_sensor_data['P_waterjacket[bar]']

elif args.filename == ".nofile":
    data = filterbank.gen_complex_chirp()
    data += np.random.randn(len(data))
    run_kurtosis(data, nfft, decimate_by, overlap_fraction,
                 info="Generated chirp",
                 whiten=args.whiten,
                 twosided=args.twosided,
                 save_plot=args.save)
