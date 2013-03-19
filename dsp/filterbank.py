#!/usr/bin/python
#Needs the following libs
#sudo apt-get install python-numpy python-scipy python-matplotlib

import numpy as np
import argparse
import sys
import matplotlib.pyplot as plot
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import scipy.signal as sg
import copy

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

DECIMATE_BY = 3
FILT_CONST = 50
def gen_complex_chirp(fs=44100):
    f0=-fs/2.1
    f1=fs/2.1
    t1 = 1
    beta = (f1-f0)/float(t1)
    t = np.arange(0,t1,t1/float(fs))
    return np.exp(2j*np.pi*(.5*beta*(t**2) + f0*t))

def show_filter_response(filt, axarr, title=None):
    w,h = sg.freqz(filt)
    axarr.plot(w/max(w), np.abs(h))
    if title != None:
        axarr.set_title(title)
    axarr.set_xlabel("Normalized frequency")
    axarr.set_ylabel("Gain")

def show_specgram(input_data, fft_size=512, one_sided=False, title=None):
    split = "onesided" if one_sided else "twosided"
    plot.specgram(input_data, fft_size, sides=split)
    if title != None:
        plot.title(title)
    plot.show()

def basic_single_filter(input_data):
    filt = prototype_filter()
    filtered_data = sg.lfilter(filt, 1, input_data)
    return filtered_data, filt

def polyphase_single_filter(input_data, decimate_by, filt):
    if len(input_data) % decimate_by != 0:
        input_data = input_data[:len(input_data)-len(input_data)%decimate_by]
    head_data_stream = np.asarray(input_data[::decimate_by])
    tail_data_streams = np.asarray([np.asarray(input_data[0+i::decimate_by]) for i in range(1,decimate_by)])
    #Sorting the datastreams is CRUCIAL! Rows go in the following order
    #1
    #...
    #3
    #2
    data_streams = np.vstack((head_data_stream, tail_data_streams[::-1,:]))
    #Adding these zeros is important?
    filter_streams = np.asarray([np.asarray(filt[0+i::decimate_by]) for i in range(decimate_by)])
    filtered_data_streams = np.asarray([sg.lfilter(filter_streams[n,:], 1, data_streams[n,:]) for n in range(decimate_by)])
    filtered_data_streams = np.asarray([np.append(filtered_data_streams[i],0) if i==0 else np.insert(filtered_data_streams[i], 0, 0) for i in range(decimate_by)])
    filtered_data = np.sum(filtered_data_streams, axis=0)
    return filtered_data

def polyphase_analysis(input_data, decimate_by, filt):
    if len(input_data) % decimate_by != 0:
        input_data = input_data[:len(input_data)-len(input_data)%decimate_by]
    #decimate prototype filter
    head_data_stream = np.asarray(input_data[::decimate_by])
    tail_data_streams = np.asarray([np.asarray(input_data[0+i::decimate_by]) for i in range(1,decimate_by)])
    data_streams = np.vstack((head_data_stream, tail_data_streams[::-1,:]))
    filter_streams = np.asarray([np.asarray(filt[0+i::decimate_by]) for i in range(decimate_by)])
    filtered_data_streams = np.asarray([sg.lfilter(filter_streams[n,:], 1, data_streams[n,:]) for n in range(decimate_by)])
    filtered_data_streams = np.asarray([np.append(filtered_data_streams[i],0) if i==0 else np.insert(filtered_data_streams[i], 0, 0) for i in range(decimate_by)])
    out = np.fft.ifft(filtered_data_streams, n=decimate_by, axis=0)
    return out

if __name__=="__main__":
    try:
        args = parser.parse_args()
    except SystemExit:
        parser.print_help()
        sys.exit()

    if args.filename[-4:] == ".wav":
        print "WARNING: Plot values not guaranteed correct for .wav file input!"
        sr, data = wavfile.read(args.filename)
        data = np.asarray(data, dtype=np.complex64)[::args.endpoints[2]]
        if args.endpoints[1] == None:
            pass
        else:
            data = data[args.endpoints[0]:args.endpoints[1]]
        #data /= 32768
    else:
        #Generate chirp and add noise - fully synthetic chirp has strange looking plots
        data = gen_complex_chirp()
        data += .01*np.random.randn(len(data))

    def prototype_filter(num_taps=DECIMATE_BY*FILT_CONST, normalized_cutoff=1./(DECIMATE_BY+.1*DECIMATE_BY)):
        return sg.firwin(num_taps, normalized_cutoff)

    NFFT=512
    SIDES="twosided"
    ASPECT="normal"
    CMAP=cm.gray
    ORIGIN="lower"
    INTERPOLATION="bicubic"
    NOVERLAP=1
    XAXIS="Time (seconds)"
    YAXIS="Normalized Frequency"
    NXTICKS = 5
    NYTICKS = 5
    FS = 44100

    def format_axes(ax, freq_zoom=1, freq_bank=None):
        ax.set_xlabel(XAXIS)
        xmin, xmax = ax.get_xlim()
        xlabels = [x for x in np.linspace(0,1,NXTICKS)]
        ax.set_xlim(0, xmax)
        ax.xaxis.set_major_locator(LinearLocator(NXTICKS))
        ax.set_xticklabels(xlabels)

        ax.set_ylabel(YAXIS)
        ymin, ymax = ax.get_ylim()
        #-.49999 to keep it from displaying as -0.00
        #All other if statement values are to compensate for filter bank ordering
        ylabels = [float(y)/freq_zoom + (0 if freq_bank == None else freq_bank*(float(1.)/freq_zoom) - (1. if freq_bank > DECIMATE_BY/2 else 0))
                   for y in np.linspace(-.5,.5,NYTICKS)]
        ax.set_ylim(0, ymax)
        ylabels = ["%.2f" % y for y in ylabels]
        ax.yaxis.set_major_locator(LinearLocator(NYTICKS))
        ax.set_yticklabels(ylabels)

    f1, axarr1 = plot.subplots(5)
    plot.tight_layout()
    pxx, freqs, bins, im = axarr1[0].specgram(data, NFFT, noverlap=NOVERLAP)
    #This specgram, imshow runaround seems to be necessary to eliminate blank space at the end of regular matplotlib specgram calls?
    #If anyone knows a better fix, let me know
    axarr1[0].imshow(np.ma.log(abs(pxx)), aspect=ASPECT, cmap=CMAP, origin=ORIGIN, interpolation=INTERPOLATION)
    axarr1[0].set_title("Specgram of original data")
    format_axes(axarr1[0])

    basic, filt = basic_single_filter(data)
    show_filter_response(filt, axarr1[1], title="Lowpass filter response")

    pxx, freqs, bins, im = axarr1[2].specgram(basic, NFFT, noverlap=NOVERLAP)
    axarr1[2].imshow(np.ma.log(abs(pxx)), aspect=ASPECT, cmap=CMAP, origin=ORIGIN, interpolation=INTERPOLATION)
    axarr1[2].set_title("Filtered")
    format_axes(axarr1[2])

    decimated = basic[::DECIMATE_BY]
    pxx, freqs, bins, im = axarr1[3].specgram(decimated, NFFT, noverlap=NOVERLAP)
    axarr1[3].imshow(np.ma.log(abs(pxx)), aspect=ASPECT, cmap=CMAP, origin=ORIGIN, interpolation=INTERPOLATION)
    axarr1[3].set_title("Filtered, then decimated")
    format_axes(axarr1[3], freq_zoom=DECIMATE_BY)

    decimated_filtered = polyphase_single_filter(data, DECIMATE_BY, prototype_filter())
    pxx, freqs, bins, im = axarr1[4].specgram(decimated_filtered, NFFT, noverlap=NOVERLAP)
    axarr1[4].imshow(np.ma.log(abs(pxx)), aspect=ASPECT, cmap=CMAP, origin=ORIGIN, interpolation=INTERPOLATION)
    axarr1[4].set_title("Polyphase filtered data")
    format_axes(axarr1[4], freq_zoom=DECIMATE_BY)

    f2, axarr2 = plot.subplots(DECIMATE_BY)
    plot.tight_layout()
    decimated_filterbank = polyphase_analysis(data, DECIMATE_BY, prototype_filter())
    for i in range(decimated_filterbank.shape[0]):
        pxx, freqs, bins, im = axarr2[i].specgram(decimated_filterbank[i], NFFT)
        axarr2[i].imshow(np.ma.log(abs(pxx)), aspect=ASPECT, cmap=CMAP, origin=ORIGIN, interpolation=INTERPOLATION)
        axarr2[i].set_title("Filterbank output " + `i`)
        format_axes(axarr2[i], freq_zoom=DECIMATE_BY, freq_bank=i)
    plot.show()
