import wave
import os
import scipy as sp
import scipy.io.wavfile

import datetime

import numpy
from sklearn.externals import joblib
from sklearn.externals.joblib import Parallel

from fileio import save_data
from utils import create_chordDict

wav_dir = "/home/nikste/datasets/chord_recognition/wav/"
label_dir = "/home/nikste/datasets/chord_recognition/McGill-Billboard/"

def get_filenames(load_range):
    fns_wav = []
    for file in os.listdir(wav_dir):
        if file.endswith(".wav"):
            fns_wav.append(file)

    return [fns_wav[i] for i in load_range]

def get_wav_filenames(fns):
    fns_wav = []
    for i in range(0,len(fns)):
        fns_wav.append(wav_dir + fns[i])
    return fns_wav

def get_label_filenames(fns, extension):
    fns_label = []
    for i in range(0, len(fns)):
        num = fns[i].split('.')[0]
        fns_label.append(label_dir + num + "/" + extension)
    return fns_label


def load_wavs(fns_wav):
    input_data = []
    for i in range(0,len(fns_wav)):
        print "reading file:",fns_wav[i]
        sr,d = sp.io.wavfile.read(fns_wav[i])
        print "samplerate:",sr
        # for j in range(0,len(d)):
        #     print d[j]
        # wf = wave.open(file(fns_wav[i],"rb"))
        # wav_bytes = []
        # for i in range(wf.getnframes()):
        #     frame = wf.readframes(i)
        #     wav_bytes.append(frame)
        #     print frame
        input_data.append(d)
        print "read file",i+1, "of ", len(fns_wav)
    return input_data,sr


def load_label_file_contents(filename):
    f = open(filename,"r")
    d = create_chordDict()
    lines = f.readlines()

    last_element = len(lines)
    for i in range(0, last_element):
        lines[i] = lines[i][:-1].split("\t")
        if len(lines[i]) < 3:
            del lines[i]
            last_element -= 1
        else:
            lines[i][0] = float(lines[i][0])
            lines[i][1] = float(lines[i][1])
            lines[i][2] = d[lines[i][2]]
    return lines


def load_labels(fns_label):
    y = []
    for i in range(0,len(fns_label)):
        y.append(load_label_file_contents(fns_label[i]))
    return y

def convert_to_indices(Xs, sr, ys_text):
    ys = []
    stepsize_seconds =  1./float(sr)
    for i in range(0,len(Xs)):

        X = Xs[i]
        y_text = ys_text[i]
        y = []

        current_time = 0.
        current_index_y_text = 0
        next_frame_start_time = y_text[0][1]
        for j in range(0,len(X)):
            if(next_frame_start_time < current_time):
                # print "current_time:",current_time
                # print "next_frame_start_time",next_frame_start_time
                if(current_index_y_text < len(y_text) - 1):
                    current_index_y_text += 1
                    next_frame_start_time = y_text[current_index_y_text][1]
                else:
                    pass
                    # print "warning, trying to change:"
                    # print "current_time:", current_time
                    # print "next_frame_start_time", next_frame_start_time
            y.append(y_text[current_index_y_text][2])

            current_time += stepsize_seconds
        ys.append(y)
    return ys

def scale_data(X):
    # for dtype= int 16
    #scales to 0 to 1
    # -32768 to 32767
    X_out = []
    ra = 32768. + 32767.
    for i in range(0,len(X)):
        t_start = datetime.datetime.now()
        X_out_aux = numpy.zeros(len(X[i]))
        for j in range(0,len(X[i])):
            scaled = X[i][j]/ra + 0.5

            if scaled < 0.:
                print "warning:",scaled, "<", 0
                X_out_aux[j] = 0.
            if scaled > 1.:
                print "warning",scaled, ">", 1
                X_out_aux[j] = 1.
            X_out_aux[j] = scaled
        X_out.append(X_out_aux)
        print "scaled", i, "of", len(X), "took", datetime.datetime.now() - t_start
    return X_out


def scale_list(X):
    print "scaling new list",len(X)
    ra = 32768. + 32767.
    X_out = []
    for i in range(0, len(X)):
        scaled = X[i] / ra + 0.5
        if scaled < 0.:
            print "warning:", scaled, "<", 0
            # X[i] = 0.
            X_out.append(0.)
        elif scaled > 1.:
            print "warning", scaled, ">", 1
            # X[i] = 1.
            X_out.append(1.)
        else:
            # X[i] = scaled
            X_out.append(scaled)
    print "done",len(X)
    return X_out

def scale_element(x):
    ra = 32768. + 32767.
    scaled = x / ra + 0.5
    if scaled < 0.:
        print "warning:", scaled, "<", 0
        # X[i] = 0.
        return 0.
    elif scaled > 1.:
        print "warning", scaled, ">", 1
        # X[i] = 1.
        return 1.
    else:
        # X[i] = scaled
        return scaled

def scale_data_parallel_element(X):
    X_scaled = []
    for i in range(0,len(X)):
        print "scaling ",i, " of ", len(X)
        X_scaled.append(joblib.Parallel(n_jobs=1000)(joblib.delayed(scale_element) (x) for x in X[i]))
    return X_scaled

def scale_data_parallel_list(X):
    X_scaled = joblib.Parallel(n_jobs=200)(joblib.delayed(scale_list) (x) for x in X)
    return X_scaled

def load_data_wav(load_range = range(0, 740)):
    fns = get_filenames(load_range)
    fns_label = get_label_filenames(fns,'majmin.lab')
    fns_wav = get_wav_filenames(fns)
    X,sr = load_wavs(fns_wav)
    y = load_labels(fns_label)
    y = convert_to_indices(X,sr,y)
    return X,y


def process_data((f,t)):
    print "loading ", f, "to", t
    files = range(f,t)#20)

    X_org,y = load_data_wav(load_range=files)
    print "start scaling"

    t_start = datetime.datetime.now()
    X = scale_data(X_org)
    print "unthreaded scaling:",datetime.datetime.now() - t_start

    save_data(X,y,"" + str(f) + "_" + str(t))
    # print_unique_labels(y)



def load_and_preprocess_wav_to_pickle():

    from_tos = [[0, 10], [10, 20], [20, 30], [30, 40], [40, 50], [50, 60], [60, 70], [70, 80], [80, 90], [90, 100],
                [100, 110], [110, 120], [120, 130], [130, 140], [140, 150], [150, 160], [160, 170], [170, 180],
                [180, 190], [190, 200],
                [200, 210], [210, 220], [220, 230], [230, 240], [240, 250], [250, 260], [260, 270], [270, 280],
                [280, 290], [290, 300],
                [300, 310], [310, 320], [320,330],[330,340], [340, 350], [350, 360], [360, 370], [370, 380],
                [380, 390], [390, 400],
                [400, 410], [410, 420], [420, 430], [430, 440], [440, 450], [450, 460], [460, 470], [470, 480],
                [480, 390], [490, 500],
                [500, 510], [510, 520], [520, 530], [530, 540], [540, 550], [550, 560], [560, 570], [570, 580],
                [580, 390], [590, 600],
                [600, 610], [610, 620], [620, 630], [630, 640], [640, 650], [650, 660], [660, 670], [670, 680],
                [680, 390], [690, 700],
                [700, 710], [710, 720], [720, 730], [730, 740]]


    Parallel(n_jobs=7)(delayed(process_data)(el) for el in from_tos)