import datetime

from sklearn.externals import joblib
from sklearn.externals.joblib import Parallel,delayed

from preprocessing import load_data_wav, scale_data, scale_data_parallel_element, scale_data_parallel_list
from inputGenerator import DataGenerator
from fileio import save_data, load_data
from referenceClassifyier import ReferenceClassifyier
from utils import print_unique_labels, deterimine_max_and_min
from training import train_cnn, count_chords, count_chords_cathegorical
import numpy as np

X,y = load_data(chunks=range(0,1))

X_test,y_test = load_data(chunks=range(20,21))

# joblib.dump(([X[0]],[y[0]]),"one_song_for_testing")
x, y = joblib.load("one_song_for_testing")

# x = np.array(x)
# y = np.array(y)
x_test = x
y_test = y

count_chords_cathegorical(y)
count_chords_cathegorical(y_test)

# from_tos = [[0,100],[100,200],[200,300],[300,400],[400,500],[500,600],[600,700],[700,740]]
#
# for from_to in from_tos:

dg = DataGenerator(x, y, x_test, y_test)
rc = ReferenceClassifyier(y)
# print y.shape
train_cnn(dg, rc=rc)


