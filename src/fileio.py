import datetime
from sklearn.externals import joblib
import pickle

from sklearn.externals.joblib import Parallel,delayed

home_dir = "/home/nikste/datasets/chord_recognition/python_data2/"

def save_data(X,y,fname="data"):
    joblib.dump((X,y),home_dir + fname,protocol=pickle.HIGHEST_PROTOCOL,compress=9)




def load_data(chunks=range(0,74)):
    names = ["0_10", "10_20", "20_30", "30_40", "40_50", "50_60", "60_70", "70_80", "80_90", "90_100",
             "100_110", "110_120", "120_130", "130_140", "140_150", "150_160", "160_170", "170_180","180_190", "190_200",
             "200_210", "210_220", "220_230", "230_240", "240_250", "250_260", "260_270", "270_280","280_290", "290_300",
             "300_310", "310_320", "320,330","330,340", "340_350", "350_360", "360_370", "370_380","380_390", "390_400",
             "400_410", "410_420", "420_430", "430_440", "440_450", "450_460", "460_470", "470_480","480_390", "490_500",
             "500_510", "510_520", "520_530", "530_540", "540_550", "550_560", "560_570", "570_580","580_390", "590_600",
             "600_610", "610_620", "620_630", "630_640", "640_650", "650_660", "660_670", "670_680","680_390", "690_700",
             "700_710", "710_720", "720_730", "730_740"]
    fns_to_load = [names[i] for i in chunks]



    X = []
    y = []

    # t_start = datetime.datetime.now()
    # data = Parallel(n_jobs=8)(delayed(joblib.load)(home_dir + el) for el in fns_to_load)
    #
    #
    # for d in data:
    #     X.extend(d[0])
    #     y.extend(d[1])
    #
    # t_end = datetime.datetime.now()
    # print "loading took", t_end - t_start
    t_start = datetime.datetime.now()
    for fn in fns_to_load:
        print "loading",fn
        X_,y_ = joblib.load(home_dir + fn)
        X.extend(X_)
        y.extend(y_)
        t_end = datetime.datetime.now()
    print "took:",t_end - t_start

    return X, y