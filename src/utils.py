
"""
C
C#
D
D#
E
F
F#
G
G#
A
A#
H
"""


base_notes={
    "C":1,"B#":1,
    "C#":2,"Db":2,
    "D":3,
    "D#":4,"Eb":4,
    "E":5,"Fb":5,
    "F":6,"E#":6,
    "F#":7,"Gb":7,
    "G":8,
    "G#":9,"Ab":9,
    "A":10,
    "A#":11,"Bb":11,
    "B":12,"Cb":12
}



def create_chordDict(typ = "majmin"):
    chord_dict = {}
    if typ == "majmin":
        keys = base_notes.keys()
        for i in range(0,len(keys)):
            chord_dict[keys[i] + ":maj"] = base_notes[keys[i]]
        for i in range(0,len(keys)):
            chord_dict[keys[i] + ":min"] = base_notes[keys[i]] + 12
        # non-chord symbol:
        chord_dict["N"] = 0
        chord_dict["X"] = 0
    return chord_dict

def print_unique_labels(y):
    complete = []
    for i in range(0,len(y)):
        complete.extend(y[i])
    complete_set = set(complete)

    print complete_set
    print "counts"
    for i in range(0,25):
        print i,complete.count(i)

def deterimine_max_and_min(X):
    upper = []
    for i in range(0,len(X)):
        print max(X[i]),min(X[i])
        upper.extend([max(X[i]),min(X[i])])
    print "global max,min:",max(upper),min(upper)
