import numpy


class DataGenerator:

    def __init__(self, X, y, X_test=[],y_test=[], step_size = 1000, window_size = 1000, num_labels = 25):


        self._song_num = 0
        self._frame_num = 0
        self._step_size = step_size
        self._window_size = window_size
        self._epochs = 0
        self._num_labels = num_labels

        self.X = self.add_padding(X)
        self.y = self.add_padding(y)
        self.X_test = self.add_padding(X_test)
        self.y_test = y_test

    def add_padding(self,Xs):
        for i in range(0,len(Xs)):
            # add padding  window_size/2 to beginning and end of song
            Xs[i] = numpy.hstack((Xs[i],numpy.zeros(self._window_size/2)))
            Xs[i] = numpy.hstack((numpy.zeros(self._window_size/2),Xs[i]))

        return Xs

    def get_test_input_output(self):
        Xs = []
        ys = []
        for song_num in range(0,len(self.X_test)):
            X, y = self.get_points_for_song(self.X_test[song_num],self.y_test[song_num])
            Xs.extend(X)
            ys.extend(y)
        return Xs, ys

    def get_points_for_song(self,X,y):
        outputs_X = []
        outputs_y = []
        for frame_num in range(0,len(X) - self._window_size,self._step_size):
            #print frame_num,"len",len(y),len(X)
            outputs_X.append(X[frame_num : frame_num + self._window_size ])
            output_y_one_hot = numpy.zeros(self._num_labels)
            output_y_one_hot[y[frame_num]] = 1
            outputs_y.append(output_y_one_hot)
        return outputs_X,outputs_y

    def get_single_point(self):
        output_X = self.X[self._song_num][self._frame_num: self._frame_num + self._window_size]
        # output_y = self.y[self._song_num][self._frame_num + self._window_size / 2]
        output_y = self.y[self._song_num][self._frame_num]

        output_y_one_hot = numpy.zeros(self._num_labels)
        output_y_one_hot[output_y] = 1

        if (self._frame_num + self._step_size + self._window_size > len(self.X[self._song_num])):
            # print self._frame_num + self._step_size + self._window_size, "  < ", len(self.X[self._song_num])
            # if so set counter to 0
            self._frame_num = 0

            # check if all songs used
            if (self._song_num + 1 < len(self.X) - 1):
                self._song_num += 1
            else:
                # if so reset everything and add 1 to epochs
                self._song_num = 0
                self._epochs += 1
                # shuffle songs:
                perm = numpy.arange(len(self.X))
                numpy.random.shuffle(perm)
                self.X = [self.X[i] for i in perm]
                self.y = [self.y[i] for i in perm]
        else:
            self._frame_num += self._step_size
            # print "frame number:",self._frame_num, "song num:", self._song_num

        return output_X, output_y_one_hot

    def get_song_counter(self):
        return self._song_num,len(self.X)

    def get_epochs(self):
        return self._epochs

    def next_batch(self,batch_size):
        X = []
        y = []

        for i in range(0,batch_size):
            X_one, y_one = self.get_single_point()

            X.append(X_one)
            y.append(y_one)

        return X, y

    def get_test(self):
        # get frames
        # get labels
        X_t, y_t = self.get_test_input_output()
        return X_t, y_t

