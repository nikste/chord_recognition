import numpy as np



class ReferenceClassifyier():

    def __init__(self, y):
        counts = np.zeros(25)

        for el in y:
            for ell in el:
                # convert to 1 hot:
                output_y_one_hot = np.zeros(25)

                output_y_one_hot[ell] = 1
                counts += output_y_one_hot
        self._max_ind = np.argmax(counts)

    def classification_error(self, y_test):
        counts = np.zeros(25)
        for el in y_test:
            counts += el
        return counts[self._max_ind] / np.sum(counts)