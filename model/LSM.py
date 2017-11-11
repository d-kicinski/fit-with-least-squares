import numpy as np


class LSM(object):
    """ Regression with least square method .

  Paramterers
  -----------

  Attributes
  ----------
  w_ : ld-array
    Weights after fitting.

  """

    def __init__(self, x_learning, y_learning):
        self.x_ = x_learning
        self.y_ = y_learning

    def fit(self, dynamic, poly):
        self.dynamic = dynamic
        self.poly = poly

        kf = dynamic  # first calculatable sample
        kl = self.x_.size  # last sample

        M = self._calc_M_matrix(self.x_, self.y_)

        # calculating parameters of the model
        # in [0] are solutions
        self.w_ = np.linalg.lstsq(M, self.y_[kf:kl, None])[0]

        #y_arx = M @ self.w_
        #print(y_arx.shape)
        #print(self.y_[kf:, None].shape)
        #print((y_arx - self.y_[kf:, None]).shape)
        #error = (y_arx - self.y_[kf:, None]).T @ (y_arx - self.y_[kf:, None])
        #print(error)

    def predict(self, x, y):
        kf = self.dynamic  # first calculatable sample
        kl = x.size  # last sample

        y_predict = np.empty(kl)

        #print(y_predict)
        # recursive way(model OE)
        # first values are known, we use them to predict next value in
        # series and then we use that new value and previous known values excludnig
        # oldest one to calcualte next new value and so on...

        y_predict[0:kf - 1] = y[
            0:kf - 1]  # put know init values into predicton array
        #print(y)
        #print(y_predict)

        for k in range(kf, kl):
            # creating model
            # model_oe = np.array([], dtype=np.int64).reshape(2 * self.dynamic *
            # self.poly, 0)
            model_oe = []

            for i in range(1, self.dynamic + 1):
                for j in range(1, self.poly + 1):
                    model_oe = np.hstack(
                        [model_oe, x[k - i]**j, y_predict[k - i]**j])

            # calculating next value
            y_predict[k] = model_oe @ self.w_

        return y_predict.flatten(), self._calc_error(y, y_predict)

    def predict_arx(self, x, y):
        kf = self.dynamic  # first calculatable sample
        kl = x.size  # last sample
        y_predict = np.empty(kl)

        M = self._calc_M_matrix(x, y)
        y_predict = M @ self.w_
        error = self._calc_error(y[kf:], y_predict.flatten())

        return y_predict, error

    def _calc_error(self, y, y_mod):
        return (y_mod - y) @ (y_mod - y)

    def _calc_M_matrix(self, x, y):
        """
    Parameters
    ----------
    kf : uint
      First sample used for computing due to model dynamics, can be interpreted
      as 'k'(in opposite of k-1, k-2 etc)

    kl : uint
      Last sample

    Returns
    ----------
    M : {array-like}, shape = [n_samples - dynamics_degree,
                               2 * dynamics_degree * polynomial_degree]
    """

        kf = self.dynamic  # first calculatable sample
        kl = x.size  # last sample

        # initialization of empty array with proper shape
        M = np.array([], dtype=np.int64).reshape(kl - kf, 0)

        ## TODO not sure but last sample may be never used
        for i in range(1, self.dynamic + 1):
            for j in range(1, self.poly + 1):
                M = np.hstack(
                    [M, x[kf - i:kl - i, None]**j, y[kf - i:kl - i, None]**j])

        return M


if __name__ == '__main__':
    import pandas as pd
    import numpy as np

    DYNAMIC = 5  # degree of dynamic in model
    POLY = 4  # polynomial degree

    df_ler = pd.read_csv("data_learning", header=None)
    u_ler = df_ler.iloc[:, 0].values
    y_ler = df_ler.iloc[:, 1].values

    regres = LSM(u_ler, y_ler)
    regres.fit(DYNAMIC, POLY)

    df_ver = pd.read_csv("data_testing", header=None)
    u_ver = df_ver.iloc[:, 0].values
    y_ver = df_ver.iloc[:, 1].values

    y_oe, error_oe = regres.predict(u_ver, y_ver)
    y_arx, error_arx = regres.predict_arx(u_ver, y_ver)

    print("Error OE: {0}".format(error_oe))
    print("Error ARX: {0}".format(error_arx))
