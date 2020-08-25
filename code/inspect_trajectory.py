import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy import stats


def spline_traj(fname, n):

    d = np.loadtxt(fname, delimiter=',')

    xs = np.zeros((n, d.shape[1] // 2))
    ys = np.zeros((n, d.shape[1] // 2))

    rs = np.zeros((n, d.shape[1] // 2))
    vs = np.zeros((n, d.shape[1] // 2))

    for i in range(0, d.shape[1], 2):

        x = d[:, i]
        y = d[:, i + 1]

        x = x[~np.isnan(x)]
        y = y[~np.isnan(y)]

        x = x[0:np.min((x.shape[0], y.shape[0]))]
        y = y[0:np.min((x.shape[0], y.shape[0]))]

        t = np.arange(0, x.shape[0], 1) + 1

        if x.shape[0] != 0:
            xss = CubicSpline(t, x)
            yss = CubicSpline(t, y)

            ts = np.linspace(1, t[-1], n)

            xs[:, i // 2] = xss(ts)
            ys[:, i // 2] = yss(ts)

            rss = np.sqrt((xss(ts)**2 + yss(ts)**2))
            vss = np.diff(rss) / np.diff(ts)
            vss = np.concatenate(([0], vss))

            rs[:, i // 2] = rss
            vs[:, i // 2] = vss

    xs_mean = xs[:, 0:-1:2].mean(1)
    ys_mean = ys[:, 1:-1:2].mean(1)
    xs_err = stats.sem(xs[:, 0:-1:2], 1)
    ys_err = stats.sem(ys[:, 1:-1:2], 1)
    # xs_err = np.std(xs[:, 0:-1:2], 1)
    # ys_err = np.std(ys[:, 1:-1:2], 1)

    # plt.plot(xs, ys, alpha=0.5)
    # plt.xlim([-5, 5])
    # plt.show()

    # plt.plot(ts, vs, alpha=0.5)
    # plt.show()

    # x = d[:, 0::2]
    # y = d[:, 1::2]
    # t = np.arange(0, 3000, 1)
    # r = np.sqrt(x**2 + y**2)
    # v = np.diff(r, axis=0)

    # plt.plot(t[0:-1], v, alpha = 0.5)
    # plt.show()

    # plt.plot(x, y, alpha = 0.5)
    # plt.show()

    return (xs, ys, xs_mean, ys_mean, xs_err, ys_err)


n = 250
fname_base = '../datta/bayes_exp1_baseline.csv'
fname_adapt = '../datta/bayes_exp1_adaptation.csv'
fname_wash = '../datta/bayes_exp1_washout.csv'

fig, ax = plt.subplots(nrows=1, ncols=3, squeeze=False, figsize=(4, 4))
(xs_mean, ys_mean, xs_err, ys_err) = spline_traj(fname_base, sample_rate, n)
ax[0, 0].plot(xs_mean, ys_mean, color='C0')
ax[0, 0].errorbar(xs_mean,
                  ys_mean,
                  xerr=xs_err,
                  yerr=ys_err,
                  color='C0',
                  alpha=0.5)
ax[0, 0].set_xlim([-5, 5])

(xs_mean, ys_mean, xs_err, ys_err) = spline_traj(fname_adapt, sample_rate, n)
ax[0, 1].plot(xs_mean, ys_mean, color='C0')
ax[0, 1].errorbar(xs_mean,
                  ys_mean,
                  xerr=xs_err,
                  yerr=ys_err,
                  color='C0',
                  alpha=0.5)
ax[0, 1].set_xlim([-5, 5])

(xs_mean, ys_mean, xs_err, ys_err) = spline_traj(fname_wash, sample_rate, n)
ax[0, 2].plot(xs_mean, ys_mean, color='C0')
ax[0, 2].errorbar(xs_mean,
                  ys_mean,
                  xerr=xs_err,
                  yerr=ys_err,
                  color='C0',
                  alpha=0.5)
ax[0, 2].set_xlim([-5, 5])

plt.tight_layout()
plt.show()
