import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import differential_evolution
from scipy.optimize import LinearConstraint


def simulate(params, args):

    alpha = params[0]
    beta = params[1]
    gamma_1 = params[2]
    gamma_2 = params[3]
    gamma_3 = params[4]
    gamma = (gamma_1, gamma_2, gamma_3)
    sig_fb = params[5]

    r = args[0]
    sig_mp = args[1]

    n_trials = r.shape[0]

    delta_ep = np.zeros(n_trials)
    delta_mp = np.zeros(n_trials)
    x = np.zeros(n_trials)
    yff = np.zeros(n_trials)
    yfb = np.zeros(n_trials)
    y = np.zeros(n_trials)

    for i in range(n_trials - 1):

        # start to midpoint
        yff[i] = x[i]
        yfb[i] = 0.0
        y[i] = yff[i] + yfb[i]

        # midpoint to endpoint
        if sig_mp[i] != 0:
            delta_mp[i] = 0.0 - (y[i] + r[i])
            yfb[i] = gamma[sig_mp[i] - 1] * np.random.normal(
                delta_mp[i], sig_fb)
        else:
            delta_mp[i] = 0.0
            yfb[i] = 0.0

        y[i] = yff[i] + yfb[i]
        delta_ep[i] = 0.0 - (y[i] + r[i])

        # state update based on endpoint error
        x[i + 1] = beta * x[i] + alpha * delta_ep[i]

    return (y, yff, yfb)

alpha = 0.05
beta = 0.99
gamma = [0.25, 0.5, 0.75]
sig_fb = 0.1

r = np.concatenate((np.zeros(100), -20 * np.ones(100), np.zeros(100)))

sig_mp = np.concatenate(
    (np.zeros(100), np.random.choice([0, 1, 2, 3],
                                     100), np.zeros(100))).astype(int)

params = [alpha, beta, gamma[0], gamma[1], gamma[2], ]

(y, yff, yfb) = simulate(params, args)


# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
# ax.plot(np.arange(0, n_trials, 1), -r, '-k')
# ax.plot(np.arange(0, n_trials, 1), y, alpha=0.5)
# ax.plot(np.arange(0, n_trials, 1), yff, alpha=0.5)
# ax.plot(np.arange(0, n_trials, 1), yfb, alpha=0.5)
# ax.legend(['-r', 'y', 'yff', 'yfb'])
# plt.show()
