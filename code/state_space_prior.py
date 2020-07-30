import time
import os as os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import differential_evolution, minimize
from scipy.optimize import LinearConstraint
import multiprocessing as mp
from matplotlib import rc
import matplotlib as mpl
from scipy.stats import norm


def g_func(theta, theta_mu, sigma):
    if sigma != 0:
        G = np.exp(-(theta - theta_mu)**2 / (2 * sigma**2))
    else:
        G = np.zeros(11)
    return G


def simulate(p, rot, mu_f, sig_f):
    eta_pm = p[0]
    eta_pu = p[1]
    b0_mu = p[2]
    b0_sig = p[3]
    b1 = p[4]
    mu_p0 = p[5]
    sig_p0 = p[6]

    num_trials = rot.shape[0]

    delta_mu = np.zeros(num_trials)
    delta_sig = np.zeros(num_trials)

    theta_values = np.array(
        [-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150])
    theta_train_ind = np.where(theta_values == 0)[0][0]
    theta_ind = theta_train_ind * np.ones(num_trials, dtype=np.int8)

    n_tgt = theta_value.shape[0]
    mu_p = np.zeros((n_tgt, num_trials))
    sig_p = np.zeros((n_tgt, num_trials))
    mu_p[0] = mu_p0
    sig_p[0] = sig_p0

    for i in range(1, num_trials - 1):
        ind = theta_ind[i]

        sp = sig_p[ind, i]**2 / (sig_p[ind, i]**2 + sig_f[ind, i]**2)
        sf = sig_f[ind, i]**2 / (sig_p[ind, i]**2 + sig_f[ind, i]**2)**2

        delta_sig[i] = 2 * (rot[i] - mu_p[ind, i] + sp *
                            (mu_p[ind, i] - mu_f[ind, i])) * (
                                mu_p[ind, i] -
                                mu_f[ind, i]) * sf * 2 * sig_p[ind, i]

        delta_mu[i] = -2 * (mu_p[ind, i] - (1 - sp) * mu_p[ind, i] -
                            sp * mu_f[ind, i]) * (1 - sp)

        G_mu = g_func(theta_values, theta_values[theta_ind[i]], b0_mu, b1)
        G_sig = g_func(theta_values, theta_values[theta_ind[i]], b0_sig, b1)

        sig_p[:, i +
              1] = (1 - beta_f) * xf[:, i] - alpha_f * delta_sig[i] * G_sig
        mu_p[:, i + 1] = (1 - beta_s) * xs[:, i] - alpha_s * delta_mu[i] * G_mu

    return (mu_p.T, sig_p.T)


def fit_obj_func_sse(params, *args):
    x_obs = args[0]
    rot = args[1]
    x_pred = simulate(params, rot)[0]

    n_tgt = 11
    sse_rec = np.zeros(n_tgt)
    for i in range(n_tgt):
        sse_rec[i] = np.nansum((x_obs[:, i] - x_pred[:, i])**2)
        sse = np.nansum(sse_rec)
    return sse


def prep_for_fits():
    d = pd.read_csv('../data/...')

    rot = np.concatenate(
        (np.zeros(198), 30 * np.ones(110), np.nan * np.ones(66)))

    return (d, rot)


def fit_state_space():

    d, rot = prep_for_fits()

    for i in d.group.unique():
        p_rec = np.empty((0, 8))

        x_obs = d.groupby(['group', 'target', 'trial']).mean()
        x_obs.reset_index(inplace=True)
        x_obs = x_obs[x_obs['group'] == i]
        x_obs = x_obs[['hand_angle', 'target', 'trial']]
        x_obs = x_obs.pivot(index='trial',
                            columns='target',
                            values='hand_angle')
        x_obs = x_obs.values

        # for k in range(11):
        #     plt.plot(x_obs[:, k])
        #     plt.plot(rot)
        # plt.show()

        results = fit(x_obs, rot)
        p = results["x"]
        print(p)

        # x_pred = simulate_state_space_with_g_func_2_state(p, rot)[0]
        # fig, ax = plt.subplots(nrows=1, ncols=2)
        # c = cm.rainbow(np.linspace(0, 1, 11))
        # for k in range(11):
        #     ax[0].plot(x_obs[:, k], '.', color=c[k])
        #     ax[0].plot(x_pred[:, k], '-', color=c[k])
        #     ax[0].plot(rot, 'k')

        # x = np.arange(0, 11, 1)
        # y_obs = np.nanmean(x_obs[-65:-1, :], 0)
        # y_pred = np.nanmean(x_pred[-65:-1, :], 0)
        # ax[1].plot(x, y_obs)
        # ax[1].plot(x, y_pred)

        # plt.show()

        p_rec = np.append(p_rec, [p], axis=0)

        f_name_p = '../fits/fit_group_' + str(i) + '.txt'
        with open(f_name_p, 'w') as f:
            np.savetxt(f, p_rec, '%0.4f', '\n')


def fit_boot():

    n_boot_samp = 100

    d, rot = prep_for_fits()

    for i in d.group.unique():

        p_rec = -1 * np.ones((1, 10))
        for b in range(n_boot_samp):
            print(i, b)

            subs = d['subject'].unique()
            boot_subs = np.random.choice(subs,
                                         size=subs.shape[0],
                                         replace=True)

            x_boot_rec = []
            for k in boot_subs:
                x_boot_rec.append(d[d['subject'] == k])
                x_boot = pd.concat(x_boot_rec)

            x_obs = x_boot.groupby(['group', 'target', 'trial']).mean()
            x_obs.reset_index(inplace=True)
            x_obs = x_obs[x_obs['group'] == i]
            x_obs = x_obs[['hand_angle', 'target', 'trial']]
            x_obs = x_obs.pivot(index='trial',
                                columns='target',
                                values='hand_angle')
            x_obs = x_obs.values
            results = fit(x_obs, rot)
            p_rec[0, :] = results["x"]

            f_name_p = '../fits/fit_group_' + str(i) + '_boot.txt'
            with open(f_name_p, 'a') as f:
                np.savetxt(f, p_rec, '%0.4f', ',')


def fit(x_obs, rot):
    constraints = LinearConstraint(A=[[1, 0, 0, -1, 0, 0, 0, 0],
                                      [0, 1, 0, 0, -1, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0]],
                                   lb=[-1, -1, 0, 0, 0, 0, 0, 0],
                                   ub=[0, 0, 0, 0, 0, 0, 0, 0])

    args = (x_obs, rot)
    bounds = ((0, 1), (0, 1), (0, 150), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1))

    results = differential_evolution(func=fit_obj_func_sse,
                                     bounds=bounds,
                                     constraints=constraints,
                                     args=args,
                                     maxiter=800,
                                     tol=1e-15,
                                     disp=False,
                                     polish=False,
                                     updating='deferred',
                                     workers=-1)
    return results


fit_state_space()
