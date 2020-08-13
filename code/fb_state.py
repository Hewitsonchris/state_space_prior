import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import differential_evolution
from scipy.optimize import LinearConstraint


def inspect_fits():
    obs = get_observed(0)
    args = obs[0]
    x_obs_mp0 = obs[1]
    x_obs_ep0 = obs[2]

    obs = get_observed(1)
    args = obs[0]
    x_obs_mp1 = obs[1]
    x_obs_ep1 = obs[2]

    fit0 = np.loadtxt('../fits/fit_group_0.txt')
    fit1 = np.loadtxt('../fits/fit_group_1.txt')

    (y0, yff0, yfb0) = simulate(fit0[0:-1], args)
    (y1, yff1, yfb1) = simulate(fit1[0:-1], args)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    n_trials = args[0].shape[0]
    ax[0].plot(np.arange(0, n_trials, 1), -args[0], '-k')
    ax[0].plot(np.arange(0, n_trials, 1), y0, 'b', alpha=1)
    ax[0].plot(np.arange(0, n_trials, 1), yff0, 'r', alpha=1)
    ax[0].plot(np.arange(0, n_trials, 1), x_obs_mp0, 'r', alpha=0.25)
    ax[0].plot(np.arange(0, n_trials, 1), x_obs_ep0, 'b', alpha=0.25)
    ax[0].legend(['-r', 'y', 'yff', 'yfb'])
    ax[1].plot(np.arange(0, n_trials, 1), -args[0], '-k')
    ax[1].plot(np.arange(0, n_trials, 1), y1, 'b', alpha=1)
    ax[1].plot(np.arange(0, n_trials, 1), yff1, 'r', alpha=1)
    ax[1].plot(np.arange(0, n_trials, 1), x_obs_mp1, 'r', alpha=0.25)
    ax[1].plot(np.arange(0, n_trials, 1), x_obs_ep1, 'b', alpha=0.25)
    ax[1].legend(['-r', 'y', 'yff', 'yfb'])
    plt.savefig('../figures/fig_results.pdf')



def fit():

    # alpha = params[0]
    # beta = params[1]
    # gamma_1 = params[2]
    # gamma_2 = params[3]
    # gamma_3 = params[4]
    # sig_fb = params[5]

    constraints = LinearConstraint(A=[[0, 0, 1, -1, 0, 0], [0, 0, 0, 1, -1, 0],
                                      [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
                                   lb=[-1, -1, 0, 0, 0, 0],
                                   ub=[0, 0, 0, 0, 0, 0])

    bounds = ((0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1))

    group = [0, 1]
    for group in group:
        args = [group]
        results = differential_evolution(func=obj_func,
                                         bounds=bounds,
                                         args=args,
                                         disp=True,
                                         maxiter=300,
                                         tol=1e-15,
                                         polish=False,
                                         updating='deferred',
                                         workers=-1)

        f_name_p = '../fits/fit_group_' + str(group) + '.txt'
        with open(f_name_p, 'a') as f:
            np.savetxt(f, np.concatenate((results['x'], [results['fun']])),
                       '%0.4f', ',')


def obj_func(params, *args):
    '''
    params: collection of free paraameter values
    *arags: fixed parameters et al.
    '''

    group = args[0]

    obs = get_observed(group)
    args = obs[0]
    x_obs_mp = obs[1]
    x_obs_ep = obs[2]

    x_pred = simulate(params, args)
    x_pred_mp = x_pred[1]
    x_pred_ep = x_pred[0]

    sse_mp = np.sum((x_obs_mp - x_pred_mp)**2)
    sse_ep = np.sum((x_obs_ep - x_pred_ep)**2)
    sse = sse_mp + sse_ep

    return sse


def get_observed(group):
    d = pd.read_csv('../datta/Bayes_SML_EXP1_140820.csv')

    # Get args
    rot = d[d.SUBJECT == 1].ROT.values
    sig_mp = d[d.SUBJECT == 1].SIG_MP.values
    args = (rot, sig_mp)

    dd = d[['GROUP', 'TRIAL_ABS', 'HA_INITIAL', 'HA_END',
            'ROT']].groupby(['GROUP', 'TRIAL_ABS']).mean()
    dd.reset_index(inplace=True)

    # fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(6, 6))
    # ddd = dd[dd['GROUP'] == 0]
    # ax[0].plot(ddd['TRIAL_ABS'].values, -ddd['ROT'].values)
    # ax[0].plot(ddd['TRIAL_ABS'].values, ddd['HA_INITIAL'].values)
    # ax[0].plot(ddd['TRIAL_ABS'].values, ddd['HA_END'].values)
    # ddd = dd[dd['GROUP'] == 1]
    # ax[1].plot(ddd['TRIAL_ABS'].values, -ddd['ROT'].values)
    # ax[1].plot(ddd['TRIAL_ABS'].values, ddd['HA_INITIAL'].values)
    # ax[1].plot(ddd['TRIAL_ABS'].values, ddd['HA_END'].values)
    # plt.show()

    ddd = dd[dd['GROUP'] == group]

    x_obs_mp = ddd['HA_INITIAL'].values
    x_obs_ep = ddd['HA_END'].values

    return (args, x_obs_mp, x_obs_ep)


def simulate(params, args):

    alpha = params[0]
    beta = params[1]
    gamma_3 = params[2]
    gamma_2 = params[3]
    gamma_1 = params[4]
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
        if sig_mp[i] != 4:
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


fit()

# alpha = 0.05
# beta = 0.99
# gamma = [0.25, 0.5, 0.75]
# sig_fb = 0.1

# r = np.concatenate((np.zeros(100), -20 * np.ones(100), np.zeros(100)))

# sig_mp = np.concatenate(
#     (np.zeros(100), np.random.choice([0, 1, 2, 3],
#                                      100), np.zeros(100))).astype(int)

# n_trials = r.shape[0]

# params = [alpha, beta, gamma[0], gamma[1], gamma[2], sig_fb]
# args = [r, sig_mp]

# (y, yff, yfb) = simulate(params, args)

# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
# ax.plot(np.arange(0, n_trials, 1), -r, '-k')
# ax.plot(np.arange(0, n_trials, 1), y, alpha=0.5)
# ax.plot(np.arange(0, n_trials, 1), yff, alpha=0.5)
# ax.plot(np.arange(0, n_trials, 1), yfb, alpha=0.5)
# ax.legend(['-r', 'y', 'yff', 'yfb'])
# plt.show()
