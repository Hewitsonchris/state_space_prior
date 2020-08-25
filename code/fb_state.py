import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import differential_evolution
from scipy.optimize import LinearConstraint


def inspect_fits():

    obs = get_observed()

    args = obs[0]
    x_obs_mp0 = obs[1]
    x_obs_ep0 = obs[2]
    x_obs_mp1 = obs[3]
    x_obs_ep1 = obs[4]

    p = np.loadtxt('../fits/fit_group_01.txt')

    (y0, yff0, yfb0, xff0, xfb0) = simulate(p[0:-1], args, group=0)
    (y1, yff1, yfb1, xff1, xfb1) = simulate(p[0:-1], args, group=1)

    # fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(10, 6))
    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(3, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, :])
    ax = np.array([ax1, ax2, ax3, ax4, ax5])

    n_trials = args[0].shape[0]

    ax[0].plot(np.arange(0, n_trials, 1), y0, '.b', alpha=1)
    ax[0].plot(np.arange(0, n_trials, 1), yff0, '.r', alpha=1)
    ax[0].plot(np.arange(0, n_trials, 1), x_obs_ep0, 'b', alpha=0.25)
    ax[0].plot(np.arange(0, n_trials, 1), x_obs_mp0, 'r', alpha=0.25)
    ax[0].legend(['Model EP', 'Model MP', 'Human EP', 'Human MP'])
    ax[0].set_xlabel('Trial')
    ax[0].set_ylabel('Hand Angle (degrees)')
    ax[0].set_ylim([-5, 35])

    ax[1].plot(np.arange(0, n_trials, 1), y1, '.b', alpha=1)
    ax[1].plot(np.arange(0, n_trials, 1), yff1, '.r', alpha=1)
    ax[1].plot(np.arange(0, n_trials, 1), x_obs_ep1, 'b', alpha=0.25)
    ax[1].plot(np.arange(0, n_trials, 1), x_obs_mp1, 'r', alpha=0.25)
    ax[1].legend(['Model EP', 'Model MP', 'Human EP', 'Human MP'])
    ax[1].set_xlabel('Trial')
    ax[1].set_ylabel('Hand Angle (degrees)')
    ax[1].set_ylim([-5, 35])

    ax[2].plot(np.arange(0, n_trials, 1), y0, '-', alpha=1)
    ax[2].plot(np.arange(0, n_trials, 1), yff0, '-', alpha=1)
    ax[2].plot(np.arange(0, n_trials, 1), yfb0, '-', alpha=1)
    ax[2].plot(np.arange(0, n_trials, 1), xff0, '-', alpha=1)
    ax[2].plot(np.arange(0, n_trials, 1), 10 * xfb0, '-', alpha=1)
    ax[2].legend(['y', 'yff', 'yfb', 'xff', 'xfb'])
    ax[2].set_xlabel('Trial')
    ax[2].set_ylabel('Hand Angle (degrees)')
    ax[2].set_ylim([-5, 35])

    ax[3].plot(np.arange(0, n_trials, 1), y1, '-', alpha=1)
    ax[3].plot(np.arange(0, n_trials, 1), yff1, '-', alpha=1)
    ax[3].plot(np.arange(0, n_trials, 1), yfb1, '-', alpha=1)
    ax[3].plot(np.arange(0, n_trials, 1), xff1, '-', alpha=1)
    ax[3].plot(np.arange(0, n_trials, 1), 10 * xfb1, '-', alpha=1)
    ax[3].legend(['y', 'yff', 'yfb', 'xff', 'xfb'])
    ax[3].set_xlabel('Trial')
    ax[3].set_ylabel('Hand Angle (degrees)')
    ax[3].set_ylim([-5, 35])

    # NOTE: No boot
    # p[-2] /= 5
    # ax[4].bar(np.arange(1, 11, 1), p[0:-1])
    # ax[4].set_xticks(np.arange(1, 11, 1))
    # ax[4].set_xticklabels([
    #     'alpha_ff', 'beta_ff', 'alpha_fb', 'beta_fb', 'beta_ff2', 'beta_fb2',
    #     'base_fb', 'w', 'gamma_ff', 'gamma_fb'
    # ])

    # NOTE: Yes boot
    p[-2] /= 5
    p[-3] /= 5
    ax[4].scatter(np.arange(1, 11, 1), p[0:-1])

    p = np.loadtxt('../fits/fit_group_01_boot.txt')
    n_cols = 11
    n_rows = p.shape[0] // n_cols
    p = np.reshape(p, (n_rows, n_cols))

    p[:, -2] /= 5
    p[:, -3] /= 5

    ax[4].violinplot(p[:, 0:-1], np.arange(1, 11, 1))

    ax[4].set_xticks(np.arange(1, 11, 1))
    ax[4].set_xticklabels([
        'alpha_ff', 'beta_ff', 'alpha_fb', 'beta_fb', 'beta_ff2', 'beta_fb2',
        'base_fb', 'w', 'gamma_ff', 'gamma_fb'
    ])

    ax[4].plot([0, 12], [0, 0], '--')
    plt.tight_layout()
    # plt.show()
    plt.savefig('../figures/fig_results.pdf')
    plt.close()


def fit_boot():
    bounds = ((0, 0.05), (0, 1), (0, 0.01), (0, 1), (0, 1), (0, 1), (-1, 1),
              (0, 1), (-5, 5), (-5, 5))

    n_boot = 10

    for i in range(n_boot + 1):

        # fit both groups simultaneously
        # i.e., impose same parameters on all groups
        results = differential_evolution(
            func=obj_func,
            bounds=bounds,
            # args=args,
            # constraints=constraints,
            disp=True,
            maxiter=50,
            tol=1e-15,
            polish=False,
            updating='deferred',
            workers=-1)

        f_name_p = '../fits/fit_group_01_boot.txt'
        with open(f_name_p, 'a') as f:
            np.savetxt(f, np.concatenate((results['x'], [results['fun']])),
                       '%0.4f', ',')


def fit():

    # constraints = LinearConstraint(A=[[0, 0, 0, 0, -1, 1, 0, 0, 0],
    #                                   [0, 0, 0, 0, 0, -1, 1, 0, 0],
    #                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                                   [0, 0, 0, 0, 0, 0, 0, 0, 0]],
    #                                lb=[-1, -1, 0, 0, 0, 0, 0, 0, 0],
    #                                ub=[0, 0, 0, 0, 0, 0, 0, 0, 0])

    bounds = ((0, 0.05), (0, 1), (0, 0.01), (0, 1), (0, 1), (0, 1), (-1, 1),
              (0, 1), (-5, 5), (-5, 5))

    # fit both groups simultaneously
    # i.e., impose same parameters on all groups
    results = differential_evolution(
        func=obj_func,
        bounds=bounds,
        # args=args,
        # constraints=constraints,
        disp=True,
        maxiter=50,
        tol=1e-15,
        polish=False,
        updating='deferred',
        workers=-1)

    f_name_p = '../fits/fit_group_01.txt'
    with open(f_name_p, 'w') as f:
        np.savetxt(f, np.concatenate((results['x'], [results['fun']])),
                   '%0.4f', ',')


def obj_func(params, *args):
    '''
    params: collection of free paraameter values
    *arags: fixed parameters et al.
    '''

    # group = args[0]

    obs = get_observed()

    args = obs[0]
    x_obs_mp0 = obs[1]
    x_obs_ep0 = obs[2]
    x_obs_mp1 = obs[3]
    x_obs_ep1 = obs[4]

    x_pred0 = simulate(params, args, group=0)
    x_pred_mp0 = x_pred0[1]
    x_pred_ep0 = x_pred0[0]

    x_pred1 = simulate(params, args, group=1)
    x_pred_mp1 = x_pred1[1]
    x_pred_ep1 = x_pred1[0]

    # sse_mp = np.sum((x_obs_mp[11:191] - x_pred_mp[11:191])**2)
    # sse_ep = np.sum((x_obs_ep[11:191] - x_pred_ep[11:191])**2)
    sse_mp0 = np.sum((x_obs_mp0 - x_pred_mp0)**2)
    sse_ep0 = np.sum((x_obs_ep0 - x_pred_ep0)**2)
    sse_mp1 = np.sum((x_obs_mp1 - x_pred_mp1)**2)
    sse_ep1 = np.sum((x_obs_ep1 - x_pred_ep1)**2)
    w = 0.5
    sse = w * sse_mp0 + (1 - w) * sse_ep0 + w * sse_mp1 + (1 - w) * sse_ep1

    return sse


def get_observed():

    # TODO: Experiment 2
    # d = pd.read_csv('../datta/Bayes_SML_EXP2.csv')
    # d['ROT'] = -d['ROT']
    # dd = d[['ROT', 'HA_INIT', 'HA_END',
    #         'TRIAL_ABS']].groupby(['TRIAL_ABS']).mean().plot()
    # plt.show()

    d = pd.read_csv('../datta/Bayes_SML_EXP1_200820.csv')

    # Get args
    # trial events were identical for every subject, so it's okay to just grab
    # the relevant info from an arbitrary single sub
    rot = d[d.SUBJECT == 1].ROT.values
    sig_mp = d[d.SUBJECT == 1].SIG_MP.values
    args = (rot, sig_mp)

    dd = d[['GROUP', 'TRIAL_ABS', 'HA_INIT', 'HA_END', 'ROT',
            'SIG_MP']].groupby(['GROUP', 'TRIAL_ABS']).mean()
    dd.reset_index(inplace=True)

    # d_11 = d[d['SIG_MP'] == 1][d['GROUP'] == 0][d['TRIAL_ABS'] >= 11][d['TRIAL_ABS'] <= 190][['ROT', 'HA_END']]
    # d_12 = d[d['SIG_MP'] == 2][d['GROUP'] == 0][d['TRIAL_ABS'] >= 11][d['TRIAL_ABS'] <= 190][['ROT', 'HA_END']]
    # d_13 = d[d['SIG_MP'] == 3][d['GROUP'] == 0][d['TRIAL_ABS'] >= 11][d['TRIAL_ABS'] <= 190][['ROT', 'HA_END']]
    # d_21 = d[d['SIG_MP'] == 1][d['GROUP'] == 1][d['TRIAL_ABS'] >= 11][d['TRIAL_ABS'] <= 190][['ROT', 'HA_END']]
    # d_22 = d[d['SIG_MP'] == 2][d['GROUP'] == 1][d['TRIAL_ABS'] >= 11][d['TRIAL_ABS'] <= 190][['ROT', 'HA_END']]
    # d_23 = d[d['SIG_MP'] == 3][d['GROUP'] == 1][d['TRIAL_ABS'] >= 11][d['TRIAL_ABS'] <= 190][['ROT', 'HA_END']]

    # d_11 = d[d['SIG_MP'] == 1][d['GROUP'] == 0][d['TRIAL_ABS'] >= 11][d['TRIAL_ABS'] <= 190][['ROT', 'HA_INIT']]
    # d_12 = d[d['SIG_MP'] == 2][d['GROUP'] == 0][d['TRIAL_ABS'] >= 11][d['TRIAL_ABS'] <= 190][['ROT', 'HA_INIT']]
    # d_13 = d[d['SIG_MP'] == 3][d['GROUP'] == 0][d['TRIAL_ABS'] >= 11][d['TRIAL_ABS'] <= 190][['ROT', 'HA_INIT']]
    # d_21 = d[d['SIG_MP'] == 1][d['GROUP'] == 1][d['TRIAL_ABS'] >= 11][d['TRIAL_ABS'] <= 190][['ROT', 'HA_INIT']]
    # d_22 = d[d['SIG_MP'] == 2][d['GROUP'] == 1][d['TRIAL_ABS'] >= 11][d['TRIAL_ABS'] <= 190][['ROT', 'HA_INIT']]
    # d_23 = d[d['SIG_MP'] == 3][d['GROUP'] == 1][d['TRIAL_ABS'] >= 11][d['TRIAL_ABS'] <= 190][['ROT', 'HA_INIT']]

    # fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12, 6))

    # x = d_11['ROT'].values
    # y = d_11['HA_INIT'].values
    # coef = np.polyfit(x, y, 1)
    # poly1d_fn = np.poly1d(coef)
    # ax[0, 0].plot(x, y, '.')
    # ax[0, 0].plot(x, poly1d_fn(x), '-')

    # x = d_12['ROT'].values
    # y = d_12['HA_INIT'].values
    # coef = np.polyfit(x, y, 1)
    # poly1d_fn = np.poly1d(coef)
    # ax[0, 1].plot(x, y, '.')
    # ax[0, 1].plot(x, poly1d_fn(x), '-')

    # x = d_13['ROT'].values
    # y = d_13['HA_INIT'].values
    # coef = np.polyfit(x, y, 1)
    # poly1d_fn = np.poly1d(coef)
    # ax[0, 2].plot(x, y, '.')
    # ax[0, 2].plot(x, poly1d_fn(x), '-')

    # x = d_21['ROT'].values
    # y = d_21['HA_INIT'].values
    # coef = np.polyfit(x, y, 1)
    # poly1d_fn = np.poly1d(coef)
    # ax[1, 0].plot(x, y, '.')
    # ax[1, 0].plot(x, poly1d_fn(x), '-')

    # x = d_22['ROT'].values
    # y = d_22['HA_INIT'].values
    # coef = np.polyfit(x, y, 1)
    # poly1d_fn = np.poly1d(coef)
    # ax[1, 1].plot(x, y, '.')
    # ax[1, 1].plot(x, poly1d_fn(x), '-')

    # x = d_23['ROT'].values
    # y = d_23['HA_INIT'].values
    # coef = np.polyfit(x, y, 1)
    # poly1d_fn = np.poly1d(coef)
    # ax[1, 2].plot(x, y, '.')
    # ax[1, 2].plot(x, poly1d_fn(x), '-')

    # # for x in range(ax.shape[0]):
    # #     for y in range(ax.shape[1]):
    # #         ax[x, y].set_xlim([-40, -20])
    # #         ax[x, y].set_ylim([0, 40])

    # plt.show()

    x_obs_mp0 = dd[dd['GROUP'] == 0]['HA_INIT'].values
    x_obs_ep0 = dd[dd['GROUP'] == 0]['HA_END'].values
    x_obs_mp1 = dd[dd['GROUP'] == 1]['HA_INIT'].values
    x_obs_ep1 = dd[dd['GROUP'] == 1]['HA_END'].values

    return (args, x_obs_mp0, x_obs_ep0, x_obs_mp1, x_obs_ep1)


def simulate(params, args, group=None):

    alpha_ff = params[0]
    beta_ff = params[1]
    alpha_fb = params[2]
    beta_fb = params[3]
    beta_ff2 = params[4]
    beta_fb2 = params[5]
    base_fb = params[6]
    w = params[7]
    gamma_ff = params[8]
    gamma_fb = params[9]

    r = args[0]
    sig_mp = args[1]

    n_trials = r.shape[0]

    delta_ep = np.zeros(n_trials)
    delta_mp = np.zeros(n_trials)
    xff = np.zeros(n_trials)
    xfb = np.zeros(n_trials)
    yff = np.zeros(n_trials)
    yfb = np.zeros(n_trials)
    y = np.zeros(n_trials)

    xfb[0] = base_fb

    for i in range(n_trials - 1):

        # start to midpoint
        yff[i] = xff[i]
        yfb[i] = 0.0
        y[i] = yff[i] + yfb[i]

        # midpoint to endpoint
        if sig_mp[i] != 4:
            delta_mp[i] = 0.0 - (y[i] + r[i])
            yfb[i] = xfb[i] * delta_mp[i]
        else:
            delta_mp[i] = 0.0
            yfb[i] = 0.0

        y[i] = yff[i] + yfb[i]

        if sig_mp[i] == 1:
            delta_ep[i] = 0.0 - (y[i] + r[i])
        else:
            delta_ep[i] = 0.0

        if group == 1:
            delta_ep[i] = 0.0

        if sig_mp[i] != 4:
            bayes_mod_ff = bayes_int(sig_mp[i], gamma_ff)
            bayes_mod_fb = bayes_int(sig_mp[i], gamma_fb)
        else:
            bayes_mod_ff = 0.0
            bayes_mod_fb = 0.0

        # state update based on endpoint error
        if i > 191:
            xff[i + 1] = beta_ff2 * xff[i]
            xfb[i + 1] = beta_fb2 * xfb[i] + base_fb
        else:
            xff[i + 1] = beta_ff * xff[i] + bayes_mod_ff * alpha_ff * (
                w * delta_mp[i] + (1 - w) * delta_ep[i])
            xfb[i + 1] = beta_fb * xfb[i] + bayes_mod_fb * alpha_fb * delta_ep[
                i] + base_fb

    return (y, yff, yfb, xff, xfb)


def bayes_int(x, m):
    '''I want m=1 to be no differential integration'''

    # x = np.arange(1, 3, 0.01)
    # plt.plot(x, np.tanh(8 * (x - 2)) + 1)
    # plt.plot(x, np.tanh(-1 * (x - 2)) + 1)
    # plt.show()

    return np.tanh(m * (x - 2)) + 1


# fit()
# fit_boot()
# inspect_fits()
