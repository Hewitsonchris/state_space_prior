import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import differential_evolution
from scipy.optimize import LinearConstraint


def prep_data():
    pass
    # f1 = '../datta/exp1.csv'
    # f2 = '../datta/exp2.csv'
    # f3 = '../datta/exp3.csv'
    # f4 = '../datta/exp4.csv'
    # ff = [f1, f2, f3, f4]

    # pf1 = '../fits/fit_individual_1.txt'
    # pf2 = '../fits/fit_individual_2.txt'
    # pf3 = '../fits/fit_individual_3.txt'
    # pf4 = '../fits/fit_individual_4.txt'
    # pf = [pf1, pf2, pf3, pf4]

    # d1 = pd.read_csv(f1)
    # d2 = pd.read_csv(f2)
    # d3 = pd.read_csv(f3)
    # d4 = pd.read_csv(f4)

    # d1 = d1[d1['TRIAL_ABS'] < 191]
    # d2 = d2[d2['TRIAL_ABS'] < 191]
    # d3 = d3[d3['TRIAL_ABS'] < 191]
    # d3['GROUP'] = [2] * d3.shape[0]
    # d4['GROUP'] = [3] * d4.shape[0]
    # d4['ROT'] = -d4['ROT']

    # d1.to_csv('../datta/exp1.csv')
    # d2.to_csv('../datta/exp2.csv')
    # d3.to_csv('../datta/exp3.csv')
    # d4.to_csv('../datta/exp4.csv')


def fit_validate(fin, fout, bounds, maxiter):

    for i in range(len(fin)):

        p = np.loadtxt(fout[i], delimiter=',')

        d = pd.read_csv(fin[i])
        subs = d['SUBJECT'].unique()

        for s in range(subs.shape[0]):

            dd = d[d['SUBJECT'] == subs[s]]
            rot = dd['ROT'].values
            sig_mp = dd['SIG_MP'].values
            group = dd['GROUP'].values
            args = (rot, sig_mp, group)

            ## simulate data from best fitting params
            pp = p[s, :-1]
            (y, yff, yfb, xff, xfb) = simulate(pp, args)

            ## try to recover best fitting params
            x_obs_mp = yff
            x_obs_ep = y
            args = (rot, sig_mp, x_obs_mp, x_obs_ep, group)

            results = differential_evolution(func=obj_func_individual,
                                             bounds=bounds,
                                             args=args,
                                             disp=True,
                                             maxiter=maxiter,
                                             tol=1e-15,
                                             polish=True,
                                             updating='deferred',
                                             workers=-1)

            with open(fout[i][:-4] + '_val.txt', 'a') as f:
                tmp = np.concatenate((results['x'], [results['fun']]))
                tmp = np.reshape(tmp, (tmp.shape[0], 1))
                np.savetxt(f, tmp.T, '%0.4f', delimiter=',', newline='\n')


def inspect_validated_fits(fin, fout):

    for i in range(len(fin)):
        print(i)

        pin = np.loadtxt(fin[i], delimiter=',')
        pout = np.loadtxt(fout[i], delimiter=',')

        pin = pin[:, :-1]
        pout = pout[:, :-1]

        pin[:, -3:] /= 5
        pout[:, -3:] /= 5

        names = [
            'alpha_ff', 'beta_ff', 'alpha_fb', 'beta_fb', 'base_fb', 'w',
            'gamma_ff', 'gamma_fb', 'gamma_fb2'
        ]

        fig, ax = plt.subplots(3, 3, figsize=(10,10))
        ax = ax.flatten()
        for j in range(pin.shape[1]):

            ax[j].plot(pin[:, j], pout[:, j], '.')
            ax[j].plot([-1, 1], [-1, 1], '--k', alpha=0.5)
            ax[j].set_title(names[j])

        plt.tight_layout()
        # plt.show()
        plt.savefig('../figures/fit_val_' + str(i) +'.pdf')


def inspect_fits_individual_all(fin, fout):

    fig, ax = plt.subplots(len(fin), 3, figsize=(15, 10), squeeze=False)

    for i in range(len(fin)):

        d = pd.read_csv(fin[i])

        dd = d[['TRIAL_ABS', 'HA_INIT', 'HA_END', 'ROT', 'SIG_MP',
                'GROUP']].groupby(['TRIAL_ABS']).mean()
        dd.reset_index(inplace=True)

        x_obs_mp = dd['HA_INIT'].values
        x_obs_ep = dd['HA_END'].values

        p = np.loadtxt(fout[i], delimiter=',')

        subs = d['SUBJECT'].unique()
        yrec = np.zeros((subs.shape[0], dd.shape[0]))
        yffrec = np.zeros((subs.shape[0], dd.shape[0]))
        yfbrec = np.zeros((subs.shape[0], dd.shape[0]))
        xffrec = np.zeros((subs.shape[0], dd.shape[0]))
        xfbrec = np.zeros((subs.shape[0], dd.shape[0]))

        for s in range(subs.shape[0]):
            dd = d[d['SUBJECT'] == subs[s]]
            rot = dd['ROT'].values
            sig_mp = dd['SIG_MP'].values
            group = dd['GROUP'].values
            args = (rot, sig_mp, group)
            pp = p[s, :-1]
            (y, yff, yfb, xff, xfb) = simulate(pp, args)
            yrec[s, :] = y
            yffrec[s, :] = yff

        y = np.mean(yrec, axis=0)
        yff = np.mean(yffrec, axis=0)
        yfb = np.mean(yfbrec, axis=0)
        xff = np.mean(xffrec, axis=0)
        xfb = np.mean(xfbrec, axis=0)

        # ax[i, 0].plot(np.arange(0, rot.shape[0], 1), -rot, '-k', alpha=1)
        ax[i, 0].plot(np.arange(0, rot.shape[0], 1), y, '.C0', alpha=.5)
        ax[i, 0].plot(np.arange(0, rot.shape[0], 1), yff, '.C1', alpha=.5)
        ax[i, 0].plot(np.arange(0, rot.shape[0], 1), x_obs_ep, 'C0', alpha=0.5)
        ax[i, 0].plot(np.arange(0, rot.shape[0], 1), x_obs_mp, 'C1', alpha=0.5)
        # ax[i, 0].legend(['Model EP', 'Model MP', 'Human EP', 'Human MP'])
        # ax[i, 0].set_xlabel('Trial')
        # ax[i, 0].set_ylabel('Hand Angle')
        # ax[i, 0].set_ylim([-10, 35])

        ax[i, 1].plot(np.arange(0, rot.shape[0], 1), y, '-', alpha=1)
        ax[i, 1].plot(np.arange(0, rot.shape[0], 1), yff, '-', alpha=1)
        ax[i, 1].plot(np.arange(0, rot.shape[0], 1), yfb, '-', alpha=1)
        ax[i, 1].plot(np.arange(0, rot.shape[0], 1), xff, '-', alpha=1)
        ax[i, 1].plot(np.arange(0, rot.shape[0], 1), xfb, '-', alpha=1)
        ax[i, 1].legend(['y', 'yff', 'yfb', 'xff', 'xfb'], loc='upper right')
        # ax[i, 1].set_xlabel('Trial')
        # ax[i, 1].set_ylabel('Hand Angle (degrees)')

        # p[:, -3:-1] /= 10

        xticks = np.arange(1, p.shape[1], 1)
        ax[i, 2].violinplot(p[:, :-1])
        for ii in range(p.shape[0]):
            ax[i, 2].plot(xticks, p[ii, :-1], '.k')
        ax[i, 2].plot([1, p.shape[1]], [0, 0], '--k', alpha=0.25)
        ax[i, 2].set_xticks(xticks)
        ax[i, 2].set_xticklabels([
            'alpha_ff', 'beta_ff', 'alpha_fb', 'beta_fb', 'base_fb', 'w',
            'gamma_ff', 'gamma_fb', 'gamma_fb2'
        ],
                                 rotation=45,
                                 ha="right")

    # plt.show()
    plt.savefig('../figures/fit_summary.pdf')


def inspect_fits_individual():

    p = np.loadtxt('../fits/fit_individual.txt')

    fig, ax = plt.subplots(nrows=6, ncols=4, figsize=(10, 6))
    ax = ax.flatten()

    d3 = pd.read_csv('../datta/MRES_ADAPT.csv')
    subs = d3['SUBJECT'].unique()
    # subs = subs[0:2]
    for i in range(subs.shape[0]):

        sub = subs[i]

        dd3 = d3[d3['SUBJECT'] == sub][[
            'ROTATION', 'HA_INIT', 'HA_END', 'TRIAL_ABS'
        ]]

        dd3.reset_index(inplace=True)

        rot3 = -dd3.ROTATION.values
        sig_mp3 = d3[d3.SUBJECT == sub].SIG_MP.values

        x_obs_mp3 = dd3['HA_INIT'].values
        x_obs_ep3 = dd3['HA_END'].values

        n_params = 10
        start = i * n_params
        stop = start + n_params

        (y3, yff3, yfb3, xff3, xfb3) = simulate(p[start:stop], (rot3, sig_mp3),
                                                group=3)

        ax[i].plot(np.arange(0, rot3.shape[0], 1), y3, '.C0', alpha=.1)
        ax[i].plot(np.arange(0, rot3.shape[0], 1), yff3, '.C1', alpha=.1)
        ax[i].plot(np.arange(0, rot3.shape[0], 1), x_obs_ep3, 'C0', alpha=0.1)
        ax[i].plot(np.arange(0, rot3.shape[0], 1), x_obs_mp3, 'C1', alpha=0.1)
        # ax[i].legend(['Model EP', 'Model MP', 'Human EP', 'Human MP'])
        # ax[i].set_xlabel('Trial')
        # ax[i].set_ylabel('Hand Angle (degrees)')
        # ax[i].set_ylim([-5, 35])

    # plt.tight_layout()
    # plt.show()
    # plt.savefig('../figures/fig_results_individual.pdf')
    # plt.close()

    p = np.loadtxt('../fits/fit_individual.txt')
    n_cols = n_params
    n_rows = p.shape[0] // n_cols
    p = np.reshape(p, (n_rows, n_cols))

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.violinplot(p[:, 0:-1], np.arange(1, n_params, 1))
    plt.plot([1, n_params], [0, 0], '--k', alpha=0.25)
    ax.set_xticks(np.arange(1, n_cols, 1))
    ax.set_xticklabels([
        'alpha_ff', 'beta_ff', 'alpha_fb', 'beta_fb', 'base_fb', 'w',
        'gamma_ff', 'gamma_fb', 'gamma_fb2'
    ])
    plt.show()


def inspect_fits():

    obs = get_observed()

    rot0 = obs[0][0]
    sig_mp0 = obs[0][1]
    x_obs_mp0 = obs[0][2]
    x_obs_ep0 = obs[0][3]

    rot1 = obs[1][0]
    sig_mp1 = obs[1][1]
    x_obs_mp1 = obs[1][2]
    x_obs_ep1 = obs[1][3]

    rot2 = obs[2][0]
    sig_mp2 = obs[2][1]
    x_obs_mp2 = obs[2][2]
    x_obs_ep2 = obs[2][3]

    p = np.loadtxt('../fits/fit.txt')

    (y0, yff0, yfb0, xff0, xfb0) = simulate(p[0:-1], (rot0, sig_mp0), group=0)
    (y1, yff1, yfb1, xff1, xfb1) = simulate(p[0:-1], (rot1, sig_mp1), group=1)
    (y2, yff2, yfb2, xff2, xfb2) = simulate(p[0:-1], (rot2, sig_mp2), group=2)

    # fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(10, 6))
    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(3, 3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])
    ax7 = fig.add_subplot(gs[2, :])
    ax = np.array([ax1, ax2, ax3, ax4, ax5, ax6, ax7])

    ax[0].plot(np.arange(0, rot0.shape[0], 1), y0, '.C0', alpha=1)
    ax[0].plot(np.arange(0, rot0.shape[0], 1), yff0, '.C1', alpha=1)
    ax[0].plot(np.arange(0, rot0.shape[0], 1), x_obs_ep0, 'C0', alpha=0.25)
    ax[0].plot(np.arange(0, rot0.shape[0], 1), x_obs_mp0, 'C1', alpha=0.25)
    ax[0].legend(['Model EP', 'Model MP', 'Human EP', 'Human MP'])
    ax[0].set_xlabel('Trial')
    ax[0].set_ylabel('Hand Angle (degrees)')
    ax[0].set_ylim([-5, 35])

    ax[1].plot(np.arange(0, rot1.shape[0], 1), y1, '.C0', alpha=1)
    ax[1].plot(np.arange(0, rot1.shape[0], 1), yff1, '.C1', alpha=1)
    ax[1].plot(np.arange(0, rot1.shape[0], 1), x_obs_ep1, 'C0', alpha=0.25)
    ax[1].plot(np.arange(0, rot1.shape[0], 1), x_obs_mp1, 'C1', alpha=0.25)
    ax[1].legend(['Model EP', 'Model MP', 'Human EP', 'Human MP'])
    ax[1].set_xlabel('Trial')
    ax[1].set_ylabel('Hand Angle (degrees)')
    ax[1].set_ylim([-5, 35])

    ax[2].plot(np.arange(0, rot2.shape[0], 1), y2, '.C0', alpha=1)
    ax[2].plot(np.arange(0, rot2.shape[0], 1), yff2, '.C1', alpha=1)
    ax[2].plot(np.arange(0, rot2.shape[0], 1), x_obs_ep2, 'C0', alpha=0.25)
    ax[2].plot(np.arange(0, rot2.shape[0], 1), x_obs_mp2, 'C1', alpha=0.25)
    ax[2].legend(['Model EP', 'Model MP', 'Human EP', 'Human MP'])
    ax[2].set_xlabel('Trial')
    ax[2].set_ylabel('Hand Angle (degrees)')
    ax[2].set_ylim([-5, 35])

    ax[3].plot(np.arange(0, rot0.shape[0], 1), y0, '-', alpha=1)
    ax[3].plot(np.arange(0, rot0.shape[0], 1), yff0, '-', alpha=1)
    ax[3].plot(np.arange(0, rot0.shape[0], 1), yfb0, '-', alpha=1)
    ax[3].plot(np.arange(0, rot0.shape[0], 1), xff0, '-', alpha=1)
    ax[3].plot(np.arange(0, rot0.shape[0], 1), xfb0, '-', alpha=1)
    ax[3].legend(['y', 'yff', 'yfb', 'xff', 'xfb'])
    ax[3].set_xlabel('Trial')
    ax[3].set_ylabel('Hand Angle (degrees)')
    ax[3].set_ylim([-5, 35])

    ax[4].plot(np.arange(0, rot1.shape[0], 1), y1, '-', alpha=1)
    ax[4].plot(np.arange(0, rot1.shape[0], 1), yff1, '-', alpha=1)
    ax[4].plot(np.arange(0, rot1.shape[0], 1), yfb1, '-', alpha=1)
    ax[4].plot(np.arange(0, rot1.shape[0], 1), xff1, '-', alpha=1)
    ax[4].plot(np.arange(0, rot1.shape[0], 1), xfb1, '-', alpha=1)
    ax[4].legend(['y', 'yff', 'yfb', 'xff', 'xfb'])
    ax[4].set_xlabel('Trial')
    ax[4].set_ylabel('Hand Angle (degrees)')
    ax[4].set_ylim([-5, 35])

    ax[5].plot(np.arange(0, rot2.shape[0], 1), y2, '-', alpha=1)
    ax[5].plot(np.arange(0, rot2.shape[0], 1), yff2, '-', alpha=1)
    ax[5].plot(np.arange(0, rot2.shape[0], 1), yfb2, '-', alpha=1)
    ax[5].plot(np.arange(0, rot2.shape[0], 1), xff2, '-', alpha=1)
    ax[5].plot(np.arange(0, rot2.shape[0], 1), xfb2, '-', alpha=1)
    ax[5].legend(['y', 'yff', 'yfb', 'xff', 'xfb'])
    ax[5].set_xlabel('Trial')
    ax[5].set_ylabel('Hand Angle (degrees)')
    ax[5].set_ylim([-5, 35])

    ax[6].scatter(np.arange(1, 10, 1), p[:-1])
    ax[6].set_xticks(np.arange(1, 10, 1))
    ax[6].set_xticklabels([
        'alpha_ff', 'beta_ff', 'alpha_fb', 'beta_fb', 'base_fb', 'w',
        'gamma_ff', 'gamma_fb', 'gamma_fb2'
    ])

    boot = True
    if boot == True:

        p = np.loadtxt('../fits/fit_boot.txt')
        n_cols = 10
        n_rows = p.shape[0] // n_cols
        p = np.reshape(p, (n_rows, n_cols))

        ax[6].violinplot(p[:, 0:-1], np.arange(1, 10, 1))

    ax[6].plot([1, 10], [0, 0], '--k', alpha=0.25)
    plt.tight_layout()
    plt.show()
    # plt.savefig('../figures/fig_results.pdf')
    # plt.close()


def fit_boot():

    bounds = ((0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (-10, 10),
              (-10, 10), (-10, 10))

    for i in range(n_boot + 1):

        start = time.time()
        print(i)

        args = get_observed_boot()

        # fit both groups simultaneously
        # i.e., impose same parameters on all groups
        results = differential_evolution(func=obj_func,
                                         bounds=bounds,
                                         args=args,
                                         disp=True,
                                         maxiter=300,
                                         tol=1e-15,
                                         polish=False,
                                         updating='deferred',
                                         workers=-1)

        f_name_p = '../fits/fit_boot.txt'
        with open(f_name_p, 'a') as f:
            np.savetxt(f, np.concatenate((results['x'], [results['fun']])),
                       '%0.4f', ',')

        end = time.time()
        print(end - start)


def callback(xk, convergence):
    print(xk)
    print('\n')

    return False


def fit_individual(fin, fout, bounds, maxiter):
    for i in range(len(fin)):
        d = pd.read_csv(fin[i])
        for sub in d['SUBJECT'].unique():

            dd = d[d['SUBJECT'] == sub][[
                'ROT', 'HA_INIT', 'HA_END', 'TRIAL_ABS', 'GROUP'
            ]]

            dd.reset_index(inplace=True)

            rot = dd.ROT.values
            sig_mp = d[d.SUBJECT == sub].SIG_MP.values
            group = dd.GROUP.values
            x_obs_mp = dd['HA_INIT'].values
            x_obs_ep = dd['HA_END'].values

            args = (rot, sig_mp, x_obs_mp, x_obs_ep, group)

            results = differential_evolution(
                func=obj_func_individual,
                bounds=bounds,
                args=args,
                # callback=callback,
                disp=True,
                maxiter=maxiter,
                tol=1e-15,
                polish=True,
                updating='deferred',
                workers=-1)

            # print(results['x'])
            # (y, yff, yfb, xff, xfb) = simulate(results['x'], args)
            # fig, ax = plt.subplots(2, 1)
            # ax[0].plot(rot, 'k', alpha = 1)
            # ax[0].plot(x_obs_mp, 'C1', alpha = 0.2)
            # ax[0].plot(x_obs_ep, 'C0', alpha = 0.2)
            # ax[0].plot(yff, '.C1', alpha = 0.2)
            # ax[0].plot(y, '.C0', alpha = 0.2)
            # ax[1].plot(xff, '.C1', alpha = 0.2)
            # ax[1].plot(xfb, '.C0', alpha = 0.2)
            # plt.show()

            with open(fout[i], 'a') as f:
                tmp = np.concatenate((results['x'], [results['fun']]))
                tmp = np.reshape(tmp, (tmp.shape[0], 1))
                np.savetxt(f, tmp.T, '%0.4f', delimiter=',', newline='\n')


def fit():

    # alpha_ff = params[0]
    # beta_ff = params[1]
    # alpha_fb = params[2]
    # beta_fb = params[3]
    # base_fb = params[4]
    # w = params[5]
    # gamma_ff = params[6]
    # gamma_fb = params[7]
    # gamma_fb2 = params[8]

    bounds = ((0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (-10, 10),
              (-10, 10), (-10, 10))

    args = get_observed()

    # fit both groups simultaneously
    # i.e., impose same parameters on all groups
    results = differential_evolution(func=obj_func,
                                     bounds=bounds,
                                     args=args,
                                     disp=True,
                                     maxiter=300,
                                     tol=1e-15,
                                     polish=False,
                                     updating='deferred',
                                     workers=-1)

    f_name_p = '../fits/fit.txt'
    with open(f_name_p, 'w') as f:
        np.savetxt(f, np.concatenate((results['x'], [results['fun']])),
                   '%0.4f', ',')


def obj_func(params, *args):

    obs = args

    rot0 = obs[0][0]
    sig_mp0 = obs[0][1]
    x_obs_mp0 = obs[0][2]
    x_obs_ep0 = obs[0][3]

    rot1 = obs[1][0]
    sig_mp1 = obs[1][1]
    x_obs_mp1 = obs[1][2]
    x_obs_ep1 = obs[1][3]

    rot2 = obs[2][0]
    sig_mp2 = obs[2][1]
    x_obs_mp2 = obs[2][2]
    x_obs_ep2 = obs[2][3]

    args0 = (rot0, sig_mp0)
    args1 = (rot1, sig_mp1)
    args2 = (rot2, sig_mp2)

    x_pred0 = simulate(params, args0, group=0)
    x_pred_mp0 = x_pred0[1]
    x_pred_ep0 = x_pred0[0]

    x_pred1 = simulate(params, args1, group=1)
    x_pred_mp1 = x_pred1[1]
    x_pred_ep1 = x_pred1[0]

    x_pred2 = simulate(params, args2, group=2)
    x_pred_mp2 = x_pred2[1]
    x_pred_ep2 = x_pred2[0]

    sse_mp0 = np.sum((x_obs_mp0 - x_pred_mp0)**2)
    sse_ep0 = np.sum((x_obs_ep0 - x_pred_ep0)**2)
    sse_mp1 = np.sum((x_obs_mp1 - x_pred_mp1)**2)
    sse_ep1 = np.sum((x_obs_ep1 - x_pred_ep1)**2)
    sse_mp2 = np.sum((x_obs_mp2 - x_pred_mp2)**2)
    sse_ep2 = np.sum((x_obs_ep2 - x_pred_ep2)**2)

    w = 0.5

    sse = w * sse_mp0 + (1 - w) * sse_ep0 + w * sse_mp1 + (
        1 - w) * sse_ep1 + w * sse_mp2 + (1 - w) * sse_ep2

    return sse


def obj_func_individual(params, *args):

    obs = args

    rot = obs[0]
    sig_mp = obs[1]
    x_obs_mp = obs[2]
    x_obs_ep = obs[3]
    group = obs[4]

    args = (rot, sig_mp, group)

    x_pred = simulate(params, args)
    x_pred_mp = x_pred[1]
    x_pred_ep = x_pred[0]

    sse_mp = np.sum((x_obs_mp - x_pred_mp)**2)
    sse_ep = np.sum((x_obs_ep - x_pred_ep)**2)

    sse = sse_mp + sse_ep

    return sse


def get_observed():

    # read exp 1 data
    d1 = pd.read_csv('../datta/Bayes_SML_EXP1_200820.csv')
    d1 = d1[d1['TRIAL_ABS'] < 191]
    rot1 = d1[d1.SUBJECT == 1].ROT.values
    sig_mp1 = d1[d1.SUBJECT == 1].SIG_MP.values
    dd1 = d1[['GROUP', 'TRIAL_ABS', 'HA_INIT', 'HA_END', 'ROT',
              'SIG_MP']].groupby(['GROUP', 'TRIAL_ABS']).mean()
    dd1.reset_index(inplace=True)

    # read exp 2 data
    d2 = pd.read_csv('../datta/Bayes_SML_EXP2.csv')
    d2 = d2[d2['TRIAL_ABS'] < 191]
    rot2 = d2[d2.SUBJECT == 1].ROT.values
    sig_mp2 = d2[d2.SUBJECT == 1].SIG_MP.values
    dd2 = d2[['ROT', 'HA_INIT', 'HA_END',
              'TRIAL_ABS']].groupby(['TRIAL_ABS']).mean()
    dd2.reset_index(inplace=True)

    x_obs_mp0 = dd1[dd1['GROUP'] == 0]['HA_INIT'].values
    x_obs_ep0 = dd1[dd1['GROUP'] == 0]['HA_END'].values
    x_obs_mp1 = dd1[dd1['GROUP'] == 1]['HA_INIT'].values
    x_obs_ep1 = dd1[dd1['GROUP'] == 1]['HA_END'].values
    x_obs_mp2 = dd2['HA_INIT'].values
    x_obs_ep2 = dd2['HA_END'].values

    args1 = (rot1, sig_mp1, x_obs_mp0, x_obs_ep0)
    args2 = (rot1, sig_mp1, x_obs_mp1, x_obs_ep1)
    args3 = (rot2, sig_mp2, x_obs_mp2, x_obs_ep2)

    obs = (args1, args2, args3)

    return obs


def get_observed_boot():

    # read exp 1 data
    d1 = pd.read_csv('../datta/Bayes_SML_EXP1_200820.csv')
    rot1 = d1[d1.SUBJECT == 1].ROT.values
    sig_mp1 = d1[d1.SUBJECT == 1].SIG_MP.values

    # sample with replacement for bootstrapping
    subs1 = d1.SUBJECT.unique()
    subs_boot1 = np.random.choice(subs1, subs1.shape[0], replace=True)
    d_boot_rec1 = []
    for i in subs_boot1:
        d_boot_rec1.append(d1[d1['SUBJECT'] == i])
    d_boot1 = pd.concat(d_boot_rec1)
    dd1 = d_boot1[['GROUP', 'TRIAL_ABS', 'HA_INIT', 'HA_END', 'ROT',
                   'SIG_MP']].groupby(['GROUP', 'TRIAL_ABS']).mean()
    dd1.reset_index(inplace=True)

    # read exp 2 data
    d2 = pd.read_csv('../datta/Bayes_SML_EXP2.csv')
    rot2 = d2[d2.SUBJECT == 1].ROT.values
    sig_mp2 = d2[d2.SUBJECT == 1].SIG_MP.values

    dd2 = d2[['ROT', 'HA_INIT', 'HA_END',
              'TRIAL_ABS']].groupby(['TRIAL_ABS']).mean()
    dd2.reset_index(inplace=True)

    # sample with replacement for bootstrapping
    subs2 = d2.SUBJECT.unique()
    subs_boot2 = np.random.choice(subs2, subs2.shape[0], replace=True)
    d_boot_rec2 = []
    for i in subs_boot2:
        d_boot_rec2.append(d2[d2['SUBJECT'] == i])
    d_boot2 = pd.concat(d_boot_rec2)
    dd2 = d_boot2[['TRIAL_ABS', 'HA_INIT', 'HA_END', 'ROT',
                   'SIG_MP']].groupby(['TRIAL_ABS']).mean()
    dd2.reset_index(inplace=True)

    x_obs_mp0 = dd1[dd1['GROUP'] == 0]['HA_INIT'].values
    x_obs_ep0 = dd1[dd1['GROUP'] == 0]['HA_END'].values
    x_obs_mp1 = dd1[dd1['GROUP'] == 1]['HA_INIT'].values
    x_obs_ep1 = dd1[dd1['GROUP'] == 1]['HA_END'].values
    x_obs_mp2 = dd2['HA_INIT'].values
    x_obs_ep2 = dd2['HA_END'].values

    args1 = (rot1, sig_mp1, x_obs_mp0, x_obs_ep0)
    args2 = (rot1, sig_mp1, x_obs_mp1, x_obs_ep1)
    args3 = (rot2, sig_mp2, x_obs_mp2, x_obs_ep2)

    obs = (args1, args2, args3)

    return obs


def plot_slopes():
    pass
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


def simulate(params, args):

    alpha_ff = params[0]
    beta_ff = params[1]
    alpha_fb = params[2]
    beta_fb = params[3]
    base_fb = params[4]
    w = params[5]
    gamma_ff = params[6]
    gamma_fb = params[7]
    gamma_fb2 = params[8]

    r = args[0]
    sig_mp = args[1]
    group = args[2]

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

        if sig_mp[i] != 4:
            bayes_mod_ff = bayes_int(sig_mp[i], gamma_ff)
            bayes_mod_fb = bayes_int(sig_mp[i], gamma_fb)
            bayes_mod_fb2 = bayes_int(sig_mp[i], gamma_fb2)
        else:
            bayes_mod_ff = 0.0
            bayes_mod_fb = 0.0
            bayes_mod_fb2 = 0.0

        # midpoint to endpoint
        if sig_mp[i] != 4:
            delta_mp[i] = 0.0 - (y[i] + r[i])
            yfb[i] = xfb[i] * delta_mp[i] * bayes_mod_fb2
        else:
            delta_mp[i] = 0.0
            yfb[i] = 0.0

        y[i] = yff[i] + yfb[i]

        if sig_mp[i] == 1:
            delta_ep[i] = 0.0 - (y[i] + r[i])
        else:
            delta_ep[i] = 0.0

        if group[i] == 1:
            delta_ep[i] = 0.0

        xff[i + 1] = beta_ff * xff[i] + bayes_mod_ff * alpha_ff * (
            w * delta_mp[i] + (1 - w) * delta_ep[i])

        xfb[i + 1] = beta_fb * xfb[i] + bayes_mod_fb * alpha_fb * delta_ep[
            i] + base_fb

        xfb = np.clip(xfb, -1.5, 1.5)

    return (y, yff, yfb, xff, xfb)


def bayes_int(x, m):

    # x = np.arange(1, 3, 0.01)
    # plt.plot(x, np.tanh(1 * (x - 2)) / 2 + 0.5)
    # plt.show()

    return np.tanh(m * (x - 2)) / 2 + 0.5


def bootstrap_ci(x, n, alpha):
    x_boot = np.zeros(n)
    for i in range(n):
        x_boot[i] = np.random.choice(x, x.shape, replace=True).mean()
        ci = np.percentile(x_boot, [alpha / 2, 1.0 - alpha / 2])
    return (ci)


def bootstrap_t(x_obs, y_obs, x_samp_dist, y_samp_dist, n):
    d_obs = x_obs - y_obs

    d_boot = np.zeros(n)
    xs = np.random.choice(x_samp_dist, n, replace=True)
    ys = np.random.choice(y_samp_dist, n, replace=True)
    d_boot = xs - ys
    d_boot = d_boot - d_boot.mean()

    p_null = (1 + np.sum(np.abs(d_boot) > np.abs(d_obs))) / (n + 1)
    return (p_null)


# [alpha_ff, beta_ff, alpha_fb, beta_fb, base_fb, w, gamma_ff, gamma_fb, gamma_fb2]
b = ((0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (-5, 5), (-5, 5), (-5, 5))
m = 1000

fin1 = '../datta/exp1.csv'
fin2 = '../datta/exp2.csv'
fin3 = '../datta/exp3.csv'
fin4 = '../datta/exp4.csv'
fin = [fin1, fin2, fin3, fin4]

fout1 = '../fits/fit_individual_1.txt'
fout2 = '../fits/fit_individual_2.txt'
fout3 = '../fits/fit_individual_3.txt'
fout4 = '../fits/fit_individual_4.txt'
fout = [fout1, fout2, fout3, fout4]

# fit_individual(fin, fout, b, m)
inspect_fits_individual_all(fin, fout)
# fit_validate(fin, fout, b, m)

fin1 = '../fits/fit_individual_1.txt'
fin2 = '../fits/fit_individual_2.txt'
fin3 = '../fits/fit_individual_3.txt'
fin4 = '../fits/fit_individual_4.txt'
fin = [fin1, fin2, fin3, fin4]

fout1 = '../fits/fit_individual_1_val.txt'
fout2 = '../fits/fit_individual_2_val.txt'
fout3 = '../fits/fit_individual_3_val.txt'
fout4 = '../fits/fit_individual_4_val.txt'
fout = [fout1, fout2, fout3, fout4]

inspect_validated_fits(fin, fout)
