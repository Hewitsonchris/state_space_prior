import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import differential_evolution
from scipy.optimize import LinearConstraint
from scipy.stats import ttest_1samp
from scipy.stats import sem
from scipy.stats import norm


def load_all_data():
    d1 = pd.read_csv('../datta/exp1.csv')
    d2 = pd.read_csv('../datta/exp2.csv')
    d3 = pd.read_csv('../datta/exp3.csv')
    d4 = pd.read_csv('../datta/exp4.csv')
    d5 = pd.read_csv('../datta/exp5.csv')
    d6 = pd.read_csv('../datta/exp6.csv')

    d = pd.concat((d1, d2, d3, d4, d5, d6), sort=False)

    return d


def inspect_behaviour():

    d = load_all_data()

    dd = d.groupby(['GROUP', 'TRIAL_ABS'])[['HA_INIT',
                                            'HA_END']].mean().reset_index()

    # Group 0: sparse EP + no feedback washout
    # Group 1: no EP + no feedback washout
    # Group 2: all EP + 0 deg uniform washout
    # Group 3: KW rep + right hand transfer
    # Group 4: KW rep + left hand transfer
    # Group 5: KW rep + left hand transfer + opposite perturb
    # Group 6: Same as Group 0 + relearn + left hand 0 deg uniform
    # Group 7: unimodal -- uni likelihood -- (N=20, groups 7 and 8) no fb wash
    # Group 8: unimodal -- uni likelihood -- (N=20, groups 7 and 8) no fb wash
    # Group 9: bimodal predictable (N=20, groups 9 and10) + no fb wash
    # Group 10: bimodal predictable (N=20, groups 9 and10 + no fb wash
    # Group 11: bimodal stochastic (N=12, groups 11 and 12) + no fb wash
    # Group 12: bimodal stochastic (N=12, groups 11 and 12) + no fb wash

    fig = plt.figure(figsize=(10, 5))
    gs = fig.add_gridspec(3, 5)
    g = 0
    for i in range(3):
        for j in range(5):
            ax = fig.add_subplot(gs[i, j])
            ddd = dd[dd['GROUP'] == g]
            ax.plot(ddd['TRIAL_ABS'], ddd['HA_END'], '.', alpha=0.75)
            ax.plot(ddd['TRIAL_ABS'], ddd['HA_INIT'], '.', alpha=0.75)
            ax.set_title('Group ' + str(ddd['GROUP'].unique()))
            # ax.set_ylim([-30, 30])
            g += 1
    plt.tight_layout()
    plt.show()


def fit_validate(d, bounds, maxiter, polish, froot):

    for sub in d['SUBJECT'].unique():

        dd = d[d['SUBJECT'] == sub][[
            'ROT', 'HA_INIT', 'HA_END', 'TRIAL_ABS', 'GROUP', 'SIG_MP'
        ]]

        rot = d.ROT.values
        sig_mp = dd.SIG_MP.values
        group = d.GROUP.values
        x_obs_mp = d['HA_INIT'].values
        x_obs_ep = d['HA_END'].values
        args = (rot, sig_mp, x_obs_mp, x_obs_ep, group)

        p = np.loadtxt(froot + str(sub) + '.txt', delimiter=',')

        # simulate data from best fitting params
        (y, yff, yfb, xff, xfb) = simulate(p, args)

        args = (rot, sig_mp, yff, y, group)

        results = differential_evolution(func=obj_func,
                                         bounds=bounds,
                                         args=args,
                                         disp=True,
                                         maxiter=maxiter,
                                         tol=1e-15,
                                         polish=p,
                                         updating='deferred',
                                         workers=-1)

        fout = froot + str(sub) + '_val.txt'
        with open(fout, 'w') as f:
            tmp = np.concatenate((results['x'], [results['fun']]))
            tmp = np.reshape(tmp, (tmp.shape[0], 1))
            np.savetxt(f, tmp.T, '%0.4f', delimiter=',', newline='\n')


def inspect_fits_validate(d, froot):

    for sub in d['SUBJECT'].unique():

        pin = np.loadtxt(froot + str(sub) + '.txt', delimiter=',')
        pout = np.loadtxt(froot + str(sub) + '_val.txt', delimiter=',')

        pin = pin[:, :-1]
        pout = pout[:, :-1]

        names = [
            'alpha_ff', 'beta_ff', 'alpha_fb', 'beta_fb', 'w', 'gamma_ff',
            'gamma_fb', 'gamma_fb2', 'xfb_init'
        ]

        fig, ax = plt.subplots(3, 3, figsize=(10, 10))
        ax = ax.flatten()
        for j in range(pin.shape[1]):

            ax[j].plot(pin[:, j], pout[:, j], '.')
            ax[j].plot([-1, 1], [-1, 1], '--k', alpha=0.5)
            ax[j].set_title(names[j])

        plt.tight_layout()
        # plt.show()
        plt.savefig('../figures/fit_val_' + str(i) + '.pdf')


def inspect_fits_individual(d, froot):

    n_params = 9

    fig = plt.figure(figsize=(10, 5))
    gs = fig.add_gridspec(1, 1)
    ax1 = fig.add_subplot(gs[0, :])
    ax = np.array([ax1])

    dd = d[['TRIAL_ABS', 'HA_INIT', 'HA_END', 'ROT', 'SIG_MP',
            'GROUP']].groupby(['TRIAL_ABS']).mean()
    dd.reset_index(inplace=True)

    x_obs_mp = dd['HA_INIT'].values
    x_obs_ep = dd['HA_END'].values

    subs = d['SUBJECT'].unique()
    prec = np.zeros((subs.shape[0], n_params))
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

        fname = froot + str(s + 1) + '.txt'
        p = np.loadtxt(fname, delimiter=',')
        (y, yff, yfb, xff, xfb) = simulate(p[:-1], args)

        prec[s, :] = p[:-1]
        yrec[s, :] = y
        yffrec[s, :] = yff
        yfbrec[s, :] = yfb
        xffrec[s, :] = xff
        xfbrec[s, :] = xfb

    y = np.mean(yrec, axis=0)
    yff = np.mean(yffrec, axis=0)
    yfb = np.mean(yfbrec, axis=0)
    xff = np.mean(xffrec, axis=0)
    xfb = np.mean(xfbrec, axis=0)

    y[-1] = y[-2]
    yff[-1] = yff[-2]
    yfb[-1] = yfb[-2]
    xff[-1] = xff[-2]
    xfb[-1] = xfb[-2]

    ax[0].plot(np.arange(0, rot.shape[0], 1), y, '.C0', alpha=.5)
    ax[0].plot(np.arange(0, rot.shape[0], 1), yff, '.C1', alpha=.5)
    ax[0].plot(np.arange(0, rot.shape[0], 1), x_obs_ep, 'C0', alpha=0.5)
    ax[0].plot(np.arange(0, rot.shape[0], 1), x_obs_mp, 'C1', alpha=0.5)
    ax[0].legend(['Model EP', 'Model MP', 'Human EP', 'Human MP'])
    ax[0].set_xlabel('Trial')
    ax[0].set_ylabel('Hand Angle')

    # plt.show()
    plt.savefig('../figures/results_group.pdf')
    plt.close('all')

    # report R-squared
    print('\n')

    ss_tot_mp = np.nansum((x_obs_mp - np.nanmean(x_obs_mp))**2)
    ss_reg_mp = np.nansum((yff - np.nanmean(x_obs_mp))**2)
    ss_res_mp = np.nansum((x_obs_mp - yff)**2)
    r_squared_mp = 1 - ss_res_mp / ss_tot_mp
    print('MP: $R^2 = {:3.2f}$'.format(r_squared_mp))

    ss_tot_ep = np.nansum((x_obs_ep - np.nanmean(x_obs_ep))**2)
    ss_reg_ep = np.nansum((y - np.nanmean(x_obs_ep))**2)
    ss_res_ep = np.nansum((x_obs_ep - y)**2)
    r_squared_ep = 1 - ss_res_ep / ss_tot_ep
    print('EP: $R^2 = {:3.2f}$'.format(r_squared_ep))

    print('SSE = ', str(prec[:, -1].mean()))

    # report statistics
    tstat, pval = ttest_1samp(prec, popmean=0, axis=0)
    d = np.mean(prec, axis=0) / np.std(prec, axis=0, ddof=1)

    pname = [
        'alpha_ff', 'beta_ff', 'alpha_fb', 'beta_fb', 'w', 'gamma_ff',
        'gamma_fb', 'gamma_fb2', 'base_fb', 'base_ff', 'base_fb2'
    ]

    print('\n')
    for j in range(5, 8):
        print(pname[j] + ' = ' + str(np.round(prec[:, j].mean(), 2)) + ': t(' +
              str(prec.shape[0] - 1) + ') = ' + str(np.round(tstat[j], 2)) +
              ', p = ' + str(np.round(pval[j], 2)) + ', d = ' +
              str(np.round(d[j], 2)))


def inspect_fits_boot(group):

    d = load_all_data()


def fit_individual(d, bounds, maxiter, polish, froot):

    for sub in d['SUBJECT'].unique():

        dd = d[d['SUBJECT'] == sub][[
            'ROT', 'HA_INIT', 'HA_END', 'TRIAL_ABS', 'GROUP', 'SIG_MP'
        ]]

        rot = dd.ROT.values
        sig_mp = dd.SIG_MP.values
        group = dd.GROUP.values
        x_obs_mp = dd['HA_INIT'].values
        x_obs_ep = dd['HA_END'].values

        args = (rot, sig_mp, x_obs_mp, x_obs_ep, group)

        results = differential_evolution(func=obj_func,
                                         bounds=bounds,
                                         args=args,
                                         disp=True,
                                         maxiter=maxiter,
                                         tol=1e-15,
                                         polish=p,
                                         updating='deferred',
                                         workers=-1)

        fout = froot + str(sub) + '.txt'
        with open(fout, 'w') as f:
            tmp = np.concatenate((results['x'], [results['fun']]))
            tmp = np.reshape(tmp, (tmp.shape[0], 1))
            np.savetxt(f, tmp.T, '%0.4f', delimiter=',', newline='\n')


def fit_boot(group, bounds, maxiter, polish, n_boot_samp):

    d = load_all_data()

    for grp in group:

        dd = d[d['GROUP'] == grp]
        dd.sort_values('TRIAL_ABS', inplace=True)

        for n in range(n_boot_samp):

            boot_subs = np.random.choice(d['SUBJECT'].unique(),
                                         size=d['SUBJECT'].unique().shape[0],
                                         replace=True)

            ddd = []
            for i in range(boot_subs.shape[0]):

                ddd.append(dd[dd['SUBJECT'] == boot_subs[i]][[
                    'ROT', 'SIG_MP', 'HA_INIT', 'HA_END', 'TRIAL_ABS', 'GROUP'
                ]])

            ddd = pd.concat(ddd)
            ddd = ddd.groupby('TRIAL_ABS').mean().reset_index()
            ddd.sort_values('TRIAL_ABS', inplace=True)

            rot = ddd.ROT.values
            sig_mp = ddd.SIG_MP.values
            group = ddd.GROUP.values
            x_obs_mp = ddd['HA_INIT'].values
            x_obs_ep = ddd['HA_END'].values

            args = (rot, sig_mp, x_obs_mp, x_obs_ep, group)

            results = differential_evolution(func=obj_func,
                                             bounds=bounds,
                                             args=args,
                                             disp=True,
                                             maxiter=maxiter,
                                             tol=1e-15,
                                             polish=p,
                                             updating='deferred',
                                             workers=-1)

            fname = '...fits/fit_' + str(grp) + '_boot.txt'
            with open(fname, 'a') as f:
                tmp = np.concatenate((results['x'], [results['fun']]))
                tmp = np.reshape(tmp, (tmp.shape[0], 1))
                np.savetxt(f, tmp.T, '%0.4f', delimiter=',', newline='\n')


def obj_func(params, *args):

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

    sse_mp = 100 * np.sum((x_obs_mp[:100] - x_pred_mp[:100])**2)
    sse_ep = 100 * np.sum((x_obs_ep[:100] - x_pred_ep[:100])**2)
    sse_mp += np.sum((x_obs_mp[100:] - x_pred_mp[100:])**2)
    sse_ep += np.sum((x_obs_ep[100:] - x_pred_ep[100:])**2)

    sse = sse_mp + sse_ep

    return sse


def simulate(params, args):

    alpha_ff = params[0]
    beta_ff = params[1]
    alpha_fb = params[2]
    beta_fb = params[3]
    w = params[4]
    gamma_ff = params[5]
    gamma_fb = params[6]
    gamma_fb2 = params[7]
    # base_fb = params[8]
    # base_ff = params[9]
    # base_fb_2 = params[10]

    base_fb = 0.0
    base_ff = 0.0
    base_fb_2 = 0.0

    xfb_init = params[8]

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

    # xfb[0] = base_fb
    xfb[0] = xfb_init

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
            yfb[i] = xfb[i] * delta_mp[i] * bayes_mod_fb2 + base_fb_2
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
            w * delta_mp[i] + (1 - w) * delta_ep[i]) - alpha_ff * base_ff

        xfb[i + 1] = beta_fb * xfb[i] + bayes_mod_fb * alpha_fb * delta_ep[
            i] - alpha_fb * base_fb

        xfb = np.clip(xfb, -2, 2)

    return (y, yff, yfb, xff, xfb)


def bayes_int(x, m):

    # x = np.arange(1, 3, 0.01)
    # plt.plot(x, np.tanh(-1 * (x - 2)) / 2 + 0.5)
    # plt.plot(x, np.tanh(-2 * (x - 2)) / 2 + 0.5)
    # plt.plot(x, np.tanh(-5 * (x - 2)) / 2 + 0.5)
    # plt.plot(x, np.tanh(-10 * (x - 2)) / 2 + 0.5)
    # plt.ylim([-0.01, 1.01])
    # plt.ylabel('f(x)')
    # plt.xlabel('\sigma')
    # plt.legend(['\gamma=-1', '\gamma=-2', '\gamma=-5', '\gamma=-10'])
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


b = ((0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (-50, 50), (-50, 50), (-50, 50),
     (-2, 2))
p = False
nboot = -1

d = load_all_data()
dd = d.set_index('GROUP', drop=False).loc[[3, 4, 5]]
dd = dd.set_index('PHASE', drop=False).loc[['ADAPTATION']]

m = 1000
froot = '../fits/fit_kw_adapt_1000'
# fit_individual(dd, b, m, p, froot)
inspect_fits_individual(dd, froot)

m = 3000
froot = '../fits/fit_kw_adapt_3000'
# fit_individual(dd, b, m, p, froot)
inspect_fits_individual(dd, froot)

m = 5000
froot = '../fits/fit_kw_adapt_5000'
# fit_individual(dd, b, m, p, froot)
inspect_fits_individual(dd, froot)

# fit_validate(dd, b, m, p, froot)
# inspect_fits_individual(dd, froot)
# inspect_fits_validate(dd, froot)

# d = load_all_data()
# dd = d.set_index('GROUP', drop=False).loc[[3, 4, 5]]
# dd = dd.set_index('PHASE', drop=False).loc[['ADAPTATION']]
# dd = dd[dd['TRIAL_ABS'] < 100]
# froot = '../fits/fit_kw_adapt_early_'
# fit_individual(dd, b, m, p, froot)
# inspect_fits_individual(dd, froot)

# fit_boot(dd, b, m, p, nboot)
# inspect_fits_boot()
# inspect_behaviour()

