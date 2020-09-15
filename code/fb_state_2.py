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

    d = pd.concat((d1, d2, d3, d4, d5, d6))

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


def fit_validate_2(fin, fout, bounds, maxiter, polish):

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
            pp[-1:] = 0.0
            print(pp)
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
                                             polish=polish,
                                             updating='deferred',
                                             workers=-1)

            with open(fout[i][:-4] + '_val_2.txt', 'a') as f:
                tmp = np.concatenate((results['x'], [results['fun']]))
                tmp = np.reshape(tmp, (tmp.shape[0], 1))
                np.savetxt(f, tmp.T, '%0.4f', delimiter=',', newline='\n')


def fit_validate(fin, fout, bounds, maxiter, polish):

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
                                             polish=polish,
                                             updating='deferred',
                                             workers=-1)

            with open(fout[i][:-4] + '_val.txt', 'a') as f:
                tmp = np.concatenate((results['x'], [results['fun']]))
                tmp = np.reshape(tmp, (tmp.shape[0], 1))
                np.savetxt(f, tmp.T, '%0.4f', delimiter=',', newline='\n')


def inspect_validated_fits(fin, fout):

    for i in range(len(fin)):
        print(fin[i])

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

        fig, ax = plt.subplots(3, 3, figsize=(10, 10))
        ax = ax.flatten()
        for j in range(pin.shape[1]):

            ax[j].plot(pin[:, j], pout[:, j], '.')
            ax[j].plot([-1, 1], [-1, 1], '--k', alpha=0.5)
            ax[j].set_title(names[j])

        plt.tight_layout()
        # plt.show()
        plt.savefig('../figures/fit_val_' + str(i) + '.pdf')


def inspect_fits_individual(group):

    d = load_all_data()

    for grp in group:

        fig = plt.figure(figsize=(5, 10))
        gs = fig.add_gridspec(3, 1)
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, :])
        ax3 = fig.add_subplot(gs[2, :])
        ax = np.array([ax1, ax2, ax3])

        dd = d[d['GROUP'] == grp]

        ddd = d[['TRIAL_ABS', 'HA_INIT', 'HA_END', 'ROT', 'SIG_MP',
                 'GROUP']].groupby(['TRIAL_ABS']).mean()
        ddd.reset_index(inplace=True)

        x_obs_mp = ddd['HA_INIT'].values
        x_obs_ep = ddd['HA_END'].values

        subs = d['SUBJECT'].unique()
        yrec = np.zeros((subs.shape[0], ddd.shape[0]))
        yffrec = np.zeros((subs.shape[0], ddd.shape[0]))
        yfbrec = np.zeros((subs.shape[0], ddd.shape[0]))
        xffrec = np.zeros((subs.shape[0], ddd.shape[0]))
        xfbrec = np.zeros((subs.shape[0], ddd.shape[0]))

        for s in range(subs.shape[0]):

            ddd = d[d['SUBJECT'] == subs[s]]
            rot = ddd['ROT'].values
            sig_mp = ddd['SIG_MP'].values
            group = ddd['GROUP'].values
            args = (rot, sig_mp, group)

            fname = 'fit_' + str(grp) + '_' + str(sub) + '.txt'
            p = np.loadtxt(fname, delimiter=',')
            pp = p[:-1]
            (y, yff, yfb, xff, xfb) = simulate(pp, args)

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

        # res = yfb - (x_obs_ep - x_obs_mp)
        # ax[1].hist(res, alpha=.75)
        # tstat, pval = ttest_1samp(res, popmean=0)
        # print('\n')
        # print(res.mean())
        # print((tstat, pval))

        ax[1].plot(np.arange(0, rot.shape[0], 1), xfb, '-C0', alpha=.5)
        # ax[1, 0].plot(np.arange(0, rot.shape[0], 1),
        #               x_obs_ep - x_obs_mp,
        #               '-C1',
        #               alpha=.5)

        pp = p[:, 6:9]
        x = np.arange(1, pp.shape[1] + 1, 1)
        xticks = np.arange(1, pp.shape[1] + 1, 1)
        ax[2].plot(x, pp.mean(0), '.C0')
        ax[2].errorbar(x,
                       pp.mean(axis=0),
                       yerr=sem(pp, axis=0),
                       fmt='none',
                       ecolor='C0')
        ax[2].plot([1, pp.shape[1]], [0, 0], '--k', alpha=0.25)
        ax[2].set_xticks(xticks)
        ax[2].set_xticklabels(['gamma_ff', 'gamma_fb', 'gamma_fb2'],
                              rotation=45,
                              ha="right")

        # plt.show()
        plt.savefig('../figures/results_group_' + str(i + 1) + '.pdf')
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

        # report statistics
        tstat, pval = ttest_1samp(param, popmean=0, axis=0)
        d = np.mean(param, axis=0) / np.std(param, axis=0, ddof=1)

        pname = [
            'alpha_ff', 'beta_ff', 'alpha_fb', 'beta_fb', 'base_fb', 'w',
            'gamma_ff', 'gamma_fb', 'gamma_fb2'
        ]

        print('\n')
        for j in range(6, 9):
            print(pname[j] + ': t(' + str(param.shape[0] - 1) + ') = ' +
                  str(np.round(tstat[j], 2)) + ', p = ' +
                  str(np.round(pval[j], 2)) + ', d = ' +
                  str(np.round(d[j], 2)))


def inspect_fits_boot(group):

    d = load_all_data()


def fit_individual(group, bounds, maxiter, polish):

    d = load_all_data()

    for grp in group:

        dd = d[(d['GROUP'] == grp) & (d['TRIAL_ABS'] < 1000)]

        for sub in d['SUBJECT'].unique():

            ddd = dd[dd['SUBJECT'] == sub][[
                'ROT', 'HA_INIT', 'HA_END', 'TRIAL_ABS', 'GROUP'
            ]]

            rot = ddd.ROT.values
            sig_mp = d[d.SUBJECT == sub].SIG_MP.values
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

            fname = '../fits/fit_' + str(grp) + '_' + str(sub) + '.txt'
            with open(fname, 'w') as f:
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

    sse_mp = np.sum((x_obs_mp - x_pred_mp)**2)
    sse_ep = np.sum((x_obs_ep - x_pred_ep)**2)

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
            # yfb[i] = xfb[i] * delta_mp[i] * bayes_mod_fb2 + base_fb_2
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

        # xff[i + 1] = beta_ff * xff[i] + bayes_mod_ff * alpha_ff * (
        #     w * delta_mp[i] + (1 - w) * delta_ep[i]) - alpha_ff * base_ff

        # xfb[i + 1] = beta_fb * xfb[i] + bayes_mod_fb * alpha_fb * delta_ep[
        #     i] - alpha_fb * base_fb

        xff[i + 1] = beta_ff * xff[i] + bayes_mod_ff * alpha_ff * (
            w * delta_mp[i] + (1 - w) * delta_ep[i])

        xfb[i + 1] = beta_fb * xfb[i] + bayes_mod_fb * alpha_fb * delta_ep[i]

        xfb = np.clip(xfb, -5, 5)

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


# alpha_ff = params[0]
# beta_ff = params[1]
# alpha_fb = params[2]
# beta_fb = params[3]
# w = params[4]
# gamma_ff = params[5]
# gamma_fb = params[6]
# gamma_fb2 = params[7]
# base_fb = params[8]
# base_ff = params[9]
# base_fb_2 = params[10]
b = ((0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (-50, 50), (-50, 50), (-50, 50))
m = 1
p = False
nboot = 2

# fit_individual([3], b, m, p)
# inspect_fits_individual([3])
# fit_boot([0], b, m, p, nboot)
# inspect_fits_boot([0, 1, 2])
inspect_behaviour()

# TODO: Make separate groups so this < 1000 isn't needed
# TODO: test all code after major refactor
# TODO: clean up the fit_validates
