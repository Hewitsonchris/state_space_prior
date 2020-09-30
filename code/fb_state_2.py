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

    d = load_all_data()

    # d[d['TRIAL_ABS'] < 100].groupby('SIG_MP')['HA_END', 'ROT'].plot(
    #     x='HA_END', y='ROT', kind='scatter')
    # plt.show()

    dd = d.groupby(['GROUP', 'TRIAL_ABS',
                    'SIG_MP'])[['HA_INIT', 'HA_MID', 'HA_END',
                                'ROT']].mean().reset_index()

    # dd1 = dd[dd['GROUP'] == 1][11:190]
    # d1 = dd1[dd1['SIG_MP'] == 1]['HA_INIT'].values
    # r1 = dd1[dd1['SIG_MP'] == 1]['ROT'].values
    # d2 = dd1[dd1['SIG_MP'] == 2]['HA_INIT'].values
    # r2 = dd1[dd1['SIG_MP'] == 2]['ROT'].values
    # d3 = dd1[dd1['SIG_MP'] == 3]['HA_INIT'].values
    # r3 = dd1[dd1['SIG_MP'] == 3]['ROT'].values
    # d4 = dd1[dd1['SIG_MP'] == 4]['HA_INIT'].values
    # r4 = dd1[dd1['SIG_MP'] == 4]['ROT'].values
    # e1 = r1 - d1
    # e2 = r2 - d2
    # e3 = r3 - d3
    # e4 = r4 - d4
    # # plt.plot(r1[0:-1], d1[1:], 'o')
    # # plt.plot(r2[0:-1], d2[1:], 'o')
    # # plt.plot(r3[0:-1], d3[1:], 'o')
    # # plt.plot(r4[0:-1], d4[1:], 'o')
    # plt.plot(e1, np.diff(d1, prepend=0), 'o')
    # plt.plot(e2, np.diff(d2, prepend=0), 'o')
    # plt.plot(e3, np.diff(d3, prepend=0), 'o')
    # plt.plot(e4, np.diff(d4, prepend=0), 'o')
    # # plt.legend(['SIG_MP = 1', 'SIG_MP = 2', 'SIG_MP = 3', 'SIG_MP = 4'])
    # plt.show()

    fig, ax = plt.subplots(1, 1)
    dd[dd['GROUP'] == 7].plot.scatter(x='TRIAL_ABS',
                                      y='HA_INIT',
                                      c='C0',
                                      marker='o',
                                      ax=ax)
    dd[dd['GROUP'] == 7].plot.scatter(x='TRIAL_ABS',
                                      y='HA_END',
                                      c='C0',
                                      marker='v',
                                      ax=ax)
    dd[dd['GROUP'] == 8].plot.scatter(x='TRIAL_ABS',
                                      y='HA_INIT',
                                      c='C1',
                                      marker='o',
                                      ax=ax)
    dd[dd['GROUP'] == 8].plot.scatter(x='TRIAL_ABS',
                                      y='HA_END',
                                      c='C1',
                                      marker='v',
                                      ax=ax)
    plt.show()

    dd[dd['GROUP'] == 1].plot.scatter(x='TRIAL_ABS',
                                      y='HA_INIT',
                                      c='SIG_MP',
                                      colormap='viridis')
    plt.show()

    # delta hand_angle as function of error size
    ha_ep = dd[dd['GROUP'] == 7]['HA_END'].values
    ha_mp = dd[dd['GROUP'] == 7]['HA_INIT'].values
    rot = dd[dd['GROUP'] == 7]['ROT'].values
    sig_mp = dd[dd['GROUP'] == 7]['SIG_MP'].values
    err_ha_mp = ha_mp - rot
    err_ha_ep = ha_ep - rot
    delta_ha_mp = np.diff(ha_mp, prepend=[0])
    delta_ha_ep = np.diff(ha_ep, prepend=[0])

    fig, ax = plt.subplots(2, 2)

    c = ['C0', 'C1', 'C2', 'C3', 'C4']
    for i in np.unique(sig_mp):
        ax[0, 0].plot(err_ha_mp[sig_mp == i],
                      delta_ha_mp[sig_mp == i],
                      'o',
                      alpha=0.5)
        ax[0, 0].set_xlabel('MP error size')
        ax[0, 0].set_ylabel('delta hand angle MP')

        ax[0, 1].plot(err_ha_mp[sig_mp == i],
                      delta_ha_ep[sig_mp == i],
                      'o',
                      alpha=0.5)
        ax[0, 1].set_xlabel('MP error size')
        ax[0, 1].set_ylabel('delta hand angle EP')

        ax[1, 0].plot(err_ha_ep[sig_mp == i],
                      delta_ha_mp[sig_mp == i],
                      'o',
                      alpha=0.5)
        ax[1, 0].set_xlabel('EP error size')
        ax[1, 0].set_ylabel('delta hand angle MP')

        ax[1, 1].plot(err_ha_ep[sig_mp == i],
                      delta_ha_ep[sig_mp == i],
                      'o',
                      alpha=0.5)
        ax[1, 1].set_xlabel('EP error size')
        ax[1, 1].set_ylabel('delta hand angle EP')

    plt.tight_layout()
    plt.show()

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
            ax.set_ylim([-30, 30])
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


def inspect_fits_individual_model_compare(d):

    n_params = 10

    subs = d['SUBJECT'].unique()

    xlabel = [
        'All nonzero', 'Zero FF learning', 'Zero FB learning',
        'Zero FB control', 'Nonzero FF Learning', 'Nonzero FB learning',
        'Nonzero FB control', 'All zero'
    ]

    n_models = len(xlabel)
    print(n_models)
    aic = np.zeros((n_models, subs.shape[0]))
    bic = np.zeros((n_models, subs.shape[0]))
    k_list = [9, 8, 8, 8, 7, 7, 7, 6]
    for i in range(n_models):
        froot = '../fits/fit_kw_adapt_' + str(i) + '_'

        prec = np.zeros((subs.shape[0], n_params))

        for s in range(subs.shape[0]):
            dd = d[d['SUBJECT'] == s + 1]

            x_obs_mp = dd['HA_INIT'].values
            x_obs_ep = dd['HA_END'].values
            rot = dd['ROT'].values
            sig_mp = dd['SIG_MP'].values
            group = dd['GROUP'].values
            args = (rot, sig_mp, group)

            fname = froot + str(s + 1) + '.txt'
            p = np.loadtxt(fname, delimiter=',')
            prec[s, :] = p
            (y, yff, yfb, xff, xfb) = simulate(p[:-1], args)
            ss_tot_mp = np.nansum((x_obs_mp - np.nanmean(x_obs_mp))**2)
            ss_reg_mp = np.nansum((yff - np.nanmean(x_obs_mp))**2)
            ss_res_mp = np.nansum((x_obs_mp - yff)**2)
            ss_tot_ep = np.nansum((x_obs_ep - np.nanmean(x_obs_ep))**2)
            ss_reg_ep = np.nansum((y - np.nanmean(x_obs_ep))**2)
            ss_res_ep = np.nansum((x_obs_ep - y)**2)
            r_squared = 1 - (ss_res_ep + ss_res_mp) / (ss_tot_ep + ss_tot_mp)

            n = 1080
            k = k_list[i]
            bic[i, s] = compute_bic(r_squared, n, k)

        pname = [
            'alpha_ff', 'beta_ff', 'alpha_fb', 'beta_fb', 'w', 'gamma_ff',
            'gamma_fb', 'gamma_fb2', 'xfb_init', 'sse'
        ]

        x = np.arange(1, n_params, 1)
        plt.plot([1, n_params], [0, 0], '--')
        plt.violinplot(prec[:, :-1])
        plt.xticks(x, pname[:-1])
        for jj in range(prec.shape[0]):
            plt.plot(x, prec[jj, :-1], '.', alpha=0.5)
        plt.show()

        tstat, pval = ttest_1samp(prec, popmean=0, axis=0)
        cd = np.mean(prec, axis=0) / np.std(prec, axis=0, ddof=1)

        print('\n')
        inds = [5, 6, 7, 9]
        for j in inds:
            print(pname[j] + ' = ' + str(np.round(prec[:, j].mean(), 2)) +
                  ': t(' + str(prec.shape[0] - 1) + ') = ' +
                  str(np.round(tstat[j], 2)) + ', p = ' +
                  str(np.round(pval[j], 2)) + ', d = ' +
                  str(np.round(cd[j], 2)))

    summed_bic = bic.sum(1)
    summed_bic = summed_bic - summed_bic[0]

    pbic = np.zeros(n_models)
    for i in range(n_models):
        pbic[i] = np.exp(-0.5 * summed_bic[i]) / np.sum(
            np.exp(-0.5 * summed_bic))

    # print(summed_bic.shape)
    # print(pbic)
    # print(pbic.sum())

    # fig, ax = plt.subplots(1, 2)
    # x = np.arange(1, 8, 1)
    # ax[0].plot(x, summed_bic, 'o', alpha=1)
    # ax[0].set_xticks(x)
    # ax[0].set_xticklabels(xlabel, rotation=30)
    # ax[0].set_ylabel('Summed BIC')
    # ax[1].plot(x, pbic, 'o', alpha=1)
    # ax[1].set_xticks(x)
    # ax[1].set_xticklabels(xlabel, rotation=30)
    # ax[1].set_ylabel('P(Model | Data)')
    # plt.show()

    pbic = np.zeros((n_models, subs.shape[0]))
    for s in range(subs.shape[0]):
        for i in range(n_models):
            pbic[i, s] = np.exp(-0.5 * bic[i, s]) / np.sum(
                np.exp(-0.5 * bic[:, s]))

    b = pd.DataFrame(pbic.T)
    b.plot(kind='bar')
    plt.show()


def inspect_fits_individual(d, froot):

    n_params = 10

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

        prec[s, :] = p
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


def fit_individual(d, fit_args, froot):

    obj_func = fit_args['obj_func']
    bounds = fit_args['bounds']
    maxiter = fit_args['maxiter']
    disp = fit_args['disp']
    tol = fit_args['tol']
    polish = fit_args['polish']
    updating = fit_args['updating']
    workers = fit_args['workers']
    popsize = fit_args['popsize']
    mutation = fit_args['mutation']
    recombination = fit_args['recombination']

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
                                         disp=disp,
                                         maxiter=maxiter,
                                         popsize=popsize,
                                         mutation=mutation,
                                         recombination=recombination,
                                         tol=tol,
                                         polish=polish,
                                         updating=updating,
                                         workers=workers)

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

        # xff[i + 1] = beta_ff * xff[i] + bayes_mod_ff * alpha_ff * (
        #     w * delta_mp[i] + (1 - w) * delta_ep[i]) - alpha_ff * base_ff

        # xff[i + 1] = beta_ff * xff[i] + alpha_ff * (
        #     w * bayes_mod_ff * delta_mp[i] +
        #     (1 - w) * delta_ep[i]) - alpha_ff * base_ff

        xff[i + 1] = beta_ff * xff[i] + alpha_ff * (
            w * delta_mp[i] +
            (1 - w) * bayes_mod_ff * delta_ep[i]) - alpha_ff * base_ff

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


def compute_aic(rsq, n, k):
    # aic = 2 * k + n * np.log(sse / n)
    aic = n * np.log(1 - rsq) + k * 2
    return aic


def compute_bic(rsq, n, k):
    # bic = np.log(n) * k + n * np.log(sse / n)
    bic = n * np.log(1 - rsq) + k * np.log(n)
    return bic


nboot = -1
d = load_all_data()
dd = d.set_index('GROUP', drop=False).loc[[3, 4, 5]]
dd = dd.set_index('PHASE', drop=False).loc[['ADAPTATION']]

b_list = [((0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (-50, 50), (-50, 50),
           (-50, 50), (-2, 2)),
          ((0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 0), (-50, 50),
           (-50, 50), (-2, 2)),
          ((0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (-50, 50), (0, 0),
           (-50, 50), (-2, 2)),
          ((0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (-50, 50), (-50, 50),
           (0, 0), (-2, 2)),
          ((0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (-50, 50), (0, 0), (0, 0),
           (-2, 2)),
          ((0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 0), (-50, 50), (0, 0),
           (-2, 2)),
          ((0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 0), (0, 0), (-50, 50),
           (-2, 2)),
          ((0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 0), (0, 0), (0, 0),
           (-2, 2))]

for i in range(len(b_list)):

    b = b_list[i]

    fit_args = {
        'obj_func': obj_func,
        'bounds': b,
        'disp': True,
        'maxiter': 1000,
        'popsize': 15,
        'mutation': 0.5,
        'recombination': 0.7,
        'tol': 1e-15,
        'polish': False,
        'updating': 'deferred',
        'workers': -1
    }

    froot = '../fits/fit_kw_adapt_' + str(i) + '_'
    # fit_individual(dd, fit_args, froot)

# inspect_fits_individual_model_compare(dd)

# inspect_fits_individual(dd, froot)
# fit_boot(dd, b, m, p, nboot)
# inspect_fits_boot()
inspect_behaviour()
