# Import things that we will make use of
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import differential_evolution
from scipy.optimize import LinearConstraint


def fit():

    group = 1

    # eta_mu = params[0]
    # eta_sig = params[1]
    # b0_mu = params[2]
    # b0_sig = params[3]
    # b1 = params[4]
    # mu_init = params[5]
    # sig_init = params[6]

    # constraints = LinearConstraint(A=[[1, 0, 0, -1, 0, 0, 0, 0],
    #                                   [0, 1, 0, 0, -1, 0, 0, 0],
    #                                   [0, 0, 0, 0, 0, 0, 0, 0],
    #                                   [0, 0, 0, 0, 0, 0, 0, 0],
    #                                   [0, 0, 0, 0, 0, 0, 0, 0],
    #                                   [0, 0, 0, 0, 0, 0, 0, 0],
    #                                   [0, 0, 0, 0, 0, 0, 0, 0],
    #                                   [0, 0, 0, 0, 0, 0, 0, 0]],
    #                                lb=[-1, -1, 0, 0, 0, 0, 0, 0],
    #                                ub=[0, 0, 0, 0, 0, 0, 0, 0])

    bounds = ((0, 1), (0, 1), (1, 10), (1, 10), (1, 10), (-45, 45), (2, 10))

    args = [group]

    differential_evolution(func=obj_func,
                           bounds=bounds,
                           args=args,
                           disp=True,
                           maxiter=300,
                           tol=1e-15,
                           disp=False,
                           polish=False,
                           updating='deferred',
                           workers=-1)


def obj_func(params, *args):
    '''
    params: collection of free paraameter values
    *arags: fixed parameters et al.
    '''

    group = args[0]

    obs = get_observed(group)
    args = obs[0]
    mu_p_obs = obs[1]
    sig_p_obs = obs[2]
    mu_p_obs = np.reshape(mu_p_obs, (1, mu_p_obs.shape[0]))
    sig_p_obs = np.reshape(sig_p_obs, (1, sig_p_obs.shape[0]))

    x_pred = simulate(params, args)
    mu_p_pred = x_pred[0]
    sig_p_pred = x_pred[1]

    # plt.plot(mu_p_obs[0, :], '-b')
    # plt.plot(mu_p_pred[0, :], '-r')
    # plt.show()

    # plt.plot(sig_p_obs[0, :], '-b')
    # plt.plot(sig_p_pred[0, :], '-r')
    # plt.show()

    sse_mu_p = np.sum((mu_p_pred - mu_p_obs)**2)
    sse_sig_p = np.sum((sig_p_pred - sig_p_obs)**2)
    sse = sse_mu_p + sse_sig_p

    return sse


def get_observed(group):
    d = pd.read_csv('../datta/Bayes_SML_EXP1_060820.csv')

    # Get args
    rot = d[d.SUBJECT == 1].ROT.values
    theta_values = np.unique(d[d.SUBJECT == 1].TARGET.values)
    mu_f = d[d.SUBJECT == 1].CA_INIT.values
    sig_f = d[d.SUBJECT == 1].SIG_MP.values
    mu_f = np.reshape(mu_f, (1, mu_f.shape[0]))
    sig_f = np.reshape(sig_f, (1, sig_f.shape[0]))
    args = (rot, theta_values, mu_f, sig_f)

    ## Add a column that indicates absolute trial number
    n_trials = d.groupby('PHASE').max()['TRIAL'].sum()
    n_subs = d['SUBJECT'].unique().shape[0]

    d.replace('Baseline', '1', inplace=True)
    d.replace('Adaptation', '2', inplace=True)
    d.replace('Washout', '3', inplace=True)

    d.sort_values(['GROUP', 'SUBJECT', 'PHASE', 'TRIAL'], inplace=True)
    d['TRIAL_ABS'] = np.tile(np.arange(1, n_trials + 1), n_subs)

    # estimate prior mean with mean across subjects HA_initial[t + 1]
    dd_mu = d.groupby(['GROUP',
                       'TRIAL_ABS']).mean()['HA_INITIAL'].reset_index()
    prior_mu_obs = dd_mu[dd_mu['GROUP'] == group]['HA_INITIAL'].values

    # plt.plot(dd[dd['GROUP'] == 0].TRIAL_ABS.values,
    #          dd[dd['GROUP'] == 0].HA_INITIAL.values, '.')
    # plt.show()

    # estimate prior uncertainty with sd across subjects HA_initial[t + 1]
    dd_sig = d.groupby(['GROUP',
                        'TRIAL_ABS']).std()['HA_INITIAL'].reset_index()
    dd_sig = dd_sig[dd_sig['GROUP'] == group]
    prior_sig_obs = dd_sig[dd_sig['GROUP'] == group]['HA_INITIAL'].values

    # plt.plot(dd[dd['GROUP'] == 0].TRIAL_ABS.values,
    #          dd[dd['GROUP'] == 0].HA_INITIAL.values, '.')
    # plt.show()

    return (args, prior_mu_obs, prior_sig_obs)


# Define a function to compute how much learning from
# the current reach will generalise to the remaining
# (not reached to) targets.
def g_func(theta_g, theta_r, b0, b1):
    '''
    theta_g should be a numpy array containing the taraget locations
    theta_r should be the scalar-valued current target location
    b0 is the parameter by the same name in the paper
    b1 is the parameter by the same name in the paper
    '''
    alpha = (b0 + np.exp(b1))
    return b0 + np.exp(b1 * np.cos(np.deg2rad(theta_g - theta_r))) / alpha


def simulate(params, args):

    # Define the parameters of the model. Ultimately, we want
    # to write this code as a function that can be passed to a
    # search algorithm in order to find the parameters that
    # best fit our data, but as a start, just getting the model
    # to run with a fixed set of parameters will be
    # instructive.
    eta_mu = params[0]
    eta_sig = params[1]
    b0_mu = params[2]
    b0_sig = params[3]
    b1 = params[4]
    mu_init = params[5]
    sig_init = params[6]

    # This model requires that we know what rotation was
    # applied on each trial (this determines the centroid of
    # the midpoint cloud). When we get the optimisation phase
    # (i.e., passing a function to an algorithm to search for
    # best fitting parameters), we willl have to read in the
    # rotation from a file that corresponds what a subject or
    # group of subjects actually experienced. Here, since we
    # are just simulatnig willy nilly, we just make something
    # up that roughly corresponds to what our participants
    # experienced.
    rot = args[0]

    # Define num_trials from the rotation array defined above
    num_trials = rot.shape[0]

    # Specifiy the number of targets. On each trial, a single
    # target will be reached to, and we will want to pull this
    # information from the actual data files once we move on to
    # fitting. For now, in familiar willy nilly form, we just
    # make something up that looks kinda right.
    theta_values = args[1]

    # Pick the training target
    theta_train_ind = np.where(theta_values == 0)[0][0]

    # construct an array that indicates what target was reached
    # to on each trial. Will need to read this in from a data
    # file when fitting, but this is another instance of willy
    # nilly for now.
    theta_ind = theta_train_ind * np.ones(num_trials, dtype=np.int8)

    # Comute the number of targets from theta_ind. The number
    # of targets is important to compute becuase it corresponds
    # to the number states that our state-space model will
    # model across trials.
    n_tgt = np.unique(theta_values).shape[0]

    # This model and paradigm require that we specify the mean
    # (i.e., centroid) and uncertainty of midpoint feedback.
    # For full model fitting awesomeness we will need to read
    # these in from behavioural data files, but for now just go
    # willy nilly.
    mu_f = args[2]
    sig_f = args[3]

    # Create an array of zeros to store the simulated prior
    # mean and prior uncertainty.
    mu_p = np.zeros((n_tgt, num_trials))
    sig_p = np.zeros((n_tgt, num_trials))

    # initialise the prior mean and prior uncertainty.
    mu_p[:, 0] = mu_init
    sig_p[:, 0] = sig_init

    # create an array of zeros that we will end up populating
    # with the experienced error gradients (it's not practical
    # to explain here, so just consult the paper for what the
    # heck an error gradient is in this model).
    delta_mu = np.zeros(num_trials)
    delta_sig = np.zeros(num_trials)

    for i in range(0, num_trials - 1):
        ind = theta_ind[i]

        # Virtually unintelligible, yet correct. Please consult the paper
        sp = sig_p[ind, i]**2 / (sig_p[ind, i]**2 + sig_f[ind, i]**2)
        sf = sig_f[ind, i]**2 / (sig_p[ind, i]**2 + sig_f[ind, i]**2)**2

        # Virtually unintelligible, yet correct. Please consult the paper
        delta_sig[i] = 2 * (rot[i] - mu_p[ind, i] + sp *
                            (mu_p[ind, i] - mu_f[ind, i])) * (
                                mu_p[ind, i] -
                                mu_f[ind, i]) * sf * 2 * sig_p[ind, i]

        # Virtually unintelligible, yet correct. Please consult the paper
        delta_mu[i] = -2 * (mu_p[ind, i] - (1 - sp) * mu_p[ind, i] -
                            sp * mu_f[ind, i]) * (1 - sp)

        # compute how much learning from the current reach will
        # generalise to the remaining (not reached to) targets.
        G_sig = g_func(theta_values, theta_values[theta_ind[i]], b0_sig, b1)
        G_mu = g_func(theta_values, theta_values[theta_ind[i]], b0_mu, b1)

        # Update the prior mean and prior uncertainty
        sig_p[:, i + 1] = sig_p[:, i] - eta_sig * delta_sig[i] * G_sig
        mu_p[:, i + 1] = mu_p[:, i] - eta_mu * delta_mu[i] * G_mu

    return (mu_p, sig_p)


# fig, ax = plt.subplots(nrows=1, ncols=2)
# c = cm.rainbow(np.linspace(0, 1, 11))

# for k in range(mu_p.shape[0]):
#     ax[0].plot(mu_p[k, :], '.', color=c[k])
#     ax[0].plot(rot, 'k')

# for k in range(sig_p.shape[0]):
#     ax[1].plot(sig_p[k, :], '.', color=c[k])
#     ax[1].plot(rot, 'k')

# plt.show()
