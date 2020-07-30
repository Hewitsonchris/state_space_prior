import numpy as np
import matplotlib.pyplot as plt


def x_normal_known_var():
    """
    Bayesian updating with true data generated from unknown mean and known
    variance. You can see this in the equations for mu_posterior and
    sigma_posterior: Both depend on sigma_true but neither depend on
    mu_true.
    """
    n = 100

    mu_true = 10.0
    sigma_true = 2.0

    mu = np.zeros(n)
    sigma = np.zeros(n)

    mu[0] = 0.0
    sigma[0] = 0.75

    x = np.zeros(n)
    for i in range(1, n):
        if i < 50:
            x[i] = np.random.normal(mu_true, sigma_true, 1)
        else:
            x[i] = np.random.normal(mu_true + 10.0, sigma_true, 1)

        mu[i] = (mu[i - 1] * sigma_true**2 + x[i - 1] * sigma[i - 1]**2) / (
            sigma_true**2 + sigma[i - 1]**2)

        sigma[i] = np.sqrt((sigma_true**2) * (sigma[i - 1]**2) /
                           (sigma_true**2 + sigma[i - 1]**2))

    plt.plot(mu[1:n])
    plt.plot(sigma[1:n])
    plt.show()


def x_normal_known_var_bimodal_one_prior():
    """
    x_normal_known_var_bimodal() but using a single prior.
    """
    n = 100

    mu_true_1 = 10.0
    sigma_true_1 = 2.0

    mu_true_2 = -10.0
    sigma_true_2 = 2.0

    mu = np.zeros(n)
    sigma = np.zeros(n)

    mu[0] = 0.0
    sigma[0] = 0.75

    x = np.zeros(n)
    for i in range(1, n):
        if np.random.rand() < 0.5:
            x[i] = np.random.normal(mu_true_1, sigma_true_1, 1)
            mu_true = mu_true_1
            sigma_true = sigma_true_1

        else:
            x[i] = np.random.normal(mu_true_2, sigma_true_2, 1)
            mu_true = mu_true_2
            sigma_true = sigma_true_2

        mu[i] = (mu[i - 1] * sigma_true**2 + x[i - 1] *
                    sigma[i - 1]**2) / (sigma_true**2 + sigma[i - 1]**2)

        sigma[i] = np.sqrt((sigma_true**2) * (sigma[i - 1]**2) /
                            (sigma_true**2 + sigma[i - 1]**2))

    plt.plot(mu[1:n])
    plt.ylim(-10,10)
    plt.show()

    plt.hist(x)
    plt.show()


def x_normal_known_var_bimodal():
    """
    Known variance --- simple bimodal version.
    """
    n = 100

    mu_true_1 = 10.0
    sigma_true_1 = 2.0

    mu_true_2 = -10.0
    sigma_true_2 = 2.0

    mu_1 = np.zeros(n)
    sigma_1 = np.zeros(n)

    mu_2 = np.zeros(n)
    sigma_2 = np.zeros(n)

    mu_1[0] = 0.0
    sigma_1[0] = 0.75

    mu_2[0] = 0.0
    sigma_2[0] = 0.75

    x = np.zeros(n)
    for i in range(1, n):
        if np.random.rand() < 0.5:
            x[i] = np.random.normal(mu_true_1, sigma_true_1, 1)

            mu_1[i] = (mu_1[i - 1] * sigma_true_1**2 + x[i] * sigma_1[i - 1]**2
                       ) / (sigma_true_1**2 + sigma_1[i - 1]**2)

            sigma_1[i] = np.sqrt((sigma_true_1**2) * (sigma_1[i - 1]**2) /
                                 (sigma_true_1**2 + sigma_1[i - 1]**2))

            mu_2[i] = mu_2[i - 1]
            sigma_2[i] = sigma_2[i - 1]

        else:
            x[i] = np.random.normal(mu_true_2, sigma_true_2, 1)

            mu_2[i] = (mu_2[i - 1] * sigma_true_2**2 + x[i] * sigma_2[i - 1]**2
                       ) / (sigma_true_2**2 + sigma_2[i - 1]**2)

            sigma_2[i] = np.sqrt((sigma_true_2**2) * (sigma_2[i - 1]**2) /
                                 (sigma_true_2**2 + sigma_2[i - 1]**2))

            mu_1[i] = mu_1[i - 1]
            sigma_1[i] = sigma_1[i - 1]

    plt.subplot(211)
    plt.plot(mu_1[1:n])
    plt.plot(sigma_1[1:n])
    plt.title('Context 1')
    plt.subplot(212)
    plt.plot(mu_2[1:n])
    plt.plot(sigma_2[1:n])
    plt.title('Context 2')
    plt.show()


def x_normal_unknown_var():
    """
    With unknown variance, the updates to the hyperparameters do not depend on
    mu_true or sigma_true. However, just by playing around with it a bit, the
    estimate of sigma_true (i.e., ev_sigma seems quite sensitive to
    hyperparameter initial values).
    https://stats.stackexchange.com/questions/365192/bayesian-update-for-a-univariate-normal-distribution-with-unknown-mean-and-varia
    """
    n = 200

    mu_true = 12.0
    sigma_true = 4.0

    k = np.zeros(n)
    m = np.zeros(n)
    v = np.zeros(n)
    ss = np.zeros(n)
    ev_mu = np.zeros(n)
    ev_sigma = np.zeros(n)

    k[0] = 5.0
    m[0] = 5.0
    v[0] = 5.0
    ss[0] = 50.0

    x = np.zeros(n)
    for i in range(1, n):
        x[i] = np.random.normal(mu_true, sigma_true, 1)
        # if i < 500:
        #     x[i] = np.random.normal(mu_true, sigma_true, 1)
        # else:
        #     x[i] = np.random.normal(mu_true + 10.0, sigma_true + 10.0, 1)

        k[i] = k[i - 1] + 1.0
        m[i] = (k[i - 1] * m[i - 1] + x[i]) / k[i]
        v[i] = v[i - 1] + 1.0
        ss[i] = (v[i - 1] * ss[i - 1] + k[i - 1] *
                 (x[i - 1] - m[i - 1]) / k[i]) / v[i]

        ev_sigma[i] = np.sqrt((v[i] * ss[i]) / (v[i] - 2.0))
        ev_mu[i] = m[i]

    plt.subplot(211)
    plt.plot(ev_mu[1:n])
    plt.title("mu")
    plt.subplot(212)
    plt.plot(ev_sigma[1:n])
    plt.title("sigma")
    plt.show()


# x_normal_known_var()
x_normal_unknown_var()
