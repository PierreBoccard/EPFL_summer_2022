import numpy as np
from emcee import autocorr
import matplotlib.pyplot as plt


def plot_convergence_by_walker(samples_mcmc, param_mcmc, walkerRatio=10, n_iter_total=None, 
                               verbose=False, display_autocorr_time=True):
    """
    Plot parameter traces throughout MCMC samples, to check for convergence.
    """
    n_params = samples_mcmc.shape[1]
    n_walkers = walkerRatio * n_params
    n_step = int(samples_mcmc.shape[0] / n_walkers)
    if n_iter_total is None:
        n_iter_total = np.nan

    if display_autocorr_time:
        try:
            samples_unflat = samples_mcmc.reshape(n_step, n_walkers, n_params)
            # tol=0 below for always get the estimate even if it's not worthy
            tau_per_param = autocorr.integrated_time(samples_unflat, tol=0, quiet=False)
        except Exception as e:
            if verbose:
                print("Error during autocorrelation time computation:", e)
            tau_per_param = [None]*n_params
        if verbose:
            print("Integrated autocorr time per param per parameter: {} (N/50={:.2f})"
                  .format(tau_per_param, n_iter_total/50.))

    chain = np.empty((n_walkers, n_step, n_params))
    for i in np.arange(n_params):
        samples = samples_mcmc[:, i].T
        chain[:, :, i] = samples.reshape((n_step, n_walkers)).T

    mean_pos = np.zeros((n_params, n_step))
    median_pos = np.zeros((n_params, n_step))
    std_pos = np.zeros((n_params, n_step))
    q16_pos = np.zeros((n_params, n_step))
    q84_pos = np.zeros((n_params, n_step))

    # chain = np.empty((nwalker, nstep, ndim), dtype = np.double)
    for i in np.arange(n_params):
        for j in np.arange(n_step):
            mean_pos[i][j] = np.mean(chain[:, j, i])
            median_pos[i][j] = np.median(chain[:, j, i])
            std_pos[i][j] = np.std(chain[:, j, i])
            q16_pos[i][j] = np.percentile(chain[:, j, i], 16.)
            q84_pos[i][j] = np.percentile(chain[:, j, i], 84.)

    fig, ax = plt.subplots(nrows=n_params, ncols=1 , sharex=True, figsize=(16, 2 * n_params))
    if n_params == 1: ax = [ax]
    last = n_step
    burnin = int((9.*n_step) / 10.) #get the final value on the last 10% on the chain

    for i in range(n_params):
        if verbose :
            print(param_mcmc[i], '{:.4f} +/- {:.4f}'.format(median_pos[i][last - 1], (q84_pos[i][last - 1] - q16_pos[i][last - 1]) / 2))
        ax[i].plot(median_pos[i][:last], c='g')
        ax[i].axhline(np.median(median_pos[i][burnin:last]), c='r', lw=1)
        ax[i].fill_between(np.arange(last), q84_pos[i][:last], q16_pos[i][:last], alpha=0.4)
        ax[i].set_ylabel(param_mcmc[i], fontsize=10)
        ax[i].set_xlim(0, last)
        if display_autocorr_time:
            text = r"$\tau={}$".format(tau_per_param[i]) + r" (N/50={:.2f})".format(n_iter_total/50.)
            ax[i].text(0.01, 0.05, text, fontsize=20, color='black', transform=ax[i].transAxes)

    return fig