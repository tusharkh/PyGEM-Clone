"""Run the model calibration"""
# Spyder cannot run parallels, so always set -option_parallels=0 when testing in Spyder.

# Built-in libraries
import os
import glob

# External libraries
import pandas as pd
import numpy as np
import xarray as xr
import pymc
import matplotlib.pyplot as plt

from pymc import utils
from pymc.database import base

from scipy.stats.kde import gaussian_kde
from scipy.stats import norm
from scipy.stats import truncnorm
from scipy.stats import uniform
from scipy.stats import linregress
# Local libraries
import pygem_input as input
import pygemfxns_modelsetup as modelsetup
import class_mbdata


#%% ===== SCRIPT SPECIFIC INPUT DATA =====
cal_datasets = ['shean']
#cal_datasets = ['shean', 'wgms_d']

# mcmc model parameters
parameters = ['precfactor', 'tempchange', 'ddfsnow']
parameters_all = ['ddfsnow', 'precfactor', 'tempchange', 'ddfice', 'lrgcm', 'lrglac', 'precgrad', 'tempsnow']
# Autocorrelation lags
acorr_maxlags = 100

# Export option
#output_filepath = input.main_directory + '/../Output/'
suffix = '_trunc'
mcmc_data_fp = input.main_directory + '/../MCMC_data/'
mcmc_prior_fp = mcmc_data_fp + 'prior_comparison/'
mcmc_output_netcdf_fp = input.main_directory + '/../MCMC_data/netcdf' + suffix + '/'
mcmc_output_figures_fp = input.main_directory + '/../MCMC_data/figures' + suffix + '/'
mcmc_output_tables_fp = input.main_directory + '/../MCMC_data/tables/'
mcmc_output_csv_fp = input.main_directory + '/../MCMC_data/csv' + suffix + '/'
mcmc_output_hist_fp = input.main_directory + '/../MCMC_data/hist' + suffix + '/'

debug = False


def prec_transformation(precfactor_raw):
    """
    Converts raw precipitation factors from normal distribution to correct values.

    Takes raw values from normal distribution and converts them to correct precipitation factors according to:
        if x >= 0:
            f(x) = x + 1
        else:
            f(x) = 1 / (1 - x)
    i.e., normally distributed values from -2 to 2 and converts them to be 1/3 to 3.

    Parameters
    ----------
    precfactor_raw : float
        numpy array of untransformed precipitation factor values

    Returns
    -------
    precfactor : float
        array of corrected precipitation factors
    """
    precfactor = precfactor_raw.copy()
    precfactor[precfactor >= 0] = precfactor[precfactor >= 0] + 1
    precfactor[precfactor < 0] = 1 / (1 - precfactor[precfactor < 0])
    return precfactor


def effective_n(ds, vn, iters, burn):
    """
    Compute the effective sample size of a trace.

    Takes the trace and computes the effective sample size
    according to its detrended autocorrelation.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset containing mcmc traces
    vn : str
        Parameter variable name
    iters : int
        number of mcmc iterations to test
    burn : int
        number of initial iterations to throw away

    Returns
    -------
    effective_n : int
        effective sample size
    """
    # Effective sample size
    x = ds['mp_value'].sel(chain=0, mp=vn).values[burn:iters]
    # detrend trace using mean to be consistent with statistics
    # definition of autocorrelation
    x = (x - x.mean())
    # compute autocorrelation (note: only need second half since
    # they are symmetric)
    rho = np.correlate(x, x, mode='full')
    rho = rho[len(rho)//2:]
    # normalize the autocorrelation values
    #  note: rho[0] is the variance * n_samples, so this is consistent
    #  with the statistics definition of autocorrelation on wikipedia
    # (dividing by n_samples gives you the expected value).
    rho_norm = rho / rho[0]
    # Iterate untile sum of consecutive estimates of autocorrelation is
    # negative to avoid issues with the sum being -0.5, which returns an
    # effective_n of infinity
    negative_autocorr = False
    t = 1
    n = len(x)
    while not negative_autocorr and (t < n):
        if not t % 2:
            negative_autocorr = sum(rho_norm[t-1:t+1]) < 0
        t += 1
    return int(n / (1 + 2*rho_norm[1:t].sum()))


def gelman_rubin(ds, vn, iters=1000, burn=0):
    """
    Calculate Gelman-Rubin statistic.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing MCMC iterations for a single glacier with 3 chains
    vn : str
        Parameter variable name
    iters : int
        number of MCMC iterations to test for the gelman-rubin statistic
    burn : int
        number of MCMC iterations to ignore at start of chain before performing test

    Returns
    -------
    gelman_rubin_stat : float
        gelman_rubin statistic (R_hat)
    """
    if debug:
        if len(ds.chain) != 3:
            raise ValueError('Given dataset has an incorrect number of chains')
        if iters > len(ds.chain):
            raise ValueError('iters value too high')
        if (burn >= iters):
            raise ValueError('Given iters and burn in are incompatible')

    # unpack iterations from dataset
    for n_chain in ds.chain.values:
        if n_chain == 0:
            chain = ds['mp_value'].sel(chain=n_chain, mp=vn).values[burn:iters]
            chain = np.reshape(chain, (1,len(chain)))
        else:
            chain2add = ds['mp_value'].sel(chain=n_chain, mp=vn).values[burn:iters]
            chain2add = np.reshape(chain2add, (1,chain.shape[1]))
            chain = np.append(chain, chain2add, axis=0)

    #calculate statistics with pymc in-built function
    return pymc.gelman_rubin(chain)


def MC_error(ds, vn, iters=None, burn=0, chain_no=0, batches=5):
    """
    Calculates MC Error using the batch simulation method.
    Also returns mean of trace

    Calculates the simulation standard error, accounting for non-independent
    samples. The trace is divided into batches, and the standard deviation of
    the batch means is calculated.

    With datasets of multiple chains, choses the highest MC error of all
    the chains and returns this value unless a chain number is specified

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing MCMC iterations for a single glacier with 3 chains
    vn : str
        Parameter variable name
    chain_no : int
        Number of chain to use (0,1 ror 2)
        If none, finds the highest MC error of the three chains
        and returns this value
    batches : int
        Number of batches to divide the trace in (default 5)

    """
    if iters is None:
        iters = len(ds.mp_value)

    # get iterations from ds
    trace = [ds['mp_value'].sel(chain=n_chain, mp=vn).values[burn:iters]
             for n_chain in ds.chain.values]

    result = batchsd(trace, batches)
    mean = np.mean(trace[chain_no])

    if len(ds.chain) <= chain_no or chain_no < 0:

        raise ValueError('Given chain_no is invalid')

    else:

        return (result[chain_no], mean)


def batchsd(trace, batches=5):
    """
    Calculates MC Error using the batch simulation method.

    Calculates the simulation standard error, accounting for non-independent
    samples. The trace is divided into batches, and the standard deviation of
    the batch means is calculated.
    With datasets of multiple chains, choses the highest MC error of all
    the chains and returns this value unless a chain number is specified

    Parameters
    ----------
    trace: np.ndarray
        Array representing MCMC chain
    batches : int
        Number of batches to divide the trace in (default 5)

    """
    # see if one trace or multiple
    if len(np.shape(trace)) > 1:

        return np.array([batchsd(t, batches) for t in trace])

    else:
        if batches == 1:
            return np.std(trace) / np.sqrt(len(trace))

        try:
            batched_traces = np.resize(trace, (batches, int(len(trace) / batches)))
        except ValueError:
            # If batches do not divide evenly, trim excess samples
            resid = len(trace) % batches
            batched_traces = np.resize(trace[:-resid],
                (batches, len(trace[:-resid]) / batches))

        means = np.mean(batched_traces, 1)

        return np.std(means) / np.sqrt(batches)


def summary(netcdf, glacier_cal_data, iters=[5000, 10000, 25000], alpha=0.05, start=0,
            batches=100, chain=None, roundto=3, filename='output.txt'):
        """
        Generate a pretty-printed summary of the mcmc chain for different
        chain lengths.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset containing MCMC results
        alpha : float
            The alpha level for generating posterior intervals. Defaults to
            0.05.
        start : int
          The starting index from which to summarize (each) chain. Defaults
          to zero.
        batches : int
          Batch size for calculating standard deviation for non-independent
          samples. Defaults to 100.
        chain : int
          The index for which chain to summarize. Defaults to None (all
          chains).
        roundto : int
          The number of digits to round posterior statistics.
        filename : str
            Name of the text

        Returns
        -------
        .txt file
            Summary statistics printed out to a text file of given name
        """

        # open dataset
        ds = xr.open_dataset(netcdf)

        # Extract calibration information needed for priors
        # Variables to plot
        variables = ds.mp.values[:].tolist()
        for i in parameters_all:
            if i in variables:
                variables.remove(i)
        variables.extend(parameters)
        # Observations data
        obs_type_list = []
        for x in range(glacier_cal_data.shape[0]):
            cal_idx = glacier_cal_data.index.values[x]
            obs_type = glacier_cal_data.loc[cal_idx, 'obs_type']
            obs_type_list.append(obs_type)

        # open file to write to
        file = open(filename, 'w')

        for iteration in iters:

            print('\n%s:' % (str(iteration) + ' iterations'), file=file)

            for vn in variables:

                # get trace from database
                trace = ds['mp_value'].sel(chain=0, mp=vn).values[:iteration]

                # Calculate statistics for Node
                statdict = stats(
                    trace,
                    alpha=alpha,
                    start=start,
                    batches=batches,
                    chain=chain)

                size = np.size(statdict['mean'])

                print('\n%s:' % vn, file=file)
                print(' ', file=file)

                # Initialize buffer
                buffer = []

                # Index to interval label
                iindex = [key.split()[-1] for key in statdict.keys()].index('interval')
                interval = list(statdict.keys())[iindex]

                # Print basic stats
                buffer += [
                    'Mean             SD            MC Error(percent of Mean)       %s' %
                    interval]
                buffer += ['-' * len(buffer[-1])]

                indices = range(size)
                if len(indices) == 1:
                    indices = [None]

                _format_str = lambda x, i=None, roundto=2: str(np.round(x.ravel()[i].squeeze(), roundto))

                for index in indices:
                    # Extract statistics and convert to string
                    m = _format_str(statdict['mean'], index, roundto)
                    sd = _format_str(statdict['standard deviation'], index, roundto)
                    mce = _format_str(statdict['mc error'], index, roundto)
                    hpd = str(statdict[interval].reshape(
                            (2, size))[:,index].squeeze().round(roundto))

                    # Build up string buffer of values
                    valstr = m
                    valstr += ' ' * (17 - len(m)) + sd
                    valstr += ' ' * (17 - len(sd)) + mce
                    valstr += ' ' * (len(buffer[-1]) - len(valstr) - len(hpd)) + hpd

                    buffer += [valstr]

                buffer += [''] * 2

                # Print quantiles
                buffer += ['Posterior quantiles:', '']

                buffer += [
                    '2.5             25              50              75             97.5']
                buffer += [
                    ' |---------------|===============|===============|---------------|']

                for index in indices:
                    quantile_str = ''
                    for i, q in enumerate((2.5, 25, 50, 75, 97.5)):
                        qstr = _format_str(statdict['quantiles'][q], index, roundto)
                        quantile_str += qstr + ' ' * (17 - i - len(qstr))
                    buffer += [quantile_str.strip()]

                buffer += ['']

                print('\t' + '\n\t'.join(buffer), file=file)

        file.close()


def stats(trace, alpha=0.05, start=0, batches=100,
              chain=None, quantiles=(2.5, 25, 50, 75, 97.5)):
        """
        Generate posterior statistics for node.

        Parameters
        ----------
        trace : numpy.ndarray
            single dimension array containing mcmc iterations
        alpha : float
          The alpha level for generating posterior intervals. Defaults to
          0.05.
        start : int
          The starting index from which to summarize (each) chain. Defaults
          to zero.
        batches : int
          Batch size for calculating standard deviation for non-independent
          samples. Defaults to 100.
        chain : int
          The index for which chain to summarize. Defaults to None (all
          chains).
        quantiles : tuple or list
          The desired quantiles to be calculated. Defaults to (2.5, 25, 50, 75, 97.5).

        Returns
        -------
        statdict : dict
            dict containing the following statistics of the trace (with the same key names)

            'n': length of mcmc chain
            'standard deviation':
            'mean':
            '%s%s HPD interval' % (int(100 * (1 - alpha)), '%'): utils.hpd(trace, alpha),
            'mc error': mc error as percentage of the mean
            'quantiles':

        """

        n = len(trace)

        return {
            'n': n,
            'standard deviation': trace.std(0),
            'mean': trace.mean(0),
            '%s%s HPD interval' % (int(100 * (1 - alpha)), '%'): utils.hpd(trace, alpha),
            'mc error': base.batchsd(trace, min(n, batches)) / (abs(trace.mean(0)) / 100),
            'quantiles': utils.quantiles(trace, qlist=quantiles)
        }


def write_csv_results(models, variables, distribution_type='truncnormal'):
    """
    Write parameter statistics (mean, standard deviation, effective sample number, gelman_rubin, etc.) to csv.

    Parameters
    ----------
    models : list of pymc.MCMC.MCMC
        Models containing traces of parameters, summary statistics, etc.
    distribution_type : str
        Distribution type either 'truncnormal' or 'uniform' (default truncnormal)

    Returns
    -------
    exports .csv
    """
    model = models[0]
    # Write statistics to csv
    output_csv_fn = (input.mcmc_output_csv_fp + glacier_str + '_' + distribution_type + '_statistics_' +
                     str(len(models)) + 'chain_' + str(input.mcmc_sample_no) + 'iter_' +
                     str(input.mcmc_burn_no) + 'burn' + '.csv')
    model.write_csv(output_csv_fn, variables=['massbal', 'precfactor', 'tempchange', 'ddfsnow'])
    # Import and export csv
    csv_input = pd.read_csv(output_csv_fn)
    # Add effective sample size to csv
    massbal_neff = effective_n(model, 'massbal')
    precfactor_neff = effective_n(model, 'precfactor')
    tempchange_neff = effective_n(model, 'tempchange')
    ddfsnow_neff = effective_n(model, 'ddfsnow')
    effective_n_values = [massbal_neff, precfactor_neff, tempchange_neff, ddfsnow_neff]
    csv_input['n_eff'] = effective_n_values
    # If multiple chains, add Gelman-Rubin Statistic
    if len(models) > 1:
        gelman_rubin_values = []
        for vn in variables:
            gelman_rubin_values.append(gelman_rubin(models, vn))
        csv_input['gelman_rubin'] = gelman_rubin_values
    csv_input.to_csv(output_csv_fn, index=False)


def plot_mc_results(netcdf_fn, glacier_cal_data,
                    iters=50, burn=0, distribution_type='truncnormal',
                    precfactor_mu=input.precfactor_mu, precfactor_sigma=input.precfactor_sigma,
                    precfactor_boundlow=input.precfactor_boundlow,
                    precfactor_boundhigh=input.precfactor_boundhigh,
                    tempchange_mu=input.tempchange_mu, tempchange_sigma=input.tempchange_sigma,
                    tempchange_boundlow=input.tempchange_boundlow,
                    tempchange_boundhigh=input.tempchange_boundhigh,
                    ddfsnow_mu=input.ddfsnow_mu, ddfsnow_sigma=input.ddfsnow_sigma,
                    ddfsnow_boundlow=input.ddfsnow_boundlow, ddfsnow_boundhigh=input.ddfsnow_boundhigh):
    """
    Plot trace, prior/posterior distributions, autocorrelation, and pairwise scatter for each parameter.

    Takes the output from the Markov Chain model and plots the results for the mass balance, temperature change,
    precipitation factor, and degree day factor of snow.  Also, outputs the plots associated with the model.

    Parameters
    ----------
    netcdf_fn : str
        Netcdf of MCMC methods with chains of model parameters
    iters : int
        Number of iterations associated with the Markov Chain
    burn : int
        Number of iterations to burn in with the Markov Chain
    distribution_type : str
        Distribution type either 'truncnormal' or 'uniform' (default truncnormal)
    glacier_RGIId_float : str
    precfactor_mu : float
        Mean of precipitation factor (default assigned from input)
    precfactor_sigma : float
        Standard deviation of precipitation factor (default assigned from input)
    precfactor_boundlow : float
        Lower boundary of precipitation factor (default assigned from input)
    precfactor_boundhigh : float
        Upper boundary of precipitation factor (default assigned from input)
    tempchange_mu : float
        Mean of temperature change (default assigned from input)
    tempchange_sigma : float
        Standard deviation of temperature change (default assigned from input)
    tempchange_boundlow : float
        Lower boundary of temperature change (default assigned from input)
    tempchange_boundhigh: float
        Upper boundary of temperature change (default assigned from input)
    ddfsnow_mu : float
        Mean of degree day factor of snow (default assigned from input)
    ddfsnow_sigma : float
        Standard deviation of degree day factor of snow (default assigned from input)
    ddfsnow_boundlow : float
        Lower boundary of degree day factor of snow (default assigned from input)
    ddfsnow_boundhigh : float
        Upper boundary of degree day factor of snow (default assigned from input)

    Returns
    -------
    .png files
        Saves two figures of (1) trace, histogram, and autocorrelation, and (2) pair-wise scatter plots.
    """
    # Open dataset
    ds = xr.open_dataset(netcdf_fn)
    # Create list of model output to be used with functions
    dfs = []
    for n_chain in ds.chain.values:
        dfs.append(pd.DataFrame(ds['mp_value'].sel(chain=n_chain).values[burn:burn+iters], columns=ds.mp.values))

    # Extract calibration information needed for priors
    # Variables to plot
    variables = ds.mp.values[:].tolist()
    for i in parameters_all:
        if i in variables:
            variables.remove(i)
    variables.extend(parameters)
    # Observations data
    obs_list = []
    obs_err_list_raw = []
    obs_type_list = []
    for x in range(glacier_cal_data.shape[0]):
        cal_idx = glacier_cal_data.index.values[x]
        obs_type = glacier_cal_data.loc[cal_idx, 'obs_type']
        obs_type_list.append(obs_type)
        # Mass balance comparisons
        if glacier_cal_data.loc[cal_idx, 'obs_type'].startswith('mb'):
            # Mass balance [mwea]
            t1 = glacier_cal_data.loc[cal_idx, 't1'].astype(int)
            t2 = glacier_cal_data.loc[cal_idx, 't2'].astype(int)
            observed_massbal = glacier_cal_data.loc[cal_idx,'mb_mwe'] / (t2 - t1)
            observed_error = glacier_cal_data.loc[cal_idx,'mb_mwe_err'] / (t2 - t1)
            obs_list.append(observed_massbal)
            obs_err_list_raw.append(observed_error)
    obs_err_list = [x if ~np.isnan(x) else np.nanmean(obs_err_list_raw) for x in obs_err_list_raw]

    # ===== CHAIN, HISTOGRAM, AND AUTOCORRELATION PLOTS ===========================
    plt.figure(figsize=(12, len(variables)*3))
    plt.subplots_adjust(wspace=0.3, hspace=0.5)
    plt.suptitle('mcmc_ensembles_' + glacier_str + '_' + distribution_type, y=0.94)

    # Bounds (SciPy convention)
    precfactor_a = (precfactor_boundlow - precfactor_mu) / precfactor_sigma
    precfactor_b = (precfactor_boundhigh - precfactor_mu) / precfactor_sigma
    tempchange_a = (tempchange_boundlow - tempchange_mu) / tempchange_sigma
    tempchange_b = (tempchange_boundhigh - tempchange_mu) / tempchange_sigma
    ddfsnow_a = (ddfsnow_boundlow - ddfsnow_mu) / ddfsnow_sigma
    ddfsnow_b = (ddfsnow_boundhigh - ddfsnow_mu) / ddfsnow_sigma

    # Labels for plots
    vn_label_dict = {}
    vn_label_nounits_dict = {}
    obs_count = 0
    for vn in variables:
        if vn.startswith('obs'):
            if obs_type_list[obs_count].startswith('mb'):
                vn_label_dict[vn] = 'Mass balance ' + str(n) + '\n[mwea]'
                vn_label_nounits_dict[vn] = 'MB ' + str(n)
            obs_count += 1
        elif vn == 'massbal':
            vn_label_dict[vn] = 'Mass balance\n[mwea]'
            vn_label_nounits_dict[vn] = 'MB'
        elif vn == 'precfactor':
            vn_label_dict[vn] = 'Precipitation factor\n[-]'
            vn_label_nounits_dict[vn] = 'Prec factor'
        elif vn == 'tempchange':
            vn_label_dict[vn] = 'Temperature bias\n[degC]'
            vn_label_nounits_dict[vn] = 'Temp bias'
        elif vn == 'ddfsnow':
            vn_label_dict[vn] = 'DDFsnow\n[mwe $degC^{-1} d^{-1}$]'
            vn_label_nounits_dict[vn] = 'DDFsnow'

    for count, vn in enumerate(variables):
        # ===== Chain =====
        plt.subplot(len(variables), 3, 3*count+1)
        chain_legend = []
        for n_df, df in enumerate(dfs):
            chain = df[vn].values
            runs = np.arange(0,chain.shape[0])
            if n_df == 0:
                plt.plot(runs, chain, color='b')
            elif n_df == 1:
                plt.plot(runs, chain, color='r')
            else:
                plt.plot(runs, chain, color='y')
            chain_legend.append('chain' + str(n_df + 1))
        plt.legend(chain_legend)
        plt.xlabel('Step Number', size=10)
        plt.ylabel(vn_label_dict[vn], size=10)
        # ===== Prior and posterior distributions =====
        plt.subplot(len(variables), 3, 3*count+2)
        # Prior distribution
        z_score = np.linspace(norm.ppf(0.01), norm.ppf(0.99), 100)
        if vn.startswith('obs'):
            observed_massbal = obs_list[int(vn.split('_')[1])]
            observed_error = obs_err_list[int(vn.split('_')[1])]
            x_values = observed_massbal + observed_error * z_score
            y_values = norm.pdf(x_values, loc=observed_massbal, scale=observed_error)
        elif vn == 'massbal':
            observed_massbal = obs_list[0]
            observed_error = obs_err_list[0]
            x_values = observed_massbal + observed_error * z_score
            y_values = norm.pdf(x_values, loc=observed_massbal, scale=observed_error)
        elif vn == 'precfactor':
            if distribution_type == 'truncnormal':
                z_score = np.linspace(truncnorm.ppf(0.01, precfactor_a, precfactor_b),
                                      truncnorm.ppf(0.99, precfactor_a, precfactor_b), 100)
                x_values_raw = precfactor_mu + precfactor_sigma * z_score
                y_values = truncnorm.pdf(x_values_raw, precfactor_a, precfactor_b, loc=precfactor_mu,
                                         scale=precfactor_sigma)
            elif distribution_type == 'uniform':
                z_score = np.linspace(uniform.ppf(0.01), uniform.ppf(0.99), 100)
                x_values_raw = precfactor_boundlow + z_score * (precfactor_boundhigh - precfactor_boundlow)
                y_values = uniform.pdf(x_values_raw, loc=precfactor_boundlow,
                                       scale=(precfactor_boundhigh - precfactor_boundlow))
            # transform the precfactor values from the truncated normal to the actual values
            x_values = prec_transformation(x_values_raw)
        elif vn == 'tempchange':
            if distribution_type == 'truncnormal':
                z_score = np.linspace(truncnorm.ppf(0.01, tempchange_a, tempchange_b),
                                      truncnorm.ppf(0.99, tempchange_a, tempchange_b), 100)
                x_values = tempchange_mu + tempchange_sigma * z_score
                y_values = truncnorm.pdf(x_values, tempchange_a, tempchange_b, loc=tempchange_mu,
                                         scale=tempchange_sigma)
            elif distribution_type == 'uniform':
                z_score = np.linspace(uniform.ppf(0.01), uniform.ppf(0.99), 100)
                x_values = tempchange_boundlow + z_score * (tempchange_boundhigh - tempchange_boundlow)
                y_values = uniform.pdf(x_values, loc=tempchange_boundlow,
                                       scale=(tempchange_boundhigh - tempchange_boundlow))
        elif vn == 'ddfsnow':
            if distribution_type == 'truncnormal':
                z_score = np.linspace(truncnorm.ppf(0.01, ddfsnow_a, ddfsnow_b),
                                      truncnorm.ppf(0.99, ddfsnow_a, ddfsnow_b), 100)
                x_values = ddfsnow_mu + ddfsnow_sigma * z_score
                y_values = truncnorm.pdf(x_values, ddfsnow_a, ddfsnow_b, loc=ddfsnow_mu, scale=ddfsnow_sigma)
            elif distribution_type == 'uniform':
                z_score = np.linspace(uniform.ppf(0.01), uniform.ppf(0.99), 100)
                x_values = ddfsnow_boundlow + z_score * (ddfsnow_boundhigh - ddfsnow_boundlow)
                y_values = uniform.pdf(x_values, loc=ddfsnow_boundlow,
                                       scale=(ddfsnow_boundhigh - ddfsnow_boundlow))
        plt.plot(x_values, y_values, color='k')
        # Ensemble/Posterior distribution
        # extents
        if chain.min() < x_values.min():
            x_min = chain.min()
        else:
            x_min = x_values.min()
        if chain.max() > x_values.max():
            x_max = chain.max()
        else:
            x_max = x_values.max()
        # Chain legend
        if vn.startswith('obs'):
            chain_legend = ['observed']
        else:
            chain_legend = ['prior']
        # Loop through models
        for n_chain, df in enumerate(dfs):
            chain = df[vn].values
            # gaussian distribution
            if vn.startswith('obs'):
                kde = gaussian_kde(chain)
                x_values_kde = np.linspace(x_min, x_max, 100)
                y_values_kde = kde(x_values_kde)
                chain_legend.append('ensemble' + str(n_chain + 1))
            elif vn == 'precfactor':
                kde = gaussian_kde(chain)
                x_values_kde = x_values.copy()
                y_values_kde = kde(x_values_kde)
                chain_legend.append('posterior' + str(n_chain + 1))
            else:
                kde = gaussian_kde(chain)
                x_values_kde = x_values.copy()
                y_values_kde = kde(x_values_kde)
                chain_legend.append('posterior' + str(n_chain + 1))
            if n_chain == 0:
                plt.plot(x_values_kde, y_values_kde, color='b')
            elif n_chain == 1:
                plt.plot(x_values_kde, y_values_kde, color='r')
            else:
                plt.plot(x_values_kde, y_values_kde, color='y')
            plt.xlabel(vn_label_dict[vn], size=10)
            plt.ylabel('PDF', size=10)
            plt.legend(chain_legend)

        # ===== Normalized autocorrelation ======
        plt.subplot(len(variables), 3, 3*count+3)
        chain_norm = chain - chain.mean()
        if chain.shape[0] <= acorr_maxlags:
            acorr_lags = chain.shape[0] - 1
        else:
            acorr_lags = acorr_maxlags
        plt.acorr(chain_norm, maxlags=acorr_lags)
        plt.xlim(0,acorr_lags)
        plt.xlabel('lag')
        plt.ylabel('autocorrelation')
        chain_neff = effective_n(ds, vn, iters=iters, burn=burn)
        plt.text(int(0.6*acorr_lags), 0.85, 'n_eff=' + str(chain_neff))
    # Save figure
    plt.savefig(mcmc_output_figures_fp + glacier_str + '_' + distribution_type + '_plots_' + str(len(dfs)) + 'chain_'
                + str(iters) + 'iter_' + str(burn) + 'burn' + '.png', bbox_inches='tight')

    # ===== PAIRWISE SCATTER PLOTS ===========================================================
    fig = plt.figure(figsize=(10,12))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.suptitle('mcmc_pairwise_scatter_' + glacier_str + '_' + distribution_type, y=0.94)

    df = dfs[0]
    nvars = len(variables)
    for h, vn1 in enumerate(variables):
        v1 = chain = df[vn1].values
        for j, vn2 in enumerate(variables):
            v2 = chain = df[vn2].values
            nsub = h * nvars + j + 1
            ax = fig.add_subplot(nvars, nvars, nsub)
            if h == j:
                plt.hist(v1)
                plt.tick_params(axis='both', bottom=False, left=False, labelleft=False, labelbottom=False)
            elif h > j:
                plt.plot(v2, v1, 'o', mfc='none', mec='black')
            else:
                # Need to plot blank, so axis remain correct
                plt.plot(v2, v1, 'o', mfc='none', mec='none')
                slope, intercept, r_value, p_value, std_err = linregress(v2, v1)
                text2plot = (vn_label_nounits_dict[vn2] + '/\n' + vn_label_nounits_dict[vn1] + '\n$R^2$=' +
                             '{:.2f}'.format((r_value**2)))
                ax.text(0.5, 0.5, text2plot, transform=ax.transAxes, fontsize=14,
                        verticalalignment='center', horizontalalignment='center')
            # Plot bottom left
            if (h+1 == nvars) and (j == 0):
                plt.tick_params(axis='both', which='both', left=True, right=False, labelbottom=True,
                                labelleft=True, labelright=False)
                plt.xlabel(vn_label_dict[vn2])
                plt.ylabel(vn_label_dict[vn1])
            # Plot bottom only
            elif h + 1 == nvars:
                plt.tick_params(axis='both', which='both', left=False, right=False, labelbottom=True,
                                labelleft=False, labelright=False)
                plt.xlabel(vn_label_dict[vn2])
            # Plot left only (exclude histogram values)
            elif (h !=0) and (j == 0):
                plt.tick_params(axis='both', which='both', left=True, right=False, labelbottom=False,
                                labelleft=True, labelright=False)
                plt.ylabel(vn_label_dict[vn1])
            else:
                plt.tick_params(axis='both', left=False, right=False, labelbottom=False,
                                labelleft=False, labelright=False)
    plt.savefig(mcmc_output_figures_fp + glacier_str + '_' + distribution_type + '_pairwisescatter_' + str(len(dfs)) +
                'chain_' + str(iters) + 'iter_' + str(burn) + 'burn' + '.png', bbox_inches='tight')


def plot_mc_results2(netcdf_fn, glacier_cal_data, burns=[0,1000,3000,5000],
                     plot_res=1000, distribution_type='truncnormal'):
    """
    Plot gelman-rubin statistic, effective_n (autocorrelation with lag
    100) and markov chain error plots.

    Takes the output from the Markov Chain model and plots the results
    for the mass balance, temperature change, precipitation factor,
    and degree day factor of snow.  Also, outputs the plots associated
    with the model.

    Parameters
    ----------
    netcdf_fn : str
        Netcdf of MCMC methods with chains of model parameters
    iters : int
        Number of iterations associated with the Markov Chain
    burn : list of ints
        List of burn in values to plot for Gelman-Rubin stats
    plot_res: int
        Interval of points for which GR and MCerror statistic are calculated.
        (Lower value leads to higher plot resolution)
    glacier_RGIId_float : str
    precfactor_mu : float
        Mean of precipitation factor (default assigned from input)
    tempchange_mu : float
        Mean of temperature change (default assigned from input)
    ddfsnow_mu : float
        Mean of degree day factor of snow (default assigned from input)

    Returns
    -------
    .png files
        Saves two figures of (1) trace, histogram, and autocorrelation, and (2) pair-wise scatter plots.
    """
    # Open dataset
    ds = xr.open_dataset(netcdf_fn)

    # Extract calibration information needed for priors
    # Variables to plot
    variables = ds.mp.values[:].tolist()
    for i in parameters_all:
        if i in variables:
            variables.remove(i)
    variables.extend(parameters)
    # Observations data
    obs_type_list = []
    for x in range(glacier_cal_data.shape[0]):
        cal_idx = glacier_cal_data.index.values[x]
        obs_type = glacier_cal_data.loc[cal_idx, 'obs_type']
        obs_type_list.append(obs_type)

    # Titles for plots
    vn_title_dict = {}
    for n, vn in enumerate(variables):
        if vn.startswith('obs'):
            if obs_type_list[n].startswith('mb'):
                vn_title_dict[vn] = 'Mass Balance ' + str(n)
        elif vn == 'massbal':
            vn_title_dict[vn] = 'Mass Balance'
        elif vn == 'precfactor':
            vn_title_dict[vn] = 'Precipitation Factor'
        elif vn == 'tempchange':
            vn_title_dict[vn] = 'Temperature Bias'
        elif vn == 'ddfsnow':
            vn_title_dict[vn] = 'DDF Snow'

    # get variables and burn length for dimension
    v_len = len(variables)
    b_len = len(burns)
    c_len = len(ds.mp_value)
    no_chains = len(ds.chain)

    # hard code figure sizes
    figsize = (6.5, 5)
    dpi = 100
    wspace = 0.3
    hspace = 0.5
    title = 12
    label = 10
    suptitle = 14
    sup_y = 0.97
    nrows=2
    ncols=2

    # ====================== GELMAN-RUBIN PLOTS ===========================

    plt.figure(figsize=figsize, dpi=dpi)
    plt.subplots_adjust(wspace=wspace, hspace=hspace)
    plt.suptitle('Gelman-Rubin Statistic vs Number of MCMC Steps',
                 fontsize=suptitle, y=sup_y)

    for v_count, vn in enumerate(variables):

        plt.subplot(nrows, ncols, v_count+1)

        for b_count, burn in enumerate(burns):

            plot_list = list(range(burn+plot_res, c_len+plot_res, plot_res))
            gr_list = [gelman_rubin(ds, vn, pt, burn) for pt in plot_list]

            # plot GR
            plt.plot(plot_list, gr_list, label='Burn-In ' + str(burn))

            # plot horizontal line for benchmark
            plt.axhline(1.01, color='black', linestyle='--', linewidth=1)

            if v_count % 2 == 0:
                plt.ylabel('Gelman-Rubin Value', size=label)

            if b_count == 0:
                plt.title(vn_title_dict[vn], size=title)

            if v_count == v_len-1:
                plt.legend()

            # niceties
            plt.xlabel('Step Number', size=label)

    # Save figure
    plt.savefig(mcmc_output_figures_fp + glacier_str + '_' + distribution_type +
                '_gelman-rubin' + '_plots_' + str(no_chains) + 'chain_' +
                str(c_len) + 'iter' + '.png', bbox_inches='tight')

    # ====================== MC ERROR PLOTS ===========================

    plt.figure(figsize=figsize, dpi=dpi)
    plt.subplots_adjust(wspace=wspace, hspace=hspace)
    plt.suptitle('MC Error as Percentage of Mean vs Number of MCMC Steps', y=sup_y)

    for v_count, vn in enumerate(variables):

        plt.subplot(nrows, ncols, v_count+1)

        # points to plot at
        plot_list = list(range(0, c_len+plot_res, plot_res))

        # find mean
        total_mean = abs(np.mean(ds['mp_value'].sel(chain=0, mp=vn).values))

        mce_list = []
        #mean_list = []

        # calculate mc error and mean at each point
        for pt in plot_list:

            mce, mean = MC_error(ds, vn, iters=pt)
            mce_list.append(mce)
            #mean_list.append(abs(mean) / 100)

        # plot
        plt.plot(plot_list, mce_list / (total_mean / 100))
        plt.axhline(1, color='orange', label='1% of Mean', linestyle='--')
        plt.axhline(3, color='green', label='3% of Mean', linestyle='--')

        if v_count % 2 == 0:
            plt.ylabel('MC Error [% of mean]', size=label)

        if v_count == v_len-1:
            plt.legend()

        # niceties
        plt.xlabel('Step Number', size=label)
        plt.title(vn_title_dict[vn], size=title)

    # Save figure
    plt.savefig(mcmc_output_figures_fp + glacier_str + '_' + distribution_type +
                '_mc-error' + '_plots_' + str(no_chains) + 'chain_' +
                str(c_len) + 'iter' + '.png', bbox_inches='tight')

    # ====================== EFFECTIVE_N PLOTS ===========================

    plt.figure(figsize=figsize, dpi=dpi)
    plt.subplots_adjust(wspace=wspace, hspace=hspace)
    plt.suptitle('Effective Sample Size vs Number of MCMC Steps', y=sup_y)

    # get dataframe
    #df = ds['mp_value'].sel(chain=0).to_pandas()

    for v_count, vn in enumerate(variables):

        plt.subplot(nrows, ncols, v_count+1)

        for b_count, burn in enumerate(burns):

            # points to plot at
            plot_list = list(range(burn+plot_res, c_len+plot_res, plot_res))
            #en_list = [effective_n(df[burn:pt], vn) for pt in plot_list]
            en_list = [effective_n(ds, vn=vn, iters=pt, burn=burn) for pt in plot_list]
            # plot
            plt.plot(plot_list, en_list, label='Burn-In ' + str(burn))

            if v_count == 0:
                plt.ylabel('Effective Sample Size', size=label)

            if v_count == v_len-1:
                plt.legend()

            if b_count == 0:
                plt.title(vn_title_dict[vn], size=title)

            # niceties
            plt.xlabel('Step Number', size=label)

    # Save figure
    plt.savefig(mcmc_output_figures_fp + glacier_str + '_' + distribution_type +
                '_effective-n' + '_plots_' + str(no_chains) + 'chain_' +
                str(c_len) + 'iter' + '.png', bbox_inches='tight')


def plot_mc_results3(iters, region='all', burn=0):
    """
    Plot gelman-rubin statistic, effective_n (autocorrelation with lag
    100) and markov chain error plots.

    Takes the output from the Markov Chain model and plots the results
    for the mass balance, temperature change, precipitation factor,
    and degree day factor of snow.  Also, outputs the plots associated
    with the model.

    Parameters
    ----------
    iters : int
        Number of iterations associated with the Markov Chain
    burn : list of ints
        List of burn in values to plot for Gelman-Rubin stats

    Returns
    -------
    .png files
        saves figure showing how assessment values change with
        number of mcmc iterations
    """

    # hard code some variable names (dirty solution)
    variables = ['massbal', 'precfactor', 'tempchange', 'ddfsnow']
    vn_title_dict = {'massbal':'Mass Balance',
                     'precfactor':'Precipitation Factor',
                     'tempchange':'Temperature Bias',
                     'ddfsnow':'DDF Snow'}
    metric_title_dict = {'Gelman-Rubin':'Gelman-Rubin Statistic',
                         'MC Error': 'Monte Carlo Error',
                         'Effective N': 'Effective Sample Size'}
    metrics = ['Gelman-Rubin', 'MC Error', 'Effective N']

    # hard code font sizes
    ticks=10
    suptitle=14
    title=10
    titley = 1.05
    label=10
    plotline=2
    plotline2=1
    legend=10
    figsize=(6.5, 9)
    dpi=100
    hspace=0.6
    wspace=0.6
    sup_y = 0.97
    nrows=4
    ncols=3
    num_stds=1
    alpha = 0.7
    s_alpha = 0.5

    # bins and ticks
    bdict = {}
    tdict = {}

    plt.figure(figsize=figsize, dpi=dpi)
    plt.subplots_adjust(wspace=wspace, hspace=hspace)

    m_len = len(metrics)
    v_len = len(variables)

    # create subplot for each variable
    for v_count, vn in enumerate(variables):

        df = pd.read_csv(mcmc_output_csv_fp + 'assessment_plot_' +
                         str(region) + 'region_' + str(burn) +
                         'burn_' + str(vn) + '.csv')

        #create subplot for each metric
        for m_count, metric in enumerate(metrics):

            x = df['Iter']
            mean = df[metric + ' mean']
            std = df[metric + ' std']

            # plot histogram
            ax = plt.subplot(nrows, ncols, m_len*v_count+m_count+1)

            ax.plot(x, mean, alpha=alpha)

            #plot error region
            ax.fill_between(x, mean-(num_stds*std),
                     mean+(num_stds*std), alpha=s_alpha)

            # niceties
            if v_count == 0:
                plt.title(metric_title_dict[metric], fontsize=title, y=titley)

            # axis labels
            if metric=='MC Error':
                ylabel = metric
            else:
                ylabel = metric + ' value'

            if m_count == 0:
                ax.set_ylabel(vn_title_dict[vn] + '\n\n' + ylabel, fontsize=label, labelpad=0)
            else:
                ax.set_ylabel(ylabel, fontsize=label)

    # Save figure
    plt.savefig(mcmc_output_figures_fp + 'assessment_plot' + str(region) +
                'region_' + str(burn) + 'burn.png',
                bbox_inches='tight')

def write_table(region=15, iters=1000, burn=0):
    '''
    Writes a csv table that lists MCMC assessment values for
    each glacier (represented by a netcdf file.


    Writes out the values of effective_n (autocorrelation with
    lag 100), Gelman-Rubin Statistic, MC_error.

    Parameters
    ----------
    region : int
        number of the glacier region (13, 14 or 15)
    iters : int
        Number of iterations associated with the Markov Chain
    burn : list of ints
        List of burn in values to plot for Gelman-Rubin stats

    Returns
    -------
    dfs : list of pandas.DataFrame
        dataframes containing statistical information for all glaciers
    .csv files
        Saves tables to csv file.

    '''

    dfs=[]

    variables = ['massbal', 'precfactor', 'tempchange', 'ddfsnow']

    # find all netcdf files (representing glaciers)
    if region == 'all':
        regions = ['13', '14', '15']
        filelist = []
        for reg in regions:
            filelist.extend(glob.glob(mcmc_output_netcdf_fp + str(reg) + '*.nc'))
    else:
        filelist = glob.glob(mcmc_output_netcdf_fp + str(region) + '*.nc')

    # for testing
    filelist = filelist[10:20]

    for vn in variables:

        # create lists of each value
        glac_no = []
        effective_n_list = []
        gelman_rubin_list = []
        mc_error = []


        # iterate through each glacier
        for netcdf in filelist:
            print(netcdf)

            try:
                # open dataset
                ds = xr.open_dataset(netcdf)

                # calculate metrics
                en = effective_n(ds, vn=vn, iters=iters, burn=burn)
                mc = MC_error(ds, vn=vn, iters=iters, burn=burn)[0]


                # divide MC Error by the mean values
                mean = abs(np.mean(ds['mp_value'].sel(chain=0, mp=vn).values))
                mc /= mean
                mc *= 100.0
                mc_error.append(mc)

                if len(ds.chain) > 1:
                    gr = gelman_rubin(ds, vn=vn, iters=iters, burn=burn)

                # find values for this glacier and append to lists
                glac_no.append(netcdf[-11:-3])
                effective_n_list.append(en)
                # test if multiple chains exist
                if len(ds.chain) > 1:
                    gelman_rubin_list.append(gr)

                ds.close()

            except:
                print('Error, glacier: ', netcdf)
                pass

        # create dataframe
        data = {'Glacier': glac_no,
                'Effective N' : effective_n_list,
                'MC Error' : mc_error}
        if len(gelman_rubin_list) > 0:
            data['Gelman-Rubin'] = gelman_rubin_list
        df = pd.DataFrame(data)
        df.set_index('Glacier', inplace=True)

        # save csv
        df.to_csv(mcmc_output_csv_fp + 'region' + str(region) + '_' +
                  str(iters) + 'iterations_' + str(burn) + 'burn_' + str(vn) + '.csv')

        dfs.append(df)

    return dfs


def write_table2(iters, region='all', burn=0):
    '''
    Writes a csv table that lists mean MCMC assessment values for
    each glacier (represented by a netcdf file) for all glaciers at
    different chain lengths.

    Writes out the values of effective_n (autocorrelation with
    lag 100), Gelman-Rubin Statistic, MC_error.

    Parameters
    ----------
    region : int
        number of the glacier region (13, 14 or 15)
    iters : int
        Number of iterations associated with the Markov Chain
    burn : list of ints
        List of burn in values to plot for Gelman-Rubin stats

    Returns
    -------
    .csv files
        Saves tables to csv file.

    '''

    variables = ['massbal', 'precfactor', 'tempchange', 'ddfsnow']

    # find all netcdf files (representing glaciers)
    if region == 'all':
        regions = ['13', '14', '15']
        filelist = []
        for reg in regions:
            filelist.extend(glob.glob(mcmc_output_netcdf_fp + str(reg) + '*.nc'))
    else:
        filelist = glob.glob(mcmc_output_netcdf_fp + str(region) + '*.nc')

    # for testing
    #filelist = filelist[3:6]

    for vn in variables:

        # create lists of each value
        glac_no = []
        effective_n_list = []
        gelman_rubin_list = []
        mc_error = []


        # iterate through each glacier
        for netcdf in filelist:
            print(netcdf)

            try:
                # open dataset
                ds = xr.open_dataset(netcdf)

                # calculate metrics
                en = [effective_n(ds, vn=vn, iters=i, burn=burn) for i in iters]
                mc = [MC_error(ds, vn=vn, iters=i, burn=burn)[0] for i in iters]
                if len(ds.chain) > 1:
                    gr = [gelman_rubin(ds, vn=vn, iters=i, burn=burn) for i in iters]

                # find values for this glacier and append to lists
                glac_no.append(netcdf[-11:-3])
                effective_n_list.append(en)
                # test if multiple chains exist
                if len(ds.chain) > 1:
                    gelman_rubin_list.append(gr)

                # divide MC Error by the mean values
                mean = abs(np.mean(ds['mp_value'].sel(chain=0, mp=vn).values))
                print(vn)
                print(mean)
                print(mc)
                mc /= mean
                print(mc)
                mc *= 100.0
                print(mc)
                mc_error.append(mc)

                ds.close()

            except:
                print('Error, glacier: ', netcdf)
                pass

        # do averaging operations
        effective_n_list_mean = np.mean(effective_n_list, axis=0)
        gelman_rubin_list_mean = np.mean(gelman_rubin_list, axis=0)
        mc_error_mean = np.mean(mc_error, axis=0)
        effective_n_list_std = np.std(effective_n_list, axis=0)
        gelman_rubin_list_std = np.std(gelman_rubin_list, axis=0)
        mc_error_std = np.std(mc_error, axis=0)

        # create dataframe
        data = {'Iter': iters,
                'Effective N mean' : effective_n_list_mean,
                'MC Error mean' : mc_error_mean,
                'Effective N std' : effective_n_list_std,
                'MC Error std' : mc_error_std}
        if len(gelman_rubin_list) > 0:
            data['Gelman-Rubin mean'] = gelman_rubin_list_mean
            data['Gelman-Rubin std'] = gelman_rubin_list_std
        df = pd.DataFrame(data)
        df.set_index('Iter', inplace=True)

        # save csv
        df.to_csv(mcmc_output_csv_fp + 'assessment_plot2_' + str(region) + 'region_' +
                  str(burn) + 'burn_' + str(vn) + '.csv')


def plot_histograms(iters, burn, region=15, dfs=None):
    '''
    Plots histograms to assess mcmc chains for groups of glaciers

    Plots histograms of effective_n, gelman-rubin and mc error for
    the given number of iterations and burn-in and the given variable.

    For this function to work, the appropriate csv file must have already
    been created.

    Parameters
    ----------
    dfs : list of pandas.DataFrame
        list of dataframes containing glacier information to be plotted. If
        none, looks for appropriate csv file
    vn : str
        Name of variable (massbal, ddfsnow, precfactor, tempchange)
    iters : int
        Number of iterations associated with the Markov Chain
    burn : list of ints
        List of burn in values to plot for Gelman-Rubin stats

    Returns
    -------
    .png files
        Saves images to 3 png files.

    '''

    # hard code some variable names (dirty solution)
    variables = ['massbal', 'precfactor', 'tempchange', 'ddfsnow']
    vn_title_dict = {'massbal':'Mass Balance',
                     'precfactor':'Precipitation Factor',
                     'tempchange':'Temperature Bias',
                     'ddfsnow':'DDF Snow'}
    metrics = ['Gelman-Rubin', 'MC Error', 'Effective N']

    vn_df_dict = {}

    # read csv files
    for vn in variables:
        vn_df_dict[vn] = pd.read_csv(mcmc_output_csv_fp + 'region' +
                                     str(region) + '_' + str(iters) +
                                     'iterations_' + str(burn) + 'burn_' +
                                     str(vn) + '.csv')


    # get variables and burn length for dimension
    v_len = len(variables)

    # create plot for each metric
    for metric in metrics:

        plt.figure(figsize=(v_len*4, 3))
        plt.subplots_adjust(wspace=0.3, hspace=0.3)

        if metric is 'MC Error':
            plt.suptitle(metric + ' (as percentage of mean) Histrogram ' +
                         str(iters) + ' iterations ' + str(burn) + ' burn-in', y=1.05,
                         fontsize=14)
        else:
            plt.suptitle(metric + ' Histrogram ' +
                         str(iters) + ' iterations ' + str(burn) + ' burn-in', y=1.10,
                         fontsize=14)

        # create subplot for each variable
        for v_count, vn in enumerate(variables):

            df = vn_df_dict[vn]

            # plot histogram
            plt.subplot(1, v_len, v_count+1)
            n, bins, patches = plt.hist(x=df[metric], bins=30, alpha=.4, edgecolor='black',
                                        color='#0504aa')


            # niceties
            plt.title(vn_title_dict[vn])

            if v_count == 0:
                plt.ylabel('Frequency')


        # Save figure
        plt.savefig(mcmc_output_hist_fp + 'region' + str(region) + '_' + str(iters) +
                    'iterations_' + str(burn) + 'burn_' + str(metric.replace(' ','_')) + '.png')


def plot_histograms_2(iters, burn, region=15, dfs=None):
    '''
    Plots histograms to assess mcmc chains for groups of glaciers.
    Puts them all in one image file.

    Plots histograms of effective_n, gelman-rubin and mc error for
    the given number of iterations and burn-in and the given variable.

    For this function to work, the appropriate csv file must have already
    been created.

    Parameters
    ----------
    iters : int
        Number of iterations associated with the Markov Chain
    burn : list of ints
        List of burn in values to plot for Gelman-Rubin stats
    region : int
        RGI region number  or 'all'
    dfs : list of pandas.DataFrame
        list of dataframes containing glacier information to be plotted. If
        none, looks for appropriate csv file

    Returns
    -------
    .png files
        Saves images to 3 png files.

    '''

    # hard code some variable names (dirty solution)
    variables = ['massbal', 'precfactor', 'tempchange', 'ddfsnow']
    vn_title_dict = {'massbal':'Mass Balance',
                     'precfactor':'Precipitation Factor',
                     'tempchange':'Temperature Bias',
                     'ddfsnow':'DDF Snow'}
    metric_title_dict = {'Gelman-Rubin':'Gelman-Rubin Statistic',
                         'MC Error': 'Monte Carlo Error',
                         'Effective N': 'Effective Sample Size'}
    metrics = ['Gelman-Rubin', 'MC Error', 'Effective N']

    if region in [13, 14, 15]:
        test = pd.read_csv(mcmc_output_csv_fp + 'region' +
                           str(region) + '_' + str(iters) +
                           'iterations_' + str(burn) + 'burn_' +
                           str('massbal') + '.csv')
    elif region == 'all':
        test = pd.read_csv(mcmc_output_csv_fp + 'region' +
                           str(13) + '_' + str(iters) +
                           'iterations_' + str(burn) + 'burn_' +
                           str('massbal') + '.csv')

    # determine whether Gelman-Rubin has been computed
    if 'Gelman-Rubin' in test.columns:
        metrics = ['Gelman-Rubin', 'MC Error', 'Effective N']
    else:
        metrics = ['MC Error', 'Effective N']

    # hard code font sizes
    ticks=10
    suptitle=14
    title=11
    titley = 1.05
    label=10
    plotline=2
    plotline2=1
    legend=10
    figsize=(6.5, 9)
    dpi=100
    hspace=0.6
    wspace=0.6

    # bins and ticks
    bdict = {}
    tdict = {}

    if suffix=='_trunc':
        bdict['MC Error massbal'] = np.arange(0, 2.5, 0.125)
        bdict['MC Error tempchange'] = np.arange(0, 2.5, 0.125)
        bdict['MC Error ddfsnow'] = np.arange(0, 2.5, 0.125)
        bdict['MC Error precfactor'] = np.arange(0, 2.5, 0.125)
        bdict['Gelman-Rubin massbal'] = np.arange(1, 1.002, 0.0001)
        bdict['Gelman-Rubin precfactor'] = np.arange(1.0, 1.006, 0.0003)
        bdict['Gelman-Rubin tempchange'] = np.arange(1.00, 1.02, 0.001)
        bdict['Gelman-Rubin ddfsnow'] = np.arange(1.0, 1.006, 0.0003)
        bdict['Effective N massbal'] = 20
        bdict['Effective N ddfsnow'] = np.arange(0, 2500, 125)
        bdict['Effective N tempchange'] = np.arange(0, 1000, 50)
        bdict['Effective N precfactor'] = np.arange(0, 1600, 80)
        tdict['MC Error'] = np.arange(0, 19, 4)
        tdict['Gelman-Rubin'] = np.arange(0, 30, 5)
        tdict['Effective N'] = np.arange(0, 19, 4)
    else:
        bdict['MC Error massbal'] = np.arange(0, 3, 0.15)
        bdict['MC Error tempchange'] = np.arange(0, 3, 0.15)
        bdict['MC Error ddfsnow'] = np.arange(0, 3, 0.15)
        bdict['MC Error precfactor'] = np.arange(0, 3, 0.15)
        bdict['Gelman-Rubin massbal'] = np.arange(1, 1.002, 0.0001)
        bdict['Gelman-Rubin precfactor'] = np.arange(1.0, 1.006, 0.0003)
        bdict['Gelman-Rubin tempchange'] = np.arange(1.00, 1.02, 0.001)
        bdict['Gelman-Rubin ddfsnow'] = np.arange(1.0, 1.006, 0.0003)
        bdict['Effective N massbal'] = 20
        bdict['Effective N ddfsnow'] = np.arange(0, 2500, 125)
        bdict['Effective N tempchange'] = np.arange(0, 1000, 50)
        bdict['Effective N precfactor'] = np.arange(0, 1600, 80)
        tdict['MC Error'] = np.arange(0, 19, 4)
        tdict['Gelman-Rubin'] = np.arange(0, 30, 5)
        tdict['Effective N'] = np.arange(0, 19, 4)

    # read csv files
    vn_df_dict = {}
    if region in [13, 14, 15]:
        for vn in variables:
            vn_df_dict[vn] = pd.read_csv(mcmc_output_csv_fp + 'region' +
                                         str(region) + '_' + str(iters) +
                                         'iterations_' + str(burn) + 'burn_' +
                                         str(vn) + '.csv')
    elif region == 'all':
        for vn in variables:
            regions = [13, 14, 15]
            dfs = []
            for reg in regions:
                dfs.append(pd.read_csv(mcmc_output_csv_fp + 'region' +
                                       str(reg) + '_' + str(iters) +
                                       'iterations_' + str(burn) + 'burn_' +
                                       str(vn) + '.csv'))
            vn_df_dict[vn] = pd.concat(dfs)

    # get variables and burn length for dimension
    v_len = len(variables)
    m_len = len(metrics)

    # create figure
    fig = plt.figure(figsize=figsize, dpi=dpi)
    plt.subplots_adjust(wspace=wspace, hspace=hspace)

    # write title
    #plt.suptitle('MC Metrics Assessment Histograms ' +
                 #str(iters) + ' iterations ' + str(burn) + ' burn-in',
                 #fontsize=suptitle, y=0.97)

    #create subplot for each metric
    for m_count, metric in enumerate(metrics):

        # create subplot for each variable
        for v_count, vn in enumerate(variables):

            df = vn_df_dict[vn]

            # plot histogram
            ax = plt.subplot(v_len, m_len, m_len*v_count+m_count+1)
            ax2 = ax.twinx()

            # compute histogram and change to percentage of glaciers
            hist, bins = np.histogram(a=df[metric], bins=bdict[metric + ' ' + vn])
            hist = hist * 100.0 / hist.sum()

            # plot histogram
            ax.bar(x=bins[1:], height=hist, width=(bins[1]-bins[0]), align='center',
                   alpha=.4, edgecolor='black', color='#0504aa')

            # create uniform bins based on metric
            ax.set_yticks(tdict[metric])

            # find cumulative percentage and plot it
            cum_hist = [hist[0:i].sum() for i in range(len(hist))]

            # find 5 % point or 95 % point
            if metric=='Effective N':
                percent = 5
            else:
                percent = 95
            index = 0
            for point in cum_hist:
                if point < percent:
                    index += 1

            ax2.plot(bins[:-1], cum_hist, color='#ff6600',
                     linewidth=plotline, label='Cumulative %')
            ax2.set_yticks(np.arange(0, 110, 20))

            ax2.plot([bins[index], bins[index]],[cum_hist[index], 0], color='black',
                     linewidth=plotline2)

            # set tick sizes
            ax.tick_params(labelsize=ticks)
            ax2.tick_params(labelsize=ticks)

            ax2.set_ylim(0, 100)
            #ax.set_xlim(bins[0], bins[-1])

            # niceties
            if v_count == 0:
                plt.title(metric_title_dict[metric], fontsize=title, y=titley)

            # axis labels
            if m_count == 0:
                ax.set_ylabel(vn_title_dict[vn] + '\n\n% of Glaciers', fontsize=label, labelpad=0)
            if m_count == 2:
                ax2.set_ylabel('Cumulative %', fontsize=label, rotation = 270, labelpad=10)
            if metric=='MC Error':
                ax.set_xlabel(metric + ' (% of mean)', fontsize=label)
            else:
                ax.set_xlabel(metric + ' value', fontsize=label)

            # legend
            #if v_count==3 and m_count==2:
                #ax2.legend(loc='best', fontsize=legend)

    # Save figure
    plt.savefig(mcmc_output_hist_fp + 'region' + str(region) + '_' + str(iters) +
                'iterations_' + str(burn) + 'burn_' + suffix + '.png',
                bbox_inches='tight')


def compare_priors(filepath, region, iters=15000, burn=0):
    '''
    Compare the effects of different probabilities on
    posterior probabilities.

    Creates a csv file with the probability statistics for each
    glacier, and each different posterior distribution.

    Parameters
    ----------
    filepath : string
        path string of folder containing netcdf files
    region : int
        RGI glacier region number
    iters : int
        Number of iterations to use for each chain
    burn : int
        Number of values to burn before calculating statistics

    Returns
    -------
    .png files
        Saves images to 3 png files.

    '''
    variables = ['massbal', 'precfactor', 'tempchange', 'ddfsnow']
    vn_title_dict = {'massbal':'Mass Balance',
                     'precfactor':'Precipitation Factor',
                     'tempchange':'Temperature Bias',
                     'ddfsnow':'DDF Snow'}
    vn_label_dict = {'massbal':'Mass balance\n[mwea]',
                     'precfactor':'Precipitation factor\n[-]',
                     'tempchange':'Temperature bias\n[degC]',
                     'ddfsnow':'DDFsnow\n[mwe $degC^{-1} d^{-1}$]'}
    priors = ['trunc', 'uniform']

    df_dict = {}

    for prob in priors:

        path = filepath + 'netcdf_' + prob + '/'
        filelist = glob.glob(path + str(region) + '*.nc')

        data_dict = {'glac_no':[]}

        for file in filelist:

            print(file)

            try:
                # Open dataset
                ds = xr.open_dataset(file)

                data_dict['glac_no'].append(file[-11:-3])

                for vn in variables:

                    values = ds['mp_value'].sel(chain=0, mp=vn).values

                    mean = np.mean(values)
                    stdev = np.std(values)

                    if (vn + '_mean') not in data_dict:
                        data_dict[vn + '_mean'] = []
                    if (vn + '_stdev') not in data_dict:
                        data_dict[vn + '_stdev'] = []

                    data_dict[vn + '_mean'].append(mean)
                    data_dict[vn + '_stdev'].append(stdev)
            except:
                if (vn + '_mean') not in data_dict:
                        data_dict[vn + '_mean'] = []
                if (vn + '_stdev') not in data_dict:
                        data_dict[vn + '_stdev'] = []

                data_dict['glac_no'].append(file[-11:-3])
                data_dict[vn + '_mean'].append(np.nan)
                data_dict[vn + '_stdev'].append(np.nan)
                print('Error: ', file)

        df = pd.DataFrame(data_dict)
        df_dict[prob] = df

        # save csv
        df.to_csv(mcmc_prior_fp + 'prior' + prob + '_' + 'region' +
                  str(region) + '_' + str(iters) + 'iterations_' +
                  str(burn) + 'burn.csv')

    return df_dict


def plot_priors(filepath='../MCMC_data/prior_comparison/', region='all', iters=30000, burn=0):
    '''
    Plots scatter plots of posterior distribution statistics for
    independent variables.
    '''
    priors = ['trunc', 'uniform']
    metrics = ['stdev', 'mean']
    variables = ['massbal', 'precfactor', 'tempchange', 'ddfsnow']
    vn_title_dict = {'massbal':'Mass Balance',
                     'precfactor':'Precipitation Factor',
                     'tempchange':'Temperature Bias',
                     'ddfsnow':'DDF Snow'}
    vn_label_dict = {'massbal':'Mass balance\n[mwea]',
                     'precfactor':'Precipitation factor\n[-]',
                     'tempchange':'Temperature bias\n[degC]',
                     'ddfsnow':'DDFsnow\n[mwe $degC^{-1} d^{-1}$]'}
    prior_title_dict = {'trunc':'Truncated Normal Distribution',
                        'uniform': 'Uniform Distribution'}
    metric_title_dict = {'stdev': 'Standard Deviation',
                         'mean': 'Mean Value'}

    prior_dict = {}

    for prior in priors:
        if region in [13, 14, 15]:
            df = pd.read_csv(filepath + 'prior' + prior + '_' + 'region' +
                             str(region) + '_' + str(iters) + 'iterations_' +
                             str(burn) + 'burn.csv')
            prior_dict[prior] = df
        elif region=='all':
            regions = [13, 14, 15]
            dfs = []
            for reg in regions:
                dfs.append(pd.read_csv(filepath + 'prior' + prior + '_' + 'region' +
                             str(reg) + '_' + str(iters) + 'iterations_' +
                             str(burn) + 'burn.csv'))
            prior_dict[prior] = pd.concat(dfs)


    # hard code font sizes
    ticks=10
    suptitle=14
    supy = 0.95
    title=12
    suby = 1.05
    label=10
    plotline=0.5
    legend=10
    figsize=(6.5, 9)
    dpi=100
    hspace=0.5
    wspace=0.4
    labelpad1=5
    labelpad2=5
    v_len = len(variables)
    p_len = len(priors)
    m_len = len(metrics)

    ticks = {}
    ticks['massbal_stdev'] = np.arange(0.1, 0.51, 0.1)
    ticks['massbal_mean'] = np.arange(-2, 2.1, 2)
    ticks['precfactor_stdev'] = np.arange(0.4, 1.1, 0.2)
    ticks['precfactor_mean'] = np.arange(0.5, 2.01, 0.5)
    ticks['tempchange_stdev'] = np.arange(1.0, 6.1, 1)
    ticks['tempchange_mean'] = np.arange(-5, 6, 5)
    ticks['ddfsnow_stdev'] = np.arange(0.0012, 0.002, 0.0003)
    ticks['ddfsnow_mean'] = np.arange(0.003, 0.0046, 0.0005)

    lims = {}
    lims['massbal_stdev'] = [0.1, 0.4]
    lims['massbal_mean'] = [-3, 3]
    lims['precfactor_stdev'] = [0.3, 0.9]
    lims['precfactor_mean'] = [0.5, 2]
    lims['tempchange_stdev'] = [0.8, 6]
    lims['tempchange_mean'] = [-9,7]
    lims['ddfsnow_stdev'] = [0.00105, 0.00185]
    lims['ddfsnow_mean'] = [0.003, 0.0045]

    # create figure
    fig = plt.figure(figsize=figsize, dpi=dpi)
    plt.subplots_adjust(wspace=wspace, hspace=hspace)

    # write title
    #plt.suptitle('Prior Probability Comparison ' +
                 #str(iters) + ' iterations ' + str(burn) + ' burn-in',
                 #fontsize=suptitle, y=supy)

    # create subplot for each variable
    for v, vn in enumerate(variables):

        # create subplot for each variable
        for m, metric in enumerate(metrics):

            # plot histogram
            ax = plt.subplot(v_len, m_len, m_len*v+m+1)
            x = prior_dict['trunc'][vn + '_' + metric]
            y = prior_dict['uniform'][vn + '_' + metric]
            plt.scatter(x=x, y=y, marker='1')
            plt.plot([-10, 10], [-10, 10], color='black', linewidth=plotline)

            ylabel = 'Uniform'
            xlabel = 'Truncated Normal'

            if vn=='ddfsnow':
                ylabel += ' [$10^{-3}$]'
                xlabel += ' [$10^{-3}$]'

            if v == 0:
                plt.title(metric_title_dict[metric], fontsize=title, y=suby)
            if m == 0:
                ax.set_ylabel(vn_title_dict[vn] + '\n' + ylabel,
                              fontsize=label, labelpad=labelpad1)
            else:
                ax.set_ylabel(ylabel,
                              fontsize=label, labelpad=labelpad2)

            ax.set_xticks(ticks[vn + '_' + metric])
            ax.set_yticks(ticks[vn + '_' + metric])
            ax.set_ylim(lims[vn + '_' + metric])
            ax.set_xlim(lims[vn + '_' + metric])

            if vn=='ddfsnow':
                a = ax.get_xticks()
                b = a.copy()
                b *= 1000
                ax.set_xticks(a)
                ax.set_xticklabels(b)
                c = ax.get_yticks()
                d = c.copy()
                d *= 1000
                ax.set_yticks(c)
                ax.set_yticklabels(d)

            ax.set_xlabel(xlabel, fontsize=label)

    plt.savefig(mcmc_prior_fp + 'scatter' + 'region' +
                str(region) + '_' + str(iters) + 'iterations_' +
                str(burn) + 'burn.jpg',
                bbox_inches='tight')


'''
#%% Find files
# ===== LOAD CALIBRATION DATA =====
rgi_glac_number = []

#mcmc_output_netcdf_fp = mcmc_output_netcdf_fp + 'single_obs_inlist/'

for i in os.listdir(mcmc_output_netcdf_fp):
#for i in ['15.00621.nc']:
    glacier_str = i.replace('.nc', '')
    if glacier_str.startswith(str(input.rgi_regionsO1[0])):
        rgi_glac_number.append(glacier_str.split('.')[1])
rgi_glac_number = sorted(rgi_glac_number)


# Glacier RGI data
main_glac_rgi = modelsetup.selectglaciersrgitable(rgi_regionsO1=input.rgi_regionsO1, rgi_regionsO2 = 'all',
                                                  rgi_glac_number=rgi_glac_number)
# Glacier hypsometry [km**2], total area
main_glac_hyps = modelsetup.import_Husstable(main_glac_rgi, input.rgi_regionsO1, input.hyps_filepath,
                                             input.hyps_filedict, input.hyps_colsdrop)
# Select dates including future projections
dates_table_nospinup, start_date, end_date = modelsetup.datesmodelrun(startyear=input.startyear, endyear=input.endyear,
                                                                      spinupyears=0)
# Calibration data
cal_data = pd.DataFrame()
for dataset in cal_datasets:
    cal_subset = class_mbdata.MBData(name=dataset, rgi_regionO1=input.rgi_regionsO1[0])
    cal_subset_data = cal_subset.retrieve_mb(main_glac_rgi, main_glac_hyps, dates_table_nospinup)
    cal_data = cal_data.append(cal_subset_data, ignore_index=True)
cal_data = cal_data.sort_values(['glacno', 't1_idx'])
cal_data.reset_index(drop=True, inplace=True)

# ===== PROCESS EACH NETCDF FILE =====
for n, glac_str_noreg in enumerate(rgi_glac_number[0:4]):
    # Glacier string
    glacier_str = str(input.rgi_regionsO1[0]) + '.' + glac_str_noreg
    # Glacier number
    glacno = int(glacier_str.split('.')[1])
    # RGI information
    glacier_rgi_table = main_glac_rgi.iloc[np.where(main_glac_rgi['glacno'] == glacno)]
    # Calibration data
    glacier_cal_data = (cal_data.iloc[np.where(cal_data['glacno'] == glacno)[0],:]).copy()
    # MCMC Analysis
    #plot_mc_results(mcmc_output_netcdf_fp + glacier_str + '.nc', glacier_cal_data, iters=25000, burn=0)
    plot_mc_results2(mcmc_output_netcdf_fp + glacier_str + '.nc', glacier_cal_data, burns=[0, 2000, 5000], plot_res=500)
    #summary(mcmc_output_netcdf_fp + glacier_str + '.nc', glacier_cal_data,
    #        filename = mcmc_output_tables_fp + glacier_str + '.txt')
'''
# histogram assessments
iterations = np.arange(1000, 31000, 3000)
iterations = np.append(iterations, 30000)
#write_table2(iters=iterations, region='all', burn=0)
plot_mc_results3(iters=iterations, region='all', burn=0)
#for iters in iterations:
#    for region in ['all']:
#        write_table(region=region, iters=iters, burn=0)
        #plot_histograms(region=region, iters=iters, burn=0)
        #plot_histograms_2(region=region, iters=iters, burn=0)
        #compare_priors(mcmc_data_fp, region=region, iters=iters, burn=0)
        #plot_priors(filepath=mcmc_prior_fp, region=region, iters=iters, burn=0)
