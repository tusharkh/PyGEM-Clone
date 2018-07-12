import pyDOE as pe
import numpy as np
import copy
import pandas as pd
import v2_run_calibration_4Tushar as v2

def sample1(tempchange, precfactor, ddfsnow,
            samples=10, criterion=None):
    """
    Performes a latin_hypercube sampling of
    the given random probability distributions.

    Takes the desired random probability distributions
    (each in the form of arrays of numbers) and
    and performs a latin hypercube sampling of these
    distributions using pyDOE and the specificed
    pyDOE algorithm. Returns a pandas DataFrame with
    each row represnting an LH sampling of the parameters
    (column names tempchange, precfactor and ddfsnow)

    Note: given traces must have the same length

    Parameters
    ----------
    tempchange : numpy.array
        The trace (or markov chain) of tempchange
        outputed by the MCMC sampling. Each trace
        is represented by an array of discrete
        values or points
    ddfsnow : numpy.array
        The trace (or markov chain) of ddfsnow
        outputed by the MCMC sampling. Each trace
        is represented by an array of discrete
        values or points
    precfactor : numpy.array
        The trace (or markov chain) of precfactor
        outputed by the MCMC sampling. Each trace
        is represented by an array of discrete
        values or points
    samples : int
        Number of samples to be returned. (default: 10)
    criterion: str
        a string that tells the pyDOE lhs function
        how to sample the points (default: None,
        which simply randomizes the points within
        the intervals):
            “center” or “c”: center the points within
            the sampling intervals
            “maximin” or “m”: maximize the minimum
            distance between points, but place the
            point in a randomized location within its
            interval
            “centermaximin” or “cm”: same as “maximin”,
            but centered within the intervals
            “correlation” or “corr”: minimize the
            maximum correlation coefficient


    Returns
    -------
    pandas.dataframe
        DataFrame with each row representing a sampling.
        The column each represent a parameter of interest
        (tempchange, precfactor, ddfsnow)

    """

    # copy the given traces
    tempchange_copy = copy.deepcopy(tempchange)
    precfactor_copy = copy.deepcopy(precfactor)
    ddfsnow_copy = copy.deepcopy(ddfsnow)

    # debug
#    print('copy of tempchange:', tempchange_copy)
#    print('copy of precfactor:', precfactor_copy)
#    print('copy of ddfsnow:', ddfsnow_copy)

    traces = [tempchange_copy, precfactor_copy,
              ddfsnow_copy]

    # sort the traces
    for trace in traces:
        trace.sort()


    # debug
#    print('sorted copy of tempchange:', tempchange_copy)
#    print('sorted copy of precfactor:', precfactor_copy)
#    print('sorted copy of ddfsnow:', ddfsnow_copy)

    lhd = pe.lhs(n=3, samples=samples, criterion=criterion)


    # debug
#    print('copy of lhs samples:', lhd)

    lhd = np.ndarray.transpose(lhd)

    # debug
#    print('copy of lhs samples transposed:', lhd)

    for i in range(len(traces)):
        lhd[i] *= len(traces[i])


    lhd = lhd.astype(int)

    # debug
#    print('copy of lhs samples ints:', lhd)

    names = ['tempchange', 'precfactor', 'ddfsnow']
    samples = pd.DataFrame()

    for i in range(len(traces)):
        array = []
        for j in range(len(lhd[i])):
#            print(i, j, traces[i][lhd[i][j]])
            array.append(traces[i][lhd[i][j]])
        samples[names[i]] = array

#    print(samples, type(samples))
    return samples

def find_mass_balace(samples, replace=True):
    """
    Calculated mass balances for given set of
    parameters

    Takes a pandas dataframe of sample sets and
    adds a column for the mass balances (average
    annual change in S.W.E. over the
    time period of David Shean's observations)
    Takes a list of random probability distributions
    (each in the form of arrays of numbers) and
    and performs a latin hypercube sampling of these
    dataframes. Returns a new dataframe (can replace
    the old one if desired) with the mass balances
    included.

    Parameters
    ----------
    samples : pandas.dataframe
        A dataframe where each row represents sets of
        parameters for the model.Columns are 'tempchange',
        'precfactor' and 'ddfsnow'
    replace : boolean
        True if dataframe should be unchanged and a new
        dataframe should be returned, if False the given
        dataframe will be modified and a copy will not be
        created. default True

    Returns
    -------
    pandas.dataframe
         A dataframe where each row represents sets of
        parameters for the model and the mass balance
        that results from using these parameters.
        Columns are 'tempchange','precfactor', 'ddfsnow'
        and 'massbalance'

    """

    # make copy if replace is True
    if replace:
        samples = copy.deepcopy(samples)

    # calculate massbalances row by row
    massbalances = []
    for index, row in samples.iterrows():
        # debug
#        print(row['precfactor'], row['tempchange'], row['ddfsnow'])

        # caluclate mass balance for each set of params
        massbalance = v2.get_mass_balance(precfactor=row['precfactor'],
                                          tempchange=row['tempchange'],
                                          ddfsnow=row['ddfsnow'])

        # debug
#        print('massbalance', massbalance)

        # add to dataframe
        massbalances.append(massbalance)

    # add to dataframe
    samples['massbalance'] = massbalances

    return samples

def sample(distributions, samples=10, criterion=None):
    """
    Performes a latin_hypercube sampling of
    the given random probability distributions.

    Takes a list of random probability distributions
    (each in the form of arrays of numbers) and
    and performs a latin hypercube sampling of these
    distributions using pyDOE and the specificed
    pyDOE algorithm. Returns a list of array_like
    objects which each represent an LH sample.

    Parameters
    ----------
    distributions : numpy.ndarray
        An array of the distributions to be sampled.
        Distributions are represented by an array of
        discrete points.
    samples : int
        Number of samples to be returned. (default: 10)
    criterion: str
        a string that tells the pyDOE lhs function
        how to sample the points (default: None,
        which simply randomizes the points within
        the intervals):
            “center” or “c”: center the points within
            the sampling intervals
            “maximin” or “m”: maximize the minimum
            distance between points, but place the
            point in a randomized location within its
            interval
            “centermaximin” or “cm”: same as “maximin”,
            but centered within the intervals
            “correlation” or “corr”: minimize the
            maximum correlation coefficient


    Returns
    -------
    numpy.ndarray
        Array of arrays each representing a sampling. The
        order of values in the samples is the same as the
        order of the distributions argument passed

    """

    # sort the given arrays in distribution in order to 
    # divide distribution into equal probability secitons
    dists_copy = copy.deepcopy(distributions)

    print('copy of argument:', dists_copy)

    for dist in dists_copy:
        dist.sort()
    print('copy of sorted argument:', dists_copy)
    lhd = pe.lhs(n=len(dists_copy), samples=samples,
                 criterion=criterion)

    print('copy of lhs samples:', lhd)

    lhd = np.ndarray.transpose(lhd)


    print('copy of lhs samples transposed:', lhd)

    for i in range(0, len(dists_copy)):
        lhd[i] = len(dists_copy[i]) * lhd[i]

    lhd = lhd.astype(int)


    print('copy of lhs samples ints:', lhd)

    matrix = []
    for i in range(0, len(lhd)):
        array = []
        for j in range(0, len(lhd[i])):
            print(i, j,  dists_copy[i][lhd[i][j]])
            array.append(dists_copy[i][lhd[i][j]])
        matrix.append(np.array(array))

    result = np.array(matrix)
    result = np.ndarray.transpose(result)

    print(result, type(result), type(result[1]))
    return result

def sample2(tempchange, ddfsnow, precfactor, massbal,
            samples=10, criterion=None):
    """
    Performes a latin_hypercube sampling of
    the given random probability distributions.

    Takes the output of the MCMC sampling in the
    form of the traces of each of the three variables
    we look at (tempchange, ddfsnow, precfactor) and
    the trace of the massbalance, and performs a latin
    hypercube sampling of these traces using only the
    distribution of mass balances. This function uses
    pyDOE and the specificed pyDOE algorithm. Returns
    a pandas dataframe with each row representing one
    hypercube sampling, ie a single set of parameters
    and their respective mass balance.

    Note: given traces must have the same length

    Parameters
    ----------
    tempchange : numpy.array
        The trace (or markov chain) of tempchange
        outputed by the MCMC sampling. Each trace
        is represented by an array of discrete
        values or points
    ddfsnow : numpy.array
        The trace (or markov chain) of ddfsnow
        outputed by the MCMC sampling. Each trace
        is represented by an array of discrete
        values or points
    precfactor : numpy.array
        The trace (or markov chain) of precfactor
        outputed by the MCMC sampling. Each trace
        is represented by an array of discrete
        values or points
    massbal : numpy.array
        The trace (or markov chain) of mass balance
        outputed by the MCMC sampling. Each trace
        is represented by an array of discrete
        values or points
    distributions : numpy.ndarray
        An array of the distributions to be sampled.
        Distributions are represented by an array of
        discrete points.
    samples : int
        Number of samples to be returned. (default: 10)
    criterion: str
        a string that tells the pyDOE lhs function
        how to sample the points (default: None,
        which simply randomizes the points within
        the intervals):
            “center” or “c”: center the points within
            the sampling intervals
            “maximin” or “m”: maximize the minimum
            distance between points, but place the
            point in a randomized location within its
            interval
            “centermaximin” or “cm”: same as “maximin”,
            but centered within the intervals
            “correlation” or “corr”: minimize the
            maximum correlation coefficient


    Returns
    -------
    pandas.DataFrame
        Dataframe with each row representing a sampling, i.e.
        each row is a set of parameters and their respective
        mass balance. Includes a column for each parameter and
        for massbal

    """

    # make dataframe out of given traces
    df = pd.DataFrame({'tempchange': tempchange,
                       'precfactor': precfactor,
                       'ddfsnow': ddfsnow, 'massbal': massbal})

    # sort dataframe based on values of massbal and add
    # extra indices based on sorted order
    sort_df = df.sort_values('massbal')
    sort_df['sorted_index'] = np.arange(len(sort_df))

    #debug
    print('sorted_df\n', sort_df)

    # use pyDOE, get lhs sampling for 1 distribution
    lhd = pe.lhs(n=1, samples=samples, criterion=criterion)

    # convert lhs to indices for dataframe
    lhd = (lhd * len(df)).astype(int)
    lhd = lhd.ravel()

    #debug
    print('lhd\n', lhd)

    # take sampling with lhs indices, re-sort according to
    # original trace indices
    subset = sort_df.iloc[lhd]
    subset = subset.sort_index()

    # debug
    print('subset\n', subset)

    return subset
