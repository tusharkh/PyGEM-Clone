import pyDOE as pe
import numpy as np
import copy


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
    distributions : array_like of array_like
        A list of the distributions to be sampled.
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
    list of np.array
        List of arrays each representing a sampling. The
        order of values in the samples is the same as the
        order of the distributions arg passed

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
