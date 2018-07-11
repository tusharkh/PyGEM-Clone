import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
from pymc import *

import pygem_input as input
import v2_run_calibration_4Tushar as v2

glacier_number = int(v2.rgi_glac_number[0])
observed_massbal, observed_error, index = v2.get_glacier_data(glacier_number)

'''
glac_wide_massbaltotal, model_massbal = v2.get_mass_balance() 

print(glac_wide_massbaltotal, type(glac_wide_massbaltotal))
print('initial answer equals:', model_massbal)
print('glacier number:', glacier_number, type(glacier_number))
print('observed mass balance:', observed_massbal, type(observed_massbal))
print('observed mass balance error:', observed_error, type(observed_error))

print('\n\nRound 2')


glac_wide_massbaltotal2, model_massbal2 = v2.get_mass_balance(precfactor=2, ddfsnow=0.0051,
                                                          tempchange=3)

print(glac_wide_massbaltotal2, type(glac_wide_massbaltotal2))
print('initial answer equals:', model_massbal2)
print('glacier number:', glacier_number, type(glacier_number))
print('observed mass balance:', observed_massbal, type(observed_massbal))
print('observed mass balance error:', observed_error, type(observed_error))
'''
'''
__all__ = [
    'disasters_array',
    'switchpoint',
    'early_mean',
    'late_mean',
    'rate',
    'disasters']

disasters_array = array([4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6,
                         3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5,
                         2, 2, 3, 4, 2, 1, 3, 2, 2, 1, 1, 1, 1, 3, 0, 0,
                         1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1,
                         0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2,
                         3, 3, 1, 1, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4,
                         0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1])

# Define data and stochastics

switchpoint = DiscreteUniform(
    'switchpoint',
    lower=0,
    upper=110,
    doc='Switchpoint[year]')
early_mean = Exponential('early_mean', beta=1.)
late_mean = Exponential('late_mean', beta=1.)


@deterministic(plot=False)
def rate(s=switchpoint, e=early_mean, l=late_mean):
     Concatenate Poisson means 
    out = empty(len(disasters_array))
    out[:s] = e
    out[s:] = l
    return out

disasters = Poisson('disasters', mu=rate, value=disasters_array, observed=True)
'''

# Define data and stochastics

#Create prior probability distributions, based on
#current understanding of ranges

#Precipitation factor, based on range of 0.5 to 2
# we use gamma function to get this range, with shape parameter
# alpha=6.33 (also known as k) and rate parameter beta=6 (inverse of
# scale parameter theta)
precfactor = Gamma('precfactor', alpha=6.33, beta=6)
#Degree day of snow, based on (add reference to paper)
ddfsnow = Normal('ddfsnow', mu=0.0041, tau=444444)
#Temperature change, based on range of -5 o 5
tempchange = Normal('tempchange', mu=0, tau=0.25)

@deterministic(plot=False)
def mass_bal(p=precfactor, d=ddfsnow, t=tempchange):
    return v2.get_mass_balance(precfactor=p, ddfsnow=d, tempchange=t)

# observed distribution
#obs_massbal = Normal('obs_massbal', mu=mass_bal, tau=observed_error,
#                     value=float(observed_massbal), observed=True)
obs_massbal = Normal('obs_massbal', mu=mass_bal, tau=(1/(observed_error**2)),
                     value=float(observed_massbal), observed=True)
