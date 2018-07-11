import pymc_model as pm
import pymc
import numpy as np

samples = 3000

for i in range(5):
    print('completing run ' + str(i+1))
    name = 'testconvergence' + str(samples) + '_' + str(i+1) + '.pickle'
    m = pm.run_MCMC(iterations=samples, dbname=name)
