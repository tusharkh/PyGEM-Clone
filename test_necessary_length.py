import pymc_model as pm
import pymc
import numpy as np

for i in [1000, 3000, 5000, 10000]:
    print('completing run ' + str(i))
    name = 'testlength' + str(i) + '.pickle'
    m = pm.run_MCMC(iterations=i, dbname=name)
