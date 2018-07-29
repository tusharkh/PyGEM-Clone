import pymc_model as pm
import pymc
import numpy as np

for i in [30000]:
    print('completing run ' + str(i))
    name = 'test2length' + str(i) + '.pickle'
    m = pm.run_MCMC(iterations=i, dbname=name)
