import pymc
from pymc import MCMC
import test_model_pymc as tm
import pymc_model as pm

def run_MCMC(iterations, burn=0, thin=1, tune_interval=1000,
             tune_throughout=True, save_interval=None,
             burn_till_tuned=False, stop_tuning_after=5,
             verbose=0, progress_bar=True, dbname='trial.pickle'):
    """
    Runs the MCMC algorithm.

    Runs the MCMC algorithm to calibrate the
    probability distributions of three parameters
    for the mass balance function.

    Parameters
    ----------
    model : str
        Choice of model to use, default pymc_model
    step : str
        Choice of step method to use. default metropolis-hastings
    dbname : str
        Choice of database name the sample should be saved to.
        Default name is 'trial.pickle'
    iterations : int
        Total number of iterations to do
    burn : int
        Variables will not be tallied until this many iterations are complete, default 0
    thin : int
        Variables will be tallied at intervals of this many iterations, default 1
    tune_interval : int
        Step methods will be tuned at intervals of this many iterations, default 1000
        tune_throughout : boolean
        If true, tuning will continue after the burnin period (True); otherwise tuning
        will halt at the end of the burnin period.
        save_interval : int or None
        If given, the model state will be saved at intervals of this many iterations
    verbose : boolean
    progress_bar : boolean
        Display progress bar while sampling.
    burn_till_tuned: boolean
        If True the Sampler would burn samples until all step methods are tuned.
        A tuned step methods is one that was not tuned for the last `stop_tuning_after` tuning intervals.
        The burn-in phase will have a minimum of 'burn' iterations but could be longer if
        tuning is needed. After the phase is done the sampler will run for another
        (iter - burn) iterations, and will tally the samples according to the 'thin' argument.
        This means that the total number of iteration is update throughout the sampling
        procedure.
        If burn_till_tuned is True it also overrides the tune_thorughout argument, so no step method
        will be tuned when sample are being tallied.
    stop_tuning_after: int
        the number of untuned successive tuning interval needed to be reach in order for
        the burn-in phase to be done (If burn_till_tuned is True).



    arg1 : int
        Description of arg1
    arg2 : str
        Description of arg2

    Returns
    -------
    int
        Description of return value

    """

    #set model
    if False:
        pass
    else:
        model = MCMC(pm, db='pickle', dbname=dbname)

    # set step if specified
    if step == 'am':
        model.use_step_method(pymc.AdaptiveMetropolis,
                          [precfactor, ddfsnow, tempchange],
                          delay = 1000)

    # sample
    model.sample(iterations=iterations, burn=burn, thin=thin,
                 tune_interval=tune_interval, tune_throughout=tune_throughout,
                 save_interval=save_interval, verbose=verbose,
                 progress_bar=progress_bar)

    #close database
    model.db.close()

    return model


