"""Run a model simulation."""
# Default climate data is ERA-Interim; specify CMIP5 by specifying a filename to the argument:
#    (Command line) python run_simulation_list_multiprocess.py -gcm_file=C:\...\gcm_rcpXX_filenames.txt
#      - Default is running ERA-Interim in parallel with five processors.
#    (Spyder) %run run_simulation_list_multiprocess.py C:\...\gcm_rcpXX_filenames.txt -option_parallels=0
#      - Spyder cannot run parallels, so always set -option_parallels=0 when testing in Spyder.
# Spyder cannot run parallels, so always set -option_parallels=0 when testing in Spyder.

# Built-in libraries
import os
import argparse
import multiprocessing
import time
import inspect
from time import strftime
# External libraries
import pandas as pd
import numpy as np
import xarray as xr
import netCDF4 as nc
# Local libraries
import pygem_input as input
import pygemfxns_modelsetup as modelsetup
import pygemfxns_massbalance as massbalance
import pygemfxns_output as output
import class_climate

#%% ===== SCRIPT SPECIFIC INPUT DATA =====
# Required input
# Time period
gcm_startyear = 2000
gcm_endyear = 2015
gcm_spinupyears = 0

# Output
output_package = 0
output_filepath = input.main_directory + '/../Output/'
parallel_filepath = output_filepath + 'parallel/'

# Bias adjustment option (options defined in run_gcmbiasadj script; 0 means no correction)
option_bias_adjustment = 0
# Calibrated model parameters
#  calibrated parameters are the same for all climate datasets (only bias adjustments differ for each climate dataset)
ref_modelparams_fp = input.main_directory + '/../Calibration_datasets/'
ref_modelparams_fn = 'calibration_R15_20180403_Opt02solutionspaceexpanding_wnnbrs_20180523.csv'
gcm_modelparams_fp = input.main_directory + '/../Climate_data/cmip5/bias_adjusted_1995_2100/2018_0717/'
gcm_modelparams_fn_ending = ('_biasadj_opt' + str(option_bias_adjustment) + '_1995_2015_R' + str(input.rgi_regionsO1[0])
                             + '_' + str(strftime("%Y%m%d")) +'.csv')

# Tushar's quick and dirty option
# Select True if running using MCMC method
MCMC_option = False

# MCMC settings
MCMC_sample_no = input.mcmc_sample_no
ensemble_no = input.ensemble_no

# MCMC model parameter sets
MCMC_modelparams_fp = input.mcmc_output_netcdf_fp
MCMC_modelparams_fn = input.mcmc_output_filename

# This boolean is useful for debugging. If true, a number
# of print statements are activated through the running
# of the model
debug = True


# Synthetic simulation input
option_synthetic_sim = 0
synthetic_startyear = 1990
synthetic_endyear = 1999
synthetic_temp_adjust = 0
synthetic_prec_factor = 1

#%% FUNCTIONS
def getparser():
    """
    Use argparse to add arguments from the command line
    
    Parameters
    ----------
    gcm__file (optional) : str
        text file that contains the climate data to be used in the model simulation
    num_simultaneous_processes (optional) : int
        number of cores to use in parallels
    option_parallels (optional) : int
        switch to use parallels or not
        
    Returns
    -------
    Object containing arguments and their respective values.
    """
    parser = argparse.ArgumentParser(description="run simulations from gcm list in parallel")
    # add arguments
    parser.add_argument('-gcm_file', action='store', type=str, default=input.ref_gcm_name,
                        help='text file full of commands to run')
    parser.add_argument('-num_simultaneous_processes', action='store', type=int, default=4,
                        help='number of simultaneous processes (cores) to use')
    parser.add_argument('-option_parallels', action='store', type=int, default=1,
                        help='Switch to use or not use parallels (1 - use parallels, 0 - do not)')
    return parser


def main(list_packed_vars):
    """
    Model simulation
    
    Parameters
    ----------
    list_packed_vars : list
        list of packed variables that enable the use of parallels
        
    Returns
    -------
    netcdf files of the simulation output (specific output is dependent on the output option)
    """
    # Unpack variables
    count = list_packed_vars[0]
    chunk = list_packed_vars[1]
    main_glac_rgi_all = list_packed_vars[2]
    chunk_size = list_packed_vars[3]
    gcm_name = list_packed_vars[4]

    time_start = time.time()
    parser = getparser()
    args = parser.parse_args()
    if gcm_name != input.ref_gcm_name:
        rcp_scenario = os.path.basename(args.gcm_file).split('_')[1]

    # ===== LOAD GLACIER DATA =====
    main_glac_rgi = main_glac_rgi_all.iloc[chunk:chunk + chunk_size, :].copy()
    # Glacier hypsometry [km**2], total area
    main_glac_hyps = modelsetup.import_Husstable(main_glac_rgi, input.rgi_regionsO1, input.hyps_filepath,
                                                 input.hyps_filedict, input.hyps_colsdrop)
    # Ice thickness [m], average
    main_glac_icethickness = modelsetup.import_Husstable(main_glac_rgi, input.rgi_regionsO1, input.thickness_filepath,
                                                         input.thickness_filedict, input.thickness_colsdrop)
    main_glac_hyps[main_glac_icethickness == 0] = 0
    # Width [km], average
    main_glac_width = modelsetup.import_Husstable(main_glac_rgi, input.rgi_regionsO1, input.width_filepath,
                                                  input.width_filedict, input.width_colsdrop)
    elev_bins = main_glac_hyps.columns.values.astype(int)
    # Volume [km**3] and mean elevation [m a.s.l.]
    main_glac_rgi['Volume'], main_glac_rgi['Zmean'] = modelsetup.hypsometrystats(main_glac_hyps, main_glac_icethickness)

    # Model parameters
    if input.option_import_modelparams == 0:
        main_glac_modelparams = pd.DataFrame(np.repeat([input.lrgcm, input.lrglac, input.precfactor, input.precgrad,
            input.ddfsnow, input.ddfice, input.tempsnow, input.tempchange], main_glac_rgi_all.shape[0]).reshape(-1,
            main_glac_rgi.shape[0]).transpose(), columns=input.modelparams_colnames)
    elif (gcm_name == 'ERA-Interim') or (option_bias_adjustment == 0):
        main_glac_modelparams_all = pd.read_csv(ref_modelparams_fp + ref_modelparams_fn, index_col=0)
        main_glac_modelparams = main_glac_modelparams_all.loc[main_glac_rgi['O1Index'].values, :]
    else:
        gcm_modelparams_fn = (gcm_name + '_' + rcp_scenario + gcm_modelparams_fn_ending)
        main_glac_modelparams_all = pd.read_csv(gcm_modelparams_fp + gcm_modelparams_fn, index_col=0)

        if debug:
            print(main_glac_modelparams_all)
            print(main_glac_rgi)


        if MCMC_option:
            main_glac_modelparams = (
                    main_glac_modelparams_all.loc[main_glac_modelparams_all['RGIId'].isin(main_glac_rgi['RGIId'])])
        else:
            main_glac_modelparams = main_glac_modelparams_all.loc[main_glac_rgi['O1Index'].values, :]

        if debug:
            print(main_glac_modelparams)

    # Select dates including future projections
    dates_table, start_date, end_date = modelsetup.datesmodelrun(startyear=gcm_startyear, endyear=gcm_endyear,
                                                                 spinupyears=gcm_spinupyears)

    # Synthetic simulation dates
    if option_synthetic_sim == 1:
        dates_table_synthetic, synthetic_start, synthetic_end = modelsetup.datesmodelrun(
                startyear=synthetic_startyear, endyear=synthetic_endyear, spinupyears=0)
    
    # ===== LOAD CLIMATE DATA =====
    if gcm_name == input.ref_gcm_name:
        gcm = class_climate.GCM(name=gcm_name)
        # Check that end year is reasonable
        if (gcm_name == 'ERA-Interim') and (gcm_endyear > 2016) and (option_synthetic_sim == 0):
            print('\n\nEND YEAR BEYOND AVAILABLE DATA FOR ERA-INTERIM. CHANGE END YEAR.\n\n')
    else:
        gcm = class_climate.GCM(name=gcm_name, rcp_scenario=rcp_scenario)
    
    if option_synthetic_sim == 0:
        # Air temperature [degC]
        gcm_temp, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.temp_fn, gcm.temp_vn, main_glac_rgi, 
                                                                     dates_table)
        # Precipitation [m]
        gcm_prec, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.prec_fn, gcm.prec_vn, main_glac_rgi, 
                                                                     dates_table)
        # Elevation [m asl]
        gcm_elev = gcm.importGCMfxnearestneighbor_xarray(gcm.elev_fn, gcm.elev_vn, 
                                                         main_glac_rgi)  
        # Lapse rate
        if gcm_name == 'ERA-Interim':
            gcm_lr, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.lr_fn, gcm.lr_vn, main_glac_rgi, dates_table)
        else:
            # Mean monthly lapse rate
            ref_lr_monthly_avg_all = np.genfromtxt(gcm.lr_fp + gcm.lr_fn, delimiter=',')
            ref_lr_monthly_avg = ref_lr_monthly_avg_all[main_glac_rgi['O1Index'].values]
            gcm_lr = np.tile(ref_lr_monthly_avg, int(gcm_temp.shape[1]/12))
    elif option_synthetic_sim == 1:
        # Air temperature [degC]
        gcm_temp_tile, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.temp_fn, gcm.temp_vn, main_glac_rgi, 
                                                                          dates_table_synthetic)
        # Precipitation [m]
        gcm_prec_tile, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.prec_fn, gcm.prec_vn, main_glac_rgi, 
                                                                          dates_table_synthetic)
        # Elevation [m asl]
        gcm_elev = gcm.importGCMfxnearestneighbor_xarray(gcm.elev_fn, gcm.elev_vn, main_glac_rgi)  
        # Lapse rate
        gcm_lr_tile, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.lr_fn, gcm.lr_vn, main_glac_rgi, 
                                                                        dates_table_synthetic)
        # Future simulation based on synthetic (replicated) data; add spinup years; dataset restarts after spinupyears 
        datelength = dates_table.shape[0] - gcm_spinupyears * 12
        n_tiles = int(np.ceil(datelength / dates_table_synthetic.shape[0]))
        gcm_temp = np.append(gcm_temp_tile[:,:gcm_spinupyears*12], np.tile(gcm_temp_tile,(1,n_tiles))[:,:datelength], 
                             axis=1)
        gcm_prec = np.append(gcm_prec_tile[:,:gcm_spinupyears*12], np.tile(gcm_prec_tile,(1,n_tiles))[:,:datelength], 
                             axis=1)
        gcm_lr = np.append(gcm_lr_tile[:,:gcm_spinupyears*12], np.tile(gcm_lr_tile,(1,n_tiles))[:,:datelength], axis=1)
        # Temperature and precipitation sensitivity adjustments
        gcm_temp = gcm_temp + synthetic_temp_adjust
        gcm_prec = gcm_prec * synthetic_prec_factor

    # ===== BIAS CORRECTIONS =====
    # ERA-Interim does not have any bias corrections
    if (gcm_name == 'ERA-Interim') or (option_bias_adjustment == 0):
        gcm_temp_adj = gcm_temp
        gcm_prec_adj = gcm_prec
        gcm_elev_adj = gcm_elev
    # Option 1
    elif option_bias_adjustment == 1:
        gcm_temp_adj = gcm_temp + main_glac_modelparams['temp_adj'].values[:,np.newaxis]
        gcm_prec_adj = gcm_prec * main_glac_modelparams['prec_adj'].values[:,np.newaxis]
        gcm_elev_adj = gcm_elev
    # Option 2
    elif option_bias_adjustment == 2:
        tempvar_cols = ['tempvar_' + str(n) for n in range(1,13)]
        tempavg_cols = ['tempavg_' + str(n) for n in range(1,13)]
        tempadj_cols = ['tempadj_' + str(n) for n in range(1,13)]
        precadj_cols = ['precadj_' + str(n) for n in range(1,13)]
        bias_adj_prec = main_glac_modelparams[precadj_cols].values
        variability_monthly_std = main_glac_modelparams[tempvar_cols].values
        gcm_temp_monthly_avg = main_glac_modelparams[tempavg_cols].values
        gcm_temp_monthly_adj = main_glac_modelparams[tempadj_cols].values
        # Monthly temperature bias adjusted according to monthly average

        if debug:
            print('gcm_temp:', gcm_temp.shape)
            print(np.tile(gcm_temp_monthly_adj, int(gcm_temp.shape[1]/12)).shape)
            print()

        t_mt = gcm_temp + np.tile(gcm_temp_monthly_adj, int(gcm_temp.shape[1]/12))
        # Mean monthly temperature bias adjusted according to monthly average
        t_m25avg = np.tile(gcm_temp_monthly_avg + gcm_temp_monthly_adj, int(gcm_temp.shape[1]/12))
        # Bias adjusted temperature accounting for monthly mean and variability
        gcm_temp_adj = t_m25avg + (t_mt - t_m25avg) * np.tile(variability_monthly_std, int(gcm_temp.shape[1]/12))
        # Bias adjusted precipitation
        gcm_prec_adj = gcm_prec * np.tile(bias_adj_prec, int(gcm_temp.shape[1]/12))
        # Updated elevation, since adjusted according to reference elevation
        gcm_elev_adj = main_glac_modelparams['new_gcmelev'].values
    # Option 3
    elif option_bias_adjustment == 3:
        tempadj_cols = ['tempadj_' + str(n) for n in range(1,13)]
        precadj_cols = ['precadj_' + str(n) for n in range(1,13)]
        bias_adj_prec = main_glac_modelparams[precadj_cols].values
        bias_adj_temp = main_glac_modelparams[tempadj_cols].values
        # Bias adjusted temperature
        gcm_temp_adj = gcm_temp + np.tile(bias_adj_temp, int(gcm_temp.shape[1]/12))
        # Bias adjusted precipitation
        gcm_prec_adj = gcm_prec * np.tile(bias_adj_prec, int(gcm_temp.shape[1]/12))
        # Updated elevation, since adjusted according to reference elevation
        gcm_elev_adj = main_glac_modelparams['new_gcmelev'].values


    # ===== Get MCMC parameter sets ====

    if MCMC_option:

        # in the form of an xarray dataset
        MCMC_ds = xr.open_dataset(MCMC_modelparams_fp + MCMC_modelparams_fn)


    # ===== CREATE OUTPUT FILE =====
    if MCMC_option:
        if output_package != 0:
            if gcm_name == 'ERA-Interim':
                netcdf_fn = ('PyGEM_R' + str(input.rgi_regionsO1[0]) + '_' + gcm_name + '_' + str(gcm_startyear - 
                             gcm_spinupyears) + '_' + str(gcm_endyear) + '_' + str(count) + '.nc')
            else:
                netcdf_fn = ('PyGEM_R' + str(input.rgi_regionsO1[0]) + '_' + gcm_name + '_' + rcp_scenario + 
                             '_biasadj_opt' + str(option_bias_adjustment) + '_' + str(gcm_startyear - gcm_spinupyears) 
                             + '_' + str(gcm_endyear) + '_' + str(count) + '.nc')

            nsims = MCMC_ds.sizes['runs']
            main_glac_rgi_float = main_glac_rgi.copy()
            main_glac_rgi_float.drop(labels=['RGIId'], axis=1, inplace=True)
            output.netcdfcreate(netcdf_fn, main_glac_rgi_float, main_glac_hyps,
                                dates_table, output_filepath=parallel_filepath,
                                nsims=nsims)

    else:
        if output_package != 0:
            if gcm_name == 'ERA-Interim':
                netcdf_fn = ('PyGEM_R' + str(input.rgi_regionsO1[0]) + '_' + gcm_name + '_' + str(gcm_startyear - 
                             gcm_spinupyears) + '_' + str(gcm_endyear) + '_' + str(count) + '.nc')
            else:
                netcdf_fn = ('PyGEM_R' + str(input.rgi_regionsO1[0]) + '_' + gcm_name + '_' + rcp_scenario + 
                             '_biasadj_opt' + str(option_bias_adjustment) + '_' + str(gcm_startyear - gcm_spinupyears) 
                             + '_' + str(gcm_endyear) + '_' + str(count) + '.nc')
            if option_synthetic_sim == 1:
                netcdf_fn = ('PyGEM_R' + str(input.rgi_regionsO1[0]) + '_' + gcm_name + '_' + str(gcm_startyear - 
                             gcm_spinupyears) + '_' + str(gcm_endyear) + '_T' + 
                             str(round(float(synthetic_temp_adjust))) + 'P' + 
                             str(round(float(synthetic_prec_factor) * 100 - 100)) + '_' + str(count) +  '.nc')
                print(netcdf_fn)
            main_glac_rgi_float = main_glac_rgi.copy()
            main_glac_rgi_float.drop(labels=['RGIId'], axis=1, inplace=True)
            output.netcdfcreate(netcdf_fn, main_glac_rgi_float, main_glac_hyps, dates_table)


    # ===== RUN MASS BALANCE =====
    for glac in range(main_glac_rgi.shape[0]):
        if glac%200 == 0:
            print(gcm_name,':', main_glac_rgi.loc[main_glac_rgi.index.values[glac],'RGIId'])
        # Select subsets of data
        glacier_rgi_table = main_glac_rgi.loc[main_glac_rgi.index.values[glac], :]
        glacier_gcm_elev = gcm_elev_adj[glac]
        glacier_gcm_prec = gcm_prec_adj[glac,:]
        glacier_gcm_temp = gcm_temp_adj[glac,:]
        glacier_gcm_lrgcm = gcm_lr[glac,:]
        glacier_gcm_lrglac = glacier_gcm_lrgcm.copy()
        glacier_area_t0 = main_glac_hyps.iloc[glac,:].values.astype(float)
        icethickness_t0 = main_glac_icethickness.iloc[glac,:].values.astype(float)
        width_t0 = main_glac_width.iloc[glac,:].values.astype(float)

        # if running ensembles using netcdf parameter
        if MCMC_option:

            # get glacier number
            glacier_RGIId = main_glac_rgi.iloc[0]['RGIId'][6:]

            if debug:
                print(glacier_RGIId)

            # get DataArray for specific glacier and convert
            # to pandas DataFrame
            MCMC_da = MCMC_ds[glacier_RGIId]
            MCMC_df = MCMC_da.to_pandas()

            if debug:
                print(MCMC_df)

            # use a for loop for each model run
            for MCMC_run in range(len(MCMC_df)):

                # get model parameters
                modelparameters = []
                for colname in input.modelparams_colnames:
                    modelparameters.append(MCMC_df.loc[MCMC_run][colname])

                if debug:
                    print('modelparameters:', modelparameters)

                # run mass balance calculation
                (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt,
                 glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual,
                 glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual,
                 glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack,
                 glac_wide_area_annual, glac_wide_volume_annual, glac_wide_ELA_annual) = (
                    massbalance.runmassbalance(modelparameters, glacier_rgi_table, glacier_area_t0, icethickness_t0,
                                               width_t0, elev_bins, glacier_gcm_temp, glacier_gcm_prec, 
                                               glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, dates_table, 
                                               option_areaconstant=0))
                # Annual glacier-wide mass balance [m w.e.]
                glac_wide_massbaltotal_annual = np.sum(glac_wide_massbaltotal.reshape(-1,12), axis=1)
                # Average annual glacier-wide mass balance [m w.e.a.]
                mb_mwea = glac_wide_massbaltotal_annual.mean()
                #  units: m w.e. based on initial area
                # Volume change [%]
                if icethickness_t0.max() > 0:
                    glac_vol_change_perc = ((glac_wide_volume_annual[-1] - glac_wide_volume_annual[0]) /
                                            glac_wide_volume_annual[0] * 100)

                # write to netcdf file
                print(MCMC_run)
                if output_package != 0:
                    output.netcdfwrite(netcdf_fn, glac, modelparameters, glacier_rgi_table, elev_bins, glac_bin_temp,
                                       glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt,
                                       glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual,
                                       glac_bin_area_annual, glac_bin_icethickness_annual, glac_bin_width_annual,
                                       glac_bin_surfacetype_annual, output_filepath=parallel_filepath, sim=MCMC_run)


        else:
            modelparameters = main_glac_modelparams.loc[main_glac_modelparams.index.values[glac],
                                                        input.modelparams_colnames]
            
            if debug:
                print('modelparameters:', modelparameters, '\n')
            
            # Mass balance calcs
            (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt,
             glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual,
             glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual,
             glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack,
             glac_wide_area_annual, glac_wide_volume_annual, glac_wide_ELA_annual) = (
                massbalance.runmassbalance(modelparameters, glacier_rgi_table, glacier_area_t0, icethickness_t0,
                                           width_t0, elev_bins, glacier_gcm_temp, glacier_gcm_prec, glacier_gcm_elev,
                                           glacier_gcm_lrgcm, glacier_gcm_lrglac, dates_table, option_areaconstant=0))
            
            if debug:
                # Annual glacier-wide mass balance [m w.e.]
                glac_wide_massbaltotal_annual = np.sum(glac_wide_massbaltotal.reshape(-1,12), axis=1)
                # Average annual glacier-wide mass balance [m w.e.a.]
                mb_mwea = glac_wide_massbaltotal_annual.mean()
                #  units: m w.e. based on initial area
                # Volume change [%]
                if icethickness_t0.max() > 0:
                    glac_vol_change_perc = ((glac_wide_volume_annual[-1] - glac_wide_volume_annual[0]) /
                                            glac_wide_volume_annual[0] * 100)
                print('\n','massbalance:', round(mb_mwea,2), 'vol_change_perc:', round(glac_vol_change_perc,0))


            if output_package != 0:
                output.netcdfwrite(netcdf_fn, glac, modelparameters, glacier_rgi_table, elev_bins, glac_bin_temp,
                                   glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt,
                                   glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual,
                                   glac_bin_area_annual, glac_bin_icethickness_annual, glac_bin_width_annual,
                                   glac_bin_surfacetype_annual, output_filepath=output_filepath)

    # Export variables as global to view in variable explorer
    if (args.option_parallels == 0) or (main_glac_rgi_all.shape[0] < 2 * args.num_simultaneous_processes):
        global main_vars
        main_vars = inspect.currentframe().f_locals

    print('\nProcessing time of', gcm_name, 'for', count,':',time.time()-time_start, 's')

#%% PARALLEL PROCESSING
if __name__ == '__main__':
    time_start = time.time()
    parser = getparser()
    args = parser.parse_args()

    # Select glaciers and define chunks
    main_glac_rgi_all = modelsetup.selectglaciersrgitable(rgi_regionsO1=input.rgi_regionsO1, rgi_regionsO2 = 'all',
                                                          rgi_glac_number=input.rgi_glac_number)
    # Processing needed for netcdf files
#    main_glac_rgi_all['RGIId_float'] = (np.array([np.str.split(main_glac_rgi_all['RGIId'][x],'-')[1]
#                                              for x in range(main_glac_rgi_all.shape[0])]).astype(float))
    main_glac_rgi_all_float = main_glac_rgi_all.copy()
    main_glac_rgi_all_float.drop(labels=['RGIId'], axis=1, inplace=True)
    main_glac_hyps = modelsetup.import_Husstable(main_glac_rgi_all, input.rgi_regionsO1, input.hyps_filepath,
                                                 input.hyps_filedict, input.hyps_colsdrop)
    dates_table, start_date, end_date = modelsetup.datesmodelrun(startyear=gcm_startyear, endyear=gcm_endyear,
                                                                 spinupyears=gcm_spinupyears)

    if (args.option_parallels != 0) and (main_glac_rgi_all.shape[0] >= 2 * args.num_simultaneous_processes):
        chunk_size = int(np.ceil(main_glac_rgi_all.shape[0] / args.num_simultaneous_processes))
    else:
        chunk_size = main_glac_rgi_all.shape[0]

    # Read GCM names from command file
    if args.gcm_file == input.ref_gcm_name:
        gcm_list = [input.ref_gcm_name]
    else:
        with open(args.gcm_file, 'r') as gcm_fn:
            gcm_list = gcm_fn.read().splitlines()
            rcp_scenario = os.path.basename(args.gcm_file).split('_')[1]
            print('Found %d gcms to process'%(len(gcm_list)))

    # Loop through all GCMs
    for gcm_name in gcm_list:
        print('Processing:', gcm_name)
        # Pack variables for multiprocessing
        list_packed_vars = []
        n = 0
        for chunk in range(0, main_glac_rgi_all.shape[0], chunk_size):
            n = n + 1
            list_packed_vars.append([n, chunk, main_glac_rgi_all, chunk_size, gcm_name])

        # Parallel processing
        if (args.option_parallels != 0) and (main_glac_rgi_all.shape[0] >= 2 * args.num_simultaneous_processes):
            with multiprocessing.Pool(args.num_simultaneous_processes) as p:
                p.map(main,list_packed_vars)

        # No parallel processing
        else:
            # Loop through the chunks and export bias adjustments
            for n in range(len(list_packed_vars)):
                main(list_packed_vars[n])

        if MCMC_option:

            # in the form of an xarray dataset
            MCMC_ds = xr.open_dataset(MCMC_modelparams_fp + MCMC_modelparams_fn)
            nsims = MCMC_ds.sizes['runs']

        else:
            nsims = 0

         # Combine output into single netcdf
        if (args.option_parallels != 0) and (main_glac_rgi_all.shape[0] >= 2 * args.num_simultaneous_processes):
            # Netcdf outputs
            output_prefix = ('PyGEM_R' + str(input.rgi_regionsO1[0]) + '_' + gcm_name + '_' + rcp_scenario + 
                             '_biasadj_opt' + str(option_bias_adjustment) + '_' + str(gcm_startyear - gcm_spinupyears) 
                             + '_' + str(gcm_endyear) + '_')
            output_all_fn = ('PyGEM_R' + str(input.rgi_regionsO1[0]) + '_' + gcm_name + '_' + rcp_scenario + 
                             '_biasadj_opt' + str(option_bias_adjustment) + '_' + str(gcm_startyear - gcm_spinupyears) 
                             + '_' + str(gcm_endyear) + '_' + str(strftime("%Y%m%d")) + '_' + 
                             str(len(input.rgi_glac_number)) + 'glaciers_' + str(MCMC_sample_no) + 'samples' + 
                             str(ensemble_no) + 'ensembles_' + rcp_scenario + '.nc')

            # Select netcdf files produced in parallel
            output_list = []
            for i in os.listdir(parallel_filepath):
                # Append bias adjustment results
                if i.startswith(output_prefix) == True:
                    output_list.append(i)

            # Merge netcdfs together
            if (len(output_list) > 1) and (output_package != 0):
                # Create netcdf that will have them all together
                output.netcdfcreate(output_all_fn, main_glac_rgi_all_float, main_glac_hyps, dates_table,
                                    output_filepath=input.output_filepath, nsims=nsims)
                # Open file to write
                netcdf_output = nc.Dataset(output_filepath + output_all_fn, 'r+')

                glac_count = -1
                for n in range(len(output_list)):
                    ds = nc.Dataset(parallel_filepath + output_list[n])
                    for glac in range(ds['glac_idx'][:].shape[0]):
                        glac_count = glac_count + 1
                        if output_package == 2:
                            netcdf_output.variables['temp_glac_monthly'][glac_count,:] = (
                                    ds['temp_glac_monthly'][glac,:])
                            netcdf_output.variables['prec_glac_monthly'][glac_count,:] = (
                                    ds['prec_glac_monthly'][glac,:])
                            netcdf_output.variables['acc_glac_monthly'][glac_count,:] = (
                                    ds['acc_glac_monthly'][glac,:])
                            netcdf_output.variables['refreeze_glac_monthly'][glac_count,:] = (
                                    ds['refreeze_glac_monthly'][glac,:])
                            netcdf_output.variables['melt_glac_monthly'][glac_count,:] = (
                                    ds['melt_glac_monthly'][glac,:])
                            netcdf_output.variables['frontalablation_glac_monthly'][glac_count,:] = (
                                    ds['frontalablation_glac_monthly'][glac,:])
                            netcdf_output.variables['massbaltotal_glac_monthly'][glac_count,:] = (
                                    ds['massbaltotal_glac_monthly'][glac,:])
                            netcdf_output.variables['runoff_glac_monthly'][glac_count,:] = (
                                    ds['runoff_glac_monthly'][glac,:])
                            netcdf_output.variables['snowline_glac_monthly'][glac_count,:] = (
                                    ds['snowline_glac_monthly'][glac,:])
                            netcdf_output.variables['area_glac_annual'][glac_count,:] = (
                                    ds['area_glac_annual'][glac,:])
                            netcdf_output.variables['volume_glac_annual'][glac_count,:] = (
                                    ds['volume_glac_annual'][glac,:])
                            netcdf_output.variables['ELA_glac_annual'][glac_count,:] = (
                                    ds['ELA_glac_annual'][glac,:])
                        else:
                            print('Code merge for output package')
                    ds.close()
                    # Remove file after its been merged
                    os.remove(parallel_filepath + output_list[n])
                # Close the netcdf file
                netcdf_output.close()

    print('Total processing time:', time.time()-time_start, 's')

#%% ===== PLOTTING AND PROCESSING FOR MODEL DEVELOPMENT =====
    # Place local variables in variable explorer
    if (not MCMC_option) and ((args.option_parallels == 0) or (main_glac_rgi_all.shape[0] < 2 * 
        args.num_simultaneous_processes)):
        main_vars_list = list(main_vars.keys())
        gcm_name = main_vars['gcm_name']
#        rcp_scenario = main_vars['rcp_scenario']
        main_glac_rgi = main_vars['main_glac_rgi']
        main_glac_hyps = main_vars['main_glac_hyps']
        main_glac_icethickness = main_vars['main_glac_icethickness']
        main_glac_width = main_vars['main_glac_width']
        main_glac_modelparams = main_vars['main_glac_modelparams']
        elev_bins = main_vars['elev_bins']
        dates_table = main_vars['dates_table']
        if option_synthetic_sim == 1:
            dates_table_synthetic = main_vars['dates_table_synthetic']
            gcm_temp_tile = main_vars['gcm_temp_tile']
            gcm_prec_tile = main_vars['gcm_prec_tile']
            gcm_lr_tile = main_vars['gcm_lr_tile']
        gcm_temp = main_vars['gcm_temp']
        gcm_prec = main_vars['gcm_prec']
        gcm_elev = main_vars['gcm_elev']
        gcm_temp_adj = main_vars['gcm_temp_adj']
        gcm_prec_adj = main_vars['gcm_prec_adj']
        gcm_elev_adj = main_vars['gcm_elev_adj']
        gcm_temp_lrglac = main_vars['gcm_lr']
        modelparameters = main_vars['modelparameters']
        glac_wide_massbaltotal = main_vars['glac_wide_massbaltotal']
        glac_wide_area_annual = main_vars['glac_wide_area_annual']
        glac_wide_volume_annual = main_vars['glac_wide_volume_annual']
        glacier_rgi_table = main_vars['glacier_rgi_table']
        glacier_gcm_temp = main_vars['glacier_gcm_temp']
        glacier_gcm_prec = main_vars['glacier_gcm_prec']
        glacier_gcm_elev = main_vars['glacier_gcm_elev']
        glacier_gcm_lrgcm = main_vars['glacier_gcm_lrgcm'][gcm_spinupyears*12:]
        glacier_area_t0 = main_vars['glacier_area_t0']
        icethickness_t0 = main_vars['icethickness_t0']
        width_t0 = main_vars['width_t0']
        glac_bin_frontalablation = main_vars['glac_bin_frontalablation']
        glac_bin_area_annual = main_vars['glac_bin_area_annual']
        glac_bin_massbalclim_annual = main_vars['glac_bin_massbalclim_annual']
        glac_bin_melt = main_vars['glac_bin_melt']
        glac_bin_acc = main_vars['glac_bin_acc']
        glac_bin_refreeze = main_vars['glac_bin_refreeze']
        glac_bin_temp = main_vars['glac_bin_temp']
        glacier_gcm_lrgcm = main_vars['glacier_gcm_lrgcm']