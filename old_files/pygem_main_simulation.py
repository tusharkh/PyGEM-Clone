"""
Main script for future simulations.  Separated from main script for calibrations to be cleaner for development, but
should consider merging back in future.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import xarray as xr
import netCDF4 as nc
from time import strftime
import time
import pickle
import argparse
import multiprocessing
from scipy.optimize import minimize
from scipy.stats import linregress
#from sklearn.cluster import KMeans
#from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import cartopy
import inspect

import pygem_input as input
import pygemfxns_modelsetup as modelsetup
import pygemfxns_climate as climate
import pygemfxns_massbalance as massbalance
import pygemfxns_output as output


#%% FUNCTIONS
def getparser():
    parser = argparse.ArgumentParser(description="run gcm bias corrections from gcm list in parallel")
    # add arguments
    parser.add_argument('gcm_file', action='store', type=str, default='gcm_rcpXX_filenames.txt', 
                        help='text file full of commands to run')
    parser.add_argument('-num_simultaneous_processes', action='store', type=int, default=5, 
                        help='number of simultaneous processes (cores) to use')
    parser.add_argument('-option_parallels', action='store', type=int, default=1,
                        help='Switch to use or not use paralles (1 - use parallels, 0 - do not)')
    return parser

# Delete when done modeling and restore main()
parser = getparser()
args = parser.parse_args()
with open(args.gcm_file, 'r') as gcm_fn:
    gcm_list = gcm_fn.read().splitlines()
    # RCP scenario
    rcp_scenario = os.path.basename(args.gcm_file).split('_')[1]
print('Found %d gcms to process'%(len(gcm_list)))

gcm_name = gcm_list[0]


#def main(gcm_name):
for batman in [0]:
    print(gcm_name)
    time_start = time.time()
    parser = getparser()
    args = parser.parse_args()
    # RCP scenario
    rcp_scenario = os.path.basename(args.gcm_file).split('_')[1]

    #%% ===== LOAD GLACIER DATA =====
    # RGI glacier attributes
    main_glac_rgi = modelsetup.selectglaciersrgitable()
#    # For calibration, filter data to those that have calibration data
#    if input.option_calibration == 1:
#        # Select calibration data from geodetic mass balance from David Shean
#        main_glac_calmassbal = modelsetup.selectcalibrationdata(main_glac_rgi)
#        # Concatenate massbal data to the main glacier
#        main_glac_rgi = pd.concat([main_glac_rgi, main_glac_calmassbal], axis=1)
    #    # Drop those with nan values
    #    main_glac_calmassbal = main_glac_calmassbal.dropna()
    #    main_glac_rgi = main_glac_rgi.dropna()
    # Glacier hypsometry [km**2], total area
    main_glac_hyps = modelsetup.import_Husstable(main_glac_rgi, input.rgi_regionsO1, input.hyps_filepath, 
                                                 input.hyps_filedict, input.hyps_colsdrop)
    elev_bins = main_glac_hyps.columns.values.astype(int)
    # Ice thickness [m], average
    main_glac_icethickness = modelsetup.import_Husstable(main_glac_rgi, input.rgi_regionsO1, input.thickness_filepath, 
                                                         input.thickness_filedict, input.thickness_colsdrop)
    # Width [km], average
    main_glac_width = modelsetup.import_Husstable(main_glac_rgi, input.rgi_regionsO1, input.width_filepath, 
                                                  input.width_filedict, input.width_colsdrop)
    # Add volume [km**3] and mean elevation [m a.s.l.] to the main glaciers table
    main_glac_rgi['Volume'], main_glac_rgi['Zmean'] = modelsetup.hypsometrystats(main_glac_hyps, main_glac_icethickness)
    # Model time frame
    dates_table, start_date, end_date = modelsetup.datesmodelrun(input.startyear, input.endyear, input.spinupyears)
    # Quality control - if ice thickness = 0, glacier area = 0 (problem identified by glacier RGIV6-15.00016 03/06/2018)
    main_glac_hyps[main_glac_icethickness == 0] = 0
    # Model parameters
    main_glac_modelparams_all = pd.read_csv(input.modelparams_filepath + input.modelparams_filename) 
    main_glac_modelparams = main_glac_modelparams_all.iloc[main_glac_rgi['O1Index'].values]
    main_glac_modelparams.reset_index(drop=True, inplace=True)
    main_glac_modelparams.index.name = input.indexname
    
    #%% ===== LOAD CLIMATE DATA =====
    # Select the climate data for a given gcm
    if input.option_gcm_downscale == 1:  
        # Climate data filepaths and filenames
        gcm_fp_var = input.gcm_filepath_var + rcp_scenario + '_r1i1p1_monNG/'
        gcm_fp_fx = input.gcm_filepath_fx + rcp_scenario + '_r0i0p0_fx/'
        gcm_temp_fn = 'tas_mon_' + gcm_name + '_' + rcp_scenario + '_r1i1p1_native.nc'
        gcm_prec_fn = 'pr_mon_' + gcm_name + '_' + rcp_scenario + '_r1i1p1_native.nc'
        gcm_elev_fn = 'orog_fx_' + gcm_name + '_' + rcp_scenario + '_r0i0p0.nc'
        # Air Temperature [degC] and GCM dates
        main_glac_gcmtemp_raw, main_glac_gcmdate = climate.importGCMvarnearestneighbor_xarray(
                gcm_temp_fn, input.gcm_temp_varname, main_glac_rgi, dates_table, start_date, end_date,
                filepath=gcm_fp_var)
        # Precipitation [m] and GCM dates
        main_glac_gcmprec_raw, main_glac_gcmdate = climate.importGCMvarnearestneighbor_xarray(
                gcm_prec_fn, input.gcm_prec_varname, main_glac_rgi, dates_table, start_date, end_date,
                filepath=gcm_fp_var)
        # Elevation [m a.s.l] associated with air temperature  and precipitation data
        main_glac_gcmelev = climate.importGCMfxnearestneighbor_xarray(
                gcm_elev_fn, input.gcm_elev_varname, main_glac_rgi,
                filepath=gcm_fp_fx)
        # Add GCM time series to the dates_table
        dates_table['date_gcm'] = main_glac_gcmdate
        # Lapse rate[degC m-1] 
        if (input.option_bias_adjustment == 0) and (input.option_lapserate_fromgcm == 1):
            main_glac_gcmlapserate, main_glac_gcmdate = climate.importGCMvarnearestneighbor_xarray(
                    input.gcm_lapserate_filename, input.gcm_lapserate_varname, main_glac_rgi, dates_table, start_date, 
                    end_date)
        # If no bias adjustments, set 'raw' temp and precip as the values to be used
        if input.option_bias_adjustment == 0:
            main_glac_gcmtemp = main_glac_gcmtemp_raw
            main_glac_gcmtemp = main_glac_gcmtemp_raw
    
    #%% ===== BIAS CORRECTIONS =====
    # Future simulations require bias adjusted data
    #  (otherwise, calibration would need to be done on every individual GCM and climate scenario)
    # Load bias adjustment parameters
    if input.option_bias_adjustment == 1:
        main_glac_biasadjparams_all = pd.read_csv(
                input.biasadj_params_filepath + gcm_name + '_' + rcp_scenario + '_biasadjparams_opt1_' + 
                str(input.startyear - input.spinupyears) + '_' + str(input.endyear) + '.csv', index_col=0)
        main_glac_lapserate_monthly_avg_all = np.genfromtxt(
                input.biasadj_params_filepath + 'biasadj_mon_lravg_' + str(input.startyear - input.spinupyears) + '_' + 
                str(input.endyear) + '.csv', delimiter=',')
        # Select data from glaciers in simulation
        main_glac_biasadjparams = main_glac_biasadjparams_all.iloc[main_glac_rgi['O1Index'].values]
        main_glac_gcmlapserate = np.tile(main_glac_lapserate_monthly_avg_all[main_glac_rgi['O1Index'].values],
                                         int(dates_table.shape[0]/12))
        # Perform bias adjustments
        main_glac_gcmtemp = main_glac_gcmtemp_raw + main_glac_biasadjparams.iloc[:,0].values[:,np.newaxis]
        main_glac_gcmprec = main_glac_gcmprec_raw * main_glac_biasadjparams.iloc[:,1].values[:,np.newaxis]
        
    elif input.option_bias_adjustment == 2:
        main_glac_hh2015_mon_precadj_all = np.genfromtxt(
                input.biasadj_params_filepath + gcm_name + '_' + rcp_scenario + '_biasadjparams_hh2015_mon_precadj_' + 
                str(input.startyear - input.spinupyears) + '_' + str(2100) + '.csv', delimiter=',')
        main_glac_hh2015_mon_tempadj_all = np.genfromtxt(
                input.biasadj_params_filepath + gcm_name + '_' + rcp_scenario + '_biasadjparams_hh2015_mon_tempadj_' + 
                str(input.startyear - input.spinupyears) + '_' + str(2100) + '.csv', delimiter=',')
        main_glac_hh2015_mon_tempavg_all = np.genfromtxt(
                input.biasadj_params_filepath + gcm_name + '_' + rcp_scenario + '_biasadjparams_hh2015_mon_tempavg_' + 
                str(input.startyear - input.spinupyears) + '_' + str(2100) + '.csv', delimiter=',')
        main_glac_hh2015_mon_tempvar_all = np.genfromtxt(
                input.biasadj_params_filepath + gcm_name + '_' + rcp_scenario + '_biasadjparams_hh2015_mon_tempvar_' + 
                str(input.startyear - input.spinupyears) + '_' + str(2100) + '.csv', delimiter=',')
        main_glac_lapserate_monthly_avg_all = np.genfromtxt(
                input.biasadj_params_filepath + 'biasadj_mon_lravg_' + 
                str(input.startyear - input.spinupyears) + '_' + str(2100) + '.csv', delimiter=',')
        main_glac_gcmelev_all = np.genfromtxt(
                input.biasadj_params_filepath + 'biasadj_elev.csv', delimiter=',')
        # Select data from glaciers in simulation
        main_glac_hh2015_mon_precadj = main_glac_hh2015_mon_precadj_all[main_glac_rgi['O1Index'].values]
        main_glac_hh2015_mon_tempadj = main_glac_hh2015_mon_tempadj_all[main_glac_rgi['O1Index'].values]
        main_glac_hh2015_mon_tempavg = main_glac_hh2015_mon_tempavg_all[main_glac_rgi['O1Index'].values]
        main_glac_hh2015_mon_tempvar = main_glac_hh2015_mon_tempvar_all[main_glac_rgi['O1Index'].values]
        main_glac_gcmelev = main_glac_gcmelev_all[main_glac_rgi['O1Index'].values]
        main_glac_gcmlapserate = np.tile(main_glac_lapserate_monthly_avg_all[main_glac_rgi['O1Index'].values],
                                         int(dates_table.shape[0]/12))
        # Perform bias adjustments
        # Correct for monthly temperature bias adjusted based on the monthly average
        t_mt = main_glac_gcmtemp_raw + np.tile(main_glac_hh2015_mon_tempadj, int(dates_table.shape[0]/12))
        # Mean monthly temperature after bias adjusted for monthly average            
        t_mt_refavg = np.tile(main_glac_hh2015_mon_tempavg + main_glac_hh2015_mon_tempadj, int(dates_table.shape[0]/12))
        # Bias adjusted temperature accounting for monthly mean and variability
        main_glac_gcmtemp = (t_mt_refavg + (t_mt - t_mt_refavg) * 
                             np.tile(main_glac_hh2015_mon_tempvar, int(dates_table.shape[0]/12)))
        # Bias adjusted precipitation accounting for monthly mean
        main_glac_gcmprec = main_glac_gcmprec_raw * np.tile(main_glac_hh2015_mon_precadj, int(dates_table.shape[0]/12))

    print('Loading time:', time.time()-time_start, 's')
    
    #%%===== SIMULATION RUN =====
    # Create output netcdf file
    if input.output_package != 0:
        netcdf_fn = (input.netcdf_fn_prefix + str(input.rgi_regionsO1[0]) + '_' + gcm_name + '_' + rcp_scenario + '_' +
                     str(input.startyear) + '_' + str(input.endyear) + '_' + str(strftime("%Y%m%d")) + '.nc')
        output.netcdfcreate(netcdf_fn, main_glac_rgi, main_glac_hyps, dates_table)
        
    # Glacier specific data to output to csv
    main_glac_results_colnames = ['RGIId', 'mb_mwea', 'vol_change_perc', 'nbr_idx_count', 'nbr_mb_mean', 'nbr_mb_std']
    main_glac_results = pd.DataFrame(np.zeros((main_glac_rgi.shape[0],len(main_glac_results_colnames))))
    main_glac_results.columns = main_glac_results_colnames
    main_glac_results['RGIId'] = main_glac_rgi['RGIId']
    
    # ENTER GLACIER LOOP
    for glac in range(main_glac_rgi.shape[0]):
#    for glac in [0]:
#        print(main_glac_rgi.loc[glac,'RGIId'])
        
        # Print every 100th glacier
        if glac%100 == 0:
            print(main_glac_rgi.loc[glac,'RGIId'])
        
        # Nearest neighbors: mass balance envelope
        nbr_idx_cols = [col for col in main_glac_modelparams_all if col.startswith('nearidx')]
        nbrs_data = np.zeros((len(nbr_idx_cols),4))
        #  Col 0 - index count
        #  Col 1 - index value
        #  Col 2 - MB
        #  Col 3 - MB modeled using neighbor's parameter
        nbrs_data[:,0] = np.arange(1,len(nbr_idx_cols)+1) 
        nbrs_data[:,1] = main_glac_modelparams.loc[glac, nbr_idx_cols].values.astype(int)
        nbrs_data[:,2] = main_glac_modelparams_all.loc[nbrs_data[:,1], input.massbal_colname]
        mb_nbrs_mean = nbrs_data[:,2].mean()
        mb_nbrs_std = nbrs_data[:,2].std()
        mb_envelope_lower = mb_nbrs_mean - mb_nbrs_std
        mb_envelope_upper = mb_nbrs_mean + mb_nbrs_std
        # Record results
        main_glac_results.loc[glac,'nbr_mb_mean'] = mb_nbrs_mean
        main_glac_results.loc[glac,'nbr_mb_std'] = mb_nbrs_std

        # Glacier properties
        glac_idx = main_glac_modelparams.loc[glac,'O1Index']
        glac_zmed = main_glac_rgi.loc[glac,'Zmed']
        
        # Set nbr_idx_count to -1, since adds 1 at the start        
        nbr_idx_count = -1
        # Loop through nearest neighbors until find set of model parameters that returns MB in MB envelope
        # Break loop used to exit loop if unable to find parameter set that satisfies criteria
        break_while_loop = False
        while break_while_loop == False:
            # Nearest neighbor index
            nbr_idx_count = nbr_idx_count + 1
            main_glac_results.loc[glac, 'nbr_idx_count'] = nbr_idx_count
            # Check if cycled through all neighbors; if so, then choose neighbor with MB closest to the envelope
            if nbr_idx_count == len(nbr_idx_cols):
                break_while_loop = True
                mb_abs = np.zeros((nbrs_data.shape[0],2)) 
                mb_abs[:,0] = abs(nbrs_data[:,3] - mb_envelope_lower)
                mb_abs[:,1] = abs(nbrs_data[:,3] - mb_envelope_upper)
                nbr_idx_count = np.where(mb_abs == mb_abs.min())[0][0]    
            # Nearest neighbor index value
            nbr_idx = nbrs_data[nbr_idx_count,1]
            # Model parameters
            modelparameters = main_glac_modelparams_all.loc[nbr_idx,['lrgcm', 'lrglac', 'precfactor', 'precgrad', 
                                                                     'ddfsnow', 'ddfice', 'tempsnow', 'tempchange']]
            nbr_zmed = main_glac_modelparams_all.loc[nbr_idx,'Zmed']
            modelparameters['tempchange'] = modelparameters['tempchange'] + 0.002594 * (glac_zmed - nbr_zmed)
            modelparameters['precfactor'] = modelparameters['precfactor'] - 0.0005451 * (glac_zmed - nbr_zmed)
            modelparameters['ddfsnow'] = modelparameters['ddfsnow'] + 1.31e-06 * (glac_zmed - nbr_zmed)
            # Reset variables
            glacier_rgi_table = main_glac_rgi.loc[glac, :]
            glacier_gcm_elev = main_glac_gcmelev[glac]
            glacier_gcm_prec = main_glac_gcmprec[glac,:]
            glacier_gcm_temp = main_glac_gcmtemp[glac,:]
            glacier_gcm_lrgcm = main_glac_gcmlapserate[glac]
            glacier_gcm_lrglac = glacier_gcm_lrgcm.copy()
            glacier_area_t0 = main_glac_hyps.iloc[glac,:].values.astype(float)   
            icethickness_t0 = main_glac_icethickness.iloc[glac,:].values.astype(float)
            width_t0 = main_glac_width.iloc[glac,:].values.astype(float)
            # MASS BALANCE
            # Run the mass balance function (spinup years have been removed from output)
            (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt, 
             glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual, 
             glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual, 
             glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack, glac_wide_area_annual, 
             glac_wide_volume_annual, glac_wide_ELA_annual) = (
                massbalance.runmassbalance(modelparameters, glacier_rgi_table, glacier_area_t0, icethickness_t0, width_t0, 
                                           elev_bins, glacier_gcm_temp, glacier_gcm_prec, glacier_gcm_elev, 
                                           glacier_gcm_lrgcm, glacier_gcm_lrglac, dates_table))
            # Additional output
            # Volume change [%]
            if icethickness_t0.max() > 0:
                main_glac_results.loc[glac,'vol_change_perc'] = (
                        (glac_wide_volume_annual[-1] - glac_wide_volume_annual[0]) / glac_wide_volume_annual[0] * 100)
            # Annual glacier-wide mass balance [m w.e.]
            glac_wide_massbaltotal_annual = np.sum(glac_wide_massbaltotal.reshape(-1,12), axis=1)
            # Average annual glacier-wide mass balance [m w.e.a.]
            massbal_idx_start = 0
            massbal_idx_end = 16
            mb_mwea = glac_wide_massbaltotal_annual[massbal_idx_start:massbal_idx_end].mean()
            #  units: m w.e. based on initial area
            main_glac_results.loc[glac,'mb_mwea'] = mb_mwea 
            nbrs_data[nbr_idx_count,3] = mb_mwea
            
#            print(nbr_idx_count, main_glac_modelparams_all.loc[nbr_idx,'RGIId'])
#            print('Mass balance 2000-2015 [mwea]:', mb_mwea)      
            
            # Prior to breaking loop print RGIId and z score
            if break_while_loop == True:
                # Compute z score and print out value
                z_score = (mb_mwea - mb_nbrs_mean) / mb_nbrs_std
                if abs(z_score) > 1.96:
                    print(glacier_rgi_table.RGIId, 'z_score:', mb_mwea, np.round(z_score,2))
               
            # If mass balance falls within envelope, then end while loop
            if (mb_mwea <= mb_envelope_upper) and (mb_mwea >= mb_envelope_lower):
                break_while_loop = True
        
        # OUTPUT: Record variables according to output package
        #  must be done within glacier loop since the variables will be overwritten 
        if input.output_package != 0:
            output.netcdfwrite(netcdf_fn, glac, modelparameters, glacier_rgi_table, elev_bins, glac_bin_temp, 
                               glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt, 
                               glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, 
                               glac_bin_area_annual, glac_bin_icethickness_annual, glac_bin_width_annual,
                               glac_bin_surfacetype_annual)
    
    if input.output_package != 0:
        # Add z-score
        main_glac_results['z_score'] = ((main_glac_results['mb_mwea'] - main_glac_results['nbr_mb_mean']) / 
                                        main_glac_results['nbr_mb_std'])
        csv_output_fn = (input.netcdf_fn_prefix + str(input.rgi_regionsO1[0]) + '_' + gcm_name + '_' + rcp_scenario + 
                         '_' + str(input.startyear) + '_' + str(input.endyear) + '_' + str(strftime("%Y%m%d")) + 
                         '_main_glac_results.csv')
        main_glac_results.to_csv(input.output_filepath + csv_output_fn, sep=',')
    
    # Export variables as global to view in variable explorer
#    global main_vars
#    main_vars = inspect.currentframe().f_locals
    
    print('Processing time:', time.time()-time_start, 's')

#%% ====== PARALLEL COMPUTING =====
#if input.option_parallels == 1:
#    # ARGUMENT PARSER 
#    parser = argparse.ArgumentParser(description="run model simulation processors")
#    # add arguments
##    parser.add_argument('-gcm_file', action='store', type=str, default=input.gcm_name, 
##                        help='text file full of commands to run')
#    parser.add_argument('-num_simultaneous_processes', action='store', type=int, default=5, 
#                        help='number of simultaneous processes (cores) to use')
#    # select arguments
#    args = parser.parse_args()
#    
#    # Number of simultaneous processes
#    num_simultaneous_processes = args.num_simultaneous_processes
#    
#    # Calculate the range of the chunks
#    chunk_size = divmod(main_glac_rgi.shape[0],num_simultaneous_processes)
#    chunk_range = np.zeros((num_simultaneous_processes,2))
#    for i in range(num_simultaneous_processes):
#        chunk_range[i,0] = int(np.asarray(np.arange(0,main_glac_rgi.shape[0])[i * chunk_size[0] + 
#                               min(i, chunk_size[1]):(i + 1) * chunk_size[0] + min(i + 1, chunk_size[1])]).min())
#        chunk_range[i,1] = int(np.asarray(np.arange(0,main_glac_rgi.shape[0])[i * chunk_size[0] + 
#                               min(i, chunk_size[1]):(i + 1) * chunk_size[0] + min(i + 1, chunk_size[1])]).max())
#    chunk_range = chunk_range.astype(int)
#
#chunk_list = np.ndarray.tolist(chunk_range)
#
#def chunk_glaciers(chunk_range):
#    chunk_main_glac_rgi = main_glac_rgi.iloc[chunk_range[0]:chunk_range[1]+1,:]
#    chunk_main_glac_rgi = chunk_main_glac_rgi.reset_index(drop=True)
#    print(chunk_main_glac_rgi.loc[0,'RGIId'])
#    print(chunk_main_glac_rgi.loc[chunk_main_glac_rgi.shape[0]-1,'RGIId'])
#    return chunk_main_glac_rgi
#
#    if input.option_parallels == 1:
#        with multiprocessing.Pool(num_simultaneous_processes) as p:
#            p.map(chunk_glaciers,chunk_list)    
    
#if __name__ == '__main__':
#    time_start = time.time()
#    parser = getparser()
#    args = parser.parse_args()
#    # Read GCM names from command file
#    with open(args.gcm_file, 'r') as gcm_fn:
#        gcm_list = gcm_fn.read().splitlines()
#    print('Found %d gcms to process'%(len(gcm_list)))
#    
#    if args.option_parallels != 0:
#        with multiprocessing.Pool(args.num_simultaneous_processes) as p:
#            p.map(main,gcm_list)
#    else:
#        print('Not using parallels')
#        # Loop through GCMs and export bias adjustments
#        for n_gcm in range(len(gcm_list)):
#            gcm_name = gcm_list[n_gcm]
#            # Perform GCM Bias adjustments
#            main(gcm_name)
#            # Place local variables in variable explorer
#            vars_list = list(main_vars.keys())
#            gcm_name = main_vars['gcm_name']
#            rcp_scenario = main_vars['rcp_scenario']
#            main_glac_rgi = main_vars['main_glac_rgi']
#            main_glac_hyps = main_vars['main_glac_hyps']
#            main_glac_icethickness = main_vars['main_glac_icethickness']
#            main_glac_width = main_vars['main_glac_width']
#            elev_bins = main_vars['elev_bins']
#            main_glac_gcmtemp = main_vars['main_glac_gcmtemp']
#            main_glac_gcmprec = main_vars['main_glac_gcmprec']
#            main_glac_gcmelev = main_vars['main_glac_gcmelev']
#            main_glac_gcmlapserate = main_vars['main_glac_gcmlapserate']
#            main_glac_modelparams = main_vars['main_glac_modelparams']
#            end_date = main_vars['end_date']
#            start_date = main_vars['start_date']
#            dates_table = main_vars['dates_table']
#            glac_wide_volume_annual = main_vars['glac_wide_volume_annual']
#            glac_wide_area_annual = main_vars['glac_wide_area_annual']
#            glac_bin_area_annual = main_vars['glac_bin_area_annual']
#            glac_wide_massbaltotal = main_vars['glac_wide_massbaltotal']
#            glac_bin_massbalclim_annual = main_vars['glac_bin_massbalclim_annual']
#            biasadj_temp = main_vars['biasadj_temp']
#            biasadj_prec = main_vars['biasadj_prec']            
#                    
#    print('Total processing time:', time.time()-time_start, 's')            


#%%=== Model testing ===============================================================================
#output = nc.Dataset(input.output_filepath + netcdf_fn, 'r+')
#output = nc.Dataset(input.output_filepath + 'PyGEM_R15_MPI-ESM-LR_rcp26_2000_2100_20180518.nc', 'r+')
#dates = nc.num2date(output['time'][:], units=output['time'].units, calendar=output['time'].calendar).tolist()

