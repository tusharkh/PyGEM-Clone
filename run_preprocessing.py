"""
pygemfxns_preprocessing.py is a list of the model functions that are used to preprocess the data into the proper format.

"""

# Built-in libraries
import os
import glob
import argparse
# External libraries
import pandas as pd
import numpy as np
import xarray as xr
import netCDF4 as nc
from time import strftime
from datetime import datetime
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import matplotlib.pyplot as plt
# Local libraries
import pygem_input as input
import pygemfxns_modelsetup as modelsetup
import class_climate



#%% TO-DO LIST:
# - clean up create lapse rate input data (put it all in input.py)

#%%
def getparser():
    """
    Use argparse to add arguments from the command line
    
    Parameters
    ----------
    option_createlapserates : int
        Switch for processing lapse rates (default = 0 (no))
    option_wgms : int
        Switch for processing wgms data (default = 0 (no))
        
    Returns
    -------
    Object containing arguments and their respective values.
    """
    parser = argparse.ArgumentParser(description="select pre-processing options")
    # add arguments
    parser.add_argument('-option_createlapserates', action='store', type=int, default=0,
                        help='option to create lapse rates or not (1=yes, 0=no)')
    parser.add_argument('-option_wgms', action='store', type=int, default=0,
                        help='option to pre-process wgms data (1=yes, 0=no)')
    parser.add_argument('-option_coawstmerge', action='store', type=int, default=0,
                        help='option to merge COAWST climate data products (1=yes, 0=no)')
    parser.add_argument('-option_mbdata_fillwregional', action='store', type=int, default=0,
                        help='option to fill in missing mass balance data with regional mean and std (1=yes, 0=no)')
    return parser

parser = getparser()
args = parser.parse_args()

#%%
#rgi_regionsO1 = [13,14,15]
#main_glac_rgi_all = pd.DataFrame()
#for region in rgi_regionsO1:
#    main_glac_rgi_region = modelsetup.selectglaciersrgitable(rgi_regionsO1=[region], rgi_regionsO2='all', 
#                                                             rgi_glac_number='all')
#    main_glac_rgi_all = main_glac_rgi_all.append(main_glac_rgi_region)


#%% FILL MISSING MB DATA WITH REGIONAL MEAN AND STD
if args.option_mbdata_fillwregional == 1:
    print('Filling in missing data with regional estimates...')
    # Input data
    ds_fp = input.shean_fp
    ds_fn = input.shean_fn    
    # Load mass balance measurements and identify unique rgi regions 
    ds = pd.read_csv(ds_fp + ds_fn)
    ds = ds.sort_values('RGIId', ascending=True)
    ds.reset_index(drop=True, inplace=True)
    ds['RGIId'] = round(ds['RGIId'], 5)
    ds['rgi_regO1'] = ds['RGIId'].astype(int)
    ds['rgi_str'] = ds['RGIId'].apply(lambda x: '%.5f' % x)
    rgi_regionsO1 = sorted(ds['rgi_regO1'].unique().tolist())
    # Associate the 2nd order rgi regions with each glacier
    main_glac_rgi_all = pd.DataFrame()
    for region in rgi_regionsO1:
        main_glac_rgi_region = modelsetup.selectglaciersrgitable(rgi_regionsO1=[region], rgi_regionsO2='all', 
                                                                 rgi_glac_number='all')
        main_glac_rgi_all = main_glac_rgi_all.append(main_glac_rgi_region)
    main_glac_rgi_all.reset_index(drop=True, inplace=True)
    # Add mass balance and uncertainty to main_glac_rgi
    # Select glaciers with data such that main_glac_rgi and ds indices are aligned correctly
    main_glac_rgi = (main_glac_rgi_all.iloc[np.where(main_glac_rgi_all['RGIId_float'].isin(ds['RGIId']) == True)[0],:]
                    ).copy()
    main_glac_rgi.reset_index(drop=True, inplace=True)
    main_glac_rgi['mb_mwea'] = ds['mb_mwea']
    main_glac_rgi['mb_mwea_sigma'] = ds['mb_mwea_sigma']    
    # Regional mass balances
    mb_regional_cols = ['rgi_O1', 'rgi_O2', 'mb_mwea', 'mb_mwea_sigma']
    mb_regional = pd.DataFrame(columns=mb_regional_cols)
    reg_dict_mb = {}
    reg_dict_mb_sigma = {}
    for region in rgi_regionsO1:
        # 1st order regional mass balances
        main_glac_subset = main_glac_rgi.loc[main_glac_rgi['O1Region'] == region]
        A = pd.DataFrame(columns=mb_regional_cols)
        A.loc[0,'rgi_O1'] = region
        A.loc[0,'rgi_O2'] = 'all'
        A.loc[0,'mb_mwea'] = ((main_glac_subset['Area'] * main_glac_subset['mb_mwea']).sum() / 
                              main_glac_subset['Area'].sum())
        A.loc[0,'mb_mwea_sigma'] = ((main_glac_subset['Area'] * main_glac_subset['mb_mwea_sigma']).sum() / 
                                    main_glac_subset['Area'].sum())
        mb_regional = mb_regional.append(A, sort=False)
        # 2nd order regional mass balances
        rgi_regionsO2 = sorted(main_glac_subset['O2Region'].unique().tolist())
        for regionO2 in rgi_regionsO2:
            main_glac_subset_O2 = main_glac_subset.loc[main_glac_subset['O2Region'] == regionO2]
            A = pd.DataFrame(columns=mb_regional_cols)
            A.loc[0,'rgi_O1'] = region
            A.loc[0,'rgi_O2'] = regionO2
            A.loc[0,'mb_mwea'] = ((main_glac_subset_O2['Area'] * main_glac_subset_O2['mb_mwea']).sum() / 
                                  main_glac_subset_O2['Area'].sum())
            A.loc[0,'mb_mwea_sigma'] = ((main_glac_subset_O2['Area'] * main_glac_subset_O2['mb_mwea_sigma']).sum() / 
                                        main_glac_subset_O2['Area'].sum())
            mb_regional = mb_regional.append(A, sort=False)
            # Create dictionary 
            reg_dict_mb[region, regionO2] = A.loc[0,'mb_mwea']
            reg_dict_mb_sigma[region, regionO2] = A.loc[0,'mb_mwea_sigma']
    # Fill in mass balance of glaciers with no data using regional estimates
    main_glac_nodata = (
            main_glac_rgi_all.iloc[np.where(main_glac_rgi_all['RGIId_float'].isin(ds['RGIId']) == False)[0],:]).copy()
    main_glac_nodata.reset_index(drop=True, inplace=True)
    # Dictionary linking regions and regional mass balances
    ds_nodata = pd.DataFrame(columns=ds.columns)
    ds_nodata.drop(['rgi_regO1', 'rgi_str'], axis=1, inplace=True)
    ds_nodata['RGIId'] = main_glac_nodata['RGIId_float']
    ds_nodata['t1'] = ds['t1'].min()
    ds_nodata['t2'] = ds['t2'].max()
    ds_nodata['mb_mwea'] = (
            pd.Series(list(zip(main_glac_nodata['O1Region'], main_glac_nodata['O2Region']))).map(reg_dict_mb))
    ds_nodata['mb_mwea_sigma'] = (
            pd.Series(list(zip(main_glac_nodata['O1Region'], main_glac_nodata['O2Region']))).map(reg_dict_mb_sigma))
#    # Export csv of all glaciers including those with data and those with filled values
#    ds_export = ds.copy()
#    ds_export.drop(['rgi_regO1', 'rgi_str'], axis=1, inplace=True)
#    ds_export = ds_export.append(ds_nodata)
#    ds_export = ds_export.sort_values('RGIId', ascending=True)
#    ds_export.reset_index(drop=True, inplace=True)
#    output_fn = ds_fn.replace('.csv', '_all_filled.csv')
#    ds_export.to_csv(ds_fp + output_fn, index=False)
    

#%% COAWST Climate Data
if args.option_coawstmerge == 1:
    print('Merging COAWST climate data...')

    def coawst_merge_netcdf(vn, coawst_fp, coawst_fn_prefix):
        """
        Merge COAWST products to form a timeseries

        Parameters
        ----------
        vn : str
            variable name
        coawst_fp : str
            filepath of COAWST climate data
        
        Returns
        -------
        exports netcdf of merged climate data
        """
        # Sorted list of files to merge
        ds_list = []
        for i in os.listdir(coawst_fp):
            if i.startswith(coawst_fn_prefix):
                ds_list.append(i)
        ds_list = sorted(ds_list)
        # Merge files
        count = 0
        for i in ds_list:
            count += 1
            ds = xr.open_dataset(coawst_fp + i)
            var = ds[vn].values
            lat = ds.LAT.values
            lon = ds.LON.values
            if vn == 'HGHT':
                var_all = var
            elif count == 1:
                var_all = var
                month_start_str = i.split('_')[3].split('.')[0].split('-')[0]
            elif count == len(ds_list):
                var_all = np.append(var_all, var, axis=0)
                month_end_str = i.split('_')[3].split('.')[0].split('-')[1]
            else:
                var_all = np.append(var_all, var, axis=0)
                
            print('Max TOTPRECIP:', ds.TOTPRECIP.values.max())
            print('Max TOTRAIN:', ds.TOTRAIN.values.max())
            print('Max TOTSNOW:', ds.TOTSNOW.values.max())
                
        # Merged dataset
        if vn == 'HGHT':
            ds_all_fn = coawst_fn_prefix + vn + '.nc'
            ds_all = xr.Dataset({vn: (['x', 'y'], var)},
                        coords={'LON': (['x', 'y'], lon),
                                'LAT': (['x', 'y'], lat)},
                        attrs=ds[vn].attrs)
            ds_all[vn].attrs = ds[vn].attrs
        else:
            # reference time in format for pd.date_range
            time_ref = month_start_str[0:4] + '-' + month_start_str[4:6] + '-' + month_start_str[6:8]
            ds_all_fn = coawst_fn_prefix + vn + '_' + month_start_str + '-' + month_end_str + '.nc'
            ds_all = xr.Dataset({vn: (['time', 'x', 'y'], var_all)},
                                coords={'LON': (['x', 'y'], lon),
                                        'LAT': (['x', 'y'], lat),
                                        'time': pd.date_range(time_ref, periods=len(ds_list), freq='MS'),
                                        'reference_time': pd.Timestamp(time_ref)})
            ds_all[vn].attrs = ds[vn].attrs
        # Export to netcdf
        ds_all.to_netcdf(coawst_fp + '../' + ds_all_fn)
        ds_all.close()
        
    # Load climate data
    gcm = class_climate.GCM(name='COAWST')
    # Process each variable
    for vn in input.coawst_vns:
        coawst_merge_netcdf(vn, input.coawst_fp_unmerged, input.coawst_fn_prefix_d02)
#        coawst_merge_netcdf(vn, input.coawst_fp_unmerged, input.coawst_fn_prefix_d01)
        

#%% WGMS PRE-PROCESSING
if args.option_wgms == 1:
    print('Processing WGMS datasets...')
    # Connect the WGMS mass balance datasets with the RGIIds and relevant elevation bands
    # Note: WGMS reports the RGI in terms of V5 as opposed to V6.  Some of the glaciers have changed their RGIId between
    #       the two versions, so need to convert WGMS V5 Ids to V6 Ids using the GLIMSID.
    # PROBLEMS WITH DATASETS:
    #  - need to be careful with information describing dataset as some descriptions appear to be incorrect.
        
    # ===== Dictionaries (WGMS --> RGIID V6) =====
    # Load RGI version 5 & 6 and create dictionary linking the two
    #  -required to avoid errors associated with changes in RGIId between the two versions in some regions
    rgiv6_fn_all = glob.glob(input.rgiv6_fn_prefix)
    rgiv5_fn_all = glob.glob(input.rgiv5_fn_prefix)
    # Create dictionary of all regions
    #  - regions that didn't change between versions (ex. 13, 14, 15) will all the be same.  Others that have changed
    #    may vary greatly.
    for n in range(len(rgiv6_fn_all)):
        print('Region', n+1)
        rgiv6_fn = glob.glob(input.rgiv6_fn_prefix)[n]
        rgiv6 = pd.read_csv(rgiv6_fn, encoding='latin1')
        rgiv5_fn = glob.glob(input.rgiv5_fn_prefix)[n]
        rgiv5 = pd.read_csv(rgiv5_fn, encoding='latin1')
        # Dictionary to link versions 5 & 6
        rgi_version_compare = rgiv5[['RGIId', 'GLIMSId']].copy()
        rgi_version_compare['RGIIdv6'] = np.nan
        # Link versions 5 & 6 based on GLIMSID
        for r in range(rgiv5.shape[0]):
            try:
                # Use GLIMSID
                rgi_version_compare.iloc[r,2] = (
                        rgiv6.iloc[rgiv6['GLIMSId'].values == rgiv5.loc[r,'GLIMSId'],0].values[0])
        #        # Use Lat/Lon
        #        latlon_dif = abs(rgiv6[['CenLon', 'CenLat']].values - rgiv5[['CenLon', 'CenLat']].values[r,:])
        #        latlon_dif[abs(latlon_dif) < 1e-6] = 0
        #        rgi_version_compare.iloc[r,2] = rgiv6.iloc[np.where(latlon_dif[:,0] + latlon_dif[:,1] < 0.001)[0][0],0]
            except:
                rgi_version_compare.iloc[r,2] = np.nan
        rgiv56_dict_reg = dict(zip(rgi_version_compare['RGIId'], rgi_version_compare['RGIIdv6']))
        latdict_reg = dict(zip(rgiv6['RGIId'], rgiv6['CenLat']))
        londict_reg = dict(zip(rgiv6['RGIId'], rgiv6['CenLon']))
        rgiv56_dict = {}
        latdict = {}
        londict = {}
        rgiv56_dict.update(rgiv56_dict_reg)
        latdict.update(latdict_reg)
        londict.update(londict_reg)
    # RGI Lookup table
    rgilookup = pd.read_csv(input.rgilookup_fullfn, skiprows=2)
    rgidict = dict(zip(rgilookup['FoGId'], rgilookup['RGIId']))
    # WGMS Lookup table
    wgmslookup = pd.read_csv(input.wgms_fp + input.wgms_lookup_fn, encoding='latin1')
    wgmsdict = dict(zip(wgmslookup['WGMS_ID'], wgmslookup['RGI_ID']))
    # Manual lookup table
    mandict = {10402: 'RGI60-13.10093',
               10401: 'RGI60-15.03734',
               6846: 'RGI60-15.12707'}
    
    # ===== WGMS (D) Geodetic mass balance data =====
    if 'wgms_d' in input.wgms_datasets:
        print('Processing geodetic thickness change data')
        wgms_mb_geo_all = pd.read_csv(input.wgms_fp + input.wgms_d_fn, encoding='latin1')
        wgms_mb_geo_all['RGIId_rgidict'] = wgms_mb_geo_all['WGMS_ID'].map(rgidict)
        wgms_mb_geo_all['RGIId_mandict'] = wgms_mb_geo_all['WGMS_ID'].map(mandict)
        wgms_mb_geo_all['RGIId_wgmsdict'] = wgms_mb_geo_all['WGMS_ID'].map(wgmsdict)
        wgms_mb_geo_all['RGIId_wgmsdictv6'] = wgms_mb_geo_all['RGIId_wgmsdict'].map(rgiv56_dict)
        # Use dictionaries to convert wgms data to RGIIds
        wgms_mb_geo_RGIIds_all_raw_wdicts = wgms_mb_geo_all[['RGIId_rgidict', 'RGIId_mandict','RGIId_wgmsdictv6']]
        wgms_mb_geo_RGIIds_all_raw = (
                wgms_mb_geo_RGIIds_all_raw_wdicts.apply(lambda x: sorted(x, key=pd.isnull), 1).iloc[:,0])
        # Determine regions and glacier numbers
        wgms_mb_geo_all['RGIId'] = wgms_mb_geo_RGIIds_all_raw.values
        wgms_mb_geo_all['version'], wgms_mb_geo_all['glacno'] = wgms_mb_geo_RGIIds_all_raw.str.split('-').dropna().str
        wgms_mb_geo_all['glacno'] = wgms_mb_geo_all['glacno'].apply(pd.to_numeric)
        wgms_mb_geo_all['region'] = wgms_mb_geo_all['glacno'].apply(np.floor)
        wgms_mb_geo = wgms_mb_geo_all[np.isfinite(wgms_mb_geo_all['glacno'])].sort_values('glacno')
        wgms_mb_geo.reset_index(drop=True, inplace=True)
        # Add latitude and longitude 
        wgms_mb_geo['CenLat'] = wgms_mb_geo['RGIId'].map(latdict)
        wgms_mb_geo['CenLon'] = wgms_mb_geo['RGIId'].map(londict)

        # Export relevant information
        wgms_mb_geo_export = pd.DataFrame()
        export_cols_geo = ['RGIId', 'glacno', 'WGMS_ID', 'CenLat', 'CenLon', 'REFERENCE_DATE', 'SURVEY_DATE', 
                           'LOWER_BOUND', 'UPPER_BOUND', 'AREA_SURVEY_YEAR', 'AREA_CHANGE', 'AREA_CHANGE_UNC', 
                           'THICKNESS_CHG', 'THICKNESS_CHG_UNC', 'VOLUME_CHANGE', 'VOLUME_CHANGE_UNC', 
                           'SD_PLATFORM_METHOD', 'RD_PLATFORM_METHOD', 'REFERENCE', 'REMARKS', 'INVESTIGATOR', 
                           'SPONS_AGENCY']
        wgms_mb_geo_export = wgms_mb_geo.loc[(np.isfinite(wgms_mb_geo['THICKNESS_CHG']) | 
                                             (np.isfinite(wgms_mb_geo['VOLUME_CHANGE']))), export_cols_geo]
        # Add observation type for comparison (massbalance, snowline, etc.)
        wgms_mb_geo_export[input.wgms_obs_type_cn] = 'mb_geo'
        wgms_mb_geo_export.reset_index(drop=True, inplace=True)
        wgms_mb_geo_export_fn = input.wgms_fp + input.wgms_d_fn_preprocessed
        wgms_mb_geo_export.to_csv(wgms_mb_geo_export_fn)
    
    # ===== WGMS (EE) Glaciological mass balance data =====
    if 'wgms_ee' in input.wgms_datasets:
        print('Processing glaciological mass balance data')
        wgms_mb_glac_all = pd.read_csv(input.wgms_fp + input.wgms_ee_fn, encoding='latin1')
        wgms_mb_glac_all['RGIId_rgidict'] = wgms_mb_glac_all['WGMS_ID'].map(rgidict)
        wgms_mb_glac_all['RGIId_mandict'] = wgms_mb_glac_all['WGMS_ID'].map(mandict)
        wgms_mb_glac_all['RGIId_wgmsdict'] = wgms_mb_glac_all['WGMS_ID'].map(wgmsdict)
        wgms_mb_glac_all['RGIId_wgmsdictv6'] = wgms_mb_glac_all['RGIId_wgmsdict'].map(rgiv56_dict)
        # Use dictionaries to convert wgms data to RGIIds
        wgms_mb_glac_RGIIds_all_raw_wdicts = wgms_mb_glac_all[['RGIId_rgidict', 'RGIId_mandict','RGIId_wgmsdictv6']]
        wgms_mb_glac_RGIIds_all_raw = (
                wgms_mb_glac_RGIIds_all_raw_wdicts.apply(lambda x: sorted(x, key=pd.isnull), 1).iloc[:,0])
        # Determine regions and glacier numbers
        wgms_mb_glac_all['RGIId'] = wgms_mb_glac_RGIIds_all_raw.values
        wgms_mb_glac_all['version'], wgms_mb_glac_all['glacno'] = (
                wgms_mb_glac_RGIIds_all_raw.str.split('-').dropna().str)
        wgms_mb_glac_all['glacno'] = wgms_mb_glac_all['glacno'].apply(pd.to_numeric)
        wgms_mb_glac_all['region'] = wgms_mb_glac_all['glacno'].apply(np.floor)
        wgms_mb_glac = wgms_mb_glac_all[np.isfinite(wgms_mb_glac_all['glacno'])].sort_values('glacno')
        wgms_mb_glac.reset_index(drop=True, inplace=True)
        # Add latitude and longitude 
        wgms_mb_glac['CenLat'] = wgms_mb_glac['RGIId'].map(latdict)
        wgms_mb_glac['CenLon'] = wgms_mb_glac['RGIId'].map(londict)
        # Import MB overview data to extract survey dates
        wgms_mb_overview = pd.read_csv(input.wgms_fp + input.wgms_e_fn, encoding='latin1')
        wgms_mb_glac['BEGIN_PERIOD'] = np.nan 
        wgms_mb_glac['END_PERIOD'] = np.nan 
        wgms_mb_glac['TIME_SYSTEM'] = np.nan
        wgms_mb_glac['END_WINTER'] = np.nan
        for x in range(wgms_mb_glac.shape[0]):
            wgms_mb_glac.loc[x,'BEGIN_PERIOD'] = (
                    wgms_mb_overview[(wgms_mb_glac.loc[x,'WGMS_ID'] == wgms_mb_overview['WGMS_ID']) & 
                                     (wgms_mb_glac.loc[x,'YEAR'] == wgms_mb_overview['Year'])]['BEGIN_PERIOD'].values)
            wgms_mb_glac.loc[x,'END_WINTER'] = (
                    wgms_mb_overview[(wgms_mb_glac.loc[x,'WGMS_ID'] == wgms_mb_overview['WGMS_ID']) & 
                                     (wgms_mb_glac.loc[x,'YEAR'] == wgms_mb_overview['Year'])]['END_WINTER'].values)
            wgms_mb_glac.loc[x,'END_PERIOD'] = (
                    wgms_mb_overview[(wgms_mb_glac.loc[x,'WGMS_ID'] == wgms_mb_overview['WGMS_ID']) & 
                                     (wgms_mb_glac.loc[x,'YEAR'] == wgms_mb_overview['Year'])]['END_PERIOD'].values)
            wgms_mb_glac.loc[x,'TIME_SYSTEM'] = (
                    wgms_mb_overview[(wgms_mb_glac.loc[x,'WGMS_ID'] == wgms_mb_overview['WGMS_ID']) & 
                                     (wgms_mb_glac.loc[x,'YEAR'] == wgms_mb_overview['Year'])]['TIME_SYSTEM'].values[0])  
        # Split summer, winter, and annual into separate rows so each becomes a data point in the calibration
        #  if summer and winter exist, then discard annual to avoid double-counting the annual measurement
        export_cols_annual = ['RGIId', 'glacno', 'WGMS_ID', 'CenLat', 'CenLon', 'YEAR', 'TIME_SYSTEM', 'BEGIN_PERIOD', 
                              'END_WINTER', 'END_PERIOD', 'LOWER_BOUND', 'UPPER_BOUND', 'ANNUAL_BALANCE', 
                              'ANNUAL_BALANCE_UNC', 'REMARKS']
        export_cols_summer = ['RGIId', 'glacno', 'WGMS_ID', 'CenLat', 'CenLon', 'YEAR', 'TIME_SYSTEM', 'BEGIN_PERIOD', 
                              'END_WINTER', 'END_PERIOD', 'LOWER_BOUND', 'UPPER_BOUND', 'SUMMER_BALANCE', 
                              'SUMMER_BALANCE_UNC', 'REMARKS']
        export_cols_winter = ['RGIId', 'glacno', 'WGMS_ID', 'CenLat', 'CenLon', 'YEAR', 'TIME_SYSTEM', 'BEGIN_PERIOD', 
                              'END_WINTER', 'END_PERIOD', 'LOWER_BOUND', 'UPPER_BOUND', 'WINTER_BALANCE', 
                              'WINTER_BALANCE_UNC', 'REMARKS']
        wgms_mb_glac_annual = wgms_mb_glac.loc[((np.isnan(wgms_mb_glac['WINTER_BALANCE'])) & 
                                                (np.isnan(wgms_mb_glac['SUMMER_BALANCE']))), export_cols_annual]
        wgms_mb_glac_summer = wgms_mb_glac.loc[np.isfinite(wgms_mb_glac['SUMMER_BALANCE']), export_cols_summer]
        wgms_mb_glac_winter = wgms_mb_glac.loc[np.isfinite(wgms_mb_glac['WINTER_BALANCE']), export_cols_winter]
        # Assign a time period to each of the measurements, which will be used for comparison with model data 
        wgms_mb_glac_annual['period'] = 'annual'
        wgms_mb_glac_summer['period'] = 'summer'
        wgms_mb_glac_winter['period'] = 'winter'
        # Rename columns such that all rows are the same
        wgms_mb_glac_annual.rename(columns={'ANNUAL_BALANCE': 'BALANCE', 'ANNUAL_BALANCE_UNC': 'BALANCE_UNC'}, 
                                   inplace=True)
        wgms_mb_glac_summer.rename(columns={'SUMMER_BALANCE': 'BALANCE', 'SUMMER_BALANCE_UNC': 'BALANCE_UNC'}, 
                                   inplace=True)
        wgms_mb_glac_winter.rename(columns={'WINTER_BALANCE': 'BALANCE', 'WINTER_BALANCE_UNC': 'BALANCE_UNC'}, 
                                   inplace=True)
        # Export relevant information
        wgms_mb_glac_export = (pd.concat([wgms_mb_glac_annual, wgms_mb_glac_summer, wgms_mb_glac_winter])
                                         .sort_values(['glacno', 'YEAR']))
        # Add observation type for comparison (massbalance, snowline, etc.)
        wgms_mb_glac_export[input.wgms_obs_type_cn] = 'mb_glac'
        wgms_mb_glac_export.reset_index(drop=True, inplace=True)
        wgms_mb_glac_export_fn = input.wgms_fp + input.wgms_ee_fn_preprocessed
        wgms_mb_glac_export.to_csv(wgms_mb_glac_export_fn)


#%% Create netcdf file of lapse rates from temperature pressure level data
if args.option_createlapserates == 1:
    # Input data
    gcm_filepath = os.getcwd() + '/../Climate_data/ERA_Interim/HMA_temp_pressurelevel_data/'
    gcm_filename_prefix = 'HMA_EraInterim_temp_pressurelevels_'
    tempname = 't'
    levelname = 'level'
    latname = 'latitude'
    lonname = 'longitude'
    elev_idx_max = 1
    elev_idx_min = 10
    startyear = 1979
    endyear = 2017
    output_filepath = '../Output/'
    output_filename_prefix = 'HMA_Regions13_14_15_ERAInterim_lapserates'
    
    def lapserates_createnetcdf(gcm_filepath, gcm_filename_prefix, tempname, levelname, latname, lonname, elev_idx_max, 
                                elev_idx_min, startyear, endyear, output_filepath, output_filename_prefix):
        """
        Create netcdf of lapse rate for every latitude and longitude for each month.
        
        The lapse rates are computed based on the slope of a linear line of best fit for the temperature pressure level 
        data.  Prior to running this function, you must explore the temperature pressure level data to determine the 
        elevation range indices for a given region, variable names, etc.

        Parameters
        ----------
        gcm_filepath : str
            filepath where climate data is located
        gcm_filename_prefix : str
            prefix of filename
        tempname : str
            temperature variable name
        levelname : str
            pressure level variable name
        latname : str
            latitude variable name
        lonname : str
            longitude variable name
        elev_idx_max : int
            index of the maximum pressure level being used
        elev_idx_min : int
            index of the minimum pressure level being used
        startyear : int
            starting year
        endyear : int
            ending year
        output_filepath : str
            filepath where output is to be exported
        output_filename_prefix : str
            filename prefix of the output
        
        Returns
        -------
        exports netcdf of lapse rates
        """
        fullfilename = gcm_filepath + gcm_filename_prefix + str(startyear) + '.nc'
        data = xr.open_dataset(fullfilename)    
        # Extract the pressure levels [Pa]
        if data[levelname].attrs['units'] == 'millibars':
            # Convert pressure levels from millibars to Pa
            levels = data[levelname].values * 100
        # Compute the elevation [m a.s.l] of the pressure levels using the barometric pressure formula (pressure in Pa)
        elev = -input.R_gas*input.temp_std/(input.gravity*input.molarmass_air)*np.log(levels/input.pressure_std)
        # Netcdf file for lapse rates ('w' will overwrite existing file)
        output_fullfilename = (output_filepath + output_filename_prefix + '_' + str(startyear) + '_' + str(endyear) + 
                               '.nc')
        netcdf_output = nc.Dataset(output_fullfilename, 'w', format='NETCDF4')
        # Global attributes
        netcdf_output.description = ('Lapse rates from ERA Interim pressure level data that span the regions elevation'
                                     + 'range')
        netcdf_output.history = 'Created ' + str(strftime("%Y-%m-%d %H:%M:%S"))
        netcdf_output.source = 'ERA Interim reanalysis data downloaded February 2018'
        # Dimensions
        latitude = netcdf_output.createDimension('latitude', data['latitude'].values.shape[0])
        longitude = netcdf_output.createDimension('longitude', data['longitude'].values.shape[0])
        time = netcdf_output.createDimension('time', None)
        # Create dates in proper format for time dimension
        startdate = str(startyear) + '-01-01'
        enddate = str(endyear) + '-12-31'
        startdate = datetime(*[int(item) for item in startdate.split('-')])
        enddate = datetime(*[int(item) for item in enddate.split('-')])
        startdate = startdate.strftime('%Y-%m')
        enddate = enddate.strftime('%Y-%m')
        dates = pd.DataFrame({'date' : pd.date_range(startdate, enddate, freq='MS')})
        dates = dates['date'].astype(datetime)
        # Variables associated with dimensions 
        latitude = netcdf_output.createVariable('latitude', np.float32, ('latitude',))
        latitude.long_name = 'latitude'
        latitude.units = 'degrees_north'
        latitude[:] = data['latitude'].values
        longitude = netcdf_output.createVariable('longitude', np.float32, ('longitude',))
        longitude.long_name = 'longitude'
        longitude.units = 'degrees_east'
        longitude[:] = data['longitude'].values
        time = netcdf_output.createVariable('time', np.float64, ('time',))
        time.long_name = "time"
        time.units = "hours since 1900-01-01 00:00:00"
        time.calendar = "gregorian"
        time[:] = nc.date2num(dates, units=time.units, calendar=time.calendar)
        lapserate = netcdf_output.createVariable('lapserate', np.float64, ('time', 'latitude', 'longitude'))
        lapserate.long_name = "lapse rate"
        lapserate.units = "degC m-1"
        # Set count to keep track of time position
        count = 0
        for year in range(startyear,endyear+1):
            print(year)
            fullfilename_year = gcm_filepath + gcm_filename_prefix + str(year) + '.nc'
            data_year = xr.open_dataset(fullfilename_year)
            count = count + 1
            for lat in range(0,latitude[:].shape[0]):
                for lon in range(0,longitude[:].shape[0]):
                    data_subset = data_year[tempname].isel(level=range(elev_idx_max,elev_idx_min+1), 
                                                           latitude=lat, longitude=lon).values
                    lapserate_subset = (((elev[elev_idx_max:elev_idx_min+1] * data_subset).mean(axis=1) - 
                                         elev[elev_idx_max:elev_idx_min+1].mean() * data_subset.mean(axis=1)) / 
                                        ((elev[elev_idx_max:elev_idx_min+1]**2).mean() - 
                                         (elev[elev_idx_max:elev_idx_min+1].mean())**2))
                    lapserate[12*(count-1):12*count,lat,lon] = lapserate_subset
                    # Takes roughly 4 minutes per year to compute the lapse rate for each lat/lon combo in HMA
        netcdf_output.close()
        
        # Application of the lapserate_createnetcdf function
    print('Creating lapse rates...')
    lapserates_createnetcdf(gcm_filepath, gcm_filename_prefix, tempname, levelname, latname, lonname, elev_idx_max, 
                            elev_idx_min, startyear, endyear, output_filepath, output_filename_prefix)


#%% Write csv file from model results
# Create csv such that not importing the air temperature each time (takes 90 seconds for 13,119 glaciers)
#output_csvfullfilename = input.main_directory + '/../Output/ERAInterim_elev_15_SouthAsiaEast.csv'
#climate.createcsv_GCMvarnearestneighbor(input.gcm_prec_filename, input.gcm_prec_varname, dates_table, main_glac_rgi, 
#                                        output_csvfullfilename)
#np.savetxt(output_csvfullfilename, main_glac_gcmelev, delimiter=",") 
    

#%% NEAREST NEIGHBOR CALIBRATION PARAMETERS
## Load csv
#ds = pd.read_csv(input.main_directory + '/../Output/calibration_R15_20180403_Opt02solutionspaceexpanding.csv', 
#                 index_col='GlacNo')
## Select data of interest
#data = ds[['CenLon', 'CenLat', 'lrgcm', 'lrglac', 'precfactor', 'precgrad', 'ddfsnow', 'ddfice', 'tempsnow', 
#           'tempchange']].copy()
## Drop nan data to retain only glaciers with calibrated parameters
#data_cal = data.dropna()
#A = data_cal.mean(0)
## Select latitude and longitude of calibrated parameters for distance estimate
#data_cal_lonlat = data_cal.iloc[:,0:2].values
## Loop through each glacier and select the parameters based on the nearest neighbor
#for glac in range(data.shape[0]):
#    # Avoid applying this to any glaciers that already were optimized
#    if data.iloc[glac, :].isnull().values.any() == True:
#        # Select the latitude and longitude of the glacier's center
#        glac_lonlat = data.iloc[glac,0:2].values
#        # Set point to be compatible with cdist function (from scipy)
#        pt = [[glac_lonlat[0],glac_lonlat[1]]]
#        # scipy function to calculate distance
#        distances = cdist(pt, data_cal_lonlat)
#        # Find minimum index (could be more than one)
#        idx_min = np.where(distances == distances.min())[1]
#        # Set new parameters
#        data.iloc[glac,2:] = data_cal.iloc[idx_min,2:].values.mean(0)
#        #  use mean in case multiple points are equidistant from the glacier
## Remove latitude and longitude to create csv file
#parameters_export = data.iloc[:,2:]
## Export csv file
#parameters_export.to_csv(input.main_directory + '/../Calibration_datasets/calparams_R15_20180403_nearest.csv', 
#                         index=False)    