"""
fxns_massbalance.py is a list of functions that are used to compute the mass
associated with each glacier for PyGEM.
"""
#========= LIST OF PACKAGES ==================================================
import numpy as np
#import pandas as pd
#========= IMPORT COMMON VARIABLES FROM MODEL INPUT ==========================
import pygem_input as input

#========= FUNCTIONS (alphabetical order) ===================================
def runmassbalance(modelparameters, glacier_rgi_table, glacier_area_t0, icethickness_t0, width_t0, elev_bins, 
                   glacier_gcm_temp, glacier_gcm_prec, glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, 
                   dates_table, option_areaconstant=0, debug=False):
    """
    Runs the mass balance and mass redistribution allowing the glacier to evolve.

    Parameters
    ----------
    modelparameters : pd.Series
        Model parameters (lrgcm, lrglac, precfactor, precgrad, ddfsnow, ddfice, tempsnow, tempchange)
        Order of model parameters should not be changed as the run mass balance script relies on this order.
    glacier_rgi_table : pd.Series
        Table of glacier's RGI information
    glacier_area_t0 : np.ndarray
        Initial glacier area [km2] for each elevation bin
    icethickness_t0 : np.ndarray
        Initial ice thickness [m] for each elevation bin
    width_t0 : np.ndarray
        Initial glacier width [km] for each elevation bin
    elev_bins : np.ndarray
        Elevation bins [masl]
    glacier_gcm_temp : np.ndarray
        GCM temperature [degC] at each time step based on nearest neighbor to the glacier
    glacier_gcm_prec : np.ndarray
        GCM precipitation (solid and liquid) [m] at each time step based on nearest neighbor to the glacier
    glacier_gcm_elev : float
        GCM elevation [masl] at each time step based on nearest neighbor to the glacier
    glacier_gcm_lrgcm : np.ndarray
        GCM lapse rate [K m-1] from the GCM to the glacier for each time step based on nearest neighbor to the glacier
    glacier_gcm_lrglac : np.ndarray
        GCM lapse rate [K m-1] over the glacier for each time step based on nearest neighbor to the glacier
    dates_table : pd.DataFrame
        Table of dates, year, month, daysinmonth, wateryear, and season for each timestep
    option_areaconstant : int
        switch to keep glacier area constant or not (default 0 allows glacier area to change annually)
    debug : Boolean
        option to turn on print statements for development or debugging of code (default False)

    Returns
    -------
    glac_bin_temp : np.ndarray
        Temperature [degC] for each elevation bin and timestep
    glac_bin_prec : np.ndarray
        Precipitation (only liquid) [m] for each elevation bin and timestep
    glac_bin_acc : np.ndarray
        Accumulation (solid precipitation) [mwe] for each elevation bin and timestep
    glac_bin_refreeze : np.ndarray
        Refreeze [mwe] for each elevation bin and timestep
    glac_bin_snowpack : np.ndarray
        Snowpack [mwe] for each elevation bin and timestep
    glac_bin_melt : np.ndarray
        Melt [mwe] for each elevation bin and timestep
    glac_bin_frontalablation : np.ndarray
        Frontal ablation [mwe] for each elevation bin and timestep
    glac_bin_massbalclim : np.ndarray
        Climatic mass balance [mwe] for each elevation bin and timestep
    glac_bin_massbalclim_annual : np.ndarray
        Climatic mass balance [mwe] for each elevation bin and year
    glac_bin_area_annual : np.ndarray
        Glacier area [km2] for each elevation bin and year
    glac_bin_icethickness_annual : np.ndarray
        Ice thickness [m] for each elevation bin and year
    glac_bin_width_annual : np.ndarray
        Glacier width [km] for each elevation bin and year
    glac_bin_surfacetype_annual : np.ndarray
        Surface type [see dictionary] for each elevation bin and year
    glac_wide_massbaltotal : np.ndarray
        Glacier-wide total mass balance (climatic mass balance - frontal ablation) [mwe] for each timestep
    glac_wide_runoff : np.ndarray
        Glacier-wide runoff [m3] for each timestep
    glac_wide_snowline : np.ndarray
        Snowline altitude [masl] for each timestep
    glac_wide_snowpack : np.ndarray
        Glacier-wide snowpack [km3 we] for each timestep
    glac_wide_area_annual : np.ndarray
        Glacier-wide area [km2] for each timestep
    glac_wide_volume_annual : np.ndarray
        Glacier-wide volume [km3 ice] for each timestep
    glac_wide_ELA_annual : np.ndarray
        Equilibrium line altitude [masl] for each year
    """       
    if debug:
        print('\n\nDEBUGGING MASS BALANCE FUNCTION\n\n')
    #%%
    # Select annual divisor and columns
    if input.timestep == 'monthly':
        annual_divisor = 12
#    annual_columns = np.unique(dates_table['wateryear'].values)
    annual_columns = np.arange(0, int(dates_table.shape[0] / 12))
    # Variables to export
    glac_bin_temp = np.zeros((elev_bins.shape[0],glacier_gcm_temp.shape[0]))
    glac_bin_prec = np.zeros((elev_bins.shape[0],glacier_gcm_temp.shape[0]))
    glac_bin_acc = np.zeros((elev_bins.shape[0],glacier_gcm_temp.shape[0]))
    glac_bin_refreezepotential = np.zeros((elev_bins.shape[0],glacier_gcm_temp.shape[0]))
    glac_bin_refreeze = np.zeros((elev_bins.shape[0],glacier_gcm_temp.shape[0]))
    glac_bin_melt = np.zeros((elev_bins.shape[0],glacier_gcm_temp.shape[0]))
    glac_bin_meltsnow = np.zeros((elev_bins.shape[0],glacier_gcm_temp.shape[0]))
    glac_bin_meltrefreeze = np.zeros((elev_bins.shape[0],glacier_gcm_temp.shape[0]))
    glac_bin_meltglac = np.zeros((elev_bins.shape[0],glacier_gcm_temp.shape[0]))
    glac_bin_frontalablation = np.zeros((elev_bins.shape[0],glacier_gcm_temp.shape[0]))
    glac_bin_snowpack = np.zeros((elev_bins.shape[0],glacier_gcm_temp.shape[0]))
    glac_bin_massbalclim = np.zeros((elev_bins.shape[0],glacier_gcm_temp.shape[0]))
    glac_bin_massbalclim_annual = np.zeros((elev_bins.shape[0],annual_columns.shape[0]))
    glac_bin_surfacetype_annual = np.zeros((elev_bins.shape[0],annual_columns.shape[0]))
    glac_bin_icethickness_annual = np.zeros((elev_bins.shape[0], annual_columns.shape[0] + 1))
    glac_bin_area_annual = np.zeros((elev_bins.shape[0], annual_columns.shape[0] + 1))
    glac_bin_width_annual = np.zeros((elev_bins.shape[0], annual_columns.shape[0] + 1))
    # Local variables
    glac_bin_precsnow = np.zeros((elev_bins.shape[0],glacier_gcm_temp.shape[0]))
    refreeze_potential = np.zeros(elev_bins.shape[0])
    snowpack_remaining = np.zeros(elev_bins.shape[0])
    dayspermonth = dates_table['daysinmonth'].values
    surfacetype_ddf = np.zeros(elev_bins.shape[0])
    glac_idx_initial = glacier_area_t0.nonzero()[0]
    glac_area_initial = glacier_area_t0.copy()
    
    # Sea level for marine-terminating glaciers
    sea_level = 0
    rgi_region = int(glacier_rgi_table.RGIId.split('-')[1].split('.')[0])
    frontalablation_k0 = input.frontalablation_k0dict[rgi_region]
    # Adjust sea level to account for disagreement between ice thickness estimates and glaciers classified by RGI as
    # marine-terminating. Modify the sea level, so sea level is consistent with lowest elevation bin that has ice.
    if glacier_rgi_table.loc['TermType'] == 1:
        sea_level = elev_bins[glac_idx_initial[0]] - (elev_bins[1] - elev_bins[0]) / 2
    
    #  glac_idx_initial is used with advancing glaciers to ensure no bands are added in a discontinuous section
    # Run mass balance only on pixels that have an ice thickness (some glaciers do not have an ice thickness)
    #  otherwise causes indexing errors causing the model to fail
    if icethickness_t0.max() > 0:    
        if input.option_adjusttemp_surfelev == 1:
            # ice thickness initial is used to adjust temps to changes in surface elevation
            icethickness_initial = icethickness_t0.copy()
            icethickness_initial[0:icethickness_initial.nonzero()[0][0]] = (
                    icethickness_initial[icethickness_initial.nonzero()[0][0]])
            #  bins that advance need to have an initial ice thickness; otherwise, the temp adjustment will be based on
            #  ice thickness - 0, which is wrong  Since advancing bins take the thickness of the previous bin, set the
            #  initial ice thickness of all bins below the terminus to the ice thickness at the terminus.
        # Compute the initial surface type [0=off-glacier, 1=ice, 2=snow, 3=firn, 4=debris]
        surfacetype, firnline_idx = surfacetypebinsinitial(glacier_area_t0, glacier_rgi_table, elev_bins)
        # Create surface type DDF dictionary (manipulate this function for calibration or for each glacier)
        surfacetype_ddf_dict = surfacetypeDDFdict(modelparameters)
        
    # ANNUAL LOOP (daily or monthly timestep contained within loop)
    for year in range(0, annual_columns.shape[0]): 
        # Check ice still exists:
        if icethickness_t0.max() > 0:    
        
#            if debug:
#                print(year, 'max ice thickness [m]:', icethickness_t0.max())

            # Glacier indices
            glac_idx_t0 = glacier_area_t0.nonzero()[0]
            # Functions currently set up for monthly timestep
            #  only compute mass balance while glacier exists
            if (input.timestep == 'monthly') and (glac_idx_t0.shape[0] != 0):      
                # AIR TEMPERATURE: Downscale the gcm temperature [deg C] to each bin
                if input.option_temp2bins == 1:
                    # Downscale using gcm and glacier lapse rates
                    #  T_bin = T_gcm + lr_gcm * (z_ref - z_gcm) + lr_glac * (z_bin - z_ref) + tempchange
                    glac_bin_temp[:,12*year:12*(year+1)] = (glacier_gcm_temp[12*year:12*(year+1)] + 
                         glacier_gcm_lrgcm[12*year:12*(year+1)] * 
                         (glacier_rgi_table.loc[input.option_elev_ref_downscale] - glacier_gcm_elev) + 
                         glacier_gcm_lrglac[12*year:12*(year+1)] * (elev_bins - 
                         glacier_rgi_table.loc[input.option_elev_ref_downscale])[:,np.newaxis] + modelparameters[7])
                # Option to adjust air temperature based on changes in surface elevation
                if input.option_adjusttemp_surfelev == 1:
                    # T_air = T_air + lr_glac * (icethickness_present - icethickness_initial)
                    glac_bin_temp[:,12*year:12*(year+1)] = (glac_bin_temp[:,12*year:12*(year+1)] + 
                                                            glacier_gcm_lrglac[12*year:12*(year+1)] * 
                                                            (icethickness_t0 - icethickness_initial)[:,np.newaxis])
                # remove off-glacier values
                glac_bin_temp[surfacetype==0,12*year:12*(year+1)] = 0
                
                # PRECIPITATION/ACCUMULATION: Downscale the precipitation (liquid and solid) to each bin
                if input.option_prec2bins == 1:
                    # Precipitation using precipitation factor and precipitation gradient
                    #  P_bin = P_gcm * prec_factor * (1 + prec_grad * (z_bin - z_ref))
                    glac_bin_precsnow[:,12*year:12*(year+1)] = (glacier_gcm_prec[12*year:12*(year+1)] * 
                            modelparameters[2] * (1 + modelparameters[3] * (elev_bins - 
                            glacier_rgi_table.loc[input.option_elev_ref_downscale]))[:,np.newaxis])
                    
#                    if debug:
#                        if year >= 34:
#                            print(year)
#                            print('glac_idx_t0:', glac_idx_t0)
#                            print('modelparams[2] and [3]:', modelparameters[2], modelparameters[3])
#                            print('GCM prec:', glacier_gcm_prec[12*year:12*(year+1)])
#                            print('max prec:', glac_bin_precsnow[glac_idx_t0,12*year:12*(year+1)].max())
#                            print('prec', glac_bin_precsnow[glac_idx_t0,12*year:12*(year+1)])
                    
                    
                # Option to adjust prec of uppermost 25% of glacier for wind erosion and reduced moisture content
                if input.option_preclimit == 1:
                    # If elevation range > 1000 m, apply corrections to uppermost 25% of glacier (Huss and Hock, 2015)
                    if elev_bins[glac_idx_t0[-1]] - elev_bins[glac_idx_t0[0]] > 1000:
                        # Indices of upper 25%
                        glac_idx_upper25 = glac_idx_t0[(glac_idx_t0 - glac_idx_t0[0] + 1) / glac_idx_t0.shape[0] * 100 
                                                       > 75]   
                        # Exponential decay according to elevation difference from the 75% elevation
                        #  prec_upper25 = prec * exp(-(elev_i - elev_75%)/(elev_max- - elev_75%))
                        glac_bin_precsnow[glac_idx_upper25,12*year:12*(year+1)] = (
                                glac_bin_precsnow[glac_idx_upper25[0],12*year:12*(year+1)] * 
                                np.exp(-1*(elev_bins[glac_idx_upper25] - elev_bins[glac_idx_upper25[0]]) / 
                                   (elev_bins[glac_idx_upper25[-1]] - elev_bins[glac_idx_upper25[0]]))[:,np.newaxis])
                        # Precipitation cannot be less than 87.5% of the maximum accumulation elsewhere on the glacier
                        for month in range(0,12):
                            glac_bin_precsnow[glac_idx_upper25[(glac_bin_precsnow[glac_idx_upper25,month] < 0.875 * 
                                glac_bin_precsnow[glac_idx_t0,month].max()) & 
                                (glac_bin_precsnow[glac_idx_upper25,month] != 0)], month] = (
                                                                0.875 * glac_bin_precsnow[glac_idx_t0,month].max())
                # Separate total precipitation into liquid (glac_bin_prec) and solid (glac_bin_acc)
                if input.option_accumulation == 1:
                    # if temperature above threshold, then rain
                    glac_bin_prec[:,12*year:12*(year+1)][glac_bin_temp[:,12*year:12*(year+1)] > modelparameters[6]] = (
                        glac_bin_precsnow[:,12*year:12*(year+1)][glac_bin_temp[:,12*year:12*(year+1)] 
                                                                 > modelparameters[6]])
                    # if temperature below threshold, then snow
                    glac_bin_acc[:,12*year:12*(year+1)][glac_bin_temp[:,12*year:12*(year+1)] <= modelparameters[6]] = (
                        glac_bin_precsnow[:,12*year:12*(year+1)][glac_bin_temp[:,12*year:12*(year+1)] 
                                                                 <= modelparameters[6]])
                elif input.option_accumulation == 2:
                    # If temperature between min/max, then mix of snow/rain using linear relationship between min/max
                    glac_bin_prec[:,12*year:12*(year+1)] = ((1/2 + (glac_bin_temp[:,12*year:12*(year+1)] - 
                                 modelparameters[6]) / 2) * glac_bin_precsnow[:,12*year:12*(year+1)])
                    glac_bin_acc[:,12*year:12*(year+1)] = (glac_bin_precsnow[:,12*year:12*(year+1)] - 
                                glac_bin_prec[:,12*year:12*(year+1)])
                    # If temperature above maximum threshold, then all rain
                    (glac_bin_prec[:,12*year:12*(year+1)][glac_bin_temp[:,12*year:12*(year+1)] > modelparameters[6] + 1]
                        ) = (glac_bin_precsnow[:,12*year:12*(year+1)][glac_bin_temp[:,12*year:12*(year+1)] > 
                                                                      modelparameters[6] + 1])
                    (glac_bin_acc[:,12*year:12*(year+1)][glac_bin_temp[:,12*year:12*(year+1)] > modelparameters[6] + 1] 
                        ) = 0
                    # If temperature below minimum threshold, then all snow
                    (glac_bin_acc[:,12*year:12*(year+1)][glac_bin_temp[:,12*year:12*(year+1)] <= modelparameters[6] - 1]
                        )= (glac_bin_precsnow[:,12*year:12*(year+1)][glac_bin_temp[:,12*year:12*(year+1)] <= 
                                                                     modelparameters[6] - 1])
                    (glac_bin_prec[:,12*year:12*(year+1)][glac_bin_temp[:,12*year:12*(year+1)] <= modelparameters[6] - 
                                   1]) = 0
                # remove off-glacier values
                glac_bin_prec[surfacetype==0,12*year:12*(year+1)] = 0
                glac_bin_acc[surfacetype==0,12*year:12*(year+1)] = 0
                
                # POTENTIAL REFREEZE: compute potential refreeze [m w.e.] for each bin
                if input.option_refreezing == 1:
                    # Heat conduction approach based on Huss and Hock (2015)
                    print('Heat conduction approach has not been coded yet.  Please choose an option that exists.'
                          '\n\nExiting model run.\n\n')
                    exit()
                elif input.option_refreezing == 2:
                    # Refreeze based on air temperature based on Woodward et al. (1997)
                    bin_temp_annual = annualweightedmean_array(glac_bin_temp[:,12*year:12*(year+1)], 
                                                               dates_table.iloc[12*year:12*(year+1),:])
                    bin_refreezepotential_annual = (-0.69 * bin_temp_annual + 0.0096) * 1/100
                    #   R(m) = -0.69 * Tair + 0.0096 * (1 m / 100 cm)
                    #   Note: conversion from cm to m is included
                    # Remove negative refreezing values
                    bin_refreezepotential_annual[bin_refreezepotential_annual < 0] = 0
                    # Place annual refreezing in user-defined month for accounting and melt purposes
                    placeholder = (12 - dates_table.loc[0,'month'] + input.refreeze_month) % 12
                    glac_bin_refreezepotential[:,12*year + placeholder] = bin_refreezepotential_annual  
                # remove off-glacier values
                glac_bin_refreezepotential[surfacetype==0,12*year:12*(year+1)] = 0
                
                # ENTER MONTHLY LOOP (monthly loop required as )
                for month in range(0,12):
                    # Step is the position as a function of year and month, which improves readability
                    step = 12*year + month
                    
                    # SNOWPACK, REFREEZE, MELT, AND CLIMATIC MASS BALANCE
                    # Snowpack [m w.e.] = snow remaining + new snow
                    glac_bin_snowpack[:,step] = snowpack_remaining + glac_bin_acc[:,step]
                    # Energy available for melt [degC day]    
                    melt_energy_available = glac_bin_temp[:,step]*dayspermonth[step]
                    melt_energy_available[melt_energy_available < 0] = 0
                    # Snow melt [m w.e.]
                    glac_bin_meltsnow[:,step] = surfacetype_ddf_dict[2] * melt_energy_available
                    # snow melt cannot exceed the snow depth
                    glac_bin_meltsnow[glac_bin_meltsnow[:,step] > glac_bin_snowpack[:,step], step] = (
                            glac_bin_snowpack[glac_bin_meltsnow[:,step] > glac_bin_snowpack[:,step], step])
                    # Energy remaining after snow melt [degC day]
                    melt_energy_available = melt_energy_available - glac_bin_meltsnow[:,step] / surfacetype_ddf_dict[2]
                    # remove low values of energy available caused by rounding errors in the step above
                    melt_energy_available[abs(melt_energy_available) < input.tolerance] = 0
                    # Compute the refreeze, refreeze melt, and any changes to the snow depth
                    # Refreeze potential [m w.e.]
                    #  timing of refreeze potential will vary with the method (air temperature approach updates annual 
                    #  and heat conduction approach updates monthly), so check if refreeze is being udpated
                    if glac_bin_refreezepotential[:,step].max() > 0:
                        refreeze_potential = glac_bin_refreezepotential[:,step]
                    # Refreeze [m w.e.]
                    #  refreeze in ablation zone cannot exceed amount of snow melt (accumulation zone modified below)
                    glac_bin_refreeze[:,step] = glac_bin_meltsnow[:,step]
                    # refreeze cannot exceed refreeze potential
                    glac_bin_refreeze[glac_bin_refreeze[:,step] > refreeze_potential, step] = (
                            refreeze_potential[glac_bin_refreeze[:,step] > refreeze_potential])
                    glac_bin_refreeze[abs(glac_bin_refreeze[:,step]) < input.tolerance, step] = 0
                    # Refreeze melt [m w.e.]
                    glac_bin_meltrefreeze[:,step] = surfacetype_ddf_dict[2] * melt_energy_available
                    # refreeze melt cannot exceed the refreeze
                    glac_bin_meltrefreeze[glac_bin_meltrefreeze[:,step] > glac_bin_refreeze[:,step], step] = (
                            glac_bin_refreeze[glac_bin_meltrefreeze[:,step] > glac_bin_refreeze[:,step], step])
                    # Energy remaining after refreeze melt [degC day]
                    melt_energy_available = (melt_energy_available - glac_bin_meltrefreeze[:,step] / 
                                             surfacetype_ddf_dict[2])
                    melt_energy_available[abs(melt_energy_available) < input.tolerance] = 0
                    # Snow remaining [m w.e.]
                    snowpack_remaining = (glac_bin_snowpack[:,step] + glac_bin_refreeze[:,step] - 
                                          glac_bin_meltsnow[:,step] - glac_bin_meltrefreeze[:,step])
                    snowpack_remaining[abs(snowpack_remaining) < input.tolerance] = 0
                    # Compute melt from remaining energy, if any exits, and additional refreeze in the accumulation zone
                    # DDF based on surface type [m w.e. degC-1 day-1]
                    for surfacetype_idx in surfacetype_ddf_dict: 
                        surfacetype_ddf[surfacetype == surfacetype_idx] = surfacetype_ddf_dict[surfacetype_idx]
                    # Glacier melt [m w.e.] based on remaining energy
                    glac_bin_meltglac[:,step] = surfacetype_ddf * melt_energy_available
                    # Energy remaining after glacier surface melt [degC day]
                    #  must specify on-glacier values, otherwise this will divide by zero and cause an error
                    melt_energy_available[surfacetype != 0] = (melt_energy_available[surfacetype != 0] - 
                                         glac_bin_meltglac[surfacetype != 0, step] / surfacetype_ddf[surfacetype != 0])
                    melt_energy_available[abs(melt_energy_available) < input.tolerance] = 0
                    # Additional refreeze in the accumulation area [m w.e.]
                    #  refreeze in accumulation zone = refreeze of snow + refreeze of underlying snow/firn
                    glac_bin_refreeze[elev_bins >= elev_bins[firnline_idx], step] = (
                            glac_bin_refreeze[elev_bins >= elev_bins[firnline_idx], step] +
                            glac_bin_melt[elev_bins >= elev_bins[firnline_idx], step])
                    # refreeze cannot exceed refreeze potential
                    glac_bin_refreeze[glac_bin_refreeze[:,step] > refreeze_potential, step] = (
                            refreeze_potential[glac_bin_refreeze[:,step] > refreeze_potential])
                    # update refreeze potential
                    refreeze_potential = refreeze_potential - glac_bin_refreeze[:,step]
                    refreeze_potential[abs(refreeze_potential) < input.tolerance] = 0
                    # TOTAL MELT (snow + refreeze + glacier)
                    glac_bin_melt[:,step] = (glac_bin_meltglac[:,step] + glac_bin_meltrefreeze[:,step] + 
                                             glac_bin_meltsnow[:,step])
                    # CLIMATIC MASS BALANCE [m w.e.]
                    #  climatic mass balance = accumulation + refreeze - melt
                    glac_bin_massbalclim[:,step] = (glac_bin_acc[:,step] + glac_bin_refreeze[:,step] - 
                                                    glac_bin_melt[:,step])
                    
#                    if debug:
#                        print('\nyear:', year, step)
#                        print('accumulation:', glac_bin_acc[:,step].sum())
#                        print('melt:', glac_bin_melt[:,step].sum())
                
                # ===== RETURN TO ANNUAL LOOP =====
                # Mass loss cannot exceed glacier volume
                #  mb [mwea] = -1 * sum{area [km2] * ice thickness [m]} / total area [km2] * density_ice / density_water
                mb_max_loss = (-1 * (glacier_area_t0 * icethickness_t0 * input.density_ice / input.density_water).sum() 
                               / glacier_area_t0.sum())
                # Check annual climatic mass balance
                mb_mwea = ((glacier_area_t0 * glac_bin_massbalclim[:,12*year:12*(year+1)].sum(1)).sum() / 
                            glacier_area_t0.sum()) 
                
#                if debug:
#                    print('mb_max_loss:', mb_max_loss, 'mb_check:', mb_mwea)
                    
                # If mass loss exceeds glacier mass, reduce all components
                if mb_mwea < mb_max_loss:                    
                    glac_bin_acc[:,12*year:12*(year+1)] = glac_bin_acc[:,12*year:12*(year+1)] * mb_max_loss / mb_mwea
                    glac_bin_refreeze[:,12*year:12*(year+1)] = (
                            glac_bin_refreeze[:,12*year:12*(year+1)] * mb_max_loss / mb_mwea)
                    glac_bin_melt[:,12*year:12*(year+1)] = glac_bin_melt[:,12*year:12*(year+1)] * mb_max_loss / mb_mwea
                    
                    glac_bin_massbalclim[:,12*year:12*(year+1)] = (
                            glac_bin_acc[:,12*year:12*(year+1)] + glac_bin_refreeze[:,12*year:12*(year+1)] - 
                            glac_bin_melt[:,12*year:12*(year+1)])
                    
#                    if debug:
#                        mb_mwea_adj = ((glacier_area_t0 * glac_bin_massbalclim[:,12*year:12*(year+1)].sum(1)).sum() / 
#                                       glacier_area_t0.sum()) 
#                        print('mb adjusted:', mb_mwea_adj)
                    
                
                # FRONTAL ABLATION
                if debug:
                    print('\nyear:', year)
                    print('sea level:', sea_level, 
                          'bed elev:', round(elev_bins[glac_idx_t0[0]] + (elev_bins[1] - elev_bins[0]) / 2 - 
                                             icethickness_initial[glac_idx_t0[0]], 2))
                # Glacier bed altitude [masl]
                glacier_bedelev = (elev_bins[glac_idx_t0[0]] + (elev_bins[1] - elev_bins[0]) / 2 - 
                                   icethickness_initial[glac_idx_t0[0]])
                # If glacier bed below sea level, compute frontal ablation
                if glacier_bedelev < sea_level:
                    # Volume [m3] and bed elevation [masl] of each bin
                    glac_bin_volume = glacier_area_t0 * 10**6 * icethickness_t0
                    glac_bin_bedelev = np.zeros((glacier_area_t0.shape))
                    glac_bin_bedelev[glac_idx_t0] = (
                            elev_bins[glac_idx_t0] + (elev_bins[1] - elev_bins[0]) / 2 - 
                            icethickness_initial[glac_idx_t0])
                    
                    # Option 1: Use Huss and Hock (2015) frontal ablation parameterizations
                    if input.option_frontalablation_k == 1:
                        # Calculate frontal ablation parameter based on slope of lowest 100 m of glacier
                        # Glacier indices used for slope calculation
                        elev_bin_interval = elev_bins[1] - elev_bins[0]
                        if glac_idx_t0.shape[0] > int(100 / elev_bin_interval):
                            glac_idx_slope = glac_idx_t0[0 : 0 + int(100 / elev_bin_interval)]
                        # if glacier too small, then calculate slope over the entire glacier
                        else:
                            glac_idx_slope = glac_idx_t0.copy()
                        elev_change = (elev_bins[glac_idx_slope[-1]] - elev_bins[glac_idx_slope[0]] + 
                                       elev_bin_interval)
                        #  add elevation bin interval to be inclusive, i.e., elevation bins 5 - 95 include all
                        #  glacier area between 0 - 100 masl
                        # Length of lowest 100 m of glacier
                        length_lowest100m = (glacier_area_t0[glac_idx_slope] / width_t0[glac_idx_slope] * 1000).sum()
                        # Slope of lowest 100 m of glacier
                        slope_lowest100m = np.rad2deg(np.arctan(elev_change/length_lowest100m))
                        # Frontal ablation parameter
                        frontalablation_k = frontalablation_k0 * slope_lowest100m
                    elif input.option_frontalablation_k == 2:
                        print('add more options for frontal ablation - example calibrated for each glacier')
                    
                    # Calculate frontal ablation
                    # Bed elevation with respect to sea level
                    #  negative when bed is below sea level (Oerlemans and Nick, 2005)
                    waterdepth = sea_level - glacier_bedelev
                    # Glacier length [m]
                    length = (glacier_area_t0[width_t0 > 0] / width_t0[width_t0 > 0]).sum() * 1000
                    # Height of calving front [m]
                    height_calving_1 = input.af*length**0.5
                    height_calving_2 = input.density_water / input.density_ice * waterdepth
                    height_calving = np.max([height_calving_1, height_calving_2])
                    # Volume loss [m3] due to frontal ablation
                    frontalablation_volumeloss = (
                            np.max([0, (frontalablation_k * waterdepth * height_calving)]) * 
                            width_t0[glac_idx_t0[0]] * 1000)
                    # Maximum volume loss is volume of bins with their bed elevation below sea level
                    glac_idx_fa = np.where((glac_bin_bedelev < sea_level) & (glacier_area_t0 > 0))[0]
                    frontalablation_volumeloss_max = glac_bin_volume[glac_idx_fa].sum()
                    if frontalablation_volumeloss > frontalablation_volumeloss_max:
                        frontalablation_volumeloss = frontalablation_volumeloss_max
                    
                    if debug:
                        print('frontalablation_volumeloss [m3]:', frontalablation_volumeloss)
                        print('frontalablation_massloss [Gt]:', frontalablation_volumeloss * input.density_water / 
                              input.density_ice / 10**9)
                        print('glac_idx_fa:', glac_idx_fa)
                        print('glac_bin_volume:', glac_bin_volume)
                        print('glac_idx_fa[bin_count]:', glac_idx_fa[0])
                        print('glac_bin_volume[glac_idx_fa[bin_count]]:', glac_bin_volume[glac_idx_fa[0]])
                        print('glacier_area_t0[glac_idx_fa[bin_count]]:', glacier_area_t0[glac_idx_fa[0]])
                        print('glac_bin_frontalablation:', glac_bin_frontalablation[glac_idx_fa[0], step])
                    
                    # Frontal ablation [mwe] in each bin
                    bin_count = 0
                    while (frontalablation_volumeloss > input.tolerance) and (bin_count < len(glac_idx_fa)):
                        if frontalablation_volumeloss >= glac_bin_volume[glac_idx_fa[bin_count]]:
                            glac_bin_frontalablation[glac_idx_fa[bin_count], step] = (
                                    glac_bin_volume[glac_idx_fa[bin_count]] / 
                                    (glacier_area_t0[glac_idx_fa[bin_count]] * 10**6) 
                                    * input.density_ice / input.density_water)              
                        else:
                            glac_bin_frontalablation[glac_idx_fa[bin_count], step] = (
                                    frontalablation_volumeloss / (glacier_area_t0[glac_idx_fa[bin_count]] * 10**6)
                                    * input.density_ice / input.density_water)
                        
                        
                        frontalablation_volumeloss += (
                                -1 * glac_bin_frontalablation[glac_idx_fa[bin_count], step] * input.density_water / 
                                input.density_ice * glacier_area_t0[glac_idx_fa[bin_count]] * 10**6)
                        
#                        frontalablation_volumeloss = round(frontalablation_volumeloss - (
#                                glac_bin_frontalablation[glac_idx_fa[bin_count], step] * 
#                                input.density_water / input.density_ice *
#                                glacier_area_t0[glac_idx_fa[bin_count]] * 10**6),6)
                        
                        bin_count += 1         
                        
                        
                        if debug:
                            print('glacier idx:', glac_idx_fa[bin_count-1], 
                                  'volume loss:', (glac_bin_frontalablation[glac_idx_fa[bin_count-1], step] * 
                                  glacier_area_t0[glac_idx_fa[bin_count-1]] * input.density_water / input.density_ice * 
                                  10**6).round(0))
                            print('remaining volume loss:', frontalablation_volumeloss, 'tolerance:', input.tolerance)
                            
                    if debug:
                        print('frontalablation_volumeloss remaining [m3]:', frontalablation_volumeloss)
                        print('ice thickness:', icethickness_t0[glac_idx_fa[0]].round(0), 
                              'waterdepth:', waterdepth.round(0), 
                              'height calving front:', height_calving.round(0), 
                              'width [m]:', (width_t0[glac_idx_fa[0]] * 1000).round(0))                    
                        
                # SURFACE TYPE
                # Annual surface type [-]
                # Annual climatic mass balance [m w.e.], which is used to determine the surface type
                glac_bin_massbalclim_annual[:,year] = glac_bin_massbalclim[:,12*year:12*(year+1)].sum(1)
                
#                if debug:
#                    print('glacier indices:', glac_idx_t0)
#                    print('glac_bin_massbalclim:', glac_bin_massbalclim[:,12*year:12*(year+1)].sum(1))
#                    print('Climatic mass balance:', glac_bin_massbalclim_annual[:,year].sum())
                
                # Compute the surface type for each bin
                surfacetype, firnline_idx = surfacetypebinsannual(surfacetype, glac_bin_massbalclim_annual, year)
                
                # MASS REDISTRIBUTION
                # Mass redistribution ignored for calibration and spinup years (glacier properties constant) 
                if (option_areaconstant == 1) or (year < input.spinupyears):
                    glacier_area_t1 = glacier_area_t0
                    icethickness_t1 = icethickness_t0
                    width_t1 = width_t0
                else:
                    # First, remove volume lost to frontal ablation
                    #  changes to _t0 not _t1, since t1 will be done in the mass redistribution
                    if glac_bin_frontalablation[:,step].max() > 0:
                        # Frontal ablation loss [mwe]
                        #  fa_change tracks whether entire bin is lost or not
                        fa_change = abs(glac_bin_frontalablation[:, step] * input.density_water / input.density_ice
                                        - icethickness_t0)
                        fa_change[fa_change <= input.tolerance] = 0
                        
                        if debug:
                            bins_wfa = np.where(glac_bin_frontalablation[:,step] > 0)[0]
                            print('glacier area t0:', glacier_area_t0[bins_wfa].round(3))
                            print('ice thickness t0:', icethickness_t0[bins_wfa].round(1))
                            print('frontalablation [m ice]:', (glac_bin_frontalablation[bins_wfa, step] * 
                                  input.density_water / input.density_ice).round(1))
                            print('frontal ablation [mice] vs icethickness:', fa_change[bins_wfa].round(1))
                        
                        # Check if entire bin is removed
                        glacier_area_t0[np.where(fa_change == 0)[0]] = 0
                        icethickness_t0[np.where(fa_change == 0)[0]] = 0
                        width_t0[np.where(fa_change == 0)[0]] = 0
                        # Otherwise, reduce glacier area such that glacier retreats and ice thickness remains the same
                        #  A_1 = (V_0 - V_loss) / h_1,  units: A_1 = (m ice * km2) / (m ice) = km2
                        glacier_area_t0[np.where(fa_change != 0)[0]] = (
                                (glacier_area_t0[np.where(fa_change != 0)[0]] * 
                                 icethickness_t0[np.where(fa_change != 0)[0]] - 
                                 glacier_area_t0[np.where(fa_change != 0)[0]] * 
                                 glac_bin_frontalablation[np.where(fa_change != 0)[0], step] * input.density_water 
                                 / input.density_ice) / icethickness_t0[np.where(fa_change != 0)[0]])
    
                        if debug:
                            print('glacier area t1:', glacier_area_t0[bins_wfa].round(3))
                            print('ice thickness t1:', icethickness_t0[bins_wfa].round(1))
                    
                    # Redistribute mass if glacier was not fully removed by frontal ablation
                    if glacier_area_t0.max() > 0:
                        # Mass redistribution according to Huss empirical curves
                        glacier_area_t1, icethickness_t1, width_t1 = (
                                massredistributionHuss(glacier_area_t0, icethickness_t0, width_t0, 
                                                       glac_bin_massbalclim_annual, year, glac_idx_initial, 
                                                       glac_area_initial))
                        # update surface type for bins that have retreated
                        surfacetype[glacier_area_t1 == 0] = 0
                        # update surface type for bins that have advanced 
                        surfacetype[(surfacetype == 0) & (glacier_area_t1 != 0)] = (
                                surfacetype[glacier_area_t0.nonzero()[0][0]])
                    else:
                        glacier_area_t1 = np.zeros(glacier_area_t0.shape)
                        icethickness_t1 = np.zeros(glacier_area_t0.shape)
                        width_t1 = np.zeros(glacier_area_t0.shape)
                        surfacetype = np.zeros(glacier_area_t0.shape)
                        
                # Record glacier properties (area [km**2], thickness [m], width [km])
                # if first year, record initial glacier properties (area [km**2], ice thickness [m ice], width [km])
                if year == 0:
                    glac_bin_area_annual[:,year] = glacier_area_t0
                    glac_bin_icethickness_annual[:,year] = icethickness_t0
                    glac_bin_width_annual[:,year] = width_t0
                # record the next year's properties as well
                # 'year + 1' used so the glacier properties are consistent with mass balance computations
                glac_bin_icethickness_annual[:,year + 1] = icethickness_t1
                glac_bin_area_annual[:,year + 1] = glacier_area_t1
                glac_bin_width_annual[:,year + 1] = width_t1
                # Update glacier properties for the mass balance computations
                icethickness_t0 = icethickness_t1.copy()
                glacier_area_t0 = glacier_area_t1.copy()
                width_t0 = width_t1.copy()   
                #%%
    # Remove the spinup years of the variables that are being exported
    if input.timestep == 'monthly':
        colstart = input.spinupyears * annual_divisor
        colend = glacier_gcm_temp.shape[0] + 1
    glac_bin_temp = glac_bin_temp[:,colstart:colend]
    glac_bin_prec = glac_bin_prec[:,colstart:colend]
    glac_bin_acc = glac_bin_acc[:,colstart:colend]
    glac_bin_refreeze = glac_bin_refreeze[:,colstart:colend]
    glac_bin_snowpack = glac_bin_snowpack[:,colstart:colend]
    glac_bin_melt = glac_bin_melt[:,colstart:colend]
    glac_bin_frontalablation = glac_bin_frontalablation[:,colstart:colend]
    glac_bin_massbalclim = glac_bin_massbalclim[:,colstart:colend]
    glac_bin_massbalclim_annual = glac_bin_massbalclim_annual[:,input.spinupyears:annual_columns.shape[0]+1]
    glac_bin_area_annual = glac_bin_area_annual[:,input.spinupyears:annual_columns.shape[0]+1]
    glac_bin_icethickness_annual = glac_bin_icethickness_annual[:,input.spinupyears:annual_columns.shape[0]+1]
    glac_bin_width_annual = glac_bin_width_annual[:,input.spinupyears:annual_columns.shape[0]+1]
    glac_bin_surfacetype_annual = glac_bin_surfacetype_annual[:,input.spinupyears:annual_columns.shape[0]+1]
    # Additional output:
    glac_wide_prec = np.zeros(glac_bin_temp.shape[1])
    glac_wide_acc = np.zeros(glac_bin_temp.shape[1])
    glac_wide_refreeze = np.zeros(glac_bin_temp.shape[1])
    glac_wide_melt = np.zeros(glac_bin_temp.shape[1])
    glac_wide_frontalablation = np.zeros(glac_bin_temp.shape[1])
    # Compute desired output
    glac_bin_area = glac_bin_area_annual[:,0:glac_bin_area_annual.shape[1]-1].repeat(12,axis=1)
    glac_wide_area = glac_bin_area.sum(axis=0)
    glac_wide_prec_mkm2 = (glac_bin_prec * glac_bin_area).sum(axis=0)
    glac_wide_prec[glac_wide_prec_mkm2 > 0] = (glac_wide_prec_mkm2[glac_wide_prec_mkm2 > 0] / 
                                               glac_wide_area[glac_wide_prec_mkm2 > 0])
    glac_wide_acc_mkm2 = (glac_bin_acc * glac_bin_area).sum(axis=0)
    glac_wide_acc[glac_wide_acc_mkm2 > 0] = (glac_wide_acc_mkm2[glac_wide_acc_mkm2 > 0] / 
                                             glac_wide_area[glac_wide_acc_mkm2 > 0])
    glac_wide_snowpack = glac_wide_acc_mkm2 / 1000
    glac_wide_refreeze_mkm2 = (glac_bin_refreeze * glac_bin_area).sum(axis=0)
    glac_wide_refreeze[glac_wide_refreeze_mkm2 > 0] = (glac_wide_refreeze_mkm2[glac_wide_refreeze_mkm2 > 0] / 
                                                       glac_wide_area[glac_wide_refreeze_mkm2 > 0])
    glac_wide_melt_mkm2 = (glac_bin_melt * glac_bin_area).sum(axis=0)
    glac_wide_melt[glac_wide_melt_mkm2 > 0] = (glac_wide_melt_mkm2[glac_wide_melt_mkm2 > 0] / 
                                               glac_wide_area[glac_wide_melt_mkm2 > 0])
    glac_wide_frontalablation_mkm2 = (glac_bin_frontalablation * glac_bin_area).sum(axis=0)
    glac_wide_frontalablation[glac_wide_frontalablation_mkm2 > 0] = (
            glac_wide_frontalablation_mkm2[glac_wide_frontalablation_mkm2 > 0] / 
            glac_wide_area[glac_wide_frontalablation_mkm2 > 0])
    glac_wide_massbalclim = glac_wide_acc + glac_wide_refreeze - glac_wide_melt
    glac_wide_massbaltotal = glac_wide_massbalclim - glac_wide_frontalablation
    glac_wide_runoff = (glac_wide_prec + glac_wide_melt - glac_wide_refreeze) * glac_wide_area * (1000)**2
    #  units: (m + m w.e. - m w.e.) * km**2 * (1000 m / 1 km)**2 = m**3
    glac_wide_snowline = (glac_bin_snowpack > 0).argmax(axis=0)
    glac_wide_snowline[glac_wide_snowline > 0] = (elev_bins[glac_wide_snowline[glac_wide_snowline > 0]] - 
                                                  input.binsize/2)
    glac_wide_area_annual = glac_bin_area_annual.sum(axis=0)
    glac_wide_volume_annual = (glac_bin_area_annual * glac_bin_icethickness_annual / 1000).sum(axis=0)
    glac_wide_ELA_annual = (glac_bin_massbalclim_annual > 0).argmax(axis=0)
    glac_wide_ELA_annual[glac_wide_ELA_annual > 0] = (elev_bins[glac_wide_ELA_annual[glac_wide_ELA_annual > 0]] - 
                                                      input.binsize/2)  
    # Return the desired output
    return (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt, 
            glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual, 
            glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual, 
            glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack, glac_wide_area_annual, 
            glac_wide_volume_annual, glac_wide_ELA_annual)


#%% ===================================================================================================================
def annualweightedmean_array(var, dates_table):
    """
    Calculate annual mean of variable according to the timestep.
    
    Monthly timestep will group every 12 months, so starting month is important.
    
    Parameters
    ----------
    var : np.ndarray
        Variable with monthly or daily timestep
    dates_table : pd.DataFrame
        Table of dates, year, month, daysinmonth, wateryear, and season for each timestep

    Returns
    -------
    var_annual : np.ndarray
        Annual weighted mean of variable
    """        
    if input.timestep == 'monthly':
        dayspermonth = dates_table['daysinmonth'].values.reshape(-1,12)
        #  creates matrix (rows-years, columns-months) of the number of days per month
        daysperyear = dayspermonth.sum(axis=1)
        #  creates an array of the days per year (includes leap years)
        weights = (dayspermonth / daysperyear[:,np.newaxis]).reshape(-1)
        #  computes weights for each element, then reshapes it from matrix (rows-years, columns-months) to an array, 
        #  where each column (each monthly timestep) is the weight given to that specific month
        var_annual = (var*weights[np.newaxis,:]).reshape(-1,12).sum(axis=1).reshape(-1,daysperyear.shape[0])
        #  computes matrix (rows - bins, columns - year) of weighted average for each year
        #  explanation: var*weights[np.newaxis,:] multiplies each element by its corresponding weight; .reshape(-1,12) 
        #    reshapes the matrix to only have 12 columns (1 year), so the size is (rows*cols/12, 12); .sum(axis=1) 
        #    takes the sum of each year; .reshape(-1,daysperyear.shape[0]) reshapes the matrix back to the proper 
        #    structure (rows - bins, columns - year)
        # If averaging a single year, then reshape so it returns a 1d array
        if var_annual.shape[1] == 1:
            var_annual = var_annual.reshape(var_annual.shape[0])
    elif input.timestep == 'daily':
        print('\nError: need to code the groupbyyearsum and groupbyyearmean for daily timestep.'
              'Exiting the model run.\n')
        exit()
    return var_annual
   

def massredistributionHuss(glacier_area_t0, icethickness_t0, width_t0, glac_bin_massbalclim_annual, year, 
                           glac_idx_initial, glac_area_initial, debug=False):
    """
    Mass redistribution according to empirical equations from Huss and Hock (2015) accounting for retreat/advance.

    glac_idx_initial is required to ensure that the glacier does not advance to area where glacier did not exist before
    (e.g., retreat and advance over a vertical cliff)
    
    Parameters
    ----------
    glacier_area_t0 : np.ndarray
        Glacier area [km2] from previous year for each elevation bin
    icethickness_t0 : np.ndarray
        Ice thickness [m] from previous year for each elevation bin
    width_t0 : np.ndarray
        Glacier width [km] from previous year for each elevation bin
    glac_bin_massbalclim_annual : np.ndarray
        Climatic mass balance [m w.e.] for each elevation bin and year
    year : int
        Count of the year of model run (first year is 0)
    glac_idx_initial : np.ndarray
        Initial glacier indices
    glac_area_initial : np.ndarray
        Initial glacier array used to determine average terminus area in event that glacier is only one bin
    debug : Boolean
        option to turn on print statements for development or debugging of code (default False)

    Returns
    -------
    glacier_area_t1 : np.ndarray
        Updated glacier area [km2] for each elevation bin
    icethickness_t1 : np.ndarray
        Updated ice thickness [m] for each elevation bin
    width_t1 : np.ndarray
        Updated glacier width [km] for each elevation bin
    """        
    # Reset the annual glacier area and ice thickness
    glacier_area_t1 = np.zeros(glacier_area_t0.shape)
    icethickness_t1 = np.zeros(glacier_area_t0.shape)
    width_t1 = np.zeros(glacier_area_t0.shape)
    # Annual glacier-wide volume change [km**3]
    glacier_volumechange = ((glac_bin_massbalclim_annual[:, year] / 1000 * input.density_water / 
                             input.density_ice * glacier_area_t0).sum())
    #  units: [m w.e.] * (1 km / 1000 m) * (1000 kg / (1 m water * m**2) * (1 m ice * m**2 / 900 kg) * [km**2] 
    #         = km**3 ice          
    # If volume loss is less than the glacier volume, then redistribute mass loss/gains across the glacier;
    #  otherwise, the glacier disappears (area and thickness were already set to zero above)
    if -1 * glacier_volumechange < (icethickness_t0 / 1000 * glacier_area_t0).sum():
        # Determine where glacier exists
        
        # Check for negative glacier areas
        #  shouldn't need these three lines, but Anna mentioned she was finding negative areas somehow? 2019/01/30
        glacier_area_t0[glacier_area_t0 < 0] = 0
        icethickness_t0[glacier_area_t0 < 0] = 0
        width_t0[glacier_area_t0 < 0] = 0
        
        glac_idx_t0 = glacier_area_t0.nonzero()[0]
        # Compute ice thickness [m ice], glacier area [km**2] and ice thickness change [m ice] after 
        #  redistribution of gains/losses
        if input.option_massredistribution == 1:
            # Option 1: apply mass redistribution using Huss' empirical geometry change equations
            icethickness_t1, glacier_area_t1, width_t1, icethickness_change, glacier_volumechange_remaining = (
                    massredistributioncurveHuss(icethickness_t0, glacier_area_t0, width_t0, glac_idx_t0,
                                                glacier_volumechange, glac_bin_massbalclim_annual[:, year]))
        # Glacier retreat
        #  if glacier retreats (ice thickness < 0), then ice thickness is set to zero, and some volume change will need 
        #   to be redistributed across the rest of the glacier
        while glacier_volumechange_remaining < 0:
            glacier_area_t0_retreated = glacier_area_t1.copy()
            icethickness_t0_retreated = icethickness_t1.copy()
            width_t0_retreated = width_t1.copy()
            glacier_volumechange_remaining_retreated = glacier_volumechange_remaining.copy()
            glac_idx_t0_retreated = glacier_area_t0_retreated.nonzero()[0]            
            # Set climatic mass balance for the case when there are less than 3 bins  
            #  distribute the remaining glacier volume change over the entire glacier (remaining bins)
            massbal_clim_retreat = np.zeros(glacier_area_t0_retreated.shape)
            massbal_clim_retreat[glac_idx_t0_retreated] = (glacier_volumechange_remaining / 
                                                           glacier_area_t0_retreated.sum() * 1000)
            # Mass redistribution 
            if input.option_massredistribution == 1:
                # Option 1: apply mass redistribution using Huss' empirical geometry change equations
                icethickness_t1, glacier_area_t1, width_t1, icethickness_change, glacier_volumechange_remaining = (
                        massredistributioncurveHuss(icethickness_t0_retreated, glacier_area_t0_retreated, 
                                                    width_t0_retreated, glac_idx_t0_retreated, 
                                                    glacier_volumechange_remaining_retreated, massbal_clim_retreat))                   
        # Glacier advances
        #  if glacier advances (ice thickness change exceeds threshold), then redistribute mass gain in new bins
        while (icethickness_change > input.icethickness_advancethreshold).any() == True:  
            # Record glacier area and ice thickness before advance corrections applied
            glacier_area_t1_raw = glacier_area_t1.copy()
            icethickness_t1_raw = icethickness_t1.copy()
            width_t1_raw = width_t1.copy()
            # Index bins that are advancing
            icethickness_change[icethickness_change <= input.icethickness_advancethreshold] = 0
            glac_idx_advance = icethickness_change.nonzero()[0]
            # Update ice thickness based on maximum advance threshold [m ice]
            icethickness_t1[glac_idx_advance] = (icethickness_t1[glac_idx_advance] - 
                           (icethickness_change[glac_idx_advance] - input.icethickness_advancethreshold))
            # Update glacier area based on reduced ice thicknesses [km**2]
            if input.option_glaciershape == 1:
                # Glacier area for parabola [km**2] (A_1 = A_0 * (H_1 / H_0)**0.5)
                glacier_area_t1[glac_idx_advance] = (glacier_area_t1_raw[glac_idx_advance] * 
                               (icethickness_t1[glac_idx_advance] / icethickness_t1_raw[glac_idx_advance])**0.5)
                # Glacier width for parabola [km] (w_1 = w_0 * A_1 / A_0)
                width_t1[glac_idx_advance] = (width_t1_raw[glac_idx_advance] * glacier_area_t1[glac_idx_advance] 
                                              / glacier_area_t1_raw[glac_idx_advance])
            elif input.option_glaciershape == 2:
                # Glacier area constant for rectangle [km**2] (A_1 = A_0)
                glacier_area_t1[glac_idx_advance] = glacier_area_t1_raw[glac_idx_advance]
                # Glacier with constant for rectangle [km] (w_1 = w_0)
                width_t1[glac_idx_advance] = width_t1_raw[glac_idx_advance]
            elif input.option_glaciershape == 3:
                # Glacier area for triangle [km**2] (A_1 = A_0 * H_1 / H_0)
                glacier_area_t1[glac_idx_t0] = (glacier_area_t1_raw[glac_idx_t0] * 
                               icethickness_t1[glac_idx_t0] / icethickness_t1_raw[glac_idx_t0])
                # Glacier width for triangle [km] (w_1 = w_0 * A_1 / A_0)
                width_t1[glac_idx_advance] = (width_t1_raw[glac_idx_advance] * glacier_area_t1[glac_idx_advance] 
                                              / glacier_area_t1_raw[glac_idx_advance])
            # Advance volume [km**3]
            advance_volume = ((glacier_area_t1_raw[glac_idx_advance] * 
                              icethickness_t1_raw[glac_idx_advance] / 1000).sum() - 
                              (glacier_area_t1[glac_idx_advance] * icethickness_t1[glac_idx_advance] / 
                               1000).sum())
            # Advance characteristics
            # Indices that define the glacier terminus
            glac_idx_terminus = (glac_idx_t0[(glac_idx_t0 - glac_idx_t0[0] + 1) / 
                                             glac_idx_t0.shape[0] * 100 < input.terminus_percentage])
            if debug:
                print('glacier index terminus:',glac_idx_terminus)
                print('glacier index:',glac_idx_t0)
                print('glacier indx initial:', glac_idx_initial)
            # For glaciers with so few bands that the terminus is not identified (ex. <= 4 bands for 20% threshold),
            #  then use the information from all the bands
            if glac_idx_terminus.shape[0] <= 1:
                glac_idx_terminus = glac_idx_t0.copy()
            # Average area of glacier terminus [km**2]
            #  exclude the bin at the terminus, since this bin may need to be filled first
            try:
                terminus_area_avg = (
                        glacier_area_t0[glac_idx_terminus[1]:glac_idx_terminus[glac_idx_terminus.shape[0]-1]+1].mean())
            except:  
                glac_idx_terminus_initial = (
                        glac_idx_initial[(glac_idx_initial - glac_idx_initial[0] + 1) / glac_idx_initial.shape[0] * 100 
                                          < input.terminus_percentage])
                if glac_idx_terminus_initial.shape[0] <= 1:
                    glac_idx_terminus_initial = glac_idx_initial.copy()
                terminus_area_avg = (
                        glac_area_initial[glac_idx_terminus_initial[1]:
                                          glac_idx_terminus_initial[glac_idx_terminus_initial.shape[0]-1]+1].mean())
            # Check if the last bin's area is below the terminus' average and fill it up if it is
            if (glacier_area_t1[glac_idx_terminus[0]] < terminus_area_avg) and (icethickness_t0[glac_idx_terminus[0]] <
               icethickness_t0[glac_idx_t0].mean()):
#            if glacier_area_t1[glac_idx_terminus[0]] < terminus_area_avg:
                # Volume required to fill the bin at the terminus
                advance_volume_fillbin = (icethickness_t1[glac_idx_terminus[0]] / 1000 * (terminus_area_avg - 
                                          glacier_area_t1[glac_idx_terminus[0]]))
                # If the advance volume is less than that required to fill the bin, then fill the bin as much as
                #  possible by adding area (thickness remains the same - glacier front is only thing advancing)
                if advance_volume < advance_volume_fillbin:
                    # add advance volume to the bin (area increases, thickness and width constant)
                    glacier_area_t1[glac_idx_terminus[0]] = (glacier_area_t1[glac_idx_terminus[0]] + 
                                   advance_volume / (icethickness_t1[glac_idx_terminus[0]] / 1000))
                    # set advance volume equal to zero
                    advance_volume = 0
                else:
                    # fill the bin (area increases, thickness and width constant)
                    glacier_area_t1[glac_idx_terminus[0]] = (glacier_area_t1[glac_idx_terminus[0]] + 
                                   advance_volume_fillbin / (icethickness_t1[glac_idx_terminus[0]] / 1000))
                    advance_volume = advance_volume - advance_volume_fillbin
            # With remaining advance volume, add a bin
            if advance_volume > 0:
                # Index for additional bin below the terminus
                glac_idx_bin2add = np.array([glac_idx_terminus[0] - 1])
                # Check if bin2add is in a discontinuous section of the initial glacier
                while ((glac_idx_bin2add > glac_idx_initial.min()) & 
                       ((glac_idx_bin2add == glac_idx_initial).any() == False)):
                    # Advance should not occur in a discontinuous section of the glacier (e.g., vertical drop),
                    #  so change the bin2add to the next bin down valley
                    glac_idx_bin2add = glac_idx_bin2add - 1
                # if the added bin would be below sea-level, then volume is distributed over the glacier without
                #  any adjustments
                if glac_idx_bin2add < 0:
                    glacier_area_t1 = glacier_area_t1_raw
                    icethickness_t1 = icethickness_t1_raw
                    width_t1 = width_t1_raw
                    advance_volume = 0
                # otherwise, add a bin with thickness and width equal to the previous bin and fill it up
                else:
                    # ice thickness of new bin equals ice thickness of bin at the terminus
                    icethickness_t1[glac_idx_bin2add] = icethickness_t1[glac_idx_terminus[0]]
                    width_t1[glac_idx_bin2add] = width_t1[glac_idx_terminus[0]]
                    # volume required to fill the bin at the terminus
                    advance_volume_fillbin = icethickness_t1[glac_idx_bin2add] / 1000 * terminus_area_avg 
                    # If the advance volume is unable to fill entire bin, then fill it as much as possible
                    if advance_volume < advance_volume_fillbin:
                        # add advance volume to the bin (area increases, thickness and width constant)
                        glacier_area_t1[glac_idx_bin2add] = (advance_volume / (icethickness_t1[glac_idx_bin2add]
                                                             / 1000))
                        advance_volume = 0
                    else:
                        # fill the bin (area increases, thickness and width constant)
                        glacier_area_t1[glac_idx_bin2add] = terminus_area_avg
                        advance_volume = advance_volume - advance_volume_fillbin
            # update the glacier indices
            glac_idx_t0 = glacier_area_t1.nonzero()[0]
            massbal_clim_advance = np.zeros(glacier_area_t1.shape)
            # Record glacier area and ice thickness before advance corrections applied
            glacier_area_t1_raw = glacier_area_t1.copy()
            icethickness_t1_raw = icethickness_t1.copy()
            width_t1_raw = width_t1.copy()
            # If a full bin has been added and volume still remains, then redistribute mass across the
            #  glacier, thereby enabling the bins to get thicker once again prior to adding a new bin.
            #  This is important for glaciers that have very thin ice at the terminus as this prevents the 
            #  glacier from having a thin layer of ice advance tremendously far down valley without thickening.
            if advance_volume > 0:
                if input.option_massredistribution == 1:
                    # Option 1: apply mass redistribution using Huss' empirical geometry change equations
                    icethickness_t1, glacier_area_t1, width_t1, icethickness_change, glacier_volumechange_remaining = (
                            massredistributioncurveHuss(icethickness_t1, glacier_area_t1, width_t1, glac_idx_t0, 
                                                        advance_volume, massbal_clim_advance))
            # update ice thickness change
            icethickness_change = icethickness_t1 - icethickness_t1_raw
    return glacier_area_t1, icethickness_t1, width_t1


def massredistributioncurveHuss(icethickness_t0, glacier_area_t0, width_t0, glac_idx_t0, glacier_volumechange, 
                                massbalclim_annual):
    """
    Apply the mass redistribution curves from Huss and Hock (2015).

    This is paired with massredistributionHuss, which takes into consideration retreat and advance.
    
    To-do list
    ----------
    - volume-length scaling
    - volume-area scaling
    - pair with OGGM flow model
    
    Parameters
    ----------
    icethickness_t0 : np.ndarray
        Ice thickness [m] from previous year for each elevation bin
    glacier_area_t0 : np.ndarray
        Glacier area [km2] from previous year for each elevation bin
    width_t0 : np.ndarray
        Glacier width [km] from previous year for each elevation bin
    massbalclim_annual : np.ndarray
        Annual climatic mass balance [m w.e.] for each elevation bin for a single year
    glac_idx_t0 : np.ndarray
        glacier indices for present timestep
    glacier_volumechange : float
        glacier-wide volume change [km3] based on the annual climatic mass balance

    Returns
    -------
    glacier_area_t1 : np.ndarray
        Updated glacier area [km2] for each elevation bin
    icethickness_t1 : np.ndarray
        Updated ice thickness [m] for each elevation bin
    width_t1 : np.ndarray
        Updated glacier width [km] for each elevation bin
    icethickness_change : np.ndarray
        Ice thickness change [m] for each elevation bin
    glacier_volumechange_remaining : float
        Glacier volume change remaining, which could occur if there is less ice in a bin than melt, i.e., retreat
    """           
    # Apply Huss redistribution if there are at least 3 elevation bands; otherwise, use the mass balance
    # reset variables
    icethickness_t1 = np.zeros(glacier_area_t0.shape)
    glacier_area_t1 = np.zeros(glacier_area_t0.shape)
    width_t1 = np.zeros(glacier_area_t0.shape) 
    if glac_idx_t0.shape[0] > 3:
        # Select the factors for the normalized ice thickness change curve based on glacier area
        if glacier_area_t0.sum() > 20:
            [gamma, a, b, c] = [6, -0.02, 0.12, 0]
        elif glacier_area_t0.sum() > 5:
            [gamma, a, b, c] = [4, -0.05, 0.19, 0.01]
        else:
            [gamma, a, b, c] = [2, -0.30, 0.60, 0.09]
        # reset variables
        elevrange_norm = np.zeros(glacier_area_t0.shape)
        icethicknesschange_norm = np.zeros(glacier_area_t0.shape)
        # Normalized elevation range [-]
        #  (max elevation - bin elevation) / (max_elevation - min_elevation)
        elevrange_norm[glacier_area_t0 > 0] = (glac_idx_t0[-1] - glac_idx_t0) / (glac_idx_t0[-1] - glac_idx_t0[0])
        #  using indices as opposed to elevations automatically skips bins on the glacier that have no area
        #  such that the normalization is done only on bins where the glacier lies
        # Normalized ice thickness change [-]
        icethicknesschange_norm[glacier_area_t0 > 0] = ((elevrange_norm[glacier_area_t0 > 0] + a)**gamma + 
                                                        b*(elevrange_norm[glacier_area_t0 > 0] + a) + c)
        #  delta_h = (h_n + a)**gamma + b*(h_n + a) + c
        #  indexing is faster here
        # limit the icethicknesschange_norm to between 0 - 1 (ends of fxns not exactly 0 and 1)
        icethicknesschange_norm[icethicknesschange_norm > 1] = 1
        icethicknesschange_norm[icethicknesschange_norm < 0] = 0
        # Huss' ice thickness scaling factor, fs_huss [m ice]         
        fs_huss = glacier_volumechange / (glacier_area_t0 * icethicknesschange_norm).sum() * 1000
        #  units: km**3 / (km**2 * [-]) * (1000 m / 1 km) = m ice
        # Volume change [km**3 ice]
        bin_volumechange = icethicknesschange_norm * fs_huss / 1000 * glacier_area_t0
    # Otherwise, compute volume change in each bin based on the climatic mass balance
    else:
        bin_volumechange = massbalclim_annual / 1000 * glacier_area_t0        
    if input.option_glaciershape == 1:
        # Ice thickness at end of timestep for parabola [m ice]
        #  run in two steps to avoid errors with negative numbers and fractional exponents
        #  H_1 = (H_0**1.5 + delta_Vol * H_0**0.5 / A_0)**(2/3)
        icethickness_t1[glac_idx_t0] = ((icethickness_t0[glac_idx_t0] / 1000)**1.5 + 
                       (icethickness_t0[glac_idx_t0] / 1000)**0.5 * bin_volumechange[glac_idx_t0] / 
                       glacier_area_t0[glac_idx_t0])
        icethickness_t1[icethickness_t1 < 0] = 0
        icethickness_t1[glac_idx_t0] = icethickness_t1[glac_idx_t0]**(2/3) * 1000
        # Glacier area for parabola [km**2]
        #  A_1 = A_0 * (H_1 / H_0)**0.5
        glacier_area_t1[glac_idx_t0] = (glacier_area_t0[glac_idx_t0] * (icethickness_t1[glac_idx_t0] / 
                                        icethickness_t0[glac_idx_t0])**0.5)
        # Glacier width for parabola [km]
        #  w_1 = w_0 * (A_1 / A_0)
        width_t1[glac_idx_t0] = width_t0[glac_idx_t0] * glacier_area_t1[glac_idx_t0] / glacier_area_t0[glac_idx_t0]
    elif input.option_glaciershape == 2:
        # Ice thickness at end of timestep for rectangle [m ice]
        #  H_1 = H_0 + delta_Vol / A_0
        icethickness_t1[glac_idx_t0] = (((icethickness_t0[glac_idx_t0] / 1000) + 
                                         bin_volumechange[glac_idx_t0] / glacier_area_t0[glac_idx_t0]) * 1000)
        # Glacier area constant for rectangle [km**2]
        #  A_1 = A_0
        glacier_area_t1[glac_idx_t0] = glacier_area_t0[glac_idx_t0]
        # Glacier width constant for rectangle [km]
        #  w_1 = w_0
        width_t1[glac_idx_t0] = width_t0[glac_idx_t0]
    elif input.option_glaciershape == 3:
        # Ice thickness at end of timestep for triangle [m ice]
        #  run in two steps to avoid errors with negative numbers and fractional exponents
        icethickness_t1[glac_idx_t0] = ((icethickness_t0[glac_idx_t0] / 1000)**2 + 
                       bin_volumechange[glac_idx_t0] * (icethickness_t0[glac_idx_t0] / 1000) / 
                       glacier_area_t0[glac_idx_t0])                                   
        icethickness_t1[icethickness_t1 < 0] = 0
        icethickness_t1[glac_idx_t0] = icethickness_t1[glac_idx_t0]**(1/2) * 1000
        # Glacier area for triangle [km**2]
        #  A_1 = A_0 * H_1 / H_0
        glacier_area_t1[glac_idx_t0] = (glacier_area_t0[glac_idx_t0] * icethickness_t1[glac_idx_t0] / 
                                        icethickness_t0[glac_idx_t0])
        # Glacier width for triangle [km]
        #  w_1 = w_0 * (A_1 / A_0)
        width_t1[glac_idx_t0] = width_t0[glac_idx_t0] * glacier_area_t1[glac_idx_t0] / glacier_area_t0[glac_idx_t0]
    # Ice thickness change [m ice]
    icethickness_change = icethickness_t1 - icethickness_t0
    # Compute the remaining volume change
    bin_volumechange_remaining = bin_volumechange - ((glacier_area_t1 * icethickness_t1 - glacier_area_t0 * 
                                                      icethickness_t0) / 1000)
    # remove values below tolerance to avoid rounding errors
    bin_volumechange_remaining[abs(bin_volumechange_remaining) < input.tolerance] = 0
    # Glacier volume change remaining - if less than zero, then needed for retreat
    glacier_volumechange_remaining = bin_volumechange_remaining.sum()
    # return desired output
    return icethickness_t1, glacier_area_t1, width_t1, icethickness_change, glacier_volumechange_remaining
    

def surfacetypebinsannual(surfacetype, glac_bin_massbalclim_annual, year_index):
    """
    Update surface type according to climatic mass balance over the last five years.  
    
    If 5-year climatic balance is positive, then snow/firn.  If negative, then ice/debris.
    Convention: 0 = off-glacier, 1 = ice, 2 = snow, 3 = firn, 4 = debris
    
    Function Options:
      > 1 (default) - update surface type according to Huss and Hock (2015)
      > 2 - Radic and Hock (2011)
    Huss and Hock (2015): Initially, above median glacier elevation is firn and below is ice. Surface type updated for
      each elevation band and month depending on the specific mass balance.  If the cumulative balance since the start 
      of the mass balance year is positive, then snow is assigned. If the cumulative mass balance is negative (i.e., 
      all snow of current mass balance year has melted), then bare ice or firn is exposed. Surface type is assumed to 
      be firn if the elevation band's average annual balance over the preceding 5 years (B_t-5_avg) is positive. If
      B_t-5_avg is negative, surface type is ice.
          > climatic mass balance calculated at each bin and used with the mass balance over the last 5 years to 
            determine whether the surface is firn or ice.  Snow is separate based on each month.
    Radic and Hock (2011): "DDF_snow is used above the ELA regardless of snow cover.  Below the ELA, use DDF_ice is 
      used only when snow cover is 0.  ELA is calculated from the observed annual mass balance profiles averaged over 
      the observational period and is kept constant in time for the calibration period.  For the future projections, 
      ELA is set to the mean glacier height and is time dependent since glacier volume, area, and length are time 
      dependent (volume-area-length scaling).
      Bliss et al. (2014) uses the same as Valentina's model
    
    Parameters
    ----------
    surfacetype : np.ndarray
        Surface type for each elevation bin
    glac_bin_massbalclim_annual : np.ndarray
        Annual climatic mass balance for each year and each elevation bin
    year_index : int
        Count of the year of model run (first year is 0)

    Returns
    -------
    surfacetype : np.ndarray
        Updated surface type for each elevation bin
    firnline_idx : int
        Firn line index
    """        
    # Next year's surface type is based on the bin's average annual climatic mass balance over the last 5 years.  If 
    #  less than 5 years, then use the average of the existing years.
    if year_index < 5:
        # Calculate average annual climatic mass balance since run began
        massbal_clim_mwe_runningavg = glac_bin_massbalclim_annual[:,0:year_index+1].mean(1)
    else:
        massbal_clim_mwe_runningavg = glac_bin_massbalclim_annual[:,year_index-4:year_index+1].mean(1)
    # If the average annual specific climatic mass balance is negative, then the surface type is ice (or debris)
    surfacetype[(surfacetype !=0 ) & (massbal_clim_mwe_runningavg <= 0)] = 1
    # If the average annual specific climatic mass balance is positive, then the surface type is snow (or firn)
    surfacetype[(surfacetype != 0) & (massbal_clim_mwe_runningavg > 0)] = 2
    # Compute the firnline index
    try:
        # firn in bins >= firnline_idx
        firnline_idx = np.where(surfacetype==2)[0][0]
    except:
        # avoid errors if there is no firn, i.e., the entire glacier is melting
        firnline_idx = np.where(surfacetype!=0)[0][-1]
    # Apply surface type model options
    # If firn surface type option is included, then snow is changed to firn
    if input.option_surfacetype_firn == 1:
        surfacetype[surfacetype == 2] = 3
    if input.option_surfacetype_debris == 1:
        print('Need to code the model to include debris.  Please choose an option that currently exists.\n'
              'Exiting the model run.')
        exit()
    return surfacetype, firnline_idx


def surfacetypebinsinitial(glacier_area, glacier_table, elev_bins):
    """
    Define initial surface type according to median elevation such that the melt can be calculated over snow or ice.
    
    Convention: (0 = off-glacier, 1 = ice, 2 = snow, 3 = firn, 4 = debris).
    Function options: 1 =
    
    Function options specified in pygem_input.py:
    - option_surfacetype_initial
        > 1 (default) - use median elevation to classify snow/firn above the median and ice below
        > 2 - use mean elevation instead
    - option_surfacetype_firn = 1
        > 1 (default) - firn is included
        > 0 - firn is not included
    - option_surfacetype_debris = 0
        > 0 (default) - debris cover is not included
        > 1 - debris cover is included
    
    To-do list
    ----------
    Add option_surfacetype_initial to specify an AAR ratio and apply this to estimate initial conditions
    
    Parameters
    ----------
    glacier_area : np.ndarray
        Glacier area [km2] from previous year for each elevation bin
    glacier_table : pd.Series
        Table of glacier's RGI information
    elev_bins : np.ndarray
        Elevation bins [masl]

    Returns
    -------
    surfacetype : np.ndarray
        Updated surface type for each elevation bin
    firnline_idx : int
        Firn line index
    """        
    surfacetype = np.zeros(glacier_area.shape)
    # Option 1 - initial surface type based on the median elevation
    if input.option_surfacetype_initial == 1:
        surfacetype[(elev_bins < glacier_table.loc['Zmed']) & (glacier_area > 0)] = 1
        surfacetype[(elev_bins >= glacier_table.loc['Zmed']) & (glacier_area > 0)] = 2
    # Option 2 - initial surface type based on the mean elevation
    elif input.option_surfacetype_initial ==2:
        surfacetype[(elev_bins < glacier_table['Zmean']) & (glacier_area > 0)] = 1
        surfacetype[(elev_bins >= glacier_table['Zmean']) & (glacier_area > 0)] = 2
    else:
        print("This option for 'option_surfacetype' does not exist. Please choose an option that exists. "
              + "Exiting model run.\n")
        exit()
    # Compute firnline index
    try:
        # firn in bins >= firnline_idx
        firnline_idx = np.where(surfacetype==2)[0][0]
    except:
        # avoid errors if there is no firn, i.e., the entire glacier is melting
        firnline_idx = np.where(surfacetype!=0)[0][-1]
    # If firn is included, then specify initial firn conditions
    if input.option_surfacetype_firn == 1:
        surfacetype[surfacetype == 2] = 3
        #  everything initially considered snow is considered firn, i.e., the model initially assumes there is no snow 
        #  on the surface anywhere.
    if input.option_surfacetype_debris == 1:
        print("Need to code the model to include debris. This option does not currently exist.  Please choose an option"
              + " that exists.\nExiting the model run.")
        exit()
        # One way to include debris would be to simply have debris cover maps and state that the debris retards melting 
        # as a fraction of melt.  It could also be DDF_debris as an additional calibration tool. Lastly, if debris 
        # thickness maps are generated, could be an exponential function with the DDF_ice as a term that way DDF_debris 
        # could capture the spatial variations in debris thickness that the maps supply.
    return surfacetype, firnline_idx


def surfacetypeDDFdict(modelparameters, 
                       option_surfacetype_firn=input.option_surfacetype_firn,
                       option_DDF_firn=input.option_DDF_firn):
    """
    Create a dictionary of surface type and its respective DDF.
    
    Convention: [0=off-glacier, 1=ice, 2=snow, 3=firn, 4=debris]
    
    modelparameters[lr_gcm, lr_glac, prec_factor, prec_grad, DDF_snow, DDF_ice, T_snow]
    
    To-do list
    ----------
    Add option_surfacetype_initial to specify an AAR ratio and apply this to estimate initial conditions
    
    Parameters
    ----------
    modelparameters : pd.Series
        Model parameters (lrgcm, lrglac, precfactor, precgrad, ddfsnow, ddfice, tempsnow, tempchange)
        Order of model parameters should not be changed as the run mass balance script relies on this order
    option_surfacetype_firn : int
        Option to include or exclude firn (specified in pygem_input.py)
    option_DDF_firn : int
        Option for the degree day factor of firn to be the average of snow and ice or a different value

    Returns
    -------
    surfacetype_ddf_dict : dictionary
        Dictionary relating the surface types with their respective degree day factors
    """        
    surfacetype_ddf_dict = {
            1: modelparameters[5],
            2: modelparameters[4]}
    if option_surfacetype_firn == 1:
        if option_DDF_firn == 0:
            surfacetype_ddf_dict[3] = modelparameters[4]
        elif option_DDF_firn == 1:
            surfacetype_ddf_dict[3] = np.mean([modelparameters[4],modelparameters[5]])
    if input.option_surfacetype_debris == 1:
        surfacetype_ddf_dict[4] = input.DDF_debris
    return surfacetype_ddf_dict