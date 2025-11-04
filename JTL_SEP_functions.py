import os
import datetime as dt
import numpy as np
import pandas as pd

from scipy.optimize import curve_fit
from scipy import odr

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

from sunpy.time import parse_time
import astropy.constants as aconst
import astropy.units as u

from seppy.loader.psp import psp_isois_load
from seppy.loader.soho import soho_load
from seppy.loader.stereo import stereo_load
from solo_epd_loader import epd_load

## Solar mach
# 1. initial plot to determine if event is good
# 2. loop to get sc locations over time

################################################
## Data loaders
################################################
def weighted_bin_merge(df0, spacecraft, species, channel_list, header_label, binwidths):
    """Reading in a df with just columns to merge across the rows (one row should merge to one intensity value),
        the spacecraft and instrument, the species(just e or p), and the bin numbers."""

    if species.lower() == 'p':
        species = 'protons'
    else:
        species = 'electrons'


    time_indx = []
    merged_flux = []
    for i in df0.index:
        time_indx.append(i)

        row_flux = 0
        row_div = 0
        for n in range(len(channel_list)):
            if type(header_label)==list: #If its a double-layered header
                row_flux = row_flux + ( df0.loc[i, (header_label[0], f"{header_label[1]}{channel_list[n]}")] ) * binwidths[n]
            else:
                row_flux = row_flux + ( df0.loc[i, f"{header_label}{channel_list[n]}"] ) * binwidths[n]
            row_div = row_div + binwidths[n]
        merged_flux.append(float(row_flux / row_div))


    return merged_flux


################################################

def load_sc_data(spacecraft, proton_channels, dates, data_path, intercalibration=True, radial_scaling=True, resampling='15min'):
    """Load the data, merge the bins, make omnidirectional, resample, intercalibrate, radial scale, and return one df:
        -index: times
        - header1: sc-ins
        - [Flux, Uncertainty, Radial Distance, Longitude]"""

    spacecraft = [x.lower() for x in spacecraft]


    # Download the data and load into a dictionary with the key as the spacecraft-instrument label
    sc_dict = {}

    if 'psp' in spacecraft:
        psp_df, psp_meta = psp_isois_load(dataset='PSP_ISOIS-EPIHI_L2-HET-RATES60', # 'A_H_Flux_n' 'B_H_Uncertainty_n'
                                          startdate=dates[0], enddate=dates[1],
                                          path=data_path+'psp/', resample='1min')

        # Find channels and bin widths
        bin_list = proton_channels['PSP']['channels']
        if len(bin_list)==1:
            bin_label = f"{bin_list[0]}"
        else:
            bin_label = f"{bin_list[0]}-{bin_list[1]}"
        bin_width = []
        for n in bin_list:
            binstr = str(psp_meta['H_ENERGY_LABL'][n][0]).strip().split('-')
            #print('The bin string is now: ', binstr)

            bin_start = float(binstr[0].strip())
            bin_end = binstr[-1].split('MeV')
            bin_end = float(bin_end[0].strip())

            bin_width.append(bin_end - bin_start)
        #print('PSP bin width: ', bin_width)


        # Merge the channels
        for view in ['A','B']:
            psp_df[f"{view}_F_{bin_label}"] = weighted_bin_merge(psp_df, 'psp', 'p', bin_list, f"{view}_H_Flux_", bin_width)
            psp_df[f"{view}_Func_{bin_label}"] = weighted_bin_merge(psp_df, 'psp', 'p', bin_list, f"{view}_H_Uncertainty_", bin_width)


        # Make omnidirectional
        psp_df1 = {'Time': psp_df.index}
        flux_arr, unc_arr = ([] for i in range(2))
        for tt in psp_df.index:
            flux_arr.append( np.nanmean([psp_df.loc[tt, f"A_F_{bin_label}"], psp_df.loc[tt, f"B_F_{bin_label}"]]) )
            unc_arr.append( np.nanmean([psp_df.loc[tt, f"A_Func_{bin_label}"], psp_df.loc[tt, f"B_Func_{bin_label}"]]) )
        psp_df1["Flux"] = flux_arr
        psp_df1["Uncertainty"] = unc_arr

        psp_df2 = pd.DataFrame.from_dict(psp_df1)
        psp_df2.set_index('Time', inplace=True)

        # Resample
        psp = psp_df2.resample(resample).mean()


        # Add to the collection
        sc_dict['PSP-HET'] = psp



    if 'soho' in spacecraft:
        soho_df, soho_meta = soho_load(dataset='SOHO_ERNE-HED_L2-1MIN', # 'PH_n' 'PHC_n'
                                       startdate=startdate, enddate=enddate,
                                       path=data_path+'soho/', resample='1min',
                                       pos_timestamp='start')

        # Find channels and bin widths
        bin_list = proton_channels['SOHO']['channels']
        if len(bin_list)==1:
            bin_label = f"{bin_list[0]}"
        else:
            bin_label = f"{bin_list[0]}-{bin_list[1]}"
        bin_width = []
        for n in bin_list:
            soho_meta = soho_meta['channels_dict_df_p']
            bin_start = soho_meta.loc[n, 'lower_E']
            bin_end = soho_meta.loc[n, 'upper_E']

            bin_width.append(bin_end - bin_start)


        # Calculate the uncertainty
        for n in bin_list: # data provided as intensities (PH) or counts/min (PHC)
            soho_df[f"Uncert_{n}"] = ( (soho_df[f"PH_{n}"]) / (np.sqrt(soho_df[f"PHC_{n}"])) ) * 1.1 # adding 10% uncertainty for systematic errors (according to RV)


        # Merge the channels
        soho_df1 = {'Time': soho_df.index}
        soho_df1["Flux"] = weighted_bin_merge(soho_df, 'soho', 'p', bin_list, 'PH_', bin_width)
        soho_df1["Uncertainty"] = weighted_bin_merge(soho_df, 'soho', 'p', bin_list, 'Uncert_', bin_width)

        soho_df2 = pd.DataFrame.from_dict(soho_df1)
        soho_df2.set_index('Time', inplace=True)

        # Resample
        soho = soho_df2.resample(resample).mean()


        # Add to the collection
        sc_dict['SOHO-HED'] = soho

    if 'stereo-a' in spacecraft:
        sta_df, sta_meta = stereo_load(instrument='HET', spacecraft='ahead', # 'Proton_Flux_n' 'Proton_Sigma_n'
                                       startdate=startdate, enddate=enddate,
                                       path=data_path+'stereo/', resample='1min',
                                       pos_timestamp='start')

        # Find channels and bin widths
        bin_list = proton_channels['STEREO-A']['channels']
        if len(bin_list)==1:
            bin_label = f"{bin_list[0]}"
        else:
            bin_label = f"{bin_list[0]}-{bin_list[1]}"
        bin_width = []
        for n in bin_list:
            sta_meta = sta_meta['channels_dict_df_p']
            bin_start = soho_meta.loc[n, 'lower_E']
            bin_end = soho_meta.loc[n, 'upper_E']

            bin_width.append(bin_end - bin_start)


        # Merge the channels
        sta_df1 = {'Time': sta_df.index}
        sta_df1["Flux"] = weighted_bin_merge(sta_df, 'sta', 'p', bin_list, 'Proton_Flux_', bin_width)
        sta_df1["Uncertainty"] = weighted_bin_merge(sta_df, 'sta', 'p', bin_list, 'Proton_Sigma_', bin_width)

        sta_df2 = pd.DataFrame.from_dict(sta_df1)
        sta_df2.set_index('Time', inplace=True)

        # Resample
        sta = sta_df2.resample(resample).mean()

        # Add to the collection
        sc_dict['STEREO-A HET'] = sta


    if 'solar orbiter' in spacecraft:
        solo_dfp, solo_dfe, solo_meta = epd_load(sensor='het', level='l2', # [('H_Flux','H_Flux_n')] [('H_Uncertainty','H_Uncertainty_n')]
                                      startdate=startdate, enddate=enddate,
                                      viewing='omni', autodownload=True,
                                      pos_timestamp='start', path=data_path+'solo/')

        solo_df1 = solo_dfp.resample('1min').mean()

        # Find channels and bin widths
        bin_list = proton_channels['Solar Orbiter']['channels']
        if len(bin_list)==1:
            bin_label = f"{bin_list[0]}"
        else:
            bin_label = f"{bin_list[0]}-{bin_list[1]}"
        bin_width = []
        for n in bin_list:
            bin_width.append(solo_meta['H_Bins_Width'][n])

        # Merge the channels
        solo_df2 = {'Time': solo_df1.index}
        solo_df2['Flux'] = weighted_bin_merge(solo_df1, 'solo', 'p', bin_list, [('H_Flux','H_Flux_')], bin_width)
        solo_df2["Uncertainty"] = weighted_bin_merge(solo_df1, 'solo', 'p', bin_list, [('H_Uncertainty','H_Uncertainty_')], bin_width)

        solo_df3 = pd.DataFrame.from_dict(solo_df2)
        solo_df3.set_index('Time', inplace=True)

        # Resample
        solo = solo_df3.resample(resample).mean()

        # Add to the collection
        sc_dict['Solar Orbiter - EPD/HET'] = solo


    sc_df = pd.concat(sc_dict.keys(), axis=1, join='outer')

    return sc_df


################################################
## Inter-calibration
################################################


################################################
## Radial Scaling
################################################



################################################
## Gaussian fitting
################################################



################################################
## Plotters
################################################

