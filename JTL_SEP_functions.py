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

from solarmach import SolarMACH

mrkr_settings = {'Solar Orbiter': {'marker': 's', 'color': 'dodgerblue', 'label': 'Solar Orbiter-EPD/HET'},
                 'SOHO': {'marker': 'o', 'color': 'darkgreen', 'label': 'SOHO-ERNE/HED'},
                 'STEREO-A': {'marker': '^', 'color': 'red', 'label': 'STEREO-A/HET'},
                 'PSP':  {'marker': 'p', 'color': 'purple', 'label': 'Parker Solar Probe/HET'},
                 #'Wind': {'marker': '*', 'color': 'slategray', 'label': 'Wind'},
                 #'STEREO-B': {'marker': 'v', 'color': 'blue', 'label': 'STEREO-B'},
                 #'BepiColombo': {'marker': 'd', 'color': 'orange', 'label': 'BepiColombo'}
                }


plt.style.use('seaborn-v0_8-paper')
mpl.rcParams.update({'font.size': 10,
                     'axes.titlesize': 10, 'axes.labelsize': 10,
                     'figure.labelsize': 10,
                     'lines.markersize': 8, 'lines.linewidth': 1.4,
                     'legend.fontsize': 6, 'legend.title_fontsize': 7,
                     'savefig.transparent': False, 'savefig.bbox': 'tight',
                     'axes.grid': False, 'ytick.minor.visible': False})

LEGND_LOC = (1.02, 1.02)
DEGREE_TEXT = r'$^{\circ}$'
SIGMA_TEXT = r'$\sigma$'
LAMBDAPERP_TEXT = r'$\lambda_{\perp}$'
LAMBDAPLL_TEXT = r'$\lambda_{\parallel}$'
PM_SYMB = r"$\pm$"
SQR = r"$^2$"




## Solar mach
# 1. initial plot to determine if event is good
def solarmach_basic(startdate, data_path, coord_sys='Stonyhurst', source_location=None):
    filename = f'SolarMACH_{startdate.strftime("%d%m%Y_%H:%M")}'
    print(filename)
    print(os.listdir(data_path))
    if filename+'.csv' in os.listdir(data_path):
        print('WIN')
        smtable = pd.read_csv(data_path+filename+'.csv', index_col='Spacecraft/Body')
        return smtable
        
    datetime_parse = parse_time(startdate)

    observers = ['Earth', 'PSP', 'Solar Orbiter', 'STEREO-A', 'BepiColombo']

    sm1 = SolarMACH(datetime_parse, observers, vsw_list=[], 
                    reference_long=source_location[0],  reference_lat=source_location[1], 
                    coord_sys='Stonyhurst')

    sm1_tab = sm1.coord_table.copy()
    sm1_tab.index = sm1_tab['Spacecraft/Body']
    sm1_tab.to_csv(f'{data_path+filename}.csv')

    # Show the polar plot
    sm1.plot(plot_spirals=True, plot_sun_body_line=True, reference_vsw=400,
             transparent=False, numbered_markers=False, long_offset=270, 
             return_plot_object=False, 
             outfile=f'{data_path+filename}.png')

    return sm1

    
# 2. loop to get sc locations over time
def solarmach_loop(observers, dates, data_path, source_loc=[None,None], coord_sys='Stonyhurst'):
    filename = f'SolarMACH_{dates[0].strftime("%d%m%Y")}_loop.csv'

    if filename in os.listdir(data_path):
        sm_loop = pd.read_csv(data_path+filename, index_col=0, header=[0,1], parse_dates=True)
        return sm_loop

    # Set up the time list to iterate through
    starting_time = dt.datetime(dates[0].year, dates[0].month, dates[0].day, dates[0].hour-5, 0) #Gathering enough background too
    date_list = pd.date_range(starting_time, dates[1], freq='15min')

    sm_df = {}
    for t in date_list:
        sm10 = SolarMACH(t, observers, vsw_list=[], 
                         reference_long=source_loc[0], 
                         reference_lat=source_loc[1], 
                         coord_sys='Stonyhurst')
        sm_df[t] = sm10.coord_table

    # Create a new dict with each spacecraft df saved separately
    df1 = {}
    for obs in observers:
        
        r_dist, vsw, foot_long, foot_long_error = ([] for i in range(4))
        for tt in date_list:
            tmp_df = pd.DataFrame.from_dict(sm_df[tt])
            tmp_df.set_index('Spacecraft/Body', inplace=True)

            r_dist.append( tmp_df['Heliocentric distance (AU)'][obs] )
            vsw.append( tmp_df['Vsw'][obs] )

            

            foot_calc = move_along_parker_spiral(
                r_dist=tmp_df['Heliocentric distance (AU)'][obs], 
                loc=[float(tmp_df['Stonyhurst longitude (°)'][obs]), 
                          float(tmp_df['Stonyhurst latitude (°)'][obs])],
                vsw=tmp_df['Vsw'][obs], towards=True, err_calc=True)

            foot_long.append( tmp_df['Magnetic footpoint longitude (Stonyhurst)'][obs] )
            #print('Calculated foot: ', foot_calc)
            #print('Given foot: ', tmp_df['Magnetic footpoint longitude (Stonyhurst)'][obs])
            #jax=input('How are the feet? ')
            foot_long_error.append( foot_calc[1] )

        df1[obs] = {}
        df1[obs]['r_dist'] = r_dist
        df1[obs]['vsw'] = vsw
        df1[obs]['foot_long'] = foot_long
        df1[obs]['foot_long_error'] = foot_long_error

        
    df2 = pd.DataFrame({(outer_key, inner_key): values
                        for outer_key, inner_dict in df1.items()
                        for inner_key, values in inner_dict.items()})

    df2.index = date_list
    df2.index.name = 'Time'


    # Save to csv
    df2.to_csv(data_path+filename)

    return df2
    
def move_along_parker_spiral(r_dist, loc, vsw, towards, err_calc):
    #radial distance, location  [WE, NS],  measured Vsw, towards the Sun (True) or away (False), err_calc = True/False
    sun_radius = aconst.R_sun.to(u.AU).value
    if loc[0] < 0:
        loc[0] = loc[0] + 360

    # Calculating the solar wind parameters
    omega_ref = np.radians((14.5 - 2.87 * (np.sin(np.deg2rad(loc[1]))**2)) / (24*60*60)) # in rad/s
    if err_calc:
        omega_vsw = []
        new_loc_arr = []
        for Vsw in [vsw, vsw-50, vsw+50]:
            omega_vsw.append(omega_ref / (Vsw / 1.5e8)) # in rad/au
    else:
        omega_vsw = omega_ref / (vsw / 1.5e8) # in rad/au

    # Calulcating the new location
    if err_calc:
        for Ov in omega_vsw:
            if towards:
                newloc_tmp = np.radians(loc[0]) - Ov * (sun_radius - r_dist) * ( np.cos( np.deg2rad( loc[1] ) ) )
            else:
                newloc_tmp = np.radians(loc[0]) + Ov * (sun_radius - r_dist) * ( np.cos( np.deg2rad( loc[1] ) ) )

            newloc_t = float(np.degrees(newloc_tmp))

            # Make sure it falls within the Stonyhurst restrictions
            if newloc_t > 180:
                newloc_t = newloc_t - 360
            elif newloc_t < -180:
                newloc_t = newloc_t + 360

            # Add to the array
            new_loc_arr.append(newloc_t)

        # Find the nominal and uncertainty values
        new_loc_unc = np.std(new_loc_arr, mean=new_loc_arr[0])
        new_loc = [new_loc_arr[0], float(new_loc_unc)]
    else:
        if towards:
            newloc_tmp = np.radians(loc[0]) - omega_vsw * (sun_radius - r_dist) * ( np.cos( np.deg2rad( loc[1] ) ) )
        else:
            newloc_tmp = np.radians(loc[0]) + omega_vsw * (sun_radius - r_dist) * ( np.cos( np.deg2rad( loc[1] ) ) )

        new_loc = float(np.degrees(newloc_tmp))

        # Make sure it falls within the Stonyhurst restrictions
        if new_loc > 180:
            new_loc = new_loc - 360
        elif new_loc < -180:
            new_loc = new_loc + 360


    return new_loc
    
    
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

def load_sc_data(spacecraft, proton_channels, dates, data_path, intercalibration=True, radial_scaling=True, resampling='15min', reference_loc=[None,None]):
    """Load the data, merge the bins, make omnidirectional, resample, intercalibrate, radial scale, and return one df:
        -index: times
        - header1: sc-ins
        - [Flux, Uncertainty, Radial Distance, Longitude]"""

    # Check if the file is already made and just load that one
    filename = f"SEP_intensities_{dates[0].strftime("%d%m%Y")}.csv"
    #print(filename)
    #print(data_path)
    #print(os.listdir(data_path))
    #jax=input('Yes?')
    if filename in os.listdir(data_path):
        # Jan: Potential issue with zeroes and nans both saved as zero.
        sc_df = pd.read_csv(data_path+filename, header=[0,1], index_col=0, parse_dates=True)
        return sc_df


    # Download solarmach data for the same intervals
    sm_df = solarmach_loop(spacecraft, dates, data_path, source_loc=reference_loc, coord_sys='Stonyhurst')
    #print(sm_df)
    #jax=input('Continue?')

    
    # Download the data and load into a dictionary with the key as the spacecraft-instrument label
    sc_dict = {}
    spacecraft = [x.lower() for x in spacecraft]

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
        psp = psp_df2.resample(resampling).mean()
        psp_sm = pd.concat([psp, sm_df['PSP']], axis=1, join='outer')
        

        # Add to the collection
        sc_dict['PSP'] = psp_sm



    if 'soho' in spacecraft:
        soho_df, soho_meta = soho_load(dataset='SOHO_ERNE-HED_L2-1MIN', # 'PH_n' 'PHC_n'
                                       startdate=dates[0], enddate=dates[1],
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
        soho = soho_df2.resample(resampling).mean()
        soho_sm = pd.concat([soho, sm_df['SOHO']], axis=1, join='outer')

        # Add to the collection
        sc_dict['SOHO'] = soho_sm

    if 'stereo-a' in spacecraft:
        sta_df, sta_meta = stereo_load(instrument='HET', spacecraft='ahead', # 'Proton_Flux_n' 'Proton_Sigma_n'
                                       startdate=dates[0], enddate=dates[1],
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
        sta = sta_df2.resample(resampling).mean()
        sta_sm = pd.concat([sta, sm_df['STEREO-A']], axis=1, join='outer')

        # Add to the collection
        sc_dict['STEREO-A'] = sta_sm


    if 'solar orbiter' in spacecraft:
        solo_dfp, solo_dfe, solo_meta = epd_load(sensor='het', level='l2', # [('H_Flux','H_Flux_n')] [('H_Uncertainty','H_Uncertainty_n')]
                                      startdate=dates[0], enddate=dates[1],
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
        solo_df2["Flux"] = weighted_bin_merge(solo_df1, 'solo', 'p', bin_list, ['H_Flux','H_Flux_'], bin_width)
        solo_df2["Uncertainty"] = weighted_bin_merge(solo_df1, 'solo', 'p', bin_list, ['H_Uncertainty','H_Uncertainty_'], bin_width)

        solo_df3 = pd.DataFrame.from_dict(solo_df2)
        solo_df3.set_index('Time', inplace=True)

        # Resample
        solo = solo_df3.resample(resampling).mean()
        solo_sm = pd.concat([solo, sm_df['Solar Orbiter']], axis=1, join='outer')

        # Add to the collection
        sc_dict['Solar Orbiter'] = solo_sm


    # Merge all the relevant columns into one df
    print(sc_dict)


    sc_df = pd.concat(sc_dict, axis=1, join='outer')
    sc_df.to_csv(data_path + filename)

    return sc_df


################################################
## Inter-calibration
################################################
def intercalibration_calculation(df, observer_metadict, data_path, dates):
    # Iterate through the observers (first header in df)
    for obs, meta_data in observer_metadict.items():
        factor = meta_data['intercalibration'] # extract the intercalibration factor

        # Apply the scaling to the Flux and Uncertainty columns
        for col in ['Flux','Uncertainty']: # Both are calculated the same
            #print(df[(obs,col)])
            #jax=input('huh?')
            df[(obs, col)] *= factor

    df.to_csv(f"{data_path}SEP_intensities_{dates[0].strftime("%d%m%Y")}_IC.csv") # Save for sanity checks

    return df

################################################
## Radial Scaling
################################################
def radial_scaling_calculation(df0, data_path, scaling_values, dates):
    df = df0.copy(deep=True) # so it doesnt mess with the OG df's
    
    a = scaling_values[0]
    b = scaling_values[1]

    # Iterate through observers
    for obs, df_obs in df.groupby(level=0, axis=1): # returning the observer and their specific df

        for t in df_obs.index:
            #print(df_obs.loc[t, (obs,'Flux')])
            if pd.isna(df_obs.loc[t, (obs,'Flux')]) or pd.isna(df_obs.loc[t, (obs,'r_dist')])\
            or (df_obs.loc[t, (obs,'Flux')]==0) or (df_obs.loc[t, (obs,'r_dist')]==0):
                f_rscld = np.nan
                unc_final = np.nan
            
            else:
                # Scale the flux
                f_rscld = df_obs.loc[t, (obs,'Flux')] * (df_obs.loc[t, (obs,'r_dist')] ** a)

                # Scale the uncertainty
                ## Find the difference from the boundaries
                unc_plus = df_obs.loc[t, (obs,'Flux')] * (df_obs.loc[t, (obs,'r_dist')] **(a+b))
                unc_limit_plus = abs(f_rscld - unc_plus)
                unc_minus = df_obs.loc[t, (obs,'Flux')] * (df_obs.loc[t, (obs,'r_dist')] **(a-b))
                unc_limit_minus = abs(f_rscld - unc_minus)
    
                if (unc_limit_plus >= unc_limit_minus) and (f_rscld - unc_limit_plus > 0):
                    chosen_unc_limit = unc_limit_plus
                elif (unc_limit_minus >= unc_limit_plus) and (f_rscld - unc_limit_minus > 0):
                    chosen_unc_limit = unc_limit_minus
                else:
                    print("There's a problem with the limits")
                    print("OG flux: ", df_obs.loc[t, (obs,'Flux')])
                    print("OG rad: ", df_obs.loc[t, (obs,'r_dist')])
                    print("Scaled Flux: ", f_rscld)
                    print("Unc plus: ", unc_limit_plus)
                    print("Unc minus: ", unc_limit_minus)
                    jax = input('Continue? ')
                    chosen_unc_limit = np.nan
    
                ## Find the calculated scaled uncertainty
                unc_calculated = df_obs.loc[t, (obs,'Uncertainty')] * (df_obs.loc[t, (obs,'r_dist')] ** a)
    
                ## Merge both results for the final scaled uncertainty
                unc_final = np.sqrt((unc_calculated)**2 + (chosen_unc_limit)**2)

            df.loc[t, (obs, 'Flux')] = f_rscld
            df.loc[t, (obs, 'Uncertainty')] = unc_final


    df.to_csv(f"{data_path}SEP_intensities_{dates[0].strftime("%d%m%Y")}_RS.csv") # Save for sanity checks
    return df

################################################
## Gaussian fitting
################################################



################################################
## Plotters
################################################

def plot_timeseries_result(df, data_path, dates, background_window=[]):
    # Determine how many subplots are needed
    obs = set([col[0] for col in df.columns if len(col)==2])
    obs = list(obs)
    print(obs)

    fig, ax = plt.subplots(len(obs), 1, figsize=[5, len(obs)+2.5], dpi=300, sharex=True)
    fig.subplots_adjust(hspace=0.02)

    # Title
    ax[0].set_title(dates[0].strftime("%H:%M - %d %b, %Y"), pad=9, loc='left')
    fig.supylabel('Intensity')

    for n in range(len(obs)):
        sc = obs[n]
        mrkr = mrkr_settings[sc]

        # Show the event start time
        ax[n].axvline(x=dates[0], color='k', linestyle='dashed', linewidth=0.5)

        # Show the background window
        if len(background_window) > 1:
            ax[n].axvspan(background_window[0], background_window[1], alpha=0.2, color='grey')
            bg_txt = f'Background window:\n{background_window[0].strftime("%H:%M")} - {background_window[0].strftime("%H:%M %d %b, %Y")}'
            box_obj = AnchoredText(bg_txt, frameon=True, loc='lower right', pad=0.5, prop={'size':7})
            plt.setp(box_obj.patch, facecolor='grey', alpha=0.9)
            ax[0].add_artist(box_obj)

        # Plot the data
        ax[n].semilogy(df[(sc, 'Flux')], color=mrkr['color'], label=mrkr['label'], linestyle='solid')
        ax[n].fill_between(x = df.index,
                           y1= df[(sc, 'Flux')] - df[(sc, 'Uncertainty')],
                           y2= df[(sc, 'Flux')] + df[(sc, 'Uncertainty')],
                           alpha=0.3, color=mrkr['color'])
        ax[n].legend(loc='upper right', alignment='left')
        ax[n].yaxis.set_major_locator(mpl.ticker.LogLocator(base=10, numticks=3))
        ax[n].minorticks_on()

    xmin = dates[0] - dt.timedelta(hours=5)
    xmax = dates[0] + dt.timedelta(hours=22)
    ax[0].set_xlim([xmin,xmax])
    locator = mpl.dates.AutoDateLocator(minticks=3, maxticks=6)
    ax[n-1].xaxis.set(major_locator=locator, )
    ax[n-1].xaxis.set_major_formatter(mpl.dates.ConciseDateFormatter(locator, show_offset=False))

    label=''
    if input('Save the file? ')=='y':
        label = input('Save file key word: ')
    plt.savefig(data_path+f'SEP_Intensities_{dates[0].strftime("%d%m%y")}_{label}.png')
    plt.show()
    
        
