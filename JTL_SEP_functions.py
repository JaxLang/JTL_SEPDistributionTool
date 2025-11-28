import os
import datetime as dt
import numpy as np
import pandas as pd

from scipy.optimize import curve_fit
from scipy import odr

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

import glob
import imageio
from IPython.display import Image

#from sunpy.time import parse_time
import astropy.constants as aconst
import astropy.units as u

from seppy.loader.psp import psp_isois_load
from seppy.loader.soho import soho_load
from seppy.loader.stereo import stereo_load
from solo_epd_loader import epd_load

from solarmach import SolarMACH

marker_settings = {
    'Solar Orbiter': {'marker': 's', 'color': 'dodgerblue', 'label': 'Solar Orbiter-EPD/HET'},
    'SOHO': {'marker': 'o', 'color': 'darkgreen', 'label': 'SOHO-ERNE/HED'},
    'STEREO-A': {'marker': '^', 'color': 'red', 'label': 'STEREO-A/HET'},
    'PSP':  {'marker': 'p', 'color': 'purple', 'label': 'Parker Solar Probe/HET'},
    'Wind': {'marker': '*', 'color': 'slategray', 'label': 'Wind'},
    'STEREO-B': {'marker': 'v', 'color': 'blue', 'label': 'STEREO-B'},
    'BepiColombo': {'marker': 'd', 'color': 'orange', 'label': 'BepiColombo'}
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


test_date = dt.datetime(2021,5,29,2,30)
JAX_TESTERS = False


################################################
## Event Class
################################################
class SEPEvent:
    def __init__(self, channels, dates, resampling, data_path, **kwargs): #coord_sys='Stonyhurst', flare_loc=None):
        """channels: dict of the channels [start, end] to be used for each spacecraft (given as keys)
        dates: [start, end] dates (flare start time in the 'start')
        resampling: the interval to resample the data to
        data_path: where the data is saved to
        intercalibration: dict of ic values ('None' if user doesnt want to ic)
        radial_scaling: list of radial scaling values (a +- b) (see Farwa+2025 for more details)
        coord_sys: (default: Stonyhurst) either stonyhurst or carrington for plotting and calculations.
        flare_loc: for adding a reference to plots
        """
        self.start = dates[0]
        self.end = dates[1]
        self.channels = channels
        self.resampling = resampling

        save_path = f"{data_path}SEPEvent_{dates[0].strftime("%d%b%Y")}/"
        try:
            #os.makedirs(data_path)
            os.makedirs(save_path)
        except FileExistsError:
            pass
        self.data_path = data_path
        self.path = save_path
        if 'coord_sys' in kwargs.keys():
            coord_sys = kwargs['coord_sys']
            if coord_sys.lower().startswith('car'):
                coord_sys = 'Carrington'
            elif coord_sys.lower().startswith('sto') or coord_sys.lower() == 'earth':
                coord_sys = 'Stonyhurst'
        else:
            coord_sys = 'Stonyhurst'
        self.coord_sys = coord_sys

        if 'flare_loc' in kwargs.keys():
            self.flare_loc = kwargs['flare_loc']
        else:
            self.flare_loc = [np.nan, np.nan]

        # List and load the data
        self.spacecraft_list = list(channels.keys())

        ## Solarmach data
        self.sm_data = {}
        self._load_solarmach_loop()

        ## Observer data
        self.sc_data = {}
        self._load_spacecraft_data()


    def _load_solarmach_loop(self):
        """Download the solarmach data for the time period."""
        try:
            self.sm_data = solarmach_loop(self.spacecraft_list, [self.start, self.end],
                                              self.path, self.resampling, self.flare_loc)
        except Exception as e:
            print(f"Warning: Could not load solarmach data: {e}")

    def _load_spacecraft_data(self):
        """Download the data for each sc"""
        for sc in self.spacecraft_list:
            if True: #try:
                self.sc_data[sc] = load_sc_data(sc, self.sm_data, self.channels,
                                                [self.start, self.end],
                                                self.data_path, self.resampling)
            else: #except Exception as e:
                print(f"Warning: Could not load data for {sc}: {e}")

    def get_sc_df(self, sc_name):
        """Return the df to the user"""
        return self.sc_data.get(sc_name)

    def background_subtract(self, background_window):
        """Given a window by the user, the function calculates the average, reduces the
        intensity by this average, and results in a background reduced dataset."""
        for sc in self.spacecraft_list:
            try:
                self.sc_data[sc] = background_subtracting(self.sc_data.get(sc), self.path, background_window)
            except Exception as e:
                print(f"Warning: Could not background subtract for {sc}: {e}")

    def intercalibrate(self, intercalibration_values):
        """Adjusts the data based on the given intercalibration values."""
        for sc in self.spacecraft_list:
            try:
                self.sc_data[sc] = intercalibration_calculation(self.sc_data.get(sc), intercalibration_values[sc])
            except Exception as e:
                print(f"Warning: Could not intercalibrate for {sc}: {e}")

    def radial_scale(self, radial_scaling_factors):
        """Scales the data using the functions presented in FarwaEA2025."""
        for sc in self.spacecraft_list:
            try:
                self.sc_data[sc] = radial_scaling_calculation(self.sc_data.get(sc), radial_scaling_factors)
            except Exception as e:
                print(f"Warning: Could not radial scale for {sc}: {e}")

    def plot_intensities(self, **kwargs):
        """Plots the time series for each given observer."""
        bg_zone = []
        if 'background_window' in kwargs:
            bg_zone = kwargs['background_window']
        try:
            plot_timeseries_result(self.sc_data, self.path, self.start, bg_zone);
        except Exception as e:
            print(f"Warning: Could not plot figure: {e}")

    def get_peak_fits(self):
        """Calculates the Gaussian curve fitted to the peak intensities."""
        find_and_plot_peak_intensity(self.sc_data, self.path, self.start)

    def calc_Gaussian_fit(self):
        """Calculates the Gaussian fits at each time interval."""
        try:
            self.sc_data['Gauss'] = fit_gauss_curves_to_data(self.sc_data, self.path);
        except Exception as e:
            print(f"Warning: Could not calculate Gaussian fit: {e}")

    def plot_gauss_results(self):
        try:
            if np.isnan(self.flare_loc[0]):
                plot_gauss_fits(self.sc_data, self.path, self.start);
            else:
                plot_gauss_fits(self.sc_data, self.path, self.start, flare_loc=self.flare_loc);
        except Exception as e:
            print(f"Warning: Could not plot figure: {e}")







## Solar mach
def solarmach_loop(observers, dates, data_path, resampling, source_loc=[None,None], coord_sys='Stonyhurst'):
    """Downloads the fleet location data between the given dates with the
        given 'resampling' interval."""
    filename = f'SolarMACH_{dates[0].strftime("%d%m%Y")}_loop.csv'

    if filename in os.listdir(data_path):
        sm_loop = pd.read_csv(data_path+filename, index_col=0, header=[0,1], parse_dates=True, na_values='nan')
        return sm_loop

    # Set up the time list to iterate through
    starting_time = dt.datetime(dates[0].year, dates[0].month, dates[0].day, dates[0].hour, 0) - dt.timedelta(hours=2) #Gathering enough background too
    date_list = pd.date_range(starting_time, dates[1], freq=resampling)

    sm_df = {}
    for t in date_list:
        sm10 = SolarMACH(t, observers, vsw_list=[],
                         reference_long=source_loc[0],
                         reference_lat=source_loc[1],
                         coord_sys=coord_sys)
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
                loc=[float(tmp_df['Stonyhurst longitude (°)'][obs]), # JAX: what if carrington?
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
    """ Reads in the radial distance, location  [WE, NS],  measured Vsw,
        'towards' the Sun (True) or away (False), err_calc = True/False.
        Returns the longitude position at the requested radial distance."""

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

    full_channel_list = range(channel_list[0], channel_list[1]+1)
    time_indx = []
    merged_flux = []
    for i in df0.index:
        time_indx.append(i)

        row_flux = 0
        row_div = 0
        for n in range(len(full_channel_list)):
            if type(header_label)==list: #If its a double-layered header
                row_flux = row_flux + ( df0.loc[i, (header_label[0], f"{header_label[1]}{full_channel_list[n]}")] ) * binwidths[n]
            else:
                row_flux = row_flux + ( df0.loc[i, f"{header_label}{full_channel_list[n]}"] ) * binwidths[n]
            row_div = row_div + binwidths[n]
        merged_flux.append(float(row_flux / row_div))


    return merged_flux

def rms_mean(x_arr):
    """This function is used to 'resample' or average the uncertainty columns. Parts of this code were assisted by ChatGPT on 18Nov2025."""
    return np.sqrt(np.sum(x_arr**2)) / len(x_arr)



################################################

def load_sc_data(spacecraft, sm_df, proton_channels, dates, data_path, resampling='15min'):
    """Load the data, merge the bins, make omnidirectional, resample, and return one df:
        -index: times
        - header1: sc-ins
        - [Flux, Uncertainty, Radial Distance, Longitude]"""

    # Check if the file is already made and just load that one
    filename = f"SEP_intensities_{dates[0].strftime("%d%m%Y")}.csv"


    
    # Download the data and load into a dictionary with the key as the spacecraft-instrument label
    #sc_dict = {}
    spacecraft = spacecraft.lower()

    if 'psp' == spacecraft:
        psp_df, psp_meta = psp_isois_load(dataset='PSP_ISOIS-EPIHI_L2-HET-RATES60', # 'A_H_Flux_n' 'B_H_Uncertainty_n'
                                          startdate=dates[0], enddate=dates[1],
                                          path=data_path+'psp/', resample=None) # can do resample='1min' but its not clean.
        if JAX_TESTERS:
            psp_df = psp_df.resample('1min').mean() # Results in the index of "2021-05-28 00:20:00" a clean minute.
            psp_df.to_csv(data_path+'psp_rawdata.csv', na_rep='nan')

        # Find channels and bin widths
        bin_list = proton_channels['PSP']
        if len(bin_list)==1:
            bin_label = f"{bin_list[0]}"
            bin_list.append(bin_list[0])
        else:
            bin_label = f"{bin_list[0]}-{bin_list[1]}"
        bin_width = []
        for n in range(bin_list[0], bin_list[1]+1):
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
        psp = psp_df2.resample(resampling).agg({'Flux':'mean', 'Uncertainty': rms_mean}) # psp = psp_df2.resample(resampling).mean()
        psp_sm = pd.concat([psp, sm_df['PSP']], axis=1, join='outer')
        

        # Add to the collection
        #sc_dict['PSP'] = psp_sm
        return psp_sm



    if 'soho' == spacecraft:
        soho_df, soho_meta = soho_load(dataset='SOHO_ERNE-HED_L2-1MIN', # 'PH_n' 'PHC_n'
                                       startdate=dates[0], enddate=dates[1],
                                       path=data_path+'soho/', resample=None,
                                       pos_timestamp='start')
        if JAX_TESTERS:
            soho_df = soho_df.resample('1min').mean()
            soho_df.to_csv(data_path+'soho_rawdata.csv', na_rep='nan')

        # Find channels and bin widths
        bin_list = proton_channels['SOHO']
        if len(bin_list)==1:
            bin_label = f"{bin_list[0]}"
            bin_list.append(bin_list[0])
        else:
            bin_label = f"{bin_list[0]}-{bin_list[1]}"
        bin_width = []
        for n in range(bin_list[0], bin_list[1]+1):
            soho_meta = soho_meta['channels_dict_df_p']
            bin_start = soho_meta.loc[n, 'lower_E']
            bin_end = soho_meta.loc[n, 'upper_E']

            bin_width.append(bin_end - bin_start)


        # Calculate the uncertainty
        for n in bin_list: # data provided as intensities (PH) or counts/min (PHC)
            soho_df[f"Uncert_{n}"] = ( (soho_df[f"PH_{n}"]) / (np.sqrt(soho_df[f"PHC_{n}"])) ) * 1.1 # adding 10% uncertainty for systematic errors (according to RV)
        # print(soho_df.loc[test_date, f"PH_{n}"])
        # print(soho_df.loc[test_date, f"PHC_{n}"])
        # print(soho_df.loc[test_date, f"Uncert_{n}"])
        # jax = input("Hows the soho uncertainty calc?")


        # Merge the channels
        soho_df1 = {'Time': soho_df.index}
        soho_df1["Flux"] = weighted_bin_merge(soho_df, 'soho', 'p', bin_list, 'PH_', bin_width)
        soho_df1["Uncertainty"] = weighted_bin_merge(soho_df, 'soho', 'p', bin_list, 'Uncert_', bin_width)

        soho_df2 = pd.DataFrame.from_dict(soho_df1)
        soho_df2.set_index('Time', inplace=True)

        # print(soho_df.loc[test_date, "PH_0"])
        # print(bin_width)
        # print(soho_df2.loc[test_date, "Flux"])
        # jax = input("Hows the soho flux merge?")
        # print(soho_df.loc[test_date, "Uncert_0"])
        # print(soho_df2.loc[test_date, "Uncertainty"])
        # jax = input("Hows the soho uncertainty merge?")

        # Resample
        soho = soho_df2.resample(resampling).agg({'Flux':'mean', 'Uncertainty': rms_mean}) # soho = soho_df2.resample(resampling).mean()
        soho_sm = pd.concat([soho, sm_df['SOHO']], axis=1, join='outer')

        # print(soho_df2.head())
        # print(soho.head())
        # jax=input("Hows the resample?")

        # Add to the collection
        #sc_dict['SOHO'] = soho_sm
        return soho_sm

    if 'stereo-a' == spacecraft:
        stadf, sta_meta = stereo_load(instrument='HET', spacecraft='ahead', # 'Proton_Flux_n' 'Proton_Sigma_n'
                                       startdate=dates[0], enddate=dates[1],
                                       path=data_path+'stereo/', resample=None,
                                       pos_timestamp='start')
        if JAX_TESTERS:
            sta_df = stadf.resample('1min').mean()
            sta_df.to_csv(data_path+'sta_rawdata.csv', na_rep='nan')
        else:
            sta_df = stadf

        # Find channels and bin widths
        bin_list = proton_channels['STEREO-A']
        if len(bin_list)==1:
            bin_label = f"{bin_list[0]}"
            bin_list.append(bin_list[0])
        else:
            bin_label = f"{bin_list[0]}-{bin_list[1]}"
        bin_width = []
        for n in range(bin_list[0], bin_list[1]+1):
            sta_meta = sta_meta['channels_dict_df_p']
            bin_start = sta_meta.loc[n, 'lower_E']
            bin_end = sta_meta.loc[n, 'upper_E']

            bin_width.append(bin_end - bin_start)


        # Merge the channels
        sta_df1 = {'Time': sta_df.index}
        sta_df1["Flux"] = weighted_bin_merge(sta_df, 'sta', 'p', bin_list, 'Proton_Flux_', bin_width)
        sta_df1["Uncertainty"] = weighted_bin_merge(sta_df, 'sta', 'p', bin_list, 'Proton_Sigma_', bin_width)

        sta_df2 = pd.DataFrame.from_dict(sta_df1)
        sta_df2.set_index('Time', inplace=True)

        # print(sta_df.loc[test_date, "Proton_Flux_0"])
        # print(bin_width)
        # print(sta_df2.loc[test_date, "Flux"])
        # jax = input("Hows the sta flux merge?")
        # print(sta_df.loc[test_date, "Proton_Sigma_0"])
        # print(sta_df2.loc[test_date, "Uncertainty"])
        # jax = input("Hows the sta uncertainty merge?")

        # Resample
        sta = sta_df2.resample(resampling).agg({'Flux':'mean', 'Uncertainty': rms_mean}) # sta = sta_df2.resample(resampling).mean()
        sta_sm = pd.concat([sta, sm_df['STEREO-A']], axis=1, join='outer')

        # print(sta_df2.head())
        # print(sta.head())
        # jax=input("Hows the resample?")

        # Add to the collection
        #sc_dict['STEREO-A'] = sta_sm
        return sta_sm


    if 'solar orbiter' == spacecraft:
        solo_dfp, solo_dfe, solo_meta = epd_load(sensor='het', level='l2', # [('H_Flux','H_Flux_n')] [('H_Uncertainty','H_Uncertainty_n')]
                                      startdate=dates[0], enddate=dates[1],
                                      viewing='omni', autodownload=True,
                                      pos_timestamp='start', path=data_path+'solo/')

        if JAX_TESTERS:
            solo_df1 = solo_dfp.resample('1min').mean()
            solo_df1.to_csv(data_path+'solo_rawdata.csv', na_rep='nan')
        else:
            solo_df1 = solo_dfp

        # Find channels and bin widths
        bin_list = proton_channels['Solar Orbiter']
        if len(bin_list)==1:
            bin_label = f"{bin_list[0]}"
            bin_list.append(bin_list[0])
        else:
            bin_label = f"{bin_list[0]}-{bin_list[1]}"
        bin_width = []
        for n in range(bin_list[0], bin_list[1]+1):
            bin_width.append(solo_meta['H_Bins_Width'][n])

        # Merge the channels
        solo_df2 = {'Time': solo_df1.index}
        solo_df2["Flux"] = weighted_bin_merge(solo_df1, 'solo', 'p', bin_list, ['H_Flux','H_Flux_'], bin_width)
        solo_df2["Uncertainty"] = weighted_bin_merge(solo_df1, 'solo', 'p', bin_list, ['H_Uncertainty','H_Uncertainty_'], bin_width)

        solo_df3 = pd.DataFrame.from_dict(solo_df2)
        solo_df3.set_index('Time', inplace=True)

        # print('Check channel merge')
        # print(solo_df1.loc[test_date, ('H_Flux','H_Flux_10')])
        # print(solo_df1.loc[test_date, ('H_Flux','H_Flux_11')])
        # print(solo_df1.loc[test_date, ('H_Flux','H_Flux_12')])
        # print(bin_width)
        # print(solo_df3.loc[test_date, 'Flux'])
        # jax=input('How is solos channel merge?')

        # print(solo_df1.loc[test_date, ('H_Uncertainty','H_Uncertainty_10')])
        # print(solo_df1.loc[test_date, ('H_Uncertainty','H_Uncertainty_11')])
        # print(solo_df1.loc[test_date, ('H_Uncertainty','H_Uncertainty_12')])
        # print(bin_width)
        # print(solo_df3.loc[test_date, 'Uncertainty'])
        # jax=input('How is solos unc channel merge?')

        # Resample
        solo = solo_df3.resample(resampling).agg({'Flux':'mean', 'Uncertainty': rms_mean}) # solo = solo_df3.resample(resampling).mean()
        solo_sm = pd.concat([solo, sm_df['Solar Orbiter']], axis=1, join='outer')

        # print(solo_df3.head())
        # print(solo.head())
        # jax=input('How is the solo resample?')

        # Add to the collection
        #sc_dict['Solar Orbiter'] = solo_sm
        return solo_sm


    # Merge all the relevant columns into one df
    #print(sc_dict)


    #sc_df = pd.concat(sc_dict, axis=1, join='outer')
    #sc_df.to_csv(data_path + filename, na_rep='nan')

    #return sc_df


################################################
## Inter-calibration
################################################
def intercalibration_calculation(df, factor):
    """Apply the scaling to the Flux and Uncertainty columns"""
    for col in ['Flux','Uncertainty']: # Both are calculated the same
        #print(df[(obs,col)])
        #jax=input('huh?')
        df[col] *= factor

    #df.to_csv(f"{data_path}SEP_intensities_{dates[0].strftime("%d%m%Y")}_IC.csv", na_rep='nan') # Save for sanity checks

    return df

################################################
## Radial Scaling
################################################
def radial_scaling_calculation(df, scaling_values):
    """Scales the data based on its radial distance according to:
        I_new = I * R ^(a +- b)."""
    #df = df0.copy(deep=True) # so it doesnt mess with the OG df's
    
    a = scaling_values[0]
    b = scaling_values[1]

    for t in df.index:
        #print(df.loc[t, 'Flux')])
        if pd.isna(df.loc[t, 'Flux']) or pd.isna(df.loc[t, 'r_dist'])\
            or (df.loc[t, 'Flux']==0) or (df.loc[t, 'r_dist']==0):
            f_rscld = np.nan
            unc_final = np.nan
            
        else:
            # Scale the flux
            f_rscld = df.loc[t, 'Flux'] * (df.loc[t, 'r_dist'] ** a)

            # Scale the uncertainty
            ## Find the difference from the boundaries
            unc_plus = df.loc[t, 'Flux'] * (df.loc[t, 'r_dist'] **(a+b))
            unc_limit_plus = abs(f_rscld - unc_plus)
            unc_minus = df.loc[t, 'Flux'] * (df.loc[t, 'r_dist'] **(a-b))
            unc_limit_minus = abs(f_rscld - unc_minus)
    
            if (unc_limit_plus >= unc_limit_minus) and (f_rscld - unc_limit_plus > 0):
                chosen_unc_limit = unc_limit_plus
            elif (unc_limit_minus >= unc_limit_plus) and (f_rscld - unc_limit_minus > 0):
                chosen_unc_limit = unc_limit_minus
            else: # JAX TO FIX
                print("There's a problem with the limits")
                print("OG flux: ", df.loc[t, 'Flux'])
                print("OG rad: ", df.loc[t, 'r_dist'])
                print("Scaled Flux: ", f_rscld)
                print("Unc plus: ", unc_limit_plus)
                print("Unc minus: ", unc_limit_minus)
                jax = input('Continue? ')
                chosen_unc_limit = np.nan
    
            ## Find the calculated scaled uncertainty
            unc_calculated = df.loc[t, 'Uncertainty'] * (df.loc[t, 'r_dist'] ** a)
    
            ## Merge both results for the final scaled uncertainty
            unc_final = np.sqrt(((unc_calculated)**2) + ((chosen_unc_limit)**2))

        df.loc[t, 'Flux'] = f_rscld
        df.loc[t, 'Uncertainty'] = unc_final


    #df.to_csv(f"{data_path}SEP_intensities_{dates[0].strftime("%d%m%Y")}_RS.csv", na_rep='nan') # Save for sanity checks
    return df





################################################
## Background Subtracting
################################################
def background_subtracting(df, data_path, background_window):
    """Input: the sc dataframe, the path for saving files, the background window.
    Process: 1. Find the nanmean  and nanstd of the background window for both flux and unc.
            2. Subtract the full column by the [avg - std] (this way there's less chance to get half the values as nan).
                - The unc is calculated as: unc_adj = sqrt(unc**2 + [avg-std]**2)
            3. Return the updated df."""

    # A list of just the values within the background window
    bg_flux = df['Flux'][background_window[0]:background_window[1]]
    bg_func = df['Uncertainty'][background_window[0]:background_window[1]]

    # Find the nanmean and nanstd
    f_bg_avg = float(np.nanmean(bg_flux))
    u_bg_avg = rms_mean(bg_func) # Calculate the average background using root-mean-square function

    #print('flux bg avg - std: ', f_bg_avg)
    #print(bg_func)
    #print('unc bg avg - std: ', u_bg_avg)
    #jax=input('good?')

    # Adjust the whole column
    adj_flux = (df.loc[:, 'Flux']) - f_bg_avg
    adj_func = np.sqrt( ((df.loc[:, 'Uncertainty'])**2) + (u_bg_avg**2) )

    # Replace any negative results with 'nan'
    adj_flux = adj_flux.where(adj_flux>=0, np.nan) # new list = old list where the values are >=0, else make the value 'nan'
    adj_func = adj_func.where(adj_func>=0, np.nan)

    # Put the adjusted column in
    df['Flux'] = adj_flux
    df['Uncertainty'] = adj_func

    #df.to_csv(data_path+'backsubtest.csv', na_rep='nan')

    return df



################################################
## Gaussian fitting
################################################
def gauss_function(x, A, x0, sigma):
    """Function to calculate the Gaussian curve for scipy's methods."""
    return A * np.exp( -((x-x0)**2) / (2 * sigma**2) )

def log_gauss_function(x, logA, x0, sigma):
    """Formula to calculate the Gaussian curve to data that has been log-scaled."""
    return logA - ( (x - x0)**2 ) / ( 2 * np.log(10) * sigma**2 )


def log_gauss_function_beta(beta_params, x):
    """The same as log_gauss_function but 'curve_fit' and 'odr' require a different
    ordering of the parameters."""
    a, x0, sigma = beta_params
    result = log_gauss_function(x, a, x0, sigma)
    return result

def odr_gauss_fit(dict_1timestep, prev_results):
    """A more reliable and complex fitting function that requires the
    initial parameters to be within reason, so it calls the basic fitting function.
    Input: one timesteps worth of values, and the results of the previous timesteps fit."""

    df = pd.DataFrame(dict_1timestep)

    # Remove zero and nan values
    for i, row in df.iterrows():
        if (row['y'] == 0.0) or (np.isnan(row['y'])):
            df.drop([i], inplace=True) # Drops the whole row

    # Must have at least 3 data points to calculate
    if len(df['y']) < 3:
        return {'A': np.nan, 'X0': np.nan, 'sigma': np.nan}

    # Check for previous timestep results (if none then calculate)
    if np.isnan(prev_results['A']): # Nothing stored
        try:
            popt, pcov = curve_fit(log_gauss_function, df['x'], df['y'],
                                  p0=[max(df['y']), df['x'][ df['y'].idxmax() ], 20]) # max amplitude, position of the max amplitude, width
            prev_results = {'A': float(popt[0]), 'X0': float(popt[1]), 'sigma': float(popt[2])}
        except:
            print("No updated Gaussian parameters were found")
            prev_results = {'A': np.nan, 'X0': np.nan, 'sigma': np.nan}

    # Set up the ODR functions
    odr_model = odr.Model(log_gauss_function_beta)
    if np.isnan(df['yerr']).any(): # If there are any nan's then it won't calculate properly
        odr_data = odr.RealData(x=df['x'], y=df['y'], sx=df['xerr'])
    else:
        odr_data = odr.RealData(x=df['x'], y=df['y'], sx=df['xerr'], sy=df['yerr'])

    odr_setup = odr.ODR(odr_data, odr_model, beta0=[prev_results['A'], prev_results['X0'], prev_results['sigma']])
    out = odr_setup.run()

    # Confirm that ODR found a fitted curve
    stopreason = []
    for reasons in out.stopreason:
        if 'convergence' in reasons:
            stopreason.append('pass')
        else:
            stopreason.append('fail')

    if 'pass' not in stopreason:
        return {'A': np.nan, 'X0': np.nan, 'sigma': np.nan,
                'A err': np.nan, 'X0 err': np.nan, 'sigma err': np.nan, 'res':np.nan}

    return {'A': float(out.beta[0]), 'X0': float(out.beta[1]), 'sigma': float(out.beta[2]),
            'A err': float(out.sd_beta[0]), 'X0 err': float(out.sd_beta[1]), 'sigma err': float(out.sd_beta[2]),
            'res': float(out.res_var)}

def fit_gauss_curves_to_data(sc_dict, data_path):
    """Read in the full df, calculate the curve at each timestep, save the results to new columns."""
    # Create a folder to save the gaussian timestep figures in
    try:
        os.makedirs(data_path+'Gauss_fits/')
    except FileExistsError:
        pass

    a_arr, a_errarr, x0_arr, x0_errarr, sigma_arr, sigma_errarr, res_arr, time_arr, sc_arr = ([] for i in range(9))

    # Set up a dict for the previous Gaussian results to be stored and compared.
    prev_gauss = {'A': np.nan, 'X0': np.nan, 'sigma': np.nan}

    obs = list(sc_dict.keys())

    for i in sc_dict[obs[0]].index: # iterate through each timestep
        #if i.day != 29: #if np.isnan(sc_dict[obs[0]]['r_dist']): # Doesn't try calculate when solarmach data is not available
        #    continue
        x, xerr, y, yerr, sc_arr = ([] for i in range(5))
        #print(i)

        for sc, df in sc_dict.items():
            sc_arr.append(sc)
            y.append(float(np.log10(df.loc[i, 'Flux'])))
            yerr.append(float(np.log10(df.loc[i, 'Uncertainty'])))
            x.append(float(df.loc[i, 'foot_long']))
            xerr.append(float(df.loc[i, 'foot_long_error']))

        # Calculate the fit now
        #print(x)
        #print(y)
        timestep_dict = {'x':x, 'y':y, 'sc':sc_arr, 'xerr':xerr, 'yerr':yerr}
        #gauss_results = basic_gauss_fit(timestep_dict)
        gauss_results = odr_gauss_fit(timestep_dict, prev_gauss)
        #print(gauss_results)
        a_arr.append( gauss_results['A'] )
        x0_arr.append( gauss_results['X0'] )
        sigma_arr.append( gauss_results['sigma'] )
        time_arr.append(i)
        #jax=input()
        if 'A err' in gauss_results.keys():
            a_errarr.append(gauss_results['A err'])
            x0_errarr.append(gauss_results['X0 err'])
            sigma_errarr.append(gauss_results['sigma err'])
            res_arr.append(gauss_results['res'])
        else:
            a_errarr.append(np.nan)
            x0_errarr.append(np.nan)
            sigma_errarr.append(np.nan)
            res_arr.append(np.nan)

        # Plot the fit for this timestep
        if not np.isnan(x).any() and not np.isnan(gauss_results['X0']):
            plot_curve_and_timeseries(gauss_results, timestep_dict, sc_dict, data_path+'Gauss_fits/', i)

        prev_gauss = gauss_results

    # Build the results df
    fits_dict = {'Time': time_arr}
    fits_dict['A'] = a_arr
    fits_dict['X0'] = x0_arr
    fits_dict['sigma'] = sigma_arr
    fits_dict['A err'] = a_errarr
    fits_dict['X0 err'] = x0_errarr
    fits_dict['sigma err'] = sigma_errarr
    fits_dict['res'] = res_arr

    fits_df = pd.DataFrame(fits_dict)
    fits_df.set_index('Time', inplace=True)

    fits_df.to_csv(data_path+'Gaussian_fit_results.csv')

    # Combine all figures into a gif to display
    png_dir = data_path + 'Gauss_fits/'
    images = []
    for filename in sorted(os.listdir(png_dir)):
        if filename.endswith('.png'):
            filepath = os.path.join(png_dir, filename)
            images.append(imageio.imread(filepath))

    imageio.mimsave(png_dir+'Gauss_fits.gif', images, duration=150, loop=0)

    gif = Image(filename=png_dir+"Gauss_fits.gif")
    display(gif)

    return fits_df



################################################
## Plotters
################################################

def plot_timeseries_result(sc_dict, data_path, date, background_window=[]):
    """Plots the time series for each observer."""
    # Determine how many subplots are needed
    obs = list(sc_dict.keys())
    #print(obs)

    fig, ax = plt.subplots(len(obs), 1, figsize=[5, len(obs)+2.5], dpi=300, sharex=True)
    fig.subplots_adjust(hspace=0.02)

    # Title
    ax[0].set_title(date.strftime("%H:%M - %d %b, %Y"), pad=9, loc='left')
    fig.supylabel('Intensity')

    for n, sc in enumerate(obs):
        #sc = obs[n]
        mrkr = marker_settings[sc]

        # Show the event start time
        ax[n].axvline(x=date, color='k', linestyle='dashed', linewidth=0.5)

        # Show the background window
        if len(background_window) > 1:
            ax[n].axvspan(background_window[0], background_window[1], alpha=0.2, color='grey')
            bg_txt = f'Background window:\n{background_window[0].strftime("%H:%M")} - {background_window[0].strftime("%H:%M %d %b, %Y")}'
            box_obj = AnchoredText(bg_txt, frameon=True, loc='lower right',
                                   pad=0.5, prop={'size':7})
            plt.setp(box_obj.patch, facecolor='grey', alpha=0.9)
            ax[0].add_artist(box_obj)

        # Plot the data
        ax[n].semilogy(sc_dict[sc]['Flux'], color=mrkr['color'],
                       label=mrkr['label'], linestyle='solid')
        ax[n].fill_between(x = sc_dict[sc].index,
                           y1= sc_dict[sc]['Flux'] - sc_dict[sc]['Uncertainty'],
                           y2= sc_dict[sc]['Flux'] + sc_dict[sc]['Uncertainty'],
                           alpha=0.3, color=mrkr['color'])
        ax[n].legend(loc='upper right', alignment='left')
        ax[n].yaxis.set_major_locator(mpl.ticker.LogLocator(base=10, numticks=3))
        ax[n].minorticks_on()
        #ax[n].set_ylim(1e-5,5e0) # JAX TO FIX

    xmin = date - dt.timedelta(hours=5)
    xmax = date + dt.timedelta(hours=22)
    ax[0].set_xlim([xmin,xmax])
    locator = mpl.dates.AutoDateLocator(minticks=3, maxticks=6)
    ax[n-1].xaxis.set(major_locator=locator, )
    ax[n-1].xaxis.set_major_formatter(mpl.dates.ConciseDateFormatter(locator, show_offset=False))

    label=''
    if False: #input('Save the file? ')=='y':
        label = '_' + input('Save file key word: ')
        plt.savefig(data_path+f'SEP_Intensities_{date.strftime("%d%m%y")}{label}.png')
    plt.show()

    #plt.clf()


def find_and_plot_peak_intensity(sc_dict, data_path, date):
    """Reading in all the spacecraft datasets, this function finds the peak of each dataset
        and fits a Gaussian curve to the resulting set of peaks."""

    # Select a reasonable window for the peak to be in
    peak_window_end = date + dt.timedelta(hours=10)


    # Iterate through each spacecraft df and save the intensity and datetime of the peak
    peak_y, peak_yerr, peak_x, peak_xerr, peak_time, peak_sc = ([] for i in range(6))
    for sc, sc_df in sc_dict.items():
        peak_index = sc_df.loc[date:peak_window_end, 'Flux'].idxmax()

        peak_time.append(peak_index)
        peak_sc.append(sc)
        peak_y.append(float(np.log10(sc_df.loc[peak_index, 'Flux'])))
        peak_yerr.append(float(np.log10(sc_df.loc[peak_index, 'Uncertainty'])))
        peak_x.append(float(sc_df.loc[peak_index, 'foot_long']))
        peak_xerr.append(float(sc_df.loc[peak_index, 'foot_long_error']))

    # Fit the Gaussian curve to the data
    peak_data_dict = {'sc': peak_sc,
                      'times': peak_time,
                      'y': peak_y,
                      'yerr': peak_yerr,
                      'x': peak_x,
                      'xerr': peak_xerr}
    peak_fit_results = odr_gauss_fit(peak_data_dict, {'A': np.nan, 'X0': np.nan, 'sigma': np.nan}) #JAX: can prolly provide better beta's'
    print(peak_fit_results)
    print(peak_data_dict)

    # Plot
    fig = plt.figure(figsize=[10,3], dpi=250)
    plt.axis('off')

    grid = plt.GridSpec(nrows=1, ncols=3, wspace=0.03)

    gauss_ax = fig.add_subplot(grid[0,0])
    tseries_ax = fig.add_subplot(grid[0,1:], sharey=gauss_ax)

    gauss_ax.set_ylabel('Intensity')
    gauss_ax.set_xlabel('Footprint Longitude')
    tseries_ax.set_xlabel('Time & Date')

    # Add a text box with the energy and species - JAX: NEEDS TO BE ADAPTABLE
    box_obj = AnchoredText('Peak Fits\n14 MeV Protons',
                           frameon=True, loc='lower right', pad=0.5, prop={'size':9})
    plt.setp(box_obj.patch, facecolor='grey', alpha=0.9)
    tseries_ax.add_artist(box_obj)

    ylimits = [1e5,1e-5]
    xlimits = [0,0]

    for n, sc in enumerate(peak_sc):
        mrkr = marker_settings[sc]
        ylimits[0] = np.nanmin( [ np.nanmin(sc_dict[sc]['Flux']), ylimits[0] ] )
        ylimits[1] = np.nanmax( [ np.nanmax(sc_dict[sc]['Flux']), ylimits[1] ] )
        xlimits[0] = np.nanmin( [ np.nanmin(sc_dict[sc]['foot_long']), xlimits[0] ] )
        xlimits[1] = np.nanmax( [ np.nanmax(sc_dict[sc]['foot_long']), xlimits[1] ] )

        # Plot the markers in both plots to indicate the point values being used
        gauss_ax.errorbar(peak_x[n], 10**(peak_y[n]),
                          xerr=peak_xerr[n], yerr=10**(peak_yerr[n]),
                          color=mrkr['color'], ecolor=mrkr['color'],
                          marker=mrkr['marker'])#, label=f"{mrkr['label']} ({peak_time[n].strftime("%H:%M %d %b. %y")})")

        tseries_ax.plot(peak_time[n], 10**(peak_y[n]),
                        color=mrkr['color'], marker=mrkr['marker'],
                        label=f"{mrkr['label']} ({peak_time[n].strftime("%H:%M %d %b. %y")})")

        # Plot the full time series
        tseries_ax.semilogy(sc_dict[sc]['Flux'], color=mrkr['color'])#, label=mrkr['label'])

    tseries_ax.legend(loc='upper right')

    # Plot the Gaussian Curve
    x_curve = np.linspace(-180, 180, 200) # JAX: must be adapted
    y_curve = 10**(log_gauss_function(x_curve, peak_fit_results['A'], peak_fit_results['X0'], peak_fit_results['sigma']))

    gauss_ax.semilogy(x_curve, y_curve, color='k')
    gauss_ax.axvline(x=peak_fit_results['X0'], color='goldenrod', linewidth=1.2, alpha=0.8)
    gauss_ax.hlines(y=0.6065*(10**peak_fit_results['A']),
                    xmin=(peak_fit_results['X0']-peak_fit_results['sigma']),
                    xmax=(peak_fit_results['X0']+peak_fit_results['sigma']),
                    color='turquoise', linewidth=1.2, alpha=0.8)

    # Provide error range
    #yerr_curve =

    gauss_ax.set_xlim(xlimits)
    gauss_ax.set_ylim(ylimits)

    plt.show()








def plot_curve_and_timeseries(gauss_values, sc_df, full_df, data_path, timestep):
    """Plotting two subplots, left the fitted gaussian curve, right the time series."""


    ylimits = [1e5, 1e-5]
    xlimits= [0, 0]
    for sc in ['SOHO','PSP','STEREO-A','Solar Orbiter']:
        ylimits[0] = np.nanmin([np.nanmin(full_df[sc]['Flux']), ylimits[0]])
        ylimits[1] = np.nanmax([np.nanmax(full_df[sc]['Flux']), ylimits[1]])

        xlimits[0] = np.nanmin([np.nanmin(full_df[sc]['foot_long']), xlimits[0] ])
        xlimits[1] = np.nanmax([np.nanmax(full_df[sc]['foot_long']), xlimits[1] ])

    fig = plt.figure(figsize=[10,3], dpi=250)
    plt.axis('off')

    grid = plt.GridSpec(nrows=1, ncols=3, wspace=0.03)

    gauss_ax = fig.add_subplot(grid[0,0])
    tseries_ax = fig.add_subplot(grid[0,1:], sharey=gauss_ax)

    gauss_ax.set_ylabel('Intensity')
    gauss_ax.set_xlabel('Footprint Longitude')
    tseries_ax.set_xlabel('Time & Date')

    # Add a text box with the energy and species
    box_obj = AnchoredText('14 MeV Protons\n'+timestep.strftime("%H:%M %d %b %Y"),
                           frameon=True, loc='lower right', pad=0.5, prop={'size':9})
    plt.setp(box_obj.patch, facecolor='grey', alpha=0.9)
    tseries_ax.add_artist(box_obj)

    # Add a vertical line in the time series to see where the timestamp is
    tseries_ax.axvline(x=timestep, color='k', linewidth=1.2, alpha=0.8)

    # Calculate and plot the curve
    x_curve = np.linspace(-360,360,200)
    y_curve = 10 ** log_gauss_function(x_curve, gauss_values['A'], gauss_values['X0'], gauss_values['sigma'])
    gauss_ax.semilogy(x_curve, y_curve)

    # Show the different elements of the curve (ie the center and width)
    gauss_ax.axvline(x=gauss_values['X0'], color='goldenrod', alpha=0.8)
    gauss_ax.hlines(y=0.6065*(10**gauss_values['A']),
                    xmin=(gauss_values['X0']-gauss_values['sigma']),
                    xmax=(gauss_values['X0']+gauss_values['sigma']),
                    color='turquoise', alpha=0.8, linewidth=1.2)

    gauss_text = f"Center: {gauss_values['X0']:.2f}{DEGREE_TEXT}\n"
    gauss_text = f"{gauss_text}Width: {gauss_values['sigma']:.2f}{DEGREE_TEXT}"
    box_obj = AnchoredText(gauss_text, frameon=True, loc='upper left', pad=0.5, prop={'size':9})
    plt.setp(box_obj.patch, facecolor='grey', alpha=0.9)
    gauss_ax.add_artist(box_obj)


    for n in range(len(sc_df['sc'])):
        markers = marker_settings[sc_df['sc'][n]]
        gauss_ax.semilogy(sc_df['x'][n], 10**(sc_df['y'][n]), label=sc_df['sc'][n],
                          marker=markers['marker'], color=markers['color'])

        # Plot the timeseries data
        tseries_ax.semilogy(full_df[sc_df['sc'][n]]['Flux'],
                            color=markers['color'], label=sc_df['sc'][n])
    gauss_ax.legend(bbox_to_anchor=(0.2, 1.03, 1.0, 0.1), loc='upper left', ncols=6, fontsize=6)

    gauss_ax.set_xlim(xlimits[0]-10, xlimits[1]+10)
    tseries_ax.set_ylim(ylimits[0]*0.5, ylimits[1]*2)
    if ylimits[0] < 0:
        print(ylimits)
        jax = input()
    #tseries_ax.set_xlim(eruption_dt - dt.timedelta(hours=1), eruption_dt + dt.timedelta(days=1)) # Need flare time for this
    tseries_ax.xaxis.set_major_formatter(
        mpl.dates.ConciseDateFormatter(tseries_ax.xaxis.get_major_locator(),
                                       show_offset=False))
    tseries_ax.tick_params(bottom=True, labelbottom=True,
                           left=False, labelleft=False)


    plt.savefig(data_path+f'GaussCurve_TimeSeries_{timestep.strftime("%d%b%Y_%Hh%M")}.png')
    plt.clf()





def plot_gauss_fits(sc_dict, data_path, date, **kwargs):
    """Plots the following time series: intensity, gauss center, and gauss sigma."""

    fig, ax = plt.subplots(3, 1, figsize=[6,6], dpi=300, sharex=True)
    plt.subplots_adjust(hspace=0.02)

    ax[0].set_title(date.strftime("%H:%M - %d %b, %Y"), pad=20, loc='left')
    ax[0].set_ylabel('Intensity')
    ax[1].set_ylabel(r'Gauss $X_0$')
    ax[2].set_ylabel(r'Gauss $\sigma$')


    # Define the limits to be adjusted along the way
    ylimits = {'intensity': [1e0, 1e-5],
               'center': [np.nanmin(sc_dict['Gauss']['X0']), np.nanmax(sc_dict['Gauss']['X0'])],
               'sigma': [np.nanmin(sc_dict['Gauss']['sigma']), np.nanmax(sc_dict['Gauss']['sigma'])]}

    # Plot the intensities
    for sc, s_df in sc_dict.items():
        if sc == 'Gauss': # This is the df of gaussian fits
            continue
        mrkr = marker_settings[sc]

        ax[0].semilogy(s_df['Flux'], color=mrkr['color'], label=mrkr['label'])
        ax[0].fill_between(x = s_df.index,
                           y1= s_df['Flux'] - s_df['Uncertainty'],
                           y2= s_df['Flux'] + s_df['Uncertainty'],
                          alpha=0.3, color=mrkr['color'])

        ylimits['intensity'][0] = np.nanmin([ylimits['intensity'][0], np.nanmin(s_df['Flux'])] )
        ylimits['intensity'][1] = np.nanmax([ylimits['intensity'][1], np.nanmax(s_df['Flux'])] )

    ax[0].legend(loc='upper center', ncols=5, bbox_to_anchor=(0.5,1.15))


    # Plot the flare location if its provided
    if 'flare_loc' in kwargs.keys():
        flare = kwargs['flare_loc']
        ax[1].axhline(y=flare[0],
                      linestyle='solid', color='k', alpha=0.5,
                      label=f"Flare at [{flare[0]}, {flare[1]}]{DEGREE_TEXT}")
        ax[1].legend()

    # Plot the Gaussian results
    for i, row in sc_dict['Gauss'].iterrows():
        ax[1].errorbar(i, row['X0'], yerr=row['X0 err'],
                      color='k', ecolor='grey', marker='o', markersize=3)

        ax[2].errorbar(i, row['sigma'], yerr=row['sigma err'],
                      color='k', ecolor='grey', marker='o', markersize=3)


    for n in range(3):
        # Mark the event start time
        ax[n].axvline(x=date, color='k', linestyle='solid', linewidth=0.5)


    ax[0].set_ylim([ylimits['intensity'][0]*0.5, ylimits['intensity'][1]*1.8])
    ax[1].set_ylim([ylimits['center'][0]-20, ylimits['center'][1]+20])
    ax[2].set_ylim([ylimits['sigma'][0]-20, ylimits['sigma'][1]+20])

    ax[2].set_xlim([date - dt.timedelta(hours=1), date + dt.timedelta(hours=20)])
    ax[2].xaxis.set_major_formatter(
        mpl.dates.ConciseDateFormatter(ax[1].xaxis.get_major_locator(),
                                       show_offset=False))


    plt.savefig(f"{data_path}Intensity_Gauss_TimeProfiles.png", bbox_inches='tight')
    plt.show()





