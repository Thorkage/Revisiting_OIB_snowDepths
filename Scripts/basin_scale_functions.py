import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from pyproj import Transformer
from tqdm import tqdm

def get_w99_snowdepth(x, y,  month='April'):
    transformer2 = Transformer.from_crs(3413, 4326, always_xy=True)
    lon, lat = transformer2.transform(x,y)
    lon, lat = np.deg2rad(lon), np.deg2rad(lat)
    
    if month == 'April':
        H0 = 36.80
        A = 0.4046
        B = -0.4005
        C = 0.0256
        D = 0.0024
        E = -0.0641
        eps =  9.4
        F = -0.09
        sigma_F = 0.09 
        IAV = 6.1
        
        snow_depth = H0 + A*lon + B*lat + C*lon*lat + D*lon**2 + E*lat**2
        return snow_depth, eps
    
    
def apply_uncertaintes_cwt(wavelet_dict, loaded_uncertainties):
    max_key = max(loaded_uncertainties.keys())
    max_uncertainty = loaded_uncertainties[max_key]
    
    for date in wavelet_dict.keys():
        wavelet_dict[date]['snow_depth_uncertainty'] = wavelet_dict[date]['snow_depth'].apply(
            lambda sd: next((unc for depth, unc in sorted(loaded_uncertainties.items()) if sd <= depth), max_uncertainty)
        )
    return wavelet_dict


def apply_uncertaintes_peak(wavelet_dict, loaded_uncertainties):
    max_key = max(loaded_uncertainties.keys())
    max_uncertainty = loaded_uncertainties[max_key]
    
    for date in wavelet_dict.keys():
        wavelet_dict[date]['snow_depth_peakiness_uncertainty'] = wavelet_dict[date]['snow_depth_peak'].apply(
            lambda sd: next((unc for depth, unc in sorted(loaded_uncertainties.items()) if sd <= depth), max_uncertainty)
        )
    return wavelet_dict



def calc_ice_thickness(snow_depth, freeboard, rhoW=1024, rhoI=915, rhoS=320):
    ice_thickness = (rhoW/(rhoW-rhoI)) * freeboard - ((rhoW-rhoS)/(rhoW-rhoI) * snow_depth)
    return ice_thickness

def calc_ice_thickness_uncertainty(snow_depth, freeboard, snow_depth_unc, freeboard_unc, sigmaI=10, sigmaS=100, rhoW=1024, rhoI=915, rhoS=320):
    ice_thickness_uncertainty = np.sqrt((rhoW/(rhoW-rhoI))**2 * freeboard_unc**2 +
                                             ((rhoS-rhoW)/(rhoW-rhoI))**2 * snow_depth_unc**2 +
                                             ((snow_depth*(rhoS-rhoW) + freeboard*rhoW)/((rhoW- rhoI)**2))**2 * sigmaI**2 +
                                             ((snow_depth)/(rhoW-rhoI))**2 * sigmaS**2
                                             )
    return ice_thickness_uncertainty



def resample_native_to_qlook(wavelet_dict, qlook_dict, dates):
    
    wavelet_dict_resampled = {}
    
    for date in tqdm(dates):
        # Get the coordinates of the qlook data points
        qlook_coords = np.column_stack((qlook_dict[date]['x'], qlook_dict[date]['y']))
        
        # Interpolate snow_depth from wavelet_dict to qlook_dict coordinates
        wavelet_dict_resampled[date] = qlook_dict[date].copy()
        
        wavelet_dict_resampled[date]['cwt_snow_depth'] = griddata(
            (wavelet_dict[date]['x'], wavelet_dict[date]['y']),
            wavelet_dict[date]['snow_depth'],
            qlook_coords,
            method='linear'
        )
        
        wavelet_dict_resampled[date]['cwt_snow_depth_uncertainty'] = griddata(
            (wavelet_dict[date]['x'], wavelet_dict[date]['y']),
            wavelet_dict[date]['snow_depth_uncertainty'],
            qlook_coords,
            method='linear'
        )
        
        wavelet_dict_resampled[date]['cwt_ATM_classes'] = griddata(
            (wavelet_dict[date]['x'], wavelet_dict[date]['y']),
            wavelet_dict[date]['ATM_classes'],
            qlook_coords,
            method='linear'
        )
        
        #PEAK
        wavelet_dict_resampled[date]['peak_snow_depth'] = griddata(
            (wavelet_dict[date]['x'], wavelet_dict[date]['y']),
            wavelet_dict[date]['snow_depth_peak'],
            qlook_coords,
            method='linear'
        )
        
        wavelet_dict_resampled[date]['peak_snow_depth_uncertainty'] = griddata(
            (wavelet_dict[date]['x'], wavelet_dict[date]['y']),
            wavelet_dict[date]['snow_depth_peakiness_uncertainty'],
            qlook_coords,
            method='linear'
        )
        
        wavelet_dict_resampled[date]['MYI_concentration'] = griddata(
            (wavelet_dict[date]['x'], wavelet_dict[date]['y']),
            wavelet_dict[date]['MYI_concentration'],
            qlook_coords,
            method='linear'
        )
        
        wavelet_dict_resampled[date]['cwt_ice_thickness'] = calc_ice_thickness(wavelet_dict_resampled[date]['cwt_snow_depth'], wavelet_dict_resampled[date]['ATM_fb'])
        wavelet_dict_resampled[date]['cwt_ice_thickness_uncertainty'] = calc_ice_thickness_uncertainty(wavelet_dict_resampled[date]['cwt_snow_depth'], 
                                                                                           wavelet_dict_resampled[date]['ATM_fb'],
                                                                                           wavelet_dict_resampled[date]['cwt_snow_depth_uncertainty'],
                                                                                           wavelet_dict_resampled[date]['fb_unc']
                                                                                           )
        
        wavelet_dict_resampled[date]['peak_ice_thickness'] = calc_ice_thickness(wavelet_dict_resampled[date]['peak_snow_depth'], wavelet_dict_resampled[date]['ATM_fb'])
        wavelet_dict_resampled[date]['peak_ice_thickness_uncertainty'] = calc_ice_thickness_uncertainty(wavelet_dict_resampled[date]['peak_snow_depth'], 
                                                                                           wavelet_dict_resampled[date]['ATM_fb'],
                                                                                           wavelet_dict_resampled[date]['peak_snow_depth_uncertainty'],
                                                                                           wavelet_dict_resampled[date]['fb_unc']
                                                                                           )


    return wavelet_dict_resampled

def rolling_average_by_distance(df, columns, distance):
    result = pd.DataFrame(index=df.index, columns=columns)
    for i in range(df.shape[0]):
        start_distance = df['along_track_distance'].iloc[i]
        mask = (df['along_track_distance'] >= start_distance - distance / 2) & (df['along_track_distance'] <= start_distance + distance / 2)
        for column in columns:
            result.at[i, column] = df.loc[mask, column].mean()
    return result