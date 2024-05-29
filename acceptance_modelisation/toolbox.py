import numpy as np
import astropy.units as u
from astropy.coordinates import AltAz, SkyCoord, angular_separation
from astropy.coordinates.earth import OMEGA_EARTH, EarthLocation
from astropy.time import Time
import matplotlib.pyplot as plt
from gammapy.data import Observations
from copy import deepcopy

def compute_rotation_speed_fov(time_evaluation: Time,
                               pointing_sky: SkyCoord,
                               observatory_earth_location: EarthLocation) -> u.Quantity:
    """
    Compute the rotation speed of the FOV for a given evaluation time.

    Parameters
    ----------
    time_evaluation : astropy.time.Time
        The time at which the rotation speed should be evaluated.
    pointing_sky : astropy.coordinates.SkyCoord
        The direction pointed in the sky.
    observatory_earth_location : astropy.coordinates.EarthLocation
        The position of the observatory.

    Returns
    -------
    rotation_speed : astropy.units.Quantity
        The rotation speed of the FOV at the given time and pointing direction.
    """
    pointing_altaz = pointing_sky.transform_to(AltAz(obstime=time_evaluation,
                                                     location=observatory_earth_location))
    omega_earth = OMEGA_EARTH*u.rad
    omega = omega_earth * np.cos(observatory_earth_location.lat) * np.cos(pointing_altaz.az) / np.cos(
        pointing_altaz.alt)
    return omega

def get_unique_wobble_pointings(observations: Observations, max_angular_separation=0.4):
    """
    Compute the angular separation between pointings and return a list
    of detected wobbles with their associated similar pointings

    Parameters
    ----------
    observations : gammapy.data.observations.Observations
        The list of observations
    max_angular_separation : float
        The maximum angular separation between identified wobbles, in degrees

    Returns
    -------
    unique_wobble_list : list
        A list of the wobbles detected and their associated similar pointings (angular separation < 0.4°)
    """
    all_ra_observations = np.array([obs.get_pointing_icrs(obs.tmid).ra.to_value(u.deg) for obs in observations])
    all_dec_observations = np.array([obs.get_pointing_icrs(obs.tmid).dec.to_value(u.deg) for obs in observations])
    ra_observations = deepcopy(all_ra_observations)
    dec_observations = deepcopy(all_dec_observations)
    wobbles = np.empty(shape=len(all_ra_observations), dtype=np.object_)
    wobbles_dict = {}
    i=0
    mask_allremaining = np.ones(shape=len(all_ra_observations),dtype=bool)
    while len(ra_observations)>0:
        i=i+1
        keywobble='W'+str(i)
        mask = (angular_separation(ra_observations[0]*u.deg, dec_observations[0]*u.deg,
                                   ra_observations*u.deg, dec_observations*u.deg) < max_angular_separation*u.deg)
        mask_2 = (angular_separation(np.mean(ra_observations[mask])*u.deg, np.mean(dec_observations[mask])*u.deg,
                                     all_ra_observations*u.deg, all_dec_observations*u.deg) < max_angular_separation*u.deg)
        wobbles_dict[keywobble] = [np.mean(all_ra_observations[mask_2 & mask_allremaining]), np.mean(all_dec_observations[mask_2 & mask_allremaining])]
        wobbles[mask_2 & mask_allremaining] = keywobble
        mask_allremaining = mask_allremaining * ~mask_2
        ra_observations = all_ra_observations[mask_allremaining]
        dec_observations = all_dec_observations[mask_allremaining]

    print(f"{len(wobbles_dict)} wobbles were found: \n", wobbles_dict)
    return wobbles

def plot_coszd_binning(cos_zenith_observations,cos_zenith_bin,bin_center,livetime_observations,min_cut_per_cos_zenith_bin,cos_zenith_binning_method,zd_lim=(55,75)):
    fig,ax=plt.subplots(figsize=(5,5))
    ax.hist(cos_zenith_observations, bins=cos_zenith_bin,weights=livetime_observations,alpha=0.6)

    new_ticks_coszd=np.concatenate(([np.cos(zd_lim[1]*u.deg)],bin_center,[np.cos(zd_lim[0]*u.deg)]))

    ax.set_xticks(new_ticks_coszd)
    ax.set_xticklabels(np.round(new_ticks_coszd,2), rotation=45)

    # Create a second x-axis
    ax2 = ax.twiny()
    ax2.set_xticks(new_ticks_coszd)
    ax2.set_xticklabels(np.degrees(np.arccos(new_ticks_coszd)).astype(int), rotation=45)

    # Set labels
    ax.set_xlabel('cos(zd) bin center')
    ax.set_ylabel('livetime [s]')
    ax2.set_xlabel('cos(zd) bin center in zenith [°]')

    xlim=(np.cos(zd_lim[1]*u.deg),np.cos(zd_lim[0]*u.deg))
    ax.set_xlim(xlim)
    ax2.set_xlim(xlim)
    if 'livetime' in cos_zenith_binning_method: ax.hlines([min_cut_per_cos_zenith_bin],xlim[0],xlim[1],ls='-',color='red',label='min livetime',alpha=0.5)
    
    ylim=ax.get_ylim()    
    ax.vlines(cos_zenith_bin,ylim[0],ylim[1],ls=':',color='grey',label='bin edges',alpha=0.5)
    ax.legend(loc='best')

    plt.suptitle("Livetime per cos(zd) bin\n")
    plt.tight_layout()
    plt.show()