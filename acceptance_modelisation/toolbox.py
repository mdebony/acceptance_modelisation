import numpy as np
import astropy.units as u
from astropy.coordinates import AltAz, SkyCoord
from astropy.coordinates.earth import OMEGA_EARTH, EarthLocation
from astropy.time import Time
import matplotlib.pyplot as plt

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