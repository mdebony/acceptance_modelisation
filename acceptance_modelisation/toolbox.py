import astropy.units as u
from astropy.coordinates import AltAz, SkyCoord, angular_separation
from astropy.coordinates.earth import OMEGA_EARTH, EarthLocation
from astropy.time import Time
from gammapy.data import Observations
from copy import deepcopy
import logging
import numpy as np

logger = logging.getLogger(__name__)

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
    omega_earth = OMEGA_EARTH * u.rad
    omega = omega_earth * np.cos(observatory_earth_location.lat) * np.cos(pointing_altaz.az) / np.cos(
        pointing_altaz.alt)
    return omega


def get_unique_wobble_pointings(observations: Observations, max_angular_separation_wobble=0.4 * u.deg):
    """
    Compute the angular separation between pointings and return a list
    of detected wobbles with their associated similar pointings

    Parameters
    ----------
    observations : gammapy.data.observations.Observations
        The list of observations
    max_angular_separation_wobble : u.Quantity
        The maximum angular separation between a wobble position and associated runs, in degrees

    Returns
    -------
    unique_wobble_list : `numpy.array`
        Array of wobble name associated with each run
    """
    all_ra_observations = np.array([obs.get_pointing_icrs(obs.tmid).ra.to_value(u.deg) for obs in observations]) * u.deg
    all_dec_observations = np.array([obs.get_pointing_icrs(obs.tmid).dec.to_value(u.deg) for obs in observations]) * u.deg
    ra_observations = deepcopy(all_ra_observations)
    dec_observations = deepcopy(all_dec_observations)
    wobbles = np.empty(shape=len(all_ra_observations), dtype=np.object_)
    wobbles_dict = {}
    i = 0
    mask_all_remaining = np.ones(shape=len(all_ra_observations), dtype=bool)
    while len(ra_observations) > 0:
        i = i + 1
        keywobble = 'W' + str(i)
        mask = (angular_separation(ra_observations[0], dec_observations[0],
                                   ra_observations, dec_observations) < max_angular_separation_wobble)
        mask_2 = (angular_separation(np.mean(ra_observations[mask]),
                                     np.mean(dec_observations[mask]),
                                     all_ra_observations,
                                     all_dec_observations) < max_angular_separation_wobble)
        wobbles_dict[keywobble] = [np.mean(all_ra_observations[mask_2 & mask_all_remaining]),
                                   np.mean(all_dec_observations[mask_2 & mask_all_remaining])]
        wobbles[mask_2 & mask_all_remaining] = keywobble
        mask_all_remaining = mask_all_remaining * ~mask_2
        ra_observations = all_ra_observations[mask_all_remaining]
        dec_observations = all_dec_observations[mask_all_remaining]

    logger.info(f"{len(wobbles_dict)} wobbles were found:")
    for key, value in wobbles_dict.items():
        logger.info(f"{key}: [{value[0].round(2)}, {value[1].round(2)}]")
    return wobbles
