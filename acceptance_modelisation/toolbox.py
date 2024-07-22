import astropy.units as u
from astropy.coordinates import AltAz, SkyCoord, angular_separation
from astropy.coordinates.earth import OMEGA_EARTH, EarthLocation
from astropy.time import Time
from gammapy.data import Observations, Observation
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
    all_dec_observations = np.array(
        [obs.get_pointing_icrs(obs.tmid).dec.to_value(u.deg) for obs in observations]) * u.deg
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


def get_time_mini_irf(observation: Observation, mini_irf_time_resolution: u.Quantity):
    """
    Compute the evaluation time for each mini irfs

    Parameters
    ----------
    observation : gammapy.data.observations.Observation
        The observation for which mini irf will be generated
    mini_irf_time_resolution : u.Quantity
        The targeted time resolution for the mini irfs

    Returns
    -------
    evaluation_time : u.Quantity
        An array of the evaluation time of each mini irf, correspond to the central time of the bin
    observation_time : u.Quantity
        The observation time associated with each mini irf
    """

    total_observation_duration = observation.tstop - observation.tstart
    nb_bin = int(np.ceil((total_observation_duration / mini_irf_time_resolution).to_value(u.dimensionless_unscaled)))
    bin_duration = total_observation_duration / nb_bin
    evaluation_time = Time(np.zeros(nb_bin), format='unix')
    observation_time = np.zeros(nb_bin) * mini_irf_time_resolution.unit
    for i in range(nb_bin):
        evaluation_time[i] = observation.tstart + (i + 0.5) * bin_duration
        observation_time[i] = bin_duration

    return evaluation_time, observation_time


def generate_irf_from_mini_irf(data_cube: np.array, observation_time: u.Quantity):
    """
    Compute the final irf from the collection of mini irfs

    Parameters
    ----------
    data_cube : numpy.array
        The data cube containing all the mini irfs. Axis 0 should be the mini irfs axis
    observation_time : u.Quantity
        The observation time for each bin

    Returns
    -------
    data_cube_final : u.Quantity
        The data cube containing the final irf
    """

    # Compute the scale factor for each mini irf to get a weighted average
    scale_factor_per_bin = (observation_time / np.sum(observation_time)).to_value(u.dimensionless_unscaled)

    # Reshape the scale factor in order to be able to multiply it to the data cube
    scale_factor_per_bin_reshaped = scale_factor_per_bin.reshape(
        (scale_factor_per_bin.size, *[1 for _ in range(data_cube.ndim - 1)]))

    # Compute the weighted average of all mini irfs
    data_cube_final = np.sum(data_cube * scale_factor_per_bin_reshaped, axis=0)

    return data_cube_final
