import numpy as np
import astropy.units as u
from astropy.coordinates import AltAz
from astropy.coordinates.earth import OMEGA_EARTH


def compute_rotation_speed_fov(time_evaluation, pointing_sky, observatory_earth_location):
    """
        Rotation speed of the fov for a given evaluation time

        Parameters
        ----------
        time_evaluation : astropy.time.Time
            The time at which the rotation speed should be evaluated
        pointing_sky : astropy.coordinates.SkyCoord
            The direction pointed in the sky
        observatory_earth_location : astropy.coordinates.EarthLocation
            The position of the observatory
    """
    pointing_altaz = pointing_sky.transform_to(AltAz(obstime=time_evaluation,
                                                     location=observatory_earth_location))
    omega_earth = OMEGA_EARTH*u.rad
    omega = omega_earth * np.cos(observatory_earth_location.lat) * np.cos(pointing_altaz.az) / np.cos(
        pointing_altaz.alt)
    return omega