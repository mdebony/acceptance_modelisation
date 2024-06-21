from typing import List, Optional

import astropy.units as u
import numpy as np
from gammapy.data import Observations
from gammapy.irf import Background2D
from gammapy.maps import MapAxis
from regions import CircleAnnulusSkyRegion, CircleSkyRegion, SkyRegion

from .base_acceptance_map_creator import BaseAcceptanceMapCreator


class BaseRadialAcceptanceMapCreator(BaseAcceptanceMapCreator):

    def __init__(self,
                 energy_axis: MapAxis,
                 offset_axis: MapAxis,
                 oversample_map: int = 10,
                 exclude_regions: Optional[List[SkyRegion]] = None,
                 cos_zenith_binning_method: str = 'min_livetime',
                 cos_zenith_binning_parameter_value: int = 3600,
                 initial_cos_zenith_binning: float = 0.01,
                 max_angular_separation_wobble: u.Quantity = 0.4 * u.deg,
                 max_fraction_pixel_rotation_fov: float = 0.5,
                 time_resolution_rotation_fov: u.Quantity = 0.1 * u.s,
                 verbose: bool = False) -> None:
        """
        Create the class for calculating radial acceptance model

        Parameters
        ----------
        energy_axis : MapAxis
            The energy axis for the acceptance model
        offset_axis : MapAxis
            The offset axis for the acceptance model
        oversample_map : int, optional
            Oversample in number of pixel of the spatial axis used for the calculation
        exclude_regions : list of regions.SkyRegion, optional
            Region with known or putative gamma-ray emission, will be excluded of the calculation of the acceptance map
        cos_zenith_binning_method : str, optional
            The method used for cos zenith binning: 'min_livetime','min_livetime_per_wobble','min_n_observation'
        cos_zenith_binning_parameter_value : int, optional
            Minimum livetime (in seconds) or number of observations per zenith bins
        initial_cos_zenith_binning : float, optional
            Initial bin size for cos zenith binning
        max_angular_separation_wobble : u.Quantity, optional
            The maximum angular separation between identified wobbles, in degrees
        max_fraction_pixel_rotation_fov : float, optional
            For camera frame transformation the maximum size relative to a pixel a rotation is allowed
        time_resolution_rotation_fov : astropy.unit.Quantity, optional
            Time resolution to use for the computation of the rotation of the FoV
        verbose : bool, optional
            If True, print informations related to cos zenith binning
        """

        # If no exclusion region, default it as an empty list
        if exclude_regions is None:
            exclude_regions = []

        # Compute parameters for internal map
        self.offset_axis = offset_axis
        self.oversample_map = oversample_map
        spatial_resolution = np.min(
            np.abs(self.offset_axis.edges[1:] - self.offset_axis.edges[:-1])) / self.oversample_map
        max_offset = np.max(self.offset_axis.edges)

        # Initiate upper instance
        super().__init__(energy_axis, max_offset, spatial_resolution, exclude_regions,
                         cos_zenith_binning_method, cos_zenith_binning_parameter_value,
                         initial_cos_zenith_binning, max_angular_separation_wobble, max_fraction_pixel_rotation_fov,
                         time_resolution_rotation_fov,verbose)

    def create_acceptance_map(self, observations: Observations) -> Background2D:
        """
        Calculate a radial acceptance map

        Parameters
        ----------
        observations : Observations
            The collection of observations used to make the acceptance map

        Returns
        -------
        acceptance_map : Background2D
        """
        count_map_background, exp_map_background, exp_map_background_total, livetime = self._create_base_computation_map(
            observations)

        data_background = np.zeros((self.energy_axis.nbin, self.offset_axis.nbin)) * u.Unit('s-1 MeV-1 sr-1')
        for i in range(self.offset_axis.nbin):
            if np.isclose(0. * u.deg, self.offset_axis.edges[i]):
                selection_region = CircleSkyRegion(center=self.center_map, radius=self.offset_axis.edges[i + 1])
            else:
                selection_region = CircleAnnulusSkyRegion(center=self.center_map,
                                                          inner_radius=self.offset_axis.edges[i],
                                                          outer_radius=self.offset_axis.edges[i + 1])
            selection_map = self.geom.to_image().region_mask([selection_region])
            for j in range(self.energy_axis.nbin):
                value = u.dimensionless_unscaled * np.sum(count_map_background.data[j, :, :] * selection_map)
                value *= np.sum(exp_map_background_total.data[j, :, :] * selection_map) / np.sum(
                    exp_map_background.data[j, :, :] * selection_map)

                value /= (self.energy_axis.edges[j + 1] - self.energy_axis.edges[j])
                value /= 2. * np.pi * (np.cos(self.offset_axis.edges[i]) - np.cos(self.offset_axis.edges[i+1])) * u.steradian
                value /= livetime
                data_background[j, i] = value

        acceptance_map = Background2D(axes=[self.energy_axis, self.offset_axis], data=data_background)

        return acceptance_map
