import astropy.units as u
import numpy as np
from gammapy.irf import Background2D
from regions import CircleAnnulusSkyRegion, CircleSkyRegion

from .base_acceptance_map_creator import BaseAcceptanceMapCreator


class RadialAcceptanceMapCreator(BaseAcceptanceMapCreator):

    def __init__(self, energy_axis, offset_axis, oversample_map=10, exclude_regions=[],
                 min_run_per_cos_zenith_bin=3, initial_cos_zenith_binning=0.01,
                 max_fraction_pixel_rotation_fov=0.5, time_resolution_rotation_fov=0.1 * u.s):
        """
            Create the class for calculating radial acceptance model

            Parameters
            ----------
            energy_axis : gammapy.maps.geom.MapAxis
                The energy axis for the acceptance model
            offset_axis : gammapy.maps.geom.MapAxis
                The offset axis for the acceptance model
            oversample_map : int
                Oversample in number of pixel of the spatial axis used for the calculation
            exclude_regions : list of 'regions.SkyRegion'
                Region with known or putative gamma-ray emission, will be excluded of the calculation of the acceptance map
            min_run_per_cos_zenith_bin : int
                Minimum number of runs per zenith bins
            initial_cos_zenith_binning : float
                Initial bin size for cos zenith binning
            max_fraction_pixel_rotation_fov : float
                For camera frame transformation the maximum size relative to a pixel a rotation is allowed
            time_resolution_rotation_fov : astropy.unit.Units
                Time resolution to use for the computation of the rotation of the FoV
        """

        # Compute parameters for internal map
        self.offset_axis = offset_axis
        self.oversample_map = oversample_map
        spatial_resolution = np.min(
            np.abs(self.offset_axis.edges[1:] - self.offset_axis.edges[:-1])) / self.oversample_map
        max_offset = np.max(self.offset_axis.edges)

        # Initiate upper instance
        super().__init__(energy_axis, max_offset, spatial_resolution, exclude_regions, min_run_per_cos_zenith_bin,
                         initial_cos_zenith_binning, max_fraction_pixel_rotation_fov, time_resolution_rotation_fov)

    def create_acceptance_map(self, observations):
        """
            Calculate a radial acceptance map

            Parameters
            ----------
            observations : gammapy.data.observations.Observations
                The collection of observations used to make the acceptance map

            Returns
            -------
            background : gammapy.irf.background.Background2D
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
                value /= 2. * np.pi * (
                            self.offset_axis.edges[i + 1].to('radian') - self.offset_axis.edges[i].to('radian')) * \
                         self.offset_axis.center[i].to('radian')
                value /= livetime
                data_background[j, i] = value

        background = Background2D(axes=[self.energy_axis, self.offset_axis], data=data_background)

        return background
