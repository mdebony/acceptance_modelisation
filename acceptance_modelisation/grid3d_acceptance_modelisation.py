import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from gammapy.irf import Background3D
from gammapy.maps import MapAxis

from .base_acceptance_map_creator import BaseAcceptanceMapCreator


class Grid3DAcceptanceMapCreator(BaseAcceptanceMapCreator):

    def __init__(self, energy_axis, offset_axis, oversample_map=10, exclude_regions=[],
                 min_run_per_cos_zenith_bin=15, initial_cos_zenith_binning=0.01,
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
        if not np.allclose(self.offset_axis.bin_width, self.offset_axis.bin_width[0]):
            raise Exception('Support only regular linear bin for offset axis')
        if not np.isclose(self.offset_axis.edges[0], 0. * u.deg):
            raise Exception('Offset axis need to start at 0')
        self.oversample_map = oversample_map
        spatial_resolution = np.min(
            np.abs(self.offset_axis.edges[1:] - self.offset_axis.edges[:-1])) / self.oversample_map
        max_offset = np.max(self.offset_axis.edges)

        # Initiate upper instance
        super().__init__(energy_axis, max_offset, spatial_resolution, exclude_regions, min_run_per_cos_zenith_bin,
                         initial_cos_zenith_binning, max_fraction_pixel_rotation_fov, time_resolution_rotation_fov)

    def create_acceptance_map(self, observations):
        """
            Calculate a 3D grid acceptance map

            Parameters
            ----------
            observations : gammapy.data.observations.Observations
                The collection of observations used to make the acceptance map

            Returns
            -------
            background : gammapy.irf.background.Background3D
        """

        # Compute base data
        count_map_background, exp_map_background, exp_map_background_total, livetime = self._create_base_computation_map(
            observations)

        # Downsample map to bkg model resolution
        count_map_background_downsample = count_map_background.downsample(self.oversample_map, preserve_counts=True)
        exp_map_background_downsample = exp_map_background.downsample(self.oversample_map, preserve_counts=True)
        exp_map_background_total_downsample = exp_map_background_total.downsample(self.oversample_map, preserve_counts=True)

        # Create axis for bkg model
        edges = self.offset_axis.edges
        extended_edges = np.concatenate((-np.flip(edges), edges[1:]), axis=None)
        extended_offset_axis_x = MapAxis.from_edges(extended_edges, name='fov_lon')
        bin_width_x = np.repeat(extended_offset_axis_x.bin_width[:, np.newaxis], extended_offset_axis_x.nbin, axis=1)
        extended_offset_axis_y = MapAxis.from_edges(extended_edges, name='fov_lat')
        bin_width_y = np.repeat(extended_offset_axis_y.bin_width[np.newaxis, :], extended_offset_axis_y.nbin, axis=0)

        # Compute background
        corrected_counts = count_map_background_downsample.data * exp_map_background_total_downsample.data / exp_map_background_downsample.data
        solid_angle = 4. * (np.sin(bin_width_x / 2.) * np.sin(bin_width_y / 2.)) * u.steradian
        data_background = corrected_counts / solid_angle[np.newaxis, :, :] / self.energy_axis.bin_width[:, np.newaxis, np.newaxis] / livetime

        background = Background3D(axes=[self.energy_axis, extended_offset_axis_x, extended_offset_axis_y],
                                  data=data_background.to(u.Unit('s-1 MeV-1 sr-1')))

        return background
