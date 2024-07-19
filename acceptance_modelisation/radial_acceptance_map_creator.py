from typing import List, Optional, Tuple

import astropy.units as u
from gammapy.data import Observation
from gammapy.datasets import MapDataset
from gammapy.maps import MapAxis, WcsNDMap, WcsGeom
from regions import SkyRegion

from .base_radial_acceptance_map_creator import BaseRadialAcceptanceMapCreator


class RadialAcceptanceMapCreator(BaseRadialAcceptanceMapCreator):

    def __init__(self,
                 energy_axis: MapAxis,
                 offset_axis: MapAxis,
                 oversample_map: int = 10,
                 exclude_regions: Optional[List[SkyRegion]] = None,
                 cos_zenith_binning_method: str = 'min_livetime',
                 cos_zenith_binning_parameter_value: int = 3600,
                 initial_cos_zenith_binning: float = 0.01,
                 max_angular_separation_wobble: u.Quantity = 0.4 * u.deg,
                 zenith_binning_run_splitting: bool = False,
                 max_fraction_pixel_rotation_fov: float = 0.5,
                 time_resolution: u.Quantity = 0.1 * u.s,
                 use_mini_irf_computation: bool = False,
                 mini_irf_time_resolution: u.Quantity = 1. * u.min) -> None:

        """
        Create the class for calculating radial acceptance model
        This class should be use when strict 2D model is good enough

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
            The method used for cos zenith binning: 'min_livetime','min_n_observation'
        cos_zenith_binning_parameter_value : int, optional
            Minimum livetime (in seconds) or number of observations per zenith bins
        initial_cos_zenith_binning : float, optional
            Initial bin size for cos zenith binning
        max_angular_separation_wobble : u.Quantity, optional
            The maximum angular separation between identified wobbles, in degrees
        zenith_binning_run_splitting : bool, optional
            If true, will split each run to match zenith binning for the base model computation
            Could be computationally expensive, especially at high zenith with a high resolution zenith binning
        max_fraction_pixel_rotation_fov : float, optional
            For camera frame transformation the maximum size relative to a pixel a rotation is allowed
        time_resolution : astropy.units.Quantity, optional
            Time resolution to use for the computation of the rotation of the FoV and cut as function of the zenith bins
        use_mini_irf_computation : bool, optional
            If true, during zenith interpolation and binning will compute first mini irf for each part of the run before averaging them.
            Should improve the accuracy of the model, especially at high zenith angle. Actiate it could singificantly increase computation time.
        mini_irf_time_resolution : astropy.units.Quantity, optional
            Time resolution to use for mini irf used for computation of the final background model
        """

        # Initiate upper instance
        super().__init__(energy_axis=energy_axis,
                         offset_axis=offset_axis,
                         oversample_map=oversample_map,
                         exclude_regions=exclude_regions,
                         cos_zenith_binning_method=cos_zenith_binning_method,
                         cos_zenith_binning_parameter_value=cos_zenith_binning_parameter_value,
                         initial_cos_zenith_binning=initial_cos_zenith_binning,
                         max_fraction_pixel_rotation_fov=max_fraction_pixel_rotation_fov,
                         max_angular_separation_wobble=max_angular_separation_wobble,
                         zenith_binning_run_splitting=zenith_binning_run_splitting,
                         time_resolution=time_resolution,
                         use_mini_irf_computation=use_mini_irf_computation,
                         mini_irf_time_resolution=mini_irf_time_resolution)

    def _create_base_computation_map(self, observations: Observation) -> Tuple[WcsNDMap, WcsNDMap, WcsNDMap, u.Unit]:
        """
        From a list of observations return a stacked finely binned counts and exposure map in camera frame to compute a model

        Parameters
        ----------
        observations : gammapy.data.observations.Observations
            The list of observations

        Returns
        -------
        count_map_background : gammapy.map.WcsNDMap
            The count map
        exp_map_background : gammapy.map.WcsNDMap
            The exposure map corrected for exclusion regions
        exp_map_background_total : gammapy.map.WcsNDMap
            The exposure map without correction for exclusion regions
        livetime : astropy.unit.Unit
            The total exposure time for the model
        """
        count_map_background = WcsNDMap(geom=self.geom)
        exp_map_background = WcsNDMap(geom=self.geom, unit=u.s)
        exp_map_background_total = WcsNDMap(geom=self.geom, unit=u.s)
        livetime = 0. * u.s

        for obs in observations:
            geom = WcsGeom.create(skydir=obs.pointing.fixed_icrs, npix=(self.n_bins_map, self.n_bins_map),
                                  binsz=self.spatial_bin_size, frame="icrs", axes=[self.energy_axis])
            count_map_obs, exclusion_mask = self._create_map(obs, geom, self.exclude_regions, add_bkg=False)

            exp_map_obs = MapDataset.create(geom=count_map_obs.geoms['geom'])
            exp_map_obs_total = MapDataset.create(geom=count_map_obs.geoms['geom'])
            exp_map_obs.counts.data = obs.observation_live_time_duration.value
            exp_map_obs_total.counts.data = obs.observation_live_time_duration.value

            for i in range(count_map_obs.counts.data.shape[0]):
                count_map_obs.counts.data[i, :, :] = count_map_obs.counts.data[i, :, :] * exclusion_mask
                exp_map_obs.counts.data[i, :, :] = exp_map_obs.counts.data[i, :, :] * exclusion_mask

            count_map_background.data += count_map_obs.counts.data
            exp_map_background.data += exp_map_obs.counts.data
            exp_map_background_total.data += exp_map_obs_total.counts.data
            livetime += obs.observation_live_time_duration

        return count_map_background, exp_map_background, exp_map_background_total, livetime
