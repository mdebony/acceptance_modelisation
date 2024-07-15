import copy
import logging

from abc import ABC, abstractmethod
from typing import Tuple, List, Any, Optional

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord, AltAz, SkyOffsetFrame
from astropy.coordinates.erfa_astrom import erfa_astrom, ErfaAstromInterpolator
from astropy.time import Time
from gammapy.data import Observations, Observation
from gammapy.datasets import MapDataset
from gammapy.irf import Background2D, Background3D
from gammapy.irf.background import BackgroundIRF
from gammapy.makers import MapDatasetMaker, SafeMaskMaker, FoVBackgroundMaker
from gammapy.maps import WcsNDMap, WcsGeom, Map, MapAxis, RegionGeom
from regions import CircleSkyRegion, SkyRegion
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d

from .toolbox import compute_rotation_speed_fov, get_unique_wobble_pointings

logger = logging.getLogger(__name__)


class BaseAcceptanceMapCreator(ABC):

    def __init__(self,
                 energy_axis: MapAxis,
                 max_offset: u.Quantity,
                 spatial_resolution: u.Quantity,
                 exclude_regions: Optional[List[SkyRegion]] = None,
                 cos_zenith_binning_method: str = 'min_livetime',
                 cos_zenith_binning_parameter_value: int = 3600,
                 initial_cos_zenith_binning: float = 0.01,
                 max_angular_separation_wobble: u.Quantity = 0.4 * u.deg,
                 zenith_binning_run_splitting: bool = False,
                 max_fraction_pixel_rotation_fov: float = 0.5,
                 time_resolution_run_splitting: u.Quantity = 0.1 * u.s, ) -> None:
        """
        Create the class for calculating radial acceptance model.

        Parameters
        ----------
        energy_axis : gammapy.maps.geom.MapAxis
            The energy axis for the acceptance model
        max_offset : astropy.units.Quantity
            The offset corresponding to the edge of the model
        spatial_resolution : astropy.units.Quantity
            The spatial resolution
        exclude_regions : list of regions.SkyRegion, optional
            Regions with known or putative gamma-ray emission, will be excluded from the calculation of the acceptance map
        cos_zenith_binning_method : str, optional
            The method used for cos zenith binning: 'min_livetime', 'min_livetime_per_wobble', 'min_n_observation', 'min_n_observation_per_wobble'
        cos_zenith_binning_parameter_value : int, optional
            Minimum livetime (in seconds) or number of observations per zenith bins
        initial_cos_zenith_binning : float, optional
            Initial bin size for cos zenith binning
        max_angular_separation_wobble : u.Quantity, optional
            The maximum angular separation between identified wobbles, in degrees
        zenith_binning_run_splitting : float, optional
            If true, will split each run to match zenith binning for the base model computation
            Could be computationally expensive, especially at high zenith with a high resolution zenith binning
        max_fraction_pixel_rotation_fov : float, optional
            For camera frame transformation the maximum size relative to a pixel a rotation is allowed
        time_resolution_run_splitting : astropy.units.Quantity, optional
            Time resolution to use for the computation of the rotation of the FoV and cut as function of the zenith bins
        """

        # If no exclusion region, default it as an empty list
        if exclude_regions is None:
            exclude_regions = []

        # Store base parameter
        self.energy_axis = energy_axis
        self.max_offset = max_offset
        self.exclude_regions = exclude_regions
        self.cos_zenith_binning_method = cos_zenith_binning_method
        self.cos_zenith_binning_parameter_value = cos_zenith_binning_parameter_value
        self.initial_cos_zenith_binning = initial_cos_zenith_binning
        self.max_angular_separation_wobble = max_angular_separation_wobble

        # Calculate map parameter
        self.n_bins_map = 2 * int(np.rint((self.max_offset / spatial_resolution).to(u.dimensionless_unscaled)))
        self.spatial_bin_size = self.max_offset / (self.n_bins_map / 2)
        self.center_map = SkyCoord(ra=0. * u.deg, dec=0. * u.deg, frame='icrs')
        self.geom = WcsGeom.create(skydir=self.center_map, npix=(self.n_bins_map, self.n_bins_map),
                                   binsz=self.spatial_bin_size, frame="icrs", axes=[self.energy_axis])
        logger.info(
            'Computation will be made with a bin size of {:.3f} arcmin'.format(
                self.spatial_bin_size.to_value(u.arcmin)))

        # Store computation parameters for run splitting
        self.max_fraction_pixel_rotation_fov = max_fraction_pixel_rotation_fov
        self.time_resolution_run_splitting = time_resolution_run_splitting
        self.zenith_binning_run_splitting = zenith_binning_run_splitting

    @staticmethod
    def _transform_obs_to_camera_frame(obs: Observation) -> Observation:
        """
        Transform events and pointing of an obs from a sky frame to camera frame

        Parameters
        ----------
        obs : gammapy.data.observations.Observation
            The observation to transform

        Returns
        -------
        obs_camera_frame : gammapy.data.observations.Observation
            The observation transformed for reference in camera frame
        """

        # Transform to altaz frame
        altaz_frame = AltAz(obstime=obs.events.time,
                            location=obs.observatory_earth_location)
        events_altaz = obs.events.radec.transform_to(altaz_frame)
        pointing_altaz = obs.get_pointing_icrs(obs.events.time).transform_to(altaz_frame)

        # Rotation to transform to camera frame
        camera_frame = SkyOffsetFrame(origin=AltAz(alt=pointing_altaz.alt,
                                                   az=pointing_altaz.az,
                                                   obstime=obs.events.time,
                                                   location=obs.observatory_earth_location),
                                      rotation=[0., ] * len(obs.events.time) * u.deg)
        events_camera_frame = events_altaz.transform_to(camera_frame)

        # Formatting data for the output
        camera_frame_events = obs.events.copy()
        camera_frame_events.table['RA'] = events_camera_frame.lon
        camera_frame_events.table['DEC'] = events_camera_frame.lat
        camera_frame_obs_info = copy.deepcopy(obs.obs_info)
        camera_frame_obs_info['RA_PNT'] = 0.
        camera_frame_obs_info['DEC_PNT'] = 0.
        obs_camera_frame = Observation(obs_id=obs.obs_id,
                                       obs_info=camera_frame_obs_info,
                                       events=camera_frame_events,
                                       gti=obs.gti,
                                       aeff=obs.aeff)
        obs_camera_frame._location = obs.observatory_earth_location

        return obs_camera_frame

    def _transform_exclusion_region_to_camera_frame(self, pointing_altaz: AltAz) -> List[SkyRegion]:
        """
        Transform the list of exclusion regions in sky frame into a list in camera frame.

        Parameters
        ----------
        pointing_altaz : astropy.coordinates.AltAz
            The pointing position of the telescope.

        Returns
        -------
        exclusion_region_camera_frame : list of regions.SkyRegion
            The list of exclusion regions in camera frame.

        Raises
        ------
        Exception
            If the region type is not supported.
        """

        camera_frame = SkyOffsetFrame(origin=pointing_altaz,
                                      rotation=[0., ] * u.deg)
        exclude_region_camera_frame = []
        for region in self.exclude_regions:
            if isinstance(region, CircleSkyRegion):
                center_coordinate = region.center
                center_coordinate_altaz = center_coordinate.transform_to(pointing_altaz)
                center_coordinate_camera_frame = center_coordinate_altaz.transform_to(camera_frame)
                center_coordinate_camera_frame_arb = SkyCoord(ra=center_coordinate_camera_frame.lon[0],
                                                              dec=center_coordinate_camera_frame.lat[0])
                exclude_region_camera_frame.append(CircleSkyRegion(center=center_coordinate_camera_frame_arb,
                                                                   radius=region.radius))
            else:
                raise Exception(f'{type(region)} region type not supported')

        return exclude_region_camera_frame

    def _create_map(self,
                    obs: Observation,
                    geom: WcsGeom,
                    exclude_regions: List[SkyRegion],
                    add_bkg: bool = False
                    ) -> Tuple[MapDataset, WcsNDMap]:
        """
        Create a map and the associated exclusion mask based on the given geometry and exclusion region.

        Parameters
        ----------
        obs : gammapy.data.observations.Observation
            The observation used to make the sky map.
        geom : gammapy.maps.WcsGeom
            The geometry for the maps.
        exclude_regions : list of regions.SkyRegion
            The list of exclusion regions.
        add_bkg : bool, optional
            If true, will also add the background model to the map. Default is False.

        Returns
        -------
        map_dataset : gammapy.datasets.MapDataset
            The map dataset.
        exclusion_mask : gammapy.maps.WcsNDMap
            The exclusion mask.
        """

        maker = MapDatasetMaker(selection=["counts"])
        if add_bkg:
            maker = MapDatasetMaker(selection=["counts", "background"])

        maker_safe_mask = SafeMaskMaker(methods=["offset-max"], offset_max=self.max_offset)

        geom_image = geom.to_image()
        exclusion_mask = ~geom_image.region_mask(exclude_regions) if len(exclude_regions) > 0 else ~Map.from_geom(
            geom_image)

        map_obs = maker.run(MapDataset.create(geom=geom), obs)
        map_obs = maker_safe_mask.run(map_obs, obs)

        return map_obs, exclusion_mask

    def _create_sky_map(self,
                        obs: Observation,
                        add_bkg: bool = False
                        ) -> Tuple[MapDataset, WcsNDMap]:
        """
        Create the sky map and the associated exclusion mask based on the observation and the exclusion regions.

        Parameters
        ----------
        obs : gammapy.data.observations.Observation
            The observation used to make the sky map.
        add_bkg : bool, optional
            If true, will also add the background model to the map. Default is False.

        Returns
        -------
        map_dataset : gammapy.datasets.MapDataset
            The map dataset.
        exclusion_mask : gammapy.maps.WcsNDMap
            The exclusion mask.
        """

        geom_obs = WcsGeom.create(skydir=obs.get_pointing_icrs(obs.tmid),
                                  npix=(self.n_bins_map, self.n_bins_map),
                                  binsz=self.spatial_bin_size,
                                  frame="icrs",
                                  axes=[self.energy_axis])
        map_obs, exclusion_mask = self._create_map(obs, geom_obs, self.exclude_regions, add_bkg=add_bkg)

        return map_obs, exclusion_mask

    def _compute_time_intervals_based_on_fov_rotation(self, obs: Observation) -> Time:
        """
        Calculate time intervals based on the rotation of the Field of View (FoV).

        Parameters
        ----------
        obs : gammapy.data.observations.Observation
            The observation used to calculate time intervals.

        Returns
        -------
        time_intervals : astropy.time.Time
            The time intervals for cutting the observation into time bins.
        """

        # Determine time interval for cutting the obs as function of the rotation of the Fov
        n_bin = max(2, int(np.rint(
            ((obs.tstop - obs.tstart) / self.time_resolution_run_splitting).to_value(u.dimensionless_unscaled))))
        time_axis = np.linspace(obs.tstart, obs.tstop, num=n_bin)
        rotation_speed_fov = compute_rotation_speed_fov(time_axis, obs.get_pointing_icrs(obs.tmid),
                                                        obs.observatory_earth_location)
        rotation_fov = cumulative_trapezoid(x=time_axis.unix_tai,
                                            y=rotation_speed_fov.to_value(u.rad / u.s),
                                            initial=0.) * u.rad
        distance_rotation_fov = rotation_fov.to_value(u.rad) * np.pi * self.max_offset
        node_obs = distance_rotation_fov // (self.spatial_bin_size * self.max_fraction_pixel_rotation_fov)
        change_node = node_obs[2:] != node_obs[1:-1]
        time_interval = Time([obs.tstart, ] + [time_axis[1:-1][change_node], ] + [obs.tstop, ])

        return time_interval

    def _create_base_computation_map(self, observations: Observations) -> Tuple[WcsNDMap, WcsNDMap, WcsNDMap, u.Unit]:
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

        with erfa_astrom.set(ErfaAstromInterpolator(1000 * u.s)):
            for obs in observations:
                # Filter events in exclusion regions
                geom = RegionGeom.from_regions(self.exclude_regions)
                mask = geom.contains(obs.events.radec)
                obs._events = obs.events.select_row_subset(~mask)
                # Create a count map in camera frame
                camera_frame_obs = self._transform_obs_to_camera_frame(obs)
                count_map_obs, _ = self._create_map(camera_frame_obs, self.geom, [], add_bkg=False)
                # Create exposure maps and fill them with the obs livetime
                exp_map_obs = MapDataset.create(geom=count_map_obs.geoms['geom'])
                exp_map_obs_total = MapDataset.create(geom=count_map_obs.geoms['geom'])
                exp_map_obs.counts.data = camera_frame_obs.observation_live_time_duration.value
                exp_map_obs_total.counts.data = camera_frame_obs.observation_live_time_duration.value

                # Evaluate the average exclusion mask in camera frame
                # by evaluating it on time intervals short compared to the field of view rotation
                exclusion_mask = np.zeros(count_map_obs.counts.data.shape[1:])
                time_interval = self._compute_time_intervals_based_on_fov_rotation(obs)
                for i in range(len(time_interval) - 1):
                    # Compute the exclusion region in camera frame for the average time
                    dtime = time_interval[i + 1] - time_interval[i]
                    time = time_interval[i] + dtime / 2
                    average_alt_az_frame = AltAz(obstime=time,
                                                 location=obs.observatory_earth_location)
                    average_alt_az_pointing = obs.get_pointing_icrs(time).transform_to(average_alt_az_frame)
                    exclusion_region_camera_frame = self._transform_exclusion_region_to_camera_frame(
                        average_alt_az_pointing)
                    geom_image = self.geom.to_image()

                    exclusion_mask_t = ~geom_image.region_mask(exclusion_region_camera_frame) if len(
                        exclusion_region_camera_frame) > 0 else ~Map.from_geom(geom_image)
                    # Add the exclusion mask in camera frame weighted by the time interval duration
                    exclusion_mask += exclusion_mask_t * (dtime).value
                # Normalise the exclusion mask by the full observation duration
                exclusion_mask *= 1 / (time_interval[-1] - time_interval[0]).value

                # Correct the exposure map by the exclusion region
                for j in range(count_map_obs.counts.data.shape[0]):
                    exp_map_obs.counts.data[j, :, :] = exp_map_obs.counts.data[j, :, :] * exclusion_mask

                # Stack counts and exposure maps and livetime of all observations
                count_map_background.data += count_map_obs.counts.data
                exp_map_background.data += exp_map_obs.counts.data
                exp_map_background_total.data += exp_map_obs_total.counts.data
                livetime += camera_frame_obs.observation_live_time_duration

        return count_map_background, exp_map_background, exp_map_background_total, livetime

    @abstractmethod
    def create_acceptance_map(self, observations: Observations) -> BackgroundIRF:
        """
        Abstract method to calculate an acceptance map from a list of observations.

        Subclasses must implement this method to provide the specific algorithm for calculating the acceptance map.

        Parameters
        ----------
        observations : gammapy.data.observations.Observations
            The collection of observations used to create the acceptance map.

        Returns
        -------
        acceptance_map : gammapy.irf.background.Background2D or gammapy.irf.background.Background3D
            The acceptance map calculated using the specific algorithm implemented by the subclass.
        """
        pass

    def _normalised_model_per_run(self,
                                  observations: Observations,
                                  acceptance_map: dict[Any, BackgroundIRF]) -> dict[Any, BackgroundIRF]:
        """
        Normalised the acceptance model associated to each run to the events associated with the run

        Parameters
        ----------
        observations : gammapy.data.observations.Observations
            The collection of observations used to make the acceptance map
        acceptance_map :dict of gammapy.irf.background.Background2D or gammapy.irf.background.Background3D
            A dict with observation number as key and a background model that could be used as an acceptance model associated at each key
            This is the models that will be normalised
        Returns
        -------
        background : dict of gammapy.irf.background.Background2D or gammapy.irf.background.Background3D
            A dict with observation number as key and a background model that could be used as an acceptance model associated at each key
        """

        normalised_acceptance_map = {}
        # Fit norm of the model to the observations
        for obs in observations:
            id_observation = obs.obs_id

            # replace the background model
            modified_observation = copy.deepcopy(obs)
            modified_observation.bkg = acceptance_map[id_observation]

            # Fit the background model
            logger.info('Fit to model to run ' + str(id_observation))
            map_obs, exclusion_mask = self._create_sky_map(modified_observation, add_bkg=True)
            maker_FoV_background = FoVBackgroundMaker(method='fit', exclusion_mask=exclusion_mask)
            map_obs = maker_FoV_background.run(map_obs)

            # Extract the normalisation
            parameters = map_obs.models.to_parameters_table()
            norm_background = parameters[parameters['name'] == 'norm']['value'][0]

            if norm_background < 0.:
                logger.error(
                    'Invalid normalisation value for run ' + str(id_observation) + ' : ' + str(norm_background))
            elif norm_background > 1.5 or norm_background < 0.5:
                logger.warning(
                    'High correction of the background normalisation for run ' + str(id_observation) + ' : ' + str(
                        norm_background))

            # Apply normalisation to the background model
            normalised_acceptance_map[id_observation] = copy.deepcopy(acceptance_map[id_observation])
            normalised_acceptance_map[id_observation].data = normalised_acceptance_map[
                                                                 id_observation].data * norm_background

        return normalised_acceptance_map

    def _compute_time_intervals_based_on_zenith_bin(self, obs: Observation, edge_zenith_bin: u.Quantity) -> Time:
        """
        Calculate time intervals based on an input zenith binning

        Parameters
        ----------
        obs : gammapy.data.observations.Observation
            The observation used to calculate time intervals.
        edge_zenith_bin : astropy.units.Quantity
            The edge of the bins used for zenith binning

        Returns
        -------
        time_intervals : astropy.time.Time
            The time intervals for cutting the observation into time bins.
        """

        # Create the time axis
        n_bin = max(2, int(np.rint(
            ((obs.tstop - obs.tstart) / self.time_resolution_run_splitting).to_value(u.dimensionless_unscaled))))
        time_axis = np.linspace(obs.tstart, obs.tstop, num=n_bin)

        # Compute the zenith for each evaluation time
        altaz_coordinates = obs.get_pointing_altaz(time_axis)
        zenith_values = altaz_coordinates.zen
        if np.any(zenith_values < np.min(edge_zenith_bin)) or np.any(zenith_values > np.max(edge_zenith_bin)):
            logger.error('Run with zenith value outside of the considered range for zenith binning')

        # Split the time interval to transition between zenith bin
        id_bin = np.digitize(zenith_values, edge_zenith_bin)
        bin_transition = id_bin[2:] != id_bin[1:-1]
        time_interval = Time([obs.tstart, ] + [time_axis[1:-1][bin_transition], ] + [obs.tstop, ])

        return time_interval

    def _create_model_cos_zenith_binned(self,
                                        observations: Observations
                                        ) -> dict[Any, BackgroundIRF]:
        """
        Calculate a model for each cos zenith bin

        Parameters
        ----------
        observations : gammapy.data.observations.Observations
            The collection of observations used to make the acceptance map

        Returns
        -------
        background : dict of gammapy.irf.background.Background2D or gammapy.irf.background.Background3D
            A dict with observation number as key and a background model that could be used as an acceptance model associated at each key

        """

        # Determine binning method. Convention : per_wobble methods have negative values
        methods = {'min_livetime': 1, 'min_livetime_per_wobble': -1, 'min_n_observation': 2,
                   'min_n_observation_per_wobble': -2}
        try:
            i_method = methods[self.cos_zenith_binning_method]
        except KeyError:
            logger.error(f" KeyError : {self.cos_zenith_binning_method} not a valid zenith binning method.\nValid "
                          f"methods are {[*methods]}")
            raise
        per_wobble = i_method < 0

        # Initial binning edge
        cos_zenith_bin = np.sort(np.arange(1.0, 0. - self.initial_cos_zenith_binning, -self.initial_cos_zenith_binning))
        zenith_bin = np.rad2deg(np.arccos(cos_zenith_bin)) * u.deg

        # Cut observations if requested
        if self.zenith_binning_run_splitting:
            if abs(i_method) == 2:
                logger.warning('Using zenith bin and run splitting at the same time is not recommanded and could lead to poor model. We recommand switching to a binning requirement based on livetime.')
            compute_observations = Observations()
            for obs in observations:
                time_interval = self._compute_time_intervals_based_on_zenith_bin(obs, zenith_bin)
                for i in range(len(time_interval) - 1):
                    compute_observations.append(obs.select_time(Time([time_interval[i], time_interval[i + 1]])))
                compute_observations_len = len(compute_observations)
        else:
            compute_observations = observations

        # Determine initial binning value
        cos_zenith_observations = np.array([np.cos(obs.get_pointing_altaz(obs.tmid).zen) for obs in compute_observations])
        livetime_observations = np.array([obs.observation_live_time_duration.to_value(u.s) for obs in compute_observations])

        # Select the quantity used to count observations
        if i_method in [-1, 1]:
            cut_variable_weights = livetime_observations
        elif i_method in [-2, 2]:
            cut_variable_weights = np.ones(len(cos_zenith_observations), dtype=int)

        # Gather runs per separation angle or all together. Define the minimum multiplicity (-1) to create a zenith bin.
        if per_wobble:
            wobble_observations = np.array(
                get_unique_wobble_pointings(compute_observations, self.max_angular_separation_wobble))
            multiplicity_wob = 1
        else:
            wobble_observations = np.full(len(cos_zenith_observations), 'any', dtype=np.object_)
            multiplicity_wob = 0

        cumsum_variable = {}
        for wobble in np.unique(wobble_observations):
            # Create an array of cumulative weight of the selected variable vs cos(zenith)
            cumsum_variable[wobble] = np.cumsum(np.histogram(cos_zenith_observations[wobble_observations == wobble],
                                                             bins=cos_zenith_bin,
                                                             weights=cut_variable_weights[
                                                                 wobble_observations == wobble])[0])
        # Initiate the list of index of selected zenith bin edges
        zenith_selected = [0]

        i = 0
        n = len(cos_zenith_bin) - 2

        while i < n:
            # For each wobble, find the index of the first zenith which fulfills the zd binning criteria if any
            # Then concatenate and sort the results for all wobbles
            candidate_i_per_wobble = [np.nonzero(cum_cut_variable >= self.cos_zenith_binning_parameter_value)[0][:1]
                                      for cum_cut_variable in cumsum_variable.values()]  # nonzero is assumed sorted
            candidate_i = np.sort(np.concatenate(candidate_i_per_wobble))

            if len(candidate_i) > multiplicity_wob:
                # If the criteria is fulfilled save the correct index.
                # The first and only candidate_i in the non-per_wobble case and the second in the per_wobble case.
                i = candidate_i[multiplicity_wob]
                zenith_selected.append(i + 1)
                for wobble in np.unique(wobble_observations):
                    # Reduce the cumulative sum by the value at the selected index for the next iteration
                    cumsum_variable[wobble] -= cumsum_variable[wobble][i]
            else:
                i += 1
                # The zenith bin creation criteria is not fulfilled, the last bin edge is set to the end of the
                # cos(zenith) array
                if i == 0:
                    zenith_selected.append(n + 1)
                else:
                    zenith_selected[-1] = n + 1
                    i = n
        cos_zenith_bin = cos_zenith_bin[zenith_selected]

        # Associate each observation to the correct bin
        binned_observations = []
        for i in range((len(cos_zenith_bin) - 1)):
            binned_observations.append(Observations())
        for obs in compute_observations:
            binned_observations[np.digitize(np.cos(obs.get_pointing_altaz(obs.tmid).zen), cos_zenith_bin) - 1].append(
                obs)

        # Compute the model for each bin
        binned_model = [self.create_acceptance_map(binned_obs) for binned_obs in binned_observations]

        # Determine the center of the bin (weighted as function of the livetime of each observation)
        bin_center = []
        for i in range(len(binned_observations)):
            weighted_cos_zenith_bin_per_obs = []
            livetime_per_obs = []
            for obs in binned_observations[i]:
                weighted_cos_zenith_bin_per_obs.append(
                    obs.observation_live_time_duration * np.cos(obs.get_pointing_altaz(obs.tmid).zen))
                livetime_per_obs.append(obs.observation_live_time_duration)
            bin_center.append(np.sum([wcos.value for wcos in weighted_cos_zenith_bin_per_obs]) / np.sum(
                [livet.value for livet in livetime_per_obs]))

        logger.info("cos zenith bin edges: ", list(np.round(cos_zenith_bin, 2)))
        logger.info("cos zenith bin centers: ", list(np.round(bin_center, 2)))
        logger.info(f"observation per bin: ", list(np.histogram(cos_zenith_observations, bins=cos_zenith_bin)[0]))
        logger.info(f"livetime per bin [s]: ", list(
            np.histogram(cos_zenith_observations, bins=cos_zenith_bin, weights=livetime_observations)[0].astype(
                int)))
        if per_wobble:
            wobble_observations_bool_arr = [(np.array(wobble_observations.tolist()) == wobble) for wobble in
                                            np.unique(np.array(wobble_observations))]
            livetime_observations_and_wobble = [np.array(livetime_observations) * wobble_bool for wobble_bool in
                                                wobble_observations_bool_arr]
            for i, wobble in enumerate(np.unique(np.array(wobble_observations))):
                logger.info(
                    f"{wobble} observation per bin: {list(np.histogram(cos_zenith_observations, bins=cos_zenith_bin, weights=1 * wobble_observations_bool_arr[i])[0])}")
                logger.info(
                    f"{wobble} livetime per bin: {list(np.histogram(cos_zenith_observations, bins=cos_zenith_bin, weights=livetime_observations_and_wobble[i])[0].astype(int))}")

        # Create the dict for output of the function
        dict_binned_model = {}
        for i in range(len(binned_model)):
            dict_binned_model[np.rad2deg(np.arccos(bin_center[i]))] = binned_model[i]

        return dict_binned_model

    def create_acceptance_map_cos_zenith_binned(self,
                                                observations: Observations
                                                ) -> dict[Any, BackgroundIRF]:
        """
        Calculate an acceptance map per run using cos zenith binning

        Parameters
        ----------
        observations : gammapy.data.observations.Observations
            The collection of observations used to make the acceptance map

        Returns
        -------
        background : dict of gammapy.irf.background.Background2D or gammapy.irf.background.Background3D
            A dict with observation number as key and a background model that could be used as an acceptance model associated at each key

        """

        # Produce the binned model
        dict_binned_model = self._create_model_cos_zenith_binned(observations)
        cos_zenith_model = []
        key_model = []
        for k in np.sort(list(dict_binned_model.keys())):
            cos_zenith_model.append(np.cos(np.deg2rad(k)))
            key_model.append(k)

        # Find the closest model for each observation and associate it to each observation
        acceptance_map = {}
        if len(cos_zenith_model) <= 1:
            logger.warning('Only one zenith bin, zenith binning deactivated')
        for obs in observations:
            cos_zenith_observation = np.cos(obs.get_pointing_altaz(obs.tmid).zen)
            key_closest_model = key_model[(np.abs(cos_zenith_model - cos_zenith_observation)).argmin()]
            acceptance_map[obs.obs_id] = dict_binned_model[key_closest_model]

        return acceptance_map

    def create_acceptance_map_cos_zenith_interpolated(self,
                                                      observations: Observations
                                                      ) -> dict[Any, BackgroundIRF]:
        """
        Calculate an acceptance map per run using cos zenith binning and interpolation

        Parameters
        ----------
        observations : gammapy.data.observations.Observations
            The collection of observations used to make the acceptance map

        Returns
        -------
        background : dict of gammapy.irf.background.Background2D or gammapy.irf.background.Background3D
            A dict with observation number as key and a background model that could be used as an acceptance model associated at each key

        """

        # Produce the binned model
        dict_binned_model = self._create_model_cos_zenith_binned(observations)
        binned_model = []
        cos_zenith_model = []
        for k in np.sort(list(dict_binned_model.keys())):
            binned_model.append(dict_binned_model[k])
            cos_zenith_model.append(np.cos(np.deg2rad(k)))

        acceptance_map = {}
        if len(binned_model) <= 1:
            logger.warning('Only one zenith bin, zenith interpolation deactivated')
            for obs in observations:
                acceptance_map[obs.obs_id] = binned_model[0]
        else:
            data_cube = np.zeros(tuple([len(binned_model), ] + list(binned_model[0].data.shape))) * binned_model[0].unit
            for i in range(len(binned_model)):
                data_cube[i] = binned_model[i].data * binned_model[i].unit
            interp_func = interp1d(x=np.array(cos_zenith_model),
                                   y=np.log10(data_cube.value + np.finfo(np.float64).tiny),
                                   axis=0,
                                   fill_value='extrapolate')
            for obs in observations:
                data_obs = (10. ** interp_func(np.cos(obs.get_pointing_altaz(obs.tmid).zen)))
                data_obs[data_obs < 100 * np.finfo(np.float64).tiny] = 0.
                if type(binned_model[0]) is Background2D:
                    acceptance_map[obs.obs_id] = Background2D(axes=binned_model[0].axes,
                                                              data=data_obs * data_cube.unit)
                elif type(binned_model[0]) is Background3D:
                    acceptance_map[obs.obs_id] = Background3D(axes=binned_model[0].axes,
                                                              data=data_obs * data_cube.unit)
                else:
                    raise Exception('Unknown background format')

        return acceptance_map

    def create_acceptance_map_per_observation(self,
                                              observations: Observations,
                                              zenith_binning: bool = False,
                                              zenith_interpolation: bool = False,
                                              runwise_normalisation: bool = True,
                                              ) -> dict[Any, BackgroundIRF]:
        """
        Calculate an acceptance map with the norm adjusted for each run

        Parameters
        ----------
        observations : gammapy.data.observations.Observations
            The collection of observations used to make the acceptance map
        zenith_binning : bool, optional
            If true the acceptance maps will be generated using zenith binning
        zenith_interpolation : bool, optional
            If true the acceptance maps will be generated using zenith binning and interpolation
        runwise_normalisation : bool, optional
            If true the acceptance maps will be normalised runwise to the observations

        Returns
        -------
        background : dict of gammapy.irf.background.Background2D or gammapy.irf.background.Background3D
            A dict with observation number as key and a background model that could be used as an acceptance model associated at each key
        """

        acceptance_map = {}
        if zenith_interpolation:
            acceptance_map = self.create_acceptance_map_cos_zenith_interpolated(observations)
        elif zenith_binning:
            acceptance_map = self.create_acceptance_map_cos_zenith_binned(observations)
        else:
            unique_base_acceptance_map = self.create_acceptance_map(observations)
            for obs in observations:
                acceptance_map[obs.obs_id] = unique_base_acceptance_map

        if runwise_normalisation:
            acceptance_map = self._normalised_model_per_run(observations, acceptance_map)

        return acceptance_map
