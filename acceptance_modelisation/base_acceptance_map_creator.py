import copy
import logging
from abc import ABC, abstractmethod

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord, AltAz, SkyOffsetFrame
from astropy.time import Time
from gammapy.data import Observations, Observation
from gammapy.datasets import MapDataset
from gammapy.irf import Background2D, Background3D
from gammapy.makers import MapDatasetMaker, SafeMaskMaker, FoVBackgroundMaker
from gammapy.maps import WcsNDMap, WcsGeom, Map
from regions import CircleSkyRegion
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d

from .toolbox import compute_rotation_speed_fov


class BaseAcceptanceMapCreator(ABC):

    def __init__(self, energy_axis, max_offset, spatial_resolution, exclude_regions=[],
                 min_observation_per_cos_zenith_bin=3, initial_cos_zenith_binning=0.01,
                 max_fraction_pixel_rotation_fov=0.5, time_resolution_rotation_fov=0.1 * u.s):
        """
            Create the class for calculating radial acceptance model

            Parameters
            ----------
            energy_axis : gammapy.maps.geom.MapAxis
                The energy axis for the acceptance model
            max_offset : astropy.unit.Unit
                The offset corresponding to the edge of the model
            spatial_resolution : astropy.unit.Unit
                The spatial resolution
            exclude_regions : list of 'regions.SkyRegion'
                Region with known or putative gamma-ray emission, will be excluded of the calculation of the acceptance map
            min_observation_per_cos_zenith_bin : int
                Minimum number of runs per zenith bins
            initial_cos_zenith_binning : float
                Initial bin size for cos zenith binning
            max_fraction_pixel_rotation_fov : float
                For camera frame transformation the maximum size relative to a pixel a rotation is allowed
            time_resolution_rotation_fov : astropy.unit.Units
                Time resolution to use for the computation of the rotation of the FoV
        """

        # Store base parameter
        self.energy_axis = energy_axis
        self.max_offset = max_offset
        self.exclude_regions = exclude_regions
        self.min_observation_per_cos_zenith_bin = min_observation_per_cos_zenith_bin
        self.initial_cos_zenith_binning = initial_cos_zenith_binning

        # Calculate map parameter
        self.n_bins_map = 2 * int(np.rint((self.max_offset / spatial_resolution).to(u.dimensionless_unscaled)))
        self.spatial_bin_size = self.max_offset / (self.n_bins_map / 2)
        self.center_map = SkyCoord(ra=0. * u.deg, dec=0. * u.deg, frame='icrs')
        self.geom = WcsGeom.create(skydir=self.center_map, npix=(self.n_bins_map, self.n_bins_map),
                                   binsz=self.spatial_bin_size, frame="icrs", axes=[self.energy_axis])
        logging.info(
            'Computation will be made with a bin size of {:.3f} arcmin'.format(
                self.spatial_bin_size.to_value(u.arcmin)))

        # Store rotation computation parameters
        self.max_fraction_pixel_rotation_fov = max_fraction_pixel_rotation_fov
        self.time_resolution_rotation_fov = time_resolution_rotation_fov

    def _transform_obs_to_camera_frame(self, obs):
        """
            Transform events, pointing and exclusion regions of an obs from a sky frame to camera frame

            Parameters
            ----------
            obs : gammapy.data.observations.Observation
                The observation to transform

            Returns
            -------
            obs_camera_frame : gammapy.data.observations.Observation
                The observation transformed for reference in camera frame
            exclusion_region_camera_frame : list of region.SkyRegion
                The list of exclusion region in camera frame
        """

        # Transform to altaz frame
        altaz_frame = AltAz(obstime=obs.events.time,
                            location=obs.observatory_earth_location)
        events_altaz = obs.events.radec.transform_to(altaz_frame)
        pointing_altaz = obs.get_pointing_icrs(obs.tmid).transform_to(altaz_frame)

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

        # Compute the exclusion region in camera frame for the average time
        average_alt_az_frame = AltAz(obstime=obs_camera_frame.tmid,
                                     location=obs_camera_frame.observatory_earth_location)
        average_alt_az_pointing = obs.get_pointing_icrs(obs.tmid).transform_to(average_alt_az_frame)
        exclusion_region_camera_frame = self._transform_exclusion_region_to_camera_frame(average_alt_az_pointing)

        return obs_camera_frame, exclusion_region_camera_frame

    def _transform_exclusion_region_to_camera_frame(self, pointing_altaz):
        """
            Transform the list of exclusion region in sky frame into a list in camera frame

            Parameters
            ----------
            pointing_altaz : astropy.coordinates.AltAz
                The pointing position of the telescope

            Returns
            -------
            exclusion_region_camera_frame : list of region.SkyRegion
                The list of exclusion region in camera frame
        """

        camera_frame = SkyOffsetFrame(origin=pointing_altaz,
                                      rotation=[0., ] * u.deg)
        exclude_region_camera_frame = []
        for r in self.exclude_regions:
            if type(r) is CircleSkyRegion:
                center_coordinate = r.center
                center_coordinate_altaz = center_coordinate.transform_to(pointing_altaz)
                center_coordinate_camera_frame = center_coordinate_altaz.transform_to(camera_frame)
                center_coordinate_camera_frame_arb = SkyCoord(ra=center_coordinate_camera_frame.lon[0],
                                                              dec=center_coordinate_camera_frame.lat[0])
                exclude_region_camera_frame.append(CircleSkyRegion(center=center_coordinate_camera_frame_arb,
                                                                   radius=r.radius))
            else:
                return Exception(str(type(r)) + ' region type not supported')

        return exclude_region_camera_frame

    def _create_map(self, obs, geom, exclude_regions, add_bkg=False):
        """
            Create a map and the associated exclusion mask based on the given geometry and exclusion region

            Parameters
            ----------
            obs : gammapy.data.observations.Observation
                The observation used to make the sky map
            geom : gammapy.maps.WcsGeom
                The geometry for the maps
            exclude_regions : list of region.SkyRegion
                The list of exclusion region
            add_bkg : bool
                If true will also add the background model to the map

            Returns
            -------
            map_obs : gammapy.datasets.MapDataset
            exclusion_mask : gammapy.maps.WcsNDMap
        """

        maker = MapDatasetMaker(selection=["counts"])
        if add_bkg:
            maker = MapDatasetMaker(selection=["counts", "background"])

        maker_safe_mask = SafeMaskMaker(methods=["offset-max"], offset_max=self.max_offset)

        geom_image = geom.to_image()
        if len(exclude_regions) == 0:
            exclusion_mask = ~Map.from_geom(geom_image)
        else:
            exclusion_mask = ~geom_image.region_mask(exclude_regions)

        map_obs = maker.run(MapDataset.create(geom=geom), obs)
        map_obs = maker_safe_mask.run(map_obs, obs)

        return map_obs, exclusion_mask

    def _create_sky_map(self, obs, add_bkg=False):
        """
            Create the sky map used

            Parameters
            ----------
            obs : gammapy.data.observations.Observation
                The observation used to make the sky map
            add_bkg : bool
                If true will also add the background model to the map

            Returns
            -------
            map_obs : gammapy.datasets.MapDataset
            exclusion_mask : gammapy.maps.WcsNDMap
        """

        geom_obs = WcsGeom.create(skydir=obs.get_pointing_icrs(obs.tmid), npix=(self.n_bins_map, self.n_bins_map),
                                  binsz=self.spatial_bin_size, frame="icrs", axes=[self.energy_axis])
        map_obs, exclusion_mask = self._create_map(obs, geom_obs, self.exclude_regions, add_bkg=add_bkg)

        return map_obs, exclusion_mask

    def _create_camera_map(self, obs):
        """
            Create the sky map in camera coordinate used for computation

            Parameters
            ----------
            obs : gammapy.data.observations.Observation
                The observation used to make the sky map

            Returns
            -------
            map_obs : gammapy.datasets.MapDataset
            exclusion_mask : gammapy.maps.WcsNDMap
        """

        obs_camera_frame, exclusion_region_camera_frame = self._transform_obs_to_camera_frame(obs)
        map_obs, exclusion_mask = self._create_map(obs_camera_frame, self.geom,
                                                   exclusion_region_camera_frame, add_bkg=False)

        return map_obs, exclusion_mask

    def _compute_time_interval_cut_obs_rotation_fov(self, obs):
        """
            Create a list of time to bin the observation to take into accounts the rotation of the FoV

            Parameters
            ----------
            obs : gammapy.data.observations.Observation
                The observation used to make the sky map

            Returns
            -------
            time_interval : list of astropy.time.Time
                The time intervals to use for cutting the obs in time bins for compute
        """

        # Determine time interval for cutting the obs as function of the rotation of the Fov
        n_bin = max(2, int(np.rint(
            ((obs.tstop - obs.tstart) / self.time_resolution_rotation_fov).to_value(u.dimensionless_unscaled))))
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

    def _create_base_computation_map(self, observations):
        """
            From a list observations return a stacked finely binned counts and exposure map in camera frame to compute a model

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
            time_interval = self._compute_time_interval_cut_obs_rotation_fov(obs)
            for i in range(len(time_interval) - 1):
                cut_obs = obs.select_time(Time([time_interval[i], time_interval[i + 1]]))
                count_map_obs, exclusion_mask = self._create_camera_map(cut_obs)

                exp_map_obs = MapDataset.create(geom=count_map_obs.geoms['geom'])
                exp_map_obs_total = MapDataset.create(geom=count_map_obs.geoms['geom'])
                exp_map_obs.counts.data = cut_obs.observation_live_time_duration.value
                exp_map_obs_total.counts.data = cut_obs.observation_live_time_duration.value

                for j in range(count_map_obs.counts.data.shape[0]):
                    count_map_obs.counts.data[j, :, :] = count_map_obs.counts.data[j, :, :] * exclusion_mask
                    exp_map_obs.counts.data[j, :, :] = exp_map_obs.counts.data[j, :, :] * exclusion_mask

                count_map_background.data += count_map_obs.counts.data
                exp_map_background.data += exp_map_obs.counts.data
                exp_map_background_total.data += exp_map_obs_total.counts.data
                livetime += cut_obs.observation_live_time_duration

        return count_map_background, exp_map_background, exp_map_background_total, livetime

    @abstractmethod
    def create_acceptance_map(self, observations):
        """
            Abtract method to calculate an acceptance map from a list of observations

            Parameters
            ----------
            observations : gammapy.data.observations.Observations
                The collection of observations used to make the acceptance map

            Returns
            -------
            background : gammapy.irf.background.Background2D or gammapy.irf.background.Background3D
                The acceptance map resulting from the method
        """

    def create_acceptance_map_cos_zenith_binned(self, observations):
        """
            Calculate an acceptance map using cos zenith binning and interpolation

            Parameters
            ----------
            observations : gammapy.data.observations.Observations
                The collection of observations used to make the acceptance map

            Returns
            -------
            background : dict of gammapy.irf.background.Background2D
                A dict with observation number as key and a background model that could be used as an acceptance model associated at each key

        """

        cos_zenith_bin = np.sort(np.arange(1.0, 0. - self.initial_cos_zenith_binning, -self.initial_cos_zenith_binning))
        cos_zenith_observations = [np.cos(obs.pointing_zen) for obs in observations]
        run_per_bin = np.histogram(cos_zenith_observations, bins=cos_zenith_bin)[0]

        i = 0
        while i < len(run_per_bin):
            if run_per_bin[i] < self.min_observation_per_cos_zenith_bin and (i + 1) < len(run_per_bin):
                run_per_bin[i] += run_per_bin[i + 1]
                run_per_bin = np.delete(run_per_bin, i + 1)
                cos_zenith_bin = np.delete(cos_zenith_bin, i + 1)
            elif run_per_bin[i] < self.min_observation_per_cos_zenith_bin and (i + 1) == len(run_per_bin) and i > 0:
                run_per_bin[i - 1] += run_per_bin[i]
                run_per_bin = np.delete(run_per_bin, i)
                cos_zenith_bin = np.delete(cos_zenith_bin, i)
                i -= 1
            else:
                i += 1

        binned_observations = []
        for i in range((len(cos_zenith_bin) - 1)):
            binned_observations.append(Observations())
        for obs in observations:
            binned_observations[np.digitize(np.cos(obs.pointing_zen), cos_zenith_bin) - 1].append(obs)

        binned_model = [self.create_acceptance_map(binned_obs) for binned_obs in binned_observations]
        bin_center = []
        for i in range(len(binned_observations)):
            weighted_cos_zenith_bin_per_obs = []
            livetime_per_obs = []
            for obs in binned_observations[i]:
                weighted_cos_zenith_bin_per_obs.append(obs.observation_live_time_duration * np.cos(obs.pointing_zen))
                livetime_per_obs.append(obs.observation_live_time_duration)
            bin_center.append(
                (np.sum(weighted_cos_zenith_bin_per_obs) / np.sum(livetime_per_obs)).to(u.dimensionless_unscaled))

        acceptance_map = {}
        if len(binned_model) <= 1:
            logging.warning('Only one zenith bin, zenith interpolation deactivated')
            for obs in observations:
                acceptance_map[obs.obs_id] = binned_model[0]
        else:
            data_cube = np.zeros(tuple([len(binned_model), ] + list(binned_model[0].data.shape))) * binned_model[0].unit
            for i in range(len(binned_model)):
                data_cube[i] = binned_model[i].data * binned_model[i].unit
            interp_func = interp1d(x=np.array(bin_center),
                                   y=np.log10(data_cube.value + np.finfo(np.float64).tiny),
                                   axis=0,
                                   fill_value='extrapolate')
            for obs in observations:
                data_obs = (10. ** interp_func(np.cos(obs.pointing_zen)))
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

    def create_acceptance_map_per_observation(self, observations, zenith_bin=True):
        """
            Calculate an acceptance map with the norm adjusted for each run

            Parameters
            ----------
            observations : gammapy.data.observations.Observations
                The collection of observations used to make the acceptance map
            zenith_bin : bool,
                If true the acceptance maps will be generated using zenith binning and interpolation

            Returns
            -------
            background : dict of gammapy.irf.background.Background2D
                A dict with observation number as keu and a background model that could be used as an acceptance model associated at each key
        """

        base_acceptance_map = {}
        if zenith_bin:
            base_acceptance_map = self.create_acceptance_map_cos_zenith_binned(observations)
        else:
            unique_base_acceptance_map = self.create_acceptance_map(observations)
            for obs in observations:
                base_acceptance_map[obs.obs_id] = unique_base_acceptance_map

        acceptance_map = {}
        # Fit norm of the model to the observations
        for obs in observations:
            id_observation = obs.obs_id

            # replace the background model
            modified_observation = copy.deepcopy(obs)
            modified_observation.bkg = base_acceptance_map[id_observation]

            # Fit the background model
            logging.info('Fit to model to run ' + str(id_observation))
            map_obs, exclusion_mask = self._create_sky_map(modified_observation, add_bkg=True)
            maker_FoV_background = FoVBackgroundMaker(method='fit', exclusion_mask=exclusion_mask)
            map_obs = maker_FoV_background.run(map_obs)

            # Extract the normalisation
            parameters = map_obs.models.to_parameters_table()
            norm_background = parameters[parameters['name'] == 'norm']['value'][0]

            # Apply normalisation to the background model
            acceptance_map[id_observation] = copy.deepcopy(base_acceptance_map[id_observation])
            acceptance_map[id_observation].data = acceptance_map[id_observation].data * norm_background

        return acceptance_map
