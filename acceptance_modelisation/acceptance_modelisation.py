import copy
import numpy as np
import astropy.units as u
import logging
from scipy.interpolate import interp1d
from astropy.coordinates import SkyCoord
from gammapy.data import Observations
from gammapy.datasets import MapDataset
from gammapy.irf import Background2D
from gammapy.makers import MapDatasetMaker, SafeMaskMaker, FoVBackgroundMaker
from gammapy.maps import WcsNDMap, WcsGeom
from regions import CircleAnnulusSkyRegion


class RadialAcceptanceMapCreator:

    def __init__(self, energy_axis, offset_axis, oversample_map=10, exclude_regions=[],
                 min_run_per_cos_zenith_bin=3, initial_cos_zenith_binning=0.01):
        """
            Create the class for calculating radial acceptance model

            Parameters
            ----------
            energy_axis : gammapy.maps.geom.MapAxis
                The energy axis for the acceptance model
            offset_axis : gammapy.maps.geom.MapAxis
                The offset axis for the acceptance model
            oversample_map : int
                oversample in number of pixel of the spatial axis used for the calculation
            exclude_regions : list of 'regions.SkyRegion'
                Region with sources, will be excluded of the calculation of the acceptance map
        """

        # Store base parameter
        self.energy_axis = energy_axis
        self.offset_axis = offset_axis
        self.oversample_map = oversample_map
        self.exclude_regions = exclude_regions
        self.min_run_per_cos_zenith_bin = min_run_per_cos_zenith_bin
        self.initial_cos_zenith_binning = initial_cos_zenith_binning

        # Calculate map parameter
        self.n_bins_map = self.offset_axis.nbin * self.oversample_map * 2
        self.binsz = self.offset_axis.edges[-1] / (self.n_bins_map / 2)

    def __create_sky_map(self, obs, add_bkg=False):
        """
            Create the sky map used for computation

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

        geom_obs = WcsGeom.create(skydir=obs.pointing_radec, npix=(self.n_bins_map, self.n_bins_map),
                                  binsz=self.binsz, frame="icrs", axes=[self.energy_axis])
        maker = MapDatasetMaker(selection=["counts"])
        if add_bkg:
            maker = MapDatasetMaker(selection=["counts", "background"])

        maker_safe_mask = SafeMaskMaker(methods=["offset-max"], offset_max=np.max(self.offset_axis.edges))

        geom_image = geom_obs.to_image()
        exclusion_mask = ~geom_image.region_mask(self.exclude_regions)

        map_obs = maker.run(MapDataset.create(geom=geom_obs), obs)
        map_obs = maker_safe_mask.run(map_obs, obs)

        return map_obs, exclusion_mask

    def create_radial_acceptance_map(self, observations):
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

        center_map = SkyCoord(ra=0. * u.deg, dec=0. * u.deg, frame='icrs')

        geom = WcsGeom.create(skydir=center_map, npix=(self.n_bins_map, self.n_bins_map), binsz=self.binsz,
                              frame="icrs", axes=[self.energy_axis])

        count_map_background = WcsNDMap(geom=geom)
        exp_map_background = WcsNDMap(geom=geom, unit=u.s)
        exp_map_background_total = WcsNDMap(geom=geom, unit=u.s)
        livetime = 0. * u.s

        for obs in observations:
            count_map_obs, exclusion_mask = self.__create_sky_map(obs)

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

        data_background = np.zeros((self.energy_axis.nbin, self.offset_axis.nbin)) * u.Unit('s-1 MeV-1 sr-1')
        for i in range(self.offset_axis.nbin):
            selection_region = CircleAnnulusSkyRegion(center=center_map, inner_radius=self.offset_axis.edges[i],
                                                      outer_radius=self.offset_axis.edges[i + 1])
            selection_map = geom.to_image().region_mask([selection_region])
            for j in range(self.energy_axis.nbin):
                value = u.dimensionless_unscaled * np.sum(count_map_background.data[j, :, :] * selection_map)
                value *= np.sum(exp_map_background_total.data[j, :, :] * selection_map) / np.sum(
                    exp_map_background.data[j, :, :] * selection_map)

                value /= (self.energy_axis.edges[j + 1] - self.energy_axis.edges[j])
                value /= 2. * np.pi * \
                         (self.offset_axis.edges[i + 1].to('radian') - self.offset_axis.edges[i].to('radian')) * \
                         self.offset_axis.center[i].to('radian')
                value /= livetime
                data_background[j, i] = value

        background = Background2D(axes=[self.energy_axis, self.offset_axis], data=data_background)

        return background

    def create_radial_acceptance_map_cos_zenith_binned(self, observations):

        cos_zenith_bin = np.sort(np.arange(1.0, 0. - self.initial_cos_zenith_binning, -self.initial_cos_zenith_binning))
        cos_zenith_observations = [np.cos(obs.pointing_zen) for obs in observations]
        run_per_bin = np.histogram(cos_zenith_observations, bins=cos_zenith_bin)[0]

        i = 0
        while i < len(run_per_bin):
            if run_per_bin[i] < self.min_run_per_cos_zenith_bin and (i + 1) < len(run_per_bin):
                run_per_bin[i] += run_per_bin[i + 1]
                run_per_bin = np.delete(run_per_bin, i + 1)
                cos_zenith_bin = np.delete(cos_zenith_bin, i + 1)
            elif run_per_bin[i] < self.min_run_per_cos_zenith_bin and (i + 1) == len(run_per_bin):
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
            binned_observations[np.digitize(np.cos(obs.pointing_zen), cos_zenith_bin)-1].append(obs)

        binned_model = [self.create_radial_acceptance_map(binned_obs) for binned_obs in binned_observations]
        bin_center = []
        for i in range(len(binned_observations)):
            bin_center.append(np.mean([np.cos(obs.pointing_zen) for obs in binned_observations[i]]))

        radial_acceptance_map = {}
        if len(binned_model) <= 1:
            logging.warning('Only one zenith bin, zenith interpolation deactivated')
            for obs in observations:
                radial_acceptance_map[obs.obs_id] = binned_model[0]
        else:
            data_cube = np.zeros(tuple([len(binned_model), ] + list(binned_model[0].data.shape))) * binned_model[0].unit
            for i in range(len(binned_model)):
                data_cube[i] = binned_model[i].data * binned_model[i].unit
            interp_func = interp1d(x=np.array(bin_center),
                                   y=np.log10(data_cube.value),
                                   axis=0,
                                   fill_value='extrapolate')
            for obs in observations:
                radial_acceptance_map[obs.obs_id] = Background2D(axes=binned_model[0].axes,
                                                                         data=interp_func(np.cos(obs.pointing_zen))*data_cube.unit)

        return radial_acceptance_map

    def create_radial_acceptance_map_per_observation(self, observations, zenith_bin=True):
        """
            Calculate a radial acceptance map with the norm adjusted for each run

            Parameters
            ----------
            observations : gammapy.data.observations.Observations
                The collection of observations used to make the acceptance map
            zenith_bin : bool,
                If true the acceptance maps will be generated using zenith binning and interpolation

            Returns
            -------
            background : dict of gammapy.irf.background.Background2D
                A dict with observation number as keu and a bakground model that could be used as an acceptance model associated at each key
        """

        base_radial_acceptance_map = {}
        if zenith_bin:
            base_radial_acceptance_map = self.create_radial_acceptance_map_cos_zenith_binned(observations)
        else:
            unique_base_radial_acceptance_map = self.create_radial_acceptance_map(observations)
            for obs in observations:
                base_radial_acceptance_map[obs.obs_id] = unique_base_radial_acceptance_map

        radial_acceptance_map = {}
        # Fit norm of the model to the observations
        for obs in observations:
            id_observation = obs.obs_id

            # replace the background model
            modified_observation = copy.deepcopy(obs)
            modified_observation.bkg = base_radial_acceptance_map[id_observation]

            # Fit the background model
            map_obs, exclusion_mask = self.__create_sky_map(modified_observation, add_bkg=True)
            maker_FoV_background = FoVBackgroundMaker(method='fit', exclusion_mask=exclusion_mask)
            map_obs = maker_FoV_background.run(map_obs)

            # Extract the normalisation
            parameters = map_obs.models.to_parameters_table()
            norm_background = parameters[parameters['name'] == 'norm']['value'][0]

            # Apply normalisation to the background model
            radial_acceptance_map[id_observation] = copy.deepcopy(base_radial_acceptance_map)
            radial_acceptance_map[id_observation].data = radial_acceptance_map[id_observation].data * norm_background

        return radial_acceptance_map
