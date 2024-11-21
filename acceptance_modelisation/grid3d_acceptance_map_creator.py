import logging
from typing import Tuple, List, Optional

from astropy.coordinates import AltAz
from astropy.coordinates.erfa_astrom import erfa_astrom, ErfaAstromInterpolator
import astropy.units as u
import numpy as np
from gammapy.data import Observations, Observation
from gammapy.datasets import MapDataset
from gammapy.irf import FoVAlignment, Background3D
from gammapy.maps import WcsNDMap, Map, MapAxis, RegionGeom
from iminuit import Minuit
from regions import SkyRegion
from scipy.ndimage import rotate

from .base_acceptance_map_creator import BaseAcceptanceMapCreator
from .modeling import FIT_FUNCTION, log_factorial, log_poisson

logger = logging.getLogger(__name__)


class Grid3DAcceptanceMapCreator(BaseAcceptanceMapCreator):
    def __init__(
        self,
        energy_axis: MapAxis,
        offset_axis: MapAxis,
        oversample_map: int = 10,
        exclude_regions: Optional[List[SkyRegion]] = None,
        cos_zenith_binning_method: str = "min_livetime",
        cos_zenith_binning_parameter_value: int = 3600,
        initial_cos_zenith_binning: float = 0.01,
        max_angular_separation_wobble: u.Quantity = 0.4 * u.deg,
        zenith_binning_run_splitting: bool = False,
        max_fraction_pixel_rotation_fov: float = 0.5,
        time_resolution: u.Quantity = 0.1 * u.s,
        use_mini_irf_computation: bool = False,
        mini_irf_time_resolution: u.Quantity = 1.0 * u.min,
        method="stack",
        fit_fnc="gaussian2d",
        fit_seeds=None,
        fit_bounds=None,
        interpolation_type: str = "linear",
        activate_interpolation_cleaning: bool = False,
        interpolation_cleaning_energy_relative_threshold: float = 1e-4,
        interpolation_cleaning_spatial_relative_threshold: float = 1e-2,
    ) -> None:
        """
        Create the class for calculating 3D grid acceptance model

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
        zenith_binning_run_splitting : float, optional
            If true, will split each run to match zenith binning for the base model computation
            Could be computationally expensive, especially at high zenith with a high resolution zenith binning
        max_fraction_pixel_rotation_fov : bool, optional
            For camera frame transformation the maximum size relative to a pixel a rotation is allowed
        time_resolution : astropy.units.Quantity, optional
            Time resolution to use for the computation of the rotation of the FoV and cut as function of the zenith bins
        method : str, optional
            Decide if the acceptance is a direct event stacking or a fitted model. 'stack' or 'fit'
        fit_fnc: str or function
            Two dimensional function to be fitted. Some built-in functions are provided and selected by passing a string
            The function needs to have a size parameter (integral charge) as first parameter.
        fit_seeds: dict, can optionally be None if using a built-in function
            Seeds of the parameters of the function to fit. Normalisation parameter is ignored if given.
        fit_bounds: dict, can optionally be None if using a built-in function
            Bounds of the parameters of the function to fit. Normalisation parameter is ignored if given.
        use_mini_irf_computation : bool, optional
            If true, in case the case of zenith interpolation or binning, each run will be divided in small subrun (the slicing is based on time).
            A model will be computed for each sub run before averaging them to obtain the final model for the run.
            Should improve the accuracy of the model, especially at high zenith angle.
        mini_irf_time_resolution : astropy.units.Quantity, optional
            Time resolution to use for mini irf used for computation of the final background model
        interpolation_type: str, optional
            Select the type of interpolation to be used, could be either "log" or "linear", log tend to provided better results be could more easily create artefact that will cause issue
        activate_interpolation_cleaning: bool, optional
            If true, will activate the cleaning step after interpolation, it should help to eliminate artefact caused by interpolation
        interpolation_cleaning_energy_relative_threshold: float, optional
            To be considered value, the bin in energy need at least one adjacent bin with a relative difference within this range
        interpolation_cleaning_spatial_relative_threshold: float, optional
            To be considered value, the bin in space need at least one adjacent bin with a relative difference within this range
        """

        # If no exclusion region, default it as an empty list
        if exclude_regions is None:
            exclude_regions = []

        # Compute parameters for internal map
        self.offset_axis = offset_axis
        if not np.allclose(self.offset_axis.bin_width, self.offset_axis.bin_width[0]):
            raise Exception("Support only regular linear bin for offset axis")
        if not np.isclose(self.offset_axis.edges[0], 0.0 * u.deg):
            raise Exception("Offset axis need to start at 0")
        self.oversample_map = oversample_map
        spatial_resolution = (
            np.min(np.abs(self.offset_axis.edges[1:] - self.offset_axis.edges[:-1]))
            / self.oversample_map
        )
        max_offset = np.max(self.offset_axis.edges)

        self.method = method
        self.fit_fnc = fit_fnc
        self.fit_seeds = fit_seeds
        self.fit_bounds = fit_bounds

        # Initiate upper instance
        super().__init__(
            energy_axis=energy_axis,
            max_offset=max_offset,
            spatial_resolution=spatial_resolution,
            exclude_regions=exclude_regions,
            cos_zenith_binning_method=cos_zenith_binning_method,
            cos_zenith_binning_parameter_value=cos_zenith_binning_parameter_value,
            initial_cos_zenith_binning=initial_cos_zenith_binning,
            max_angular_separation_wobble=max_angular_separation_wobble,
            zenith_binning_run_splitting=zenith_binning_run_splitting,
            max_fraction_pixel_rotation_fov=max_fraction_pixel_rotation_fov,
            time_resolution=time_resolution,
            use_mini_irf_computation=use_mini_irf_computation,
            mini_irf_time_resolution=mini_irf_time_resolution,
            interpolation_type=interpolation_type,
            activate_interpolation_cleaning=activate_interpolation_cleaning,
            interpolation_cleaning_energy_relative_threshold=interpolation_cleaning_energy_relative_threshold,
            interpolation_cleaning_spatial_relative_threshold=interpolation_cleaning_spatial_relative_threshold,
        )

    def fit_background(self, count_map, exp_map_total, exp_map):
        centers = self.offset_axis.center.to_value(u.deg)
        centers = np.concatenate((-np.flip(centers), centers), axis=None)
        raw_seeds = {}
        bounds = {}
        if type(self.fit_fnc) == str:
            try:
                fnc = FIT_FUNCTION[self.fit_fnc]
            except KeyError:
                logger.error(
                    f"Invalid built-in fit_fnc. Use {FIT_FUNCTION.keys()} or a custom function."
                )
                raise
            raw_seeds = fnc.default_seeds.copy()
            bounds = fnc.default_bounds.copy()
        else:
            fnc = self.fit_fnc
        if self.fit_seeds is not None:
            raw_seeds.update(self.fit_seeds)
        if self.fit_bounds is not None:
            bounds.update(self.fit_bounds)

        mask = (
            exp_map > 0
        )  # Handles fully overlapping exclusion regions in the 'size' seed computation
        # Seeds the charge normalisation to the observed counts corrected for exclusion region reduction to exposure
        raw_seeds["size"] = np.sum(
            count_map[mask] * exp_map_total[mask] / exp_map[mask]
        ) / np.mean(mask)
        bounds["size"] = (raw_seeds["size"] * 0.1, raw_seeds["size"] * 10)

        # reorder seeds to fnc parameter order
        param_fnc = list(fnc.__code__.co_varnames[: fnc.__code__.co_argcount])
        param_fnc.remove("x")
        param_fnc.remove("y")
        seeds = {key: raw_seeds[key] for key in param_fnc}

        x, y = np.meshgrid(centers, centers)

        log_factorial_count_map = log_factorial(count_map)

        def f(*args):
            return -np.sum(
                log_poisson(
                    count_map,
                    fnc(x, y, *args) * exp_map / exp_map_total,
                    log_factorial_count_map,
                )
            )

        logger.info(f"seeds :\n{seeds}")
        m = Minuit(f, name=seeds.keys(), *seeds.values())
        for key, bound in bounds.items():
            if bound is None:
                m.fixed[key] = True
            else:
                m.limits[key] = bound
        m.errordef = Minuit.LIKELIHOOD
        m.simplex().migrad()
        if logger.level <= logging.INFO:
            func = fnc(x, y, **m.values.to_dict()) * exp_map / exp_map_total
            func[exp_map == 0] = 1
            rel_residuals = 100 * (count_map - func) / func
            logger.info(
                f"Fit valid : {m.valid}\n"
                f"Results ({fnc.__name__}) :\n{m.values.to_dict()}"
            )
            logger.info(
                "Average relative residuals : %.1f %%," % (np.mean(rel_residuals))
                + "Std = %.2f %%" % (np.std(rel_residuals))
                + "\n"
            )

        return fnc(x, y, **m.values.to_dict())

    def create_acceptance_map(
        self,
        observations: Observations,
        rotate_all_to_obs: Optional[Observation] = None,
        zd_correction: Optional[Background3D] = None,
    ) -> Background3D:
        """
        Calculate a 3D grid acceptance map

        Parameters
        ----------
        observations : gammapy.data.observations.Observations
            The collection of observations used to make the acceptance map
        rotate_all_to_obs : gammapy.data.observations.Observation
            All events of `observations` are rotate in camera coordinates to match
            the azimuth angle of `rotate_all_to_obs`.
            For stereoscopic telescopes, like MAGIC, the morphology changes with azimuth.
            Therefore the statistics can be enhacned with this instead of creating a background
            model for multiple azimuth bins.
        Returns
        -------
        acceptance_map : gammapy.irf.background.Background3D
        """

        # Compute base data
        count_map_background, exp_map_background, exp_map_background_total, livetime = (
            self._create_base_computation_map(
                observations, rotate_all_to_obs, zd_correction
            )
        )

        # Downsample map to bkg model resolution
        count_map_background_downsample = count_map_background.downsample(
            self.oversample_map, preserve_counts=True
        )
        exp_map_background_downsample = exp_map_background.downsample(
            self.oversample_map, preserve_counts=True
        )
        exp_map_background_total_downsample = exp_map_background_total.downsample(
            self.oversample_map, preserve_counts=True
        )

        # Create axis for bkg model
        edges = self.offset_axis.edges
        extended_edges = np.concatenate((-np.flip(edges), edges[1:]), axis=None)
        extended_offset_axis_x = MapAxis.from_edges(extended_edges, name="fov_lon")
        bin_width_x = np.repeat(
            extended_offset_axis_x.bin_width[:, np.newaxis],
            extended_offset_axis_x.nbin,
            axis=1,
        )
        extended_offset_axis_y = MapAxis.from_edges(extended_edges, name="fov_lat")
        bin_width_y = np.repeat(
            extended_offset_axis_y.bin_width[np.newaxis, :],
            extended_offset_axis_y.nbin,
            axis=0,
        )

        # Compute acceptance_map

        if self.method == "stack":
            corrected_counts = count_map_background_downsample.data * (
                exp_map_background_total_downsample.data
                / exp_map_background_downsample.data
            )
        elif self.method == "fit":
            logger.info(f"Performing the background fit using {self.fit_fnc}.")
            corrected_counts = np.empty(count_map_background_downsample.data.shape)
            for e in range(count_map_background_downsample.data.shape[0]):
                logger.info(
                    f"Energy bin : [{self.energy_axis.edges[e]:.2f},{self.energy_axis.edges[e + 1]:.2f}]"
                )
                corrected_counts[e] = self.fit_background(
                    count_map_background_downsample.data[e].astype(int),
                    exp_map_background_total_downsample.data[e],
                    exp_map_background_downsample.data[e],
                )
        else:
            raise NotImplementedError(f"Requested method '{self.method}' is not valid.")
        solid_angle = (
            4.0 * (np.sin(bin_width_x / 2.0) * np.sin(bin_width_y / 2.0)) * u.steradian
        )
        data_background = (
            corrected_counts
            / solid_angle[np.newaxis, :, :]
            / self.energy_axis.bin_width[:, np.newaxis, np.newaxis]
            / livetime
        )
        data_background = np.swapaxes(data_background, 1, 2)

        acceptance_map = Background3D(
            axes=[self.energy_axis, extended_offset_axis_x, extended_offset_axis_y],
            data=data_background.to(u.Unit("s-1 MeV-1 sr-1")),
            fov_alignment=FoVAlignment.ALTAZ,
        )

        return acceptance_map

    def _create_base_computation_map(
        self,
        observations: Observations,
        rotate_all_to_obs: Optional[Observation] = None,
        zd_correction: Optional[Background3D] = None,
    ) -> Tuple[WcsNDMap, WcsNDMap, WcsNDMap, u.Quantity]:
        """
        From a list of observations return a stacked finely binned counts and exposure map in camera frame to compute a
        model

        Parameters
        ----------
        observations : gammapy.data.observations.Observations
            The list of observations
        rotate_all_to_obs : gammapy.data.observations.Observation
            All events of `observations` are rotate in camera coordinates to match
            the azimuth angle of `rotate_all_to_obs`.
            For stereoscopic telescopes, like MAGIC, the morphology changes with azimuth.
            Therefore the statistics can be enhacned with this instead of creating a background
            model for multiple azimuth bins.

        Returns
        -------
        count_map_background : gammapy.map.WcsNDMap
            The count map
        exp_map_background : gammapy.map.WcsNDMap
            The exposure map corrected for exclusion regions
        exp_map_background_total : gammapy.map.WcsNDMap
            The exposure map without correction for exclusion regions
        livetime : astropy.unit.Quantity
            The total exposure time for the model
        """
        count_map_background = WcsNDMap(geom=self.geom)
        exp_map_background = WcsNDMap(geom=self.geom, unit=u.s)
        exp_map_background_total = WcsNDMap(geom=self.geom, unit=u.s)
        livetime = 0.0 * u.s

        with erfa_astrom.set(ErfaAstromInterpolator(1000 * u.s)):
            for obs in observations:
                # Filter events in exclusion regions
                geom = RegionGeom.from_regions(self.exclude_regions)
                mask = geom.contains(obs.events.radec)
                obs._events = obs.events.select_row_subset(~mask)
                # Create a count map in camera frame
                camera_frame_obs, rot_angle = self._transform_obs_to_camera_frame(
                    obs, rotate_all_to_obs
                )
                count_map_obs, _ = self._create_map(
                    camera_frame_obs, self.geom, [], add_bkg=False
                )

                # Remove Gradient with zd_correction
                if zd_correction is not None:
                    rot_zd_corr = rotate(
                        zd_correction.data,
                        -rot_angle.to_value("deg"),
                        axes=[2, 1],
                        reshape=False,
                        order=1,
                    )

                    count_map_obs.counts.data = (
                        count_map_obs.counts.data
                        - rot_zd_corr / 2 * count_map_obs.counts.data
                        + zd_correction.data / 2 * count_map_obs.counts.data
                    )

                # Create exposure maps and fill them with the obs livetime
                exp_map_obs = MapDataset.create(geom=count_map_obs.geoms["geom"])
                exp_map_obs_total = MapDataset.create(geom=count_map_obs.geoms["geom"])
                exp_map_obs.counts.data = (
                    camera_frame_obs.observation_live_time_duration.value
                )
                exp_map_obs_total.counts.data = (
                    camera_frame_obs.observation_live_time_duration.value
                )

                # Evaluate the average exclusion mask in camera frame
                # by evaluating it on time intervals short compared to the field of view rotation
                exclusion_mask = np.zeros(count_map_obs.counts.data.shape[1:])
                time_interval = self._compute_time_intervals_based_on_fov_rotation(obs)
                for i in range(len(time_interval) - 1):
                    # Compute the exclusion region in camera frame for the average time
                    dtime = time_interval[i + 1] - time_interval[i]
                    time = time_interval[i] + dtime / 2
                    average_alt_az_frame = AltAz(
                        obstime=time, location=obs.meta.location
                    )
                    average_alt_az_pointing = obs.get_pointing_icrs(time).transform_to(
                        average_alt_az_frame
                    )
                    exclusion_region_camera_frame = (
                        self._transform_exclusion_region_to_camera_frame(
                            average_alt_az_pointing,
                            rotate_all_to_obs,
                            rot_angle,
                        )
                    )
                    geom_image = self.geom.to_image()

                    exclusion_mask_t = (
                        ~geom_image.region_mask(exclusion_region_camera_frame)
                        if len(exclusion_region_camera_frame) > 0
                        else ~Map.from_geom(geom_image)
                    )
                    # Add the exclusion mask in camera frame weighted by the time interval duration
                    exclusion_mask += exclusion_mask_t * (dtime).value
                # Normalise the exclusion mask by the full observation duration
                exclusion_mask *= 1 / (time_interval[-1] - time_interval[0]).value

                # Correct the exposure map by the exclusion region
                for j in range(count_map_obs.counts.data.shape[0]):
                    exp_map_obs.counts.data[j, :, :] = (
                        exp_map_obs.counts.data[j, :, :] * exclusion_mask
                    )

                # Stack counts and exposure maps and livetime of all observations
                count_map_background.data += count_map_obs.counts.data
                exp_map_background.data += exp_map_obs.counts.data
                exp_map_background_total.data += exp_map_obs_total.counts.data
                livetime += camera_frame_obs.observation_live_time_duration

        return (
            count_map_background,
            exp_map_background,
            exp_map_background_total,
            livetime,
        )
