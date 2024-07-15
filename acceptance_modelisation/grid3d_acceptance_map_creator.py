import logging
from typing import List, Optional

import astropy.units as u
import numpy as np
from gammapy.data import Observations
from gammapy.irf import Background3D, FoVAlignment
from gammapy.maps import MapAxis
from iminuit import Minuit
from regions import SkyRegion

from .base_acceptance_map_creator import BaseAcceptanceMapCreator
from .modeling import FIT_FUNCTION, log_factorial, log_poisson

logger = logging.getLogger(__name__)


class Grid3DAcceptanceMapCreator(BaseAcceptanceMapCreator):

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
                 time_resolution_run_splitting: u.Quantity = 0.1 * u.s,
                 method='stack',
                 fit_fnc='gaussian2d',
                 fit_seeds=None,
                 fit_bounds=None) -> None:
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
        max_fraction_pixel_rotation_fov : float, optional
            For camera frame transformation the maximum size relative to a pixel a rotation is allowed
       time_resolution_run_splitting : astropy.units.Quantity, optional
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
        """

        # If no exclusion region, default it as an empty list
        if exclude_regions is None:
            exclude_regions = []

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

        self.method = method
        self.fit_fnc = fit_fnc
        self.fit_seeds = fit_seeds
        self.fit_bounds = fit_bounds

        # Initiate upper instance
        super().__init__(energy_axis=energy_axis,
                         max_offset=max_offset,
                         spatial_resolution=spatial_resolution,
                         exclude_regions=exclude_regions,
                         cos_zenith_binning_method=cos_zenith_binning_method,
                         cos_zenith_binning_parameter_value=cos_zenith_binning_parameter_value,
                         initial_cos_zenith_binning=initial_cos_zenith_binning,
                         max_angular_separation_wobble=max_angular_separation_wobble,
                         max_fraction_pixel_rotation_fov=max_fraction_pixel_rotation_fov,
                         zenith_binning_run_splitting=zenith_binning_run_splitting,
                         time_resolution_run_splitting=time_resolution_run_splitting)

    def fit_background(self, count_map, exp_map_total, exp_map):
        centers = self.offset_axis.center.to_value(u.deg)
        centers = np.concatenate((-np.flip(centers), centers), axis=None)
        raw_seeds = {}
        bounds = {}
        if type(self.fit_fnc) == str:
            try:
                fnc = FIT_FUNCTION[self.fit_fnc]
            except KeyError:
                logger.error(f"Invalid built-in fit_fnc. Use {FIT_FUNCTION.keys()} or a custom function.")
                raise
            raw_seeds = fnc.default_seeds.copy()
            bounds = fnc.default_bounds.copy()
        else:
            fnc = self.fit_fnc
        if self.fit_seeds is not None:
            raw_seeds.update(self.fit_seeds)
        if self.fit_bounds is not None:
            bounds.update(self.fit_bounds)

        mask = exp_map > 0  # Handles fully overlapping exclusion regions in the 'size' seed computation
        # Seeds the charge normalisation to the observed counts corrected for exclusion region reduction to exposure
        raw_seeds['size'] = np.sum(count_map[mask] * exp_map_total[mask] / exp_map[mask]) / np.mean(mask)
        bounds['size'] = (raw_seeds['size'] * 0.1, raw_seeds['size'] * 10)

        # reorder seeds to fnc parameter order
        param_fnc = list(fnc.__code__.co_varnames[:fnc.__code__.co_argcount])
        param_fnc.remove('x')
        param_fnc.remove('y')
        seeds = {key: raw_seeds[key] for key in param_fnc}

        x, y = np.meshgrid(centers, centers)

        log_factorial_count_map = log_factorial(count_map)

        def f(*args):
            return -np.sum(
                log_poisson(count_map, fnc(x, y, *args) * exp_map / exp_map_total, log_factorial_count_map))

        logger.info(f"seeds :\n{seeds}")
        m = Minuit(f,
                   name=seeds.keys(),
                   *seeds.values())
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
            logger.info(f"Fit valid : {m.valid}\n"
                        f"Results ({fnc.__name__}) :\n{m.values.to_dict()}")
            logger.info("Average relative residuals : %.1f %%," % (np.mean(rel_residuals)) +
                        "Std = %.2f %%" % (np.std(rel_residuals)) + "\n")

        return fnc(x, y, **m.values.to_dict())

    def create_acceptance_map(self, observations: Observations) -> Background3D:
        """
        Calculate a 3D grid acceptance map

        Parameters
        ----------
        observations : gammapy.data.observations.Observations
            The collection of observations used to make the acceptance map

        Returns
        -------
        acceptance_map : gammapy.irf.background.Background3D
        """

        # Compute base data
        count_map_background, exp_map_background, exp_map_background_total, livetime = self._create_base_computation_map(
            observations)

        # Downsample map to bkg model resolution
        count_map_background_downsample = count_map_background.downsample(self.oversample_map, preserve_counts=True)
        exp_map_background_downsample = exp_map_background.downsample(self.oversample_map, preserve_counts=True)
        exp_map_background_total_downsample = exp_map_background_total.downsample(self.oversample_map,
                                                                                  preserve_counts=True)

        # Create axis for bkg model
        edges = self.offset_axis.edges
        extended_edges = np.concatenate((-np.flip(edges), edges[1:]), axis=None)
        extended_offset_axis_x = MapAxis.from_edges(extended_edges, name='fov_lon')
        bin_width_x = np.repeat(extended_offset_axis_x.bin_width[:, np.newaxis], extended_offset_axis_x.nbin, axis=1)
        extended_offset_axis_y = MapAxis.from_edges(extended_edges, name='fov_lat')
        bin_width_y = np.repeat(extended_offset_axis_y.bin_width[np.newaxis, :], extended_offset_axis_y.nbin, axis=0)

        # Compute acceptance_map

        if self.method == 'stack':
            corrected_counts = count_map_background_downsample.data * (exp_map_background_total_downsample.data /
                                                                       exp_map_background_downsample.data)
        elif self.method == 'fit':
            logger.info(f"Performing the background fit using {self.fit_fnc}.")
            corrected_counts = np.empty(count_map_background_downsample.data.shape)
            for e in range(count_map_background_downsample.data.shape[0]):
                logger.info(f"Energy bin : [{self.energy_axis.edges[e]:.2f},{self.energy_axis.edges[e + 1]:.2f}]")
                corrected_counts[e] = self.fit_background(count_map_background_downsample.data[e].astype(int),
                                                          exp_map_background_total_downsample.data[e],
                                                          exp_map_background_downsample.data[e],
                                                          )
        else:
            raise NotImplementedError(f"Requested method '{self.method}' is not valid.")
        solid_angle = 4. * (np.sin(bin_width_x / 2.) * np.sin(bin_width_y / 2.)) * u.steradian
        data_background = corrected_counts / solid_angle[np.newaxis, :, :] / self.energy_axis.bin_width[:, np.newaxis,
                                                                             np.newaxis] / livetime

        acceptance_map = Background3D(axes=[self.energy_axis, extended_offset_axis_x, extended_offset_axis_y],
                                      data=data_background.to(u.Unit('s-1 MeV-1 sr-1')),
                                      fov_alignment=FoVAlignment.ALTAZ)

        return acceptance_map
