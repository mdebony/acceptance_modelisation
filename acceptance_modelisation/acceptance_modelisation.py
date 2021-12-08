import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from gammapy.datasets import MapDataset
from gammapy.irf import Background2D
from gammapy.makers import MapDatasetMaker
from gammapy.maps import WcsNDMap, WcsGeom
from regions import CircleAnnulusSkyRegion


def create_radial_acceptance_map(observations, energy_axis, offset_axis, oversample_map=10, exclude_regions=[]):
    """
        Calculate a radial acceptance map
        
        Parameters
        ----------
        observations : gammapy.data.observations.Observations
            The collection of observations used to make the acceptance map
        energy_axis : gammapy.maps.geom.MapAxis
            The energy axis for the acceptance model
        offset_axis : gammapy.maps.geom.MapAxis
            The offset axis for the acceptance model
        oversample_map : int
            oversample in number of pixel of the spatial axis used for the calculation
        exclude_regions : list of 'regions.SkyRegion'
            Region with sources, will be excluded of the calculation of the acceptance map

        Returns
        -------
        background : gammapy.irf.background.Background2D
            A bakground model that could be used as an acceptance model
    """

    n_bins_map = offset_axis.nbin * oversample_map * 2
    binsz = offset_axis.edges[-1] / (n_bins_map / 2)
    center_map = SkyCoord(ra=0. * u.deg, dec=0. * u.deg, frame='icrs')

    geom = WcsGeom.create(skydir=center_map, npix=(n_bins_map, n_bins_map), binsz=binsz, frame="icrs",
                          axes=[energy_axis])

    count_map_background = WcsNDMap(geom=geom)
    exp_map_background = WcsNDMap(geom=geom, unit=u.s)
    exp_map_background_total = WcsNDMap(geom=geom, unit=u.s)
    livetime = 0. * u.s

    for obs in observations:
        geom_obs = WcsGeom.create(skydir=obs.pointing_radec, npix=(n_bins_map, n_bins_map), binsz=binsz, frame="icrs",
                                  axes=[energy_axis])
        count_map_obs = MapDataset.create(geom=geom_obs)
        exp_map_obs = MapDataset.create(geom=geom_obs)
        exp_map_obs_total = MapDataset.create(geom=geom_obs)
        maker = MapDatasetMaker(selection=["counts"])

        geom_image = geom.to_image()
        exclusion_mask = ~geom_image.region_mask(exclude_regions)

        count_map_obs = maker.run(MapDataset.create(geom=geom_obs), obs)

        exp_map_obs.counts.data = obs.observation_live_time_duration.value
        exp_map_obs_total.counts.data = obs.observation_live_time_duration.value

        for i in range(count_map_obs.counts.data.shape[0]):
            count_map_obs.counts.data[i, :, :] = count_map_obs.counts.data[i, :, :] * exclusion_mask
            exp_map_obs.counts.data[i, :, :] = exp_map_obs.counts.data[i, :, :] * exclusion_mask

        count_map_background.data += count_map_obs.counts.data
        exp_map_background.data += exp_map_obs.counts.data
        exp_map_background_total.data += exp_map_obs_total.counts.data
        livetime += obs.observation_live_time_duration

    data_background = np.zeros((energy_axis.nbin, offset_axis.nbin)) * u.Unit('s-1 MeV-1 sr-1')
    for i in range(offset_axis.nbin):
        selection_region = CircleAnnulusSkyRegion(center=center_map, inner_radius=offset_axis.edges[i],
                                                  outer_radius=offset_axis.edges[i + 1])
        selection_map = geom.to_image().region_mask([selection_region])
        for j in range(energy_axis.nbin):
            value = u.dimensionless_unscaled * np.sum(count_map_background.data[j, :, :] * selection_map)
            value *= np.sum(exp_map_background_total.data[j, :, :] * selection_map) / np.sum(
                exp_map_background.data[j, :, :] * selection_map)

            value /= (energy_axis.edges[j + 1] - energy_axis.edges[j])
            value /= 2. * np.pi * (offset_axis.edges[i + 1].to('radian') - offset_axis.edges[i].to('radian')) * \
                     offset_axis.center[i].to('radian')
            value /= livetime
            data_background[j, i] = value

    background = Background2D(axes=[energy_axis, offset_axis], data=data_background)

    return background
