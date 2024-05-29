# Description

This package create acceptance model to be used for IACT analysis with gammapy

# Installation


```bash
git clone https://github.com/mdebony/acceptance_modelisation.git
cd acceptance_modelisation
python setup.py install
```

Dependencies :
- numpy
- scipy
- astropy
- gammapy 1.1
- regions 0.7

# Example of use

## Basic use

You could first create the acceptance model

```python
from gammapy.maps import MapAxis
from gammapy.data import DataStore
from regions import CircleSkyRegion
import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from acceptance_modelisation import RadialAcceptanceMapCreator

# The observations to use for creating the acceptance model
data_store = DataStore.from_dir("$GAMMAPY_DATA/hess-dl3-dr1")
obs_collection = data_store.get_observations([23523, 23526, 23559, 23592])

# The exclusion regions to apply during acceptance model calculation
exclude_regions = [CircleSkyRegion(center=SkyCoord.from_name('Crab'),
                                   radius=0.2 * u.deg), ]

# Define the binning of the model
e_min, e_max = 0.1 * u.TeV, 10. * u.TeV
size_fov = 2.5 * u.deg
offset_axis_acceptance = MapAxis.from_bounds(0. * u.deg, size_fov, nbin=6, name='offset')
energy_axis_acceptance = MapAxis.from_energy_bounds(e_min, e_max, nbin=6, name='energy')

acceptance_model_creator = RadialAcceptanceMapCreator(energy_axis_acceptance,
                                                          offset_axis_acceptance,
                                                          exclude_regions=exclude_regions,
                                                          oversample_map=10)
acceptance_model = acceptance_model_creator.create_acceptance_map(obs_collection)

```

You can then check the acceptance model by plotting it using

```python
acceptance_model.peek()
```

To use it with gammapy, you could first save it on a FITS file

```python
hdu_acceptance = acceptance_model.to_table_hdu()
hdu_acceptance.writeto('acceptance.fits', overwrite=True)
```

It's then possible to load the acceptance model in the current gammapy DataStore with this code.
You would need then to recreate you gammapy Observations object in order than the acceptance model is taken into account for the analysis.

```python
data_store.hdu_table.remove_rows(data_store.hdu_table['HDU_TYPE']=='bkg')
for obs_id in np.unique(data_store.hdu_table['OBS_ID']):
    data_store.hdu_table.add_row({'OBS_ID': obs_id, 
                                 'HDU_TYPE': 'bkg',
                                 "HDU_CLASS": "bkg_2d",
                                 "FILE_DIR": "",
                                 "FILE_NAME": 'acceptance.fits',
                                 "HDU_NAME": "BACKGROUND",
                                 "SIZE": hdu_acceptance.size})
data_store.hdu_table = data_store.hdu_table.copy()

obs_collection = data_store.get_observations([23523, 23526, 23559, 23592])

data_store.hdu_table
```

## Telescope position

**The observations should contain the telescope position in order to have the algorithm working.**
If the information is missing in the DL3, you could either add it or it possible to add it directly to the observation as shown in the example below.
```python
from astropy.coordinates import EarthLocation

# Your telescope position in an EarthLocation object
loc = EarthLocation.of_site('Roque de los Muchachos')

# Add telescope position to observations
for i in obs_collection:
    obs_collection[i].obs_info['GEOLON'] = loc.lon.value
    obs_collection[i].obs_info['GEOLAT'] = loc.lat.value
    obs_collection[i].obs_info['GEOALT'] = loc.height.value
    obs_collection[i]._location = loc
```

## Runwise norm of the model 

It's also possible to fit the normalisation of the model per run. For this use the method create_acceptance_map_per_observation .
In that case the output is a dictionary containing the acceptance model of each observations (with the observation Id as index).
```python
acceptance_model_creator = RadialAcceptanceMapCreator(energy_axis_acceptance,
                                                      offset_axis_acceptance,
                                                      exclude_regions=exclude_regions,
                                                      oversample_map=10)
acceptance_models = acceptance_model_creator.create_acceptance_map_per_observation(obs_collection)
```

## Zenith binned model

It's also possible to create model binned per cos zenith. For this use the method create_acceptance_map_per_observation but with the option `zenith_binning` set at True.
The width of zenith bin could be control at the creation of the object with the parameter `initial_cos_zenith_binning`.
You can chose the `cos_zenith_binning_method`: `min_livetime` or `min_n_observation`  to give a condition on minimum livetime or number of observations for each bin. The algorithm will then automatically rebin to larger bin in order to have in each bin at least `cos_zenith_binning_parameter_value` livetime (in seconds) or observation per bin.
In that case the output is a dictionary containing the acceptance model of each observations (with the observation ID as index).
Set `verbose` to True to get the binning result and a plot of the binned livetime.
```python
acceptance_model_creator = RadialAcceptanceMapCreator(energy_axis_acceptance,
                                                      offset_axis_acceptance,
                                                      exclude_regions=exclude_regions,
                                                      oversample_map=10,
                                                      cos_zenith_binning_method='min_livetime',
                                                      cos_zenith_binning_parameter_value=3600,
                                                      initial_cos_zenith_binning=0.01,
                                                      verbose=True)
acceptance_models = acceptance_model_creator.create_acceptance_map_per_observation(obs_collection,
                                                                                   zenith_binning=True)
```

## Zenith interpolated model

It's also possible to create model interpolated between the binned model per cos zenith. For this use the method create_acceptance_map_per_observation but with the option `zenith_interpolation` set at True. All the parameters controlling the cos zenith binning remain active.
```python
acceptance_model_creator = RadialAcceptanceMapCreator(energy_axis_acceptance,
                                                      offset_axis_acceptance,
                                                      exclude_regions=exclude_regions,
                                                      oversample_map=10,
                                                      min_run_per_cos_zenith_bin=3,
                                                      initial_cos_zenith_binning=0.01)
acceptance_models = acceptance_model_creator.create_acceptance_map_per_observation(obs_collection,
                                                                                   zenith_binning=True,
                                                                                   zenith_interpolation=True)
```

# Available model

All models have an identical interface. You just need to change the class used to change the model created.

There are two model currently available :

- A 2D model with hypothesis of a radial symmetry of the background across the FoV. This is the class `RadialAcceptanceMapCreator`.
    ````python
    from acceptance_modelisation import RadialAcceptanceMapCreator
    acceptance_model_creator = RadialAcceptanceMapCreator(energy_axis_acceptance,
                                                          offset_axis_acceptance,
                                                          exclude_regions=exclude_regions)
    acceptance_models = acceptance_model_creator.create_acceptance_map_per_observation(obs_collection)     
    ````
- A 3D model with a regular grid describing the FoV. This is the class `Grid3DAcceptanceMapCreator`.
    ````python
    from acceptance_modelisation import Grid3DAcceptanceMapCreator
    acceptance_model_creator = Grid3DAcceptanceMapCreator(energy_axis_acceptance,
                                                          offset_axis_acceptance,
                                                          exclude_regions=exclude_regions)
    acceptance_models = acceptance_model_creator.create_acceptance_map_per_observation(obs_collection)     
    ````