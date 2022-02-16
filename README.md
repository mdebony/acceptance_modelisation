# Description

This package create radial acceptance model to be used for IACT analysis with gammapy

# Installation


```bash
git clone https://github.com/mdebony/acceptance_modelisation.git
cd acceptance_modelisation
python setup.py install
```

Dependencies :
- numpy
- gammapy 0.19
- regions
- astropy

# Example of use

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
exclude_regions=[CircleSkyRegion(center=SkyCoord.from_name('Crab'),
                                 radius=0.2*u.deg),]

# Define the binning of the model
e_min, e_max = 0.1*u.TeV, 10.*u.TeV
size_fov = 2.5*u.deg
offset_axis_acceptance = MapAxis.from_bounds(0.*u.deg, size_fov, nbin=6, name='offset')
energy_axis_acceptance = MapAxis.from_energy_bounds(e_min, e_max, nbin=6, name='energy')


acceptance_model_creator = RadialAcceptanceMapCreator(energy_axis_acceptance,
                                                      offset_axis_acceptance,
                                                      exclude_regions=exclude_regions,
                                                      oversample_map=100)
acceptance_model = acceptance_model_creator.create_radial_acceptance_map(obs_collection)

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

It's also possible to fit the normalisation of the model per run. For this use the method create_radial_acceptance_map_per_observation .
In that case the output is a dictionary containing the acceptance model of each observations (with the observation Id as index).
```python
acceptance_model_creator = RadialAcceptanceMapCreator(energy_axis_acceptance,
                                                      offset_axis_acceptance,
                                                      exclude_regions=exclude_regions,
                                                      oversample_map=100)
acceptance_models = acceptance_model_creator.create_radial_acceptance_map_per_observation(obs_collection)
```
