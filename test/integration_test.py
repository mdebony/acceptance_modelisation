import os

import astropy.units as u
from astropy.coordinates import EarthLocation, SkyCoord
from gammapy.data import DataStore
from gammapy.irf import Background3D, Background2D
from gammapy.maps import MapAxis
from regions import CircleSkyRegion

from baccmod import RadialAcceptanceMapCreator, Grid3DAcceptanceMapCreator

import gammapy
gammapy_version = gammapy.__version__
gammapy_ver_major = int(gammapy_version.split('.')[0])
gammapy_ver_minor = int(gammapy_version.split('.')[1])
gammapy_ver_patch = 0
if len(gammapy_version.split('.')) > 2:
    gammapy_ver_patch = int(gammapy_version.split('.')[2])



class TestIntegrationClass:
    datastore = DataStore.from_dir(os.path.join(os.environ['GAMMAPY_DATA'], 'hess-dl3-dr1'))

    coordinate_pks_2155 = SkyCoord.from_name('PKS 2155-304')
    exclude_region_PKS_2155 = [CircleSkyRegion(coordinate_pks_2155, 0.4*u.deg)]
    separation_obs_pks_2155 = SkyCoord(ra=datastore.obs_table['RA_PNT'], dec=datastore.obs_table['DEC_PNT']).separation(coordinate_pks_2155)
    id_obs_pks_2155 = datastore.obs_table['OBS_ID'][separation_obs_pks_2155< 2.*u.deg]
    obs_collection_pks_2155 = datastore.get_observations(obs_id=id_obs_pks_2155, required_irf='all-optional')

    # Inject HESS site information in the run
    for i in obs_collection_pks_2155:
        if gammapy_ver_minor < 3:
            obs_collection_pks_2155[i].obs_info['GEOLON'] = 16.50004902672975
            obs_collection_pks_2155[i].obs_info['GEOLAT'] = -23.271584051253615
            obs_collection_pks_2155[i].obs_info['GEOALT'] = 1800
        obs_collection_pks_2155[i]._location = EarthLocation.from_geodetic(lon=16.50004902672975*u.deg, lat=-23.271584051253615*u.deg, height=1800.*u.m)

    energy_axis = MapAxis.from_energy_bounds(50.*u.GeV, 1.*u.TeV, nbin=5, per_decade=True, name='energy')
    offset_axis = MapAxis.from_bounds(0.*u.deg, 2.*u.deg, nbin=6, name='offset')

    def test_integration_3D(self):
        bkg_maker = Grid3DAcceptanceMapCreator(energy_axis=self.energy_axis,
                                               offset_axis=self.offset_axis,
                                               oversample_map=5,
                                               exclude_regions=self.exclude_region_PKS_2155)
        background_model = bkg_maker.create_acceptance_map(observations=self.obs_collection_pks_2155)
        assert type(background_model) is Background3D

    def test_integration_2D(self):
        bkg_maker = RadialAcceptanceMapCreator(energy_axis=self.energy_axis,
                                               offset_axis=self.offset_axis,
                                               oversample_map=5,
                                               exclude_regions=self.exclude_region_PKS_2155)
        background_model = bkg_maker.create_acceptance_map(observations=self.obs_collection_pks_2155)
        assert type(background_model) is Background2D

    def test_integration_zenith_binned_model(self):
        bkg_maker = RadialAcceptanceMapCreator(energy_axis=self.energy_axis,
                                               offset_axis=self.offset_axis,
                                               oversample_map=5,
                                               exclude_regions=self.exclude_region_PKS_2155)
        background_model = bkg_maker.create_acceptance_map_cos_zenith_binned(observations=self.obs_collection_pks_2155)
        assert type(background_model) is dict
        for id_obs in self.id_obs_pks_2155:
            assert id_obs in background_model
            assert type(background_model[id_obs]) is Background2D

    def test_integration_zenith_interpolated_model(self):
        bkg_maker = RadialAcceptanceMapCreator(energy_axis=self.energy_axis,
                                               offset_axis=self.offset_axis,
                                               oversample_map=5,
                                               exclude_regions=self.exclude_region_PKS_2155)
        background_model = bkg_maker.create_acceptance_map_cos_zenith_interpolated(observations=self.obs_collection_pks_2155)
        assert type(background_model) is dict
        for id_obs in self.id_obs_pks_2155:
            assert id_obs in background_model
            assert type(background_model[id_obs]) is Background2D

    def test_integration_zenith_interpolated_model_mini_irf_and_run_splitting(self):
        bkg_maker = RadialAcceptanceMapCreator(energy_axis=self.energy_axis,
                                               offset_axis=self.offset_axis,
                                               oversample_map=5,
                                               exclude_regions=self.exclude_region_PKS_2155,
                                               use_mini_irf_computation=True,
                                               zenith_binning_run_splitting=True)
        background_model = bkg_maker.create_acceptance_map_cos_zenith_interpolated(observations=self.obs_collection_pks_2155)
        assert type(background_model) is dict
        for id_obs in self.id_obs_pks_2155:
            assert id_obs in background_model
            assert type(background_model[id_obs]) is Background2D

    def test_integration_norm_per_run(self):
        bkg_maker = RadialAcceptanceMapCreator(energy_axis=self.energy_axis,
                                               offset_axis=self.offset_axis,
                                               oversample_map=5,
                                               exclude_regions=self.exclude_region_PKS_2155)
        background_model = bkg_maker.create_acceptance_map_per_observation(observations=self.obs_collection_pks_2155)
        assert type(background_model) is dict
        for id_obs in self.id_obs_pks_2155:
            assert id_obs in background_model
            assert type(background_model[id_obs]) is Background2D
