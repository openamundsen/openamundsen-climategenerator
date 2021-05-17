import numpy as np
import openamundsen as oa
import pandas as pd
from pathlib import Path
import scipy.stats
import tempfile
import xarray as xr
from .conf import parse_config


def min_dist(target_temp, temps, target_precip, precips):
    dists = np.sqrt((temps - target_temp)**2 + (precips - target_precip)**2).values
    year_idx, slice_idx = np.unravel_index(dists.argmin(), dists.shape)
    selected_year = temps.year.values[year_idx]
    selected_slice = temps.slice.values[slice_idx]
    return (selected_year, selected_slice)


DUMMY_DEM = '''
ncols        1
nrows        1
xllcorner    0
yllcorner    0
cellsize    {resolution}
0
'''.strip()


class ClimateGenerator:
    def __init__(self, config):
        self.config = parse_config(config)

        self.sim_dates = pd.date_range(
            start=config.sim_start_date,
            end=config.sim_end_date,
            freq=config.timestep,
        )
        self.sim_years = self.sim_dates.year.unique()

        oa_config = {
            'start_date': config.obs_start_date,
            'end_date': config.obs_end_date,
            'timestep': config.timestep,

            'input_data': {
                'meteo': {
                    'dir': config.input_dir,
                    'format': config.input_format,
                    'crs': config.crs,
                },
            },

            # Dummy fields
            'domain': 'dummy',
            'resolution': 1,
            'timezone': 0,
            'crs': 'epsg:4326',
        }

        self.oa_config = oa.parse_config(oa_config)
        self.params = ['temp', 'precip', 'rel_hum', 'sw_in', 'wind_speed']

        np.random.seed(config.random_seed)

    def initialize(self):
        with tempfile.TemporaryDirectory() as tempdir:
            oa_config = self.oa_config
            res = oa_config.resolution

            with open(f'{tempdir}/dem_dummy_{res}.asc', 'w') as f:
                f.write(DUMMY_DEM.format(resolution=res))

            oa_config['input_data']['grids'] = {'dir': tempdir}
            model = oa.OpenAmundsen(oa_config)
            model.initialize()
            self.oa_config = model.config

        data_obs = model.meteo.copy()
        data_obs = data_obs[['station_name', 'lon', 'lat', 'alt'] + self.params]

        if self.config.ref_station is None:
            num_vals = (data_obs.temp.notnull() & data_obs.precip.notnull()).sum('time')
            self.config.ref_station = str(num_vals.station[num_vals.argmax()].values)

        obs_years = data_obs.time.to_index().year.unique()
        num_slices_per_year = 365 // self.config.slice_length
        num_timesteps_per_slice = int(
            self.config.slice_length * oa.constants.HOURS_PER_DAY * oa.constants.SECONDS_PER_HOUR
            / model.timestep
        )

        self.data_obs = data_obs
        self.obs_years = obs_years
        self.stations = list(data_obs.station.values)
        self.num_slices_per_year = num_slices_per_year
        self.num_timesteps_per_slice = num_timesteps_per_slice

        self._assign_slices_obs()

        ref_data_sliced = self.data_obs_sliced.loc[self.config.ref_station, :, :, :, :]
        self.ref_obs_data_slicemean = ref_data_sliced.mean('slice_timestep')
        self.ref_obs_data_slicemean_yearmean = self.ref_obs_data_slicemean.mean('year')

        self._calc_correlations()

    def run(self):
        self._generate_reference_climate()
        self._identify_slices()
        self._assign_slices_sim()
        self._reshape_and_fill_missing()

        if self.config.output_dir is not None:
            self._write_output()

        if self.config.report_file is not None:
            from .report import generate_report
            generate_report(self, self.config.report_file)

        return self.data_sim

    def _assign_slices_obs(self):
        data_obs = self.data_obs
        num_slices_per_year = self.num_slices_per_year
        num_timesteps_per_slice = self.num_timesteps_per_slice

        data_obs_sliced = xr.DataArray(
            np.full(
                (
                    len(self.stations),
                    len(self.obs_years),
                    len(self.params),
                    num_slices_per_year,
                    num_timesteps_per_slice,
                ),
                np.nan,
            ),
            coords=[
                self.stations,
                self.obs_years,
                self.params,
                range(num_slices_per_year),
                range(num_timesteps_per_slice),
            ],
            dims=[
                'station',
                'year',
                'param',
                'slice',
                'slice_timestep',
            ],
        )

        for station in data_obs.station:
            station = str(station.values)

            for year_num, year in enumerate(self.obs_years):
                data_cur = (
                    data_obs
                    .sel(station=station)
                    .reindex(
                        time=pd.date_range(
                            start=f'{year}-01-01',
                            end=oa.conf.parse_end_date(f'{year}-12-31', self.config.timestep),
                            freq=self.config.timestep,
                        ),
                    )
                )[self.params].to_array().values

                data_cur = data_cur[:, :(num_slices_per_year * num_timesteps_per_slice)]  # e.g. for 7-day slices, remove last 1 or 2 (in a leap year) days
                data_cur = data_cur.reshape((len(self.params), num_slices_per_year, num_timesteps_per_slice))
                data_obs_sliced.loc[station, year, :, :, :] = data_cur

        self.data_obs_sliced = data_obs_sliced

        # Assign a month to each slice (take the slice center date)
        # (required for the seasonal precipitation adjustment)
        slice_dates = (
            data_obs
            .reindex(
                time=pd.date_range(
                    start=f'{self.obs_years[0]}-01-01',
                    end=oa.conf.parse_end_date(f'{self.obs_years[0]}-12-31', self.config.timestep),
                    freq=self.config.timestep,
                ),
            )
        ).time.values[:(num_slices_per_year * num_timesteps_per_slice)]
        slice_dates = slice_dates.reshape((num_slices_per_year, num_timesteps_per_slice))
        slice_center_dates = slice_dates[:, slice_dates.shape[-1] // 2]
        self.slice_months = pd.to_datetime(slice_center_dates).month

    def _calc_correlations(self):
        # Compute correlation between mean temperature and precipitation for all slices
        temp_precip_linfits = np.zeros((self.num_slices_per_year, 2))
        temp_std = np.zeros(self.num_slices_per_year)
        precip_std = np.zeros(self.num_slices_per_year)
        for slice_num in range(self.num_slices_per_year):
            linreg_x = self.ref_obs_data_slicemean.loc[:, 'temp', slice_num]
            linreg_y = self.ref_obs_data_slicemean.loc[:, 'precip', slice_num]
            pos = np.isfinite(linreg_x) & np.isfinite(linreg_y)

            slope, intercept, _, _, _ = scipy.stats.linregress(linreg_x[pos], linreg_y[pos])
            temp_precip_linfits[slice_num, :] = slope, intercept

            temp_std[slice_num] = self.ref_obs_data_slicemean.loc[
                :,
                'temp',
                slice_num,
            ].std()
            precip_std[slice_num] = self.ref_obs_data_slicemean.loc[
                :,
                'precip',
                slice_num,
            ].std()

        self.temp_precip_linfits = temp_precip_linfits
        self.temp_std = temp_std
        self.precip_std = precip_std

    def _generate_reference_climate(self):
        obs_years = self.obs_years
        sim_years = self.sim_years
        num_sim_years = len(sim_years)
        num_slices_per_year = self.num_slices_per_year

        random_temp_variation = np.random.normal(size=(num_sim_years, num_slices_per_year))
        ref_sim_temp_slicemean = np.zeros((num_sim_years, num_slices_per_year))
        ref_sim_precip_slicemean = np.zeros((num_sim_years, num_slices_per_year))

        for year_num, year in enumerate(sim_years):
            print(f'Generating reference climate for year {year}')

            for slice_num in range(num_slices_per_year):
                ref_sim_temp_slicemean[year_num, slice_num] = (
                    self.ref_obs_data_slicemean_yearmean.loc['temp', slice_num]
                    + random_temp_variation[year_num, slice_num] * self.temp_std[slice_num]
                    + self.config.temperature_change * (
                        (year - obs_years[-1] - 1) + (slice_num + 1) / num_slices_per_year
                    )
                )

                ref_sim_precip_slicemean[year_num, slice_num] = (
                    self.temp_precip_linfits[slice_num, 0]
                    * ref_sim_temp_slicemean[year_num, slice_num]
                    + self.temp_precip_linfits[slice_num, 1]
                )

        # Apply precipitation change
        mam_change = self.config.precipitation_change.MAM
        jja_change = self.config.precipitation_change.JJA
        son_change = self.config.precipitation_change.SON
        djf_change = self.config.precipitation_change.DJF
        ref_sim_precip_slicemean[:, self.slice_months.isin([3, 4, 5])] *= mam_change
        ref_sim_precip_slicemean[:, self.slice_months.isin([6, 7, 8])] *= jja_change
        ref_sim_precip_slicemean[:, self.slice_months.isin([9, 10, 11])] *= son_change
        ref_sim_precip_slicemean[:, self.slice_months.isin([12, 1, 2])] *= djf_change

        self.ref_sim_temp_slicemean = ref_sim_temp_slicemean
        self.ref_sim_precip_slicemean = ref_sim_precip_slicemean

    def _identify_slices(self):
        slices = {}

        for year_num, year in enumerate(self.sim_years):
            print(f'Identifying slices for year {year}')

            slices[year] = {}

            for slice_num in range(self.num_slices_per_year):
                slice_candidates = slice_num + np.arange(
                    -self.config.max_slice_shift,
                    self.config.max_slice_shift + 1,
                )
                slice_candidates = slice_candidates[
                    (slice_candidates >= 0)
                    & (slice_candidates < self.num_slices_per_year)
                ]

                target_temp = (
                    self.ref_sim_temp_slicemean[year_num, slice_num]
                    - self.ref_obs_data_slicemean_yearmean.loc['temp', slice_num]
                ) / self.temp_std[slice_num]
                target_precip = (
                    self.ref_sim_precip_slicemean[year_num, slice_num]
                    - self.ref_obs_data_slicemean_yearmean.loc['precip', slice_num]
                ) / self.precip_std[slice_num]
                temp_candidates = (
                    self.ref_obs_data_slicemean.loc[:, 'temp', slice_candidates]
                    - self.ref_obs_data_slicemean_yearmean.loc['temp', slice_candidates]
                ) / self.temp_std[slice_candidates]
                precip_candidates = (
                    self.ref_obs_data_slicemean.loc[:, 'precip', slice_candidates]
                    - self.ref_obs_data_slicemean_yearmean.loc['precip', slice_candidates]
                ) / self.precip_std[slice_candidates]

                selected_year, selected_slice = min_dist(
                    target_temp,
                    temp_candidates,
                    target_precip,
                    precip_candidates,
                )

                slices[year][slice_num] = (selected_year, selected_slice)

        self.slices = slices

    def _assign_slices_sim(self):
        data_sim_sliced = xr.DataArray(
            np.full(
                (
                    len(self.stations),
                    len(self.sim_years),
                    len(self.params),
                    self.num_slices_per_year,
                    self.num_timesteps_per_slice,
                ),
                np.nan,
            ),
            coords=[
                self.data_obs.station,
                self.sim_years,
                self.params,
                range(self.num_slices_per_year),
                range(self.num_timesteps_per_slice),
            ],
            dims=[
                'station',
                'year',
                'param',
                'slice',
                'slice_timestep',
            ],
        )

        for year_num, year in enumerate(self.sim_years):
            print(f'Assigning slices for year {year}')

            for slice_num in range(self.num_slices_per_year):
                selected_year, selected_slice = self.slices[year][slice_num]
                data_sim_sliced.values[:, year_num, :, slice_num, :] = self.data_obs_sliced.loc[
                    :,
                    selected_year,
                    :,
                    selected_slice,
                    :,
                ]

        self.data_sim_sliced = data_sim_sliced

    def _reshape_and_fill_missing(self):
        # To simplify missing slice assignment first generate data for entire years, then clip later
        sim_dates_fullyear = pd.date_range(
            start=f'{self.config.sim_start_date.year}-01-01',
            end=oa.conf.parse_end_date(
                f'{self.config.sim_end_date.year}-12-31',
                self.config.timestep,
            ),
            freq=self.config.timestep,
        )
        data_sim = self.data_obs.reindex(time=sim_dates_fullyear)

        data_sim_da = data_sim[self.params].to_array('param')
        data_sim_da = data_sim_da.to_dataset('station').to_array('station')  # to reorder dimensions - probably there is an easier way?
        data_sim_da.values[:] = np.nan

        for year_num, year in enumerate(self.sim_years):
            print(f'Reshaping data and filling missing records for year {year}')

            sim_dates_cur = sim_dates_fullyear[sim_dates_fullyear.year == year]
            sim_dates_cur_covered = sim_dates_cur[:(self.num_slices_per_year * self.num_timesteps_per_slice)]
            sim_dates_cur_uncovered = sim_dates_cur[(self.num_slices_per_year * self.num_timesteps_per_slice):]

            data_param_cur = (
                self.data_sim_sliced
                .loc[:, year, :, :, :]
                .values
                .reshape((len(self.stations), len(self.params), -1))
            )
            data_sim_da.loc[:, :, sim_dates_cur_covered] = data_param_cur

            # Fill missing records
            num_missing_vals = len(sim_dates_cur_uncovered)
            data_sim_da.loc[:, :, sim_dates_cur_uncovered] = data_param_cur[
                :,
                :,
                (-2 * num_missing_vals):(-num_missing_vals)
            ]

        data_sim = data_sim.merge(data_sim_da.to_dataset('param'))
        data_sim = data_sim.sel(time=self.sim_dates)
        self.data_sim = data_sim

    def _write_output(self):
        data_sim = self.data_sim
        output_dir = Path(self.config.output_dir)
        output_format = self.config.input_format

        output_dir.mkdir(parents=True, exist_ok=True)

        if output_format == 'netcdf':
            timedelta = oa.util.offset_to_timedelta(self.config.timestep)
            data_sim = data_sim.copy()
            data_sim['precip'] /= timedelta.total_seconds()
            data_sim.precip.attrs = {
                'standard_name': 'precipitation_flux',
                'units': 'kg m-2 s-1',
            }

            var_mappings = {v: k for k, v in oa.constants.NETCDF_VAR_MAPPINGS.items()}
            data_sim = data_sim.rename(var_mappings)

        for station in self.stations:
            data_station = data_sim.sel(station=station).drop_vars('station')

            if output_format == 'netcdf':
                filename = f'{output_dir}/{station}.nc'
                print(f'Writing {filename}')
                data_station.attrs['station_name'] = str(data_station.station_name.values)
                data_station = data_station.drop_vars('station_name')
                data_station.to_netcdf(filename)
            elif output_format == 'csv':
                df = (
                    data_station[self.params]
                    .to_dataframe()
                    .dropna(axis=1, how='all')
                )

                filename = f'{output_dir}/{station}.csv'
                print(f'Writing {filename}')
                df.to_csv(filename, float_format='%g')

            if output_format == 'csv':
                x, y = oa.util.transform_coords(
                    data_sim.lon,
                    data_sim.lat,
                    oa.constants.CRS_WGS84,
                    self.config.crs,
                )

                meta = pd.DataFrame(
                    index=self.stations,
                    data=dict(
                        name=data_sim.station_name,
                        x=x,
                        y=y,
                        alt=data_sim.alt,
                    ),
                )
                meta.to_csv(f'{output_dir}/stations.csv')
