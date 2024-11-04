from io import StringIO
import matplotlib.pyplot as plt
import numpy as np
import openamundsen as oa
import pandas as pd


HTML = '''
<!doctype html>
<html>
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <link href="https://unpkg.com/tailwindcss@^2/dist/tailwind.min.css" rel="stylesheet">
</head>
<body>
<div class="container mx-auto my-8 px-4">
{body}
</div>
</body>
</html>
'''

PARAM_LABELS = {
    'air_temperature': 'Air temperature',
    'precipitation_amount': 'Precipitation',
    'relative_humidity': 'Relative humidity',
    'surface_downwelling_shortwave_flux_in_air': 'Global radiation',
    'wind_speed': 'Wind speed',
}


def resample(data, rule='YE', max_missing_frac=1., sum=False):
    all_dates = pd.date_range(
        start=f'{data.index.year[0]}-01-01',
        end=oa.conf.parse_end_date(f'{data.index.year[-1]}-12-31', data.index.inferred_freq),
        freq=data.index.inferred_freq,
    )
    num_missing = data.reindex(all_dates).isna().resample(rule).sum()
    num_expected = pd.Series(index=all_dates, data=1).resample(rule).sum()

    if sum:
        data_res = data.resample(rule).sum()
    else:
        data_res = data.resample(rule).mean()

    data_res[(num_missing >= num_expected * max_missing_frac)] = np.nan

    return data_res


def save_svg(fig):
    imgdata = StringIO()
    fig.savefig(imgdata, format='svg')
    plt.close(fig)
    imgdata.seek(0)
    svg_data = imgdata.read()
    svg_data = '<svg' + svg_data.split('<svg')[1]
    return svg_data


def generate_report(cg):
    data_obs = cg.data_obs
    data_sim = cg.data_sim
    ref_station = cg.config.reference_station
    stations = list(cg.stations)
    stations.remove(ref_station)
    stations.sort()
    stations = [ref_station] + stations
    params = cg.params

    entire_period_dates = pd.date_range(
        start=min(cg.config.obs_start_date, cg.config.sim_start_date),
        end=max(cg.config.obs_end_date, cg.config.sim_end_date),
        freq=cg.config.timestep,
    )
    num_expected_yr = pd.Series(index=entire_period_dates, data=1).resample('YE').sum()

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_obs = colors[0]
    color_sim = colors[1]

    html_body = '<h1 class="text-3xl font-bold text-gray-900">Climate Generator Report</h1>\n'

    html_body += f'<h2 class="text-2xl font-bold text-gray-900 mt-4">Configuration</h2>\n'
    html_body += '<pre><code class="text-sm">\n'
    html_body += str(cg.config)
    html_body += '</code></pre>\n'

    plt.close('all')

    for station in stations:
        print(f'Creating report for station {station}')

        station_label = station
        if station == ref_station:
            station_label += ' (reference station)'
        html_body += f'<h2 class="text-2xl font-bold text-gray-900 mt-4">{station_label}</h2>\n'

        for param in params:
            data_obs_param = data_obs[param].sel(station=station).to_pandas()
            data_sim_param = data_sim[param].sel(station=station).to_pandas()

            if not data_obs_param.any():
                continue

            param_standard_name = data_obs[param].attrs.get('standard_name')
            if param_standard_name is not None:
                param_label = PARAM_LABELS.get(param_standard_name, param_standard_name)
            else:
                param_label = param
            html_body += f'<h3 class="text-xl font-bold text-gray-900 mt-4">{param_label}</h3>\n'

            data_param = pd.concat([data_obs_param, data_sim_param]).reindex(entire_period_dates)
            data_param_yr = resample(
                data_param,
                'YE',
                max_missing_frac=0.05,
                sum=(param == 'precip'),
            )

            fig, ax = plt.subplots(figsize=(5, 3))
            df = pd.DataFrame(index=data_param_yr.index, columns=['obs', 'sim'])
            df['obs'] = data_param_yr.loc[cg.config.obs_start_date:cg.config.obs_end_date]
            df['sim'] = data_param_yr.loc[cg.config.sim_start_date:cg.config.sim_end_date]
            df.obs.plot(ax=ax, marker='s', markersize=2, color=color_obs)
            df.sim.plot(ax=ax, marker='s', markersize=2, color=color_sim)
            ax.set_xlabel(None)
            ylabel = param_label
            if 'units' in data_obs[param].attrs:
                ylabel += f' ({data_obs[param].attrs["units"]})'
            ax.set_ylabel(ylabel)
            fig.canvas.draw_idle()
            svg_data = save_svg(fig)

            html_body += '<div class="w-full lg:flex items-center">\n'
            html_body += f'<div class="w-full lg:w-1/2">{svg_data}</div>\n'

            num_missing_yr = data_param.reindex(entire_period_dates).isna().resample('YE').sum()
            perc_missing_yr = 100 * num_missing_yr / num_expected_yr
            plt.close('all')
            fig, ax = plt.subplots(figsize=(5, 3))
            df = pd.DataFrame(index=perc_missing_yr.index, columns=['obs', 'sim'])
            df['obs'] = perc_missing_yr.loc[cg.config.obs_start_date:cg.config.obs_end_date]
            df['sim'] = perc_missing_yr.loc[cg.config.sim_start_date:cg.config.sim_end_date]
            df.index = df.index.year
            df.obs.plot(ax=ax, kind='bar', color=color_obs)
            df.sim.plot(ax=ax, kind='bar', color=color_sim)
            ax.set_ylim(0, 100)
            ax.set_xlabel(None)
            ax.set_ylabel('Missing values (%)')
            ax.set_xticklabels([''] * len(ax.get_xticks()))
            svg_data = save_svg(fig)

            html_body += f'<div class="w-full lg:w-1/2">{svg_data}</div>\n'
            html_body += '</div>\n'

    html = HTML.format(body=html_body)
    with open(cg.config.report_file, 'w') as f:
        f.write(html)
