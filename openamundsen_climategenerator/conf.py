import openamundsen as oa
from pathlib import Path


DATA_DIR = Path(__file__).parent / 'data'


def read_config(filename):
    return oa.read_config(filename)


def parse_config(config):
    config_schema = oa.util.read_yaml_file(f'{DATA_DIR}/configschema.yml')
    v = oa.conf.ConfigurationValidator(config_schema)
    valid = v.validate(config)

    if not valid:
        raise oa.errors.ConfigurationError('Invalid configuration\n\n' + oa.util.to_yaml(v.errors))

    full_config = oa.Configuration.fromDict(v.document)

    for key in ('obs_end_date', 'sim_end_date'):
        full_config[key] = oa.conf.parse_end_date(
            full_config[key],
            full_config['timestep'],
        )

    validate_config(full_config)

    return full_config


def validate_config(config):
    if config.report_file is not None:
        try:
            import matplotlib
        except ImportError:
            raise ConfigurationError('Report generation requires matplotlib to be installed.')


class ConfigurationError(Exception):
    pass
