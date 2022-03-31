import argparse
import openamundsen_climategenerator as oacg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config_file', help='configuration file')
    args = parser.parse_args()

    config = oacg.read_config(args.config_file)
    cg = oacg.ClimateGenerator(config)
    cg.initialize()
    cg.run()
