import yaml
import argparse
from core.train import train


def parse_args():
    parser = argparse.ArgumentParser(description="Load configuration from a YAML file.")
    parser.add_argument('-c', '--config', type=str, default='./configs/training.yml', help='Path to the configuration file.')
    return parser.parse_args()


def load_config(config_path):
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"The configuration file {config_path} does not exist.")
    except yaml.YAMLError as exc:
        raise yaml.YAMLError(f"Error parsing the YAML file: {exc}")
    return config


if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.config)
    train(config)
    