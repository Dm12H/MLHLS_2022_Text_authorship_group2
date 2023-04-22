import yaml
from typing import Dict, Any


def parse_config(filename: str = "model_config.yml") -> Dict[str, Any]:
    with open(filename, 'r') as f:
        params = yaml.safe_load(f)
    return params
