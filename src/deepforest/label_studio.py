import os
from label_studio_sdk import Client


def get_api_key():
    """Get Label Studio API key from config file."""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                               '.label_studio.config')
    if not os.path.exists(config_path):
        return None

    with open(config_path, 'r') as f:
        for line in f:
            if line.startswith('api_key'):
                return line.split('=')[1].strip()
    return None
