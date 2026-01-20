import os
import sys
import json

def load_paths():
    """
    Load paths from the JSON file specified by PYPATH_CONFIG environment variable
    and add them to sys.path.
    """
    config_path = os.environ.get("PYPATH_CONFIG")
    if not config_path:
        return

    if not os.path.exists(config_path):
        print(f"[PyPathManager] Warning: Config file not found at {config_path}")
        return

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Determine the list of paths to add
        paths_to_add = []
        if isinstance(data, list):
            paths_to_add = data
        elif isinstance(data, dict):
            # If it's a dict, we assume values are paths
            paths_to_add = list(data.values())
        else:
            print(f"[PyPathManager] Warning: Invalid JSON format in {config_path}. Expected list or dict.")
            return

        # Add paths to sys.path if not already present
        count = 0
        for path in paths_to_add:
            if path and path not in sys.path:
                sys.path.append(path)
                count += 1
        
        if count > 0:
            print(f"[PyPathManager] Added {count} paths to sys.path from {config_path}")

    except Exception as e:
        print(f"[PyPathManager] Error loading paths: {e}")

# Automatically load paths when imported
load_paths()
