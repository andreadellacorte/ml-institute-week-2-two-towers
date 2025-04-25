import json

def load_json(filepath):
    """
    Load a JSON file and return its content.
    
    Args:
        filepath (str): Path to the JSON file.
        
    Returns:
        dict: Content of the JSON file.
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def dump_json(data, filepath):
    """
    Dump data to a JSON file.
    
    Args:
        data (dict): Data to be dumped.
        filepath (str): Path to the output JSON file.
    """
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)