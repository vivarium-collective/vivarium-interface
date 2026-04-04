import numpy as np


def parse_path(path):
    """Convert a path to a list of strings.

    Accepts:
        - string: '/top/A' -> ['top', 'A'], 'top' -> ['top']
        - list/tuple: passed through as list
        - dict of paths: recursively parsed (for wires dicts)
        - None: returns empty tuple
    """
    if path is None:
        return ()
    if isinstance(path, dict):
        return {key: parse_path(value) for key, value in path.items()}
    if isinstance(path, str):
        if path.startswith('/'):
            return path.split('/')[1:]
        return [path]
    if isinstance(path, (list, tuple)):
        return list(path)
    return path


def render_path(path):
    """Convert a path (tuple/list) to a string like '/top/A'."""
    if isinstance(path, dict):
        return {key: render_path(value) for key, value in path.items()}
    if isinstance(path, str):
        return path if path.startswith('/') else '/' + path
    if isinstance(path, (tuple, list)):
        return '/' + '/'.join(str(x) for x in path)
    return path


def round_floats(data, decimals):
    """Recursively round floats in nested data structures."""
    if not decimals:
        return data
    if isinstance(data, dict):
        return {k: round_floats(v, decimals) for k, v in data.items()}
    if isinstance(data, list):
        return [round_floats(i, decimals) for i in data]
    if isinstance(data, float):
        return round(data, decimals)
    if isinstance(data, np.ndarray):
        return np.round(data, decimals=decimals)
    return data


def pad_to_length(data, length):
    """Prepend zeros to data if shorter than length."""
    if len(data) < length:
        return [0] * (length - len(data)) + data
    return data
