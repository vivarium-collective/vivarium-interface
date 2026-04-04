import numpy as np
from vivarium.utils import render_path, round_floats


def emitter_to_timeseries(results):
    """Convert a list of state snapshots into a timeseries dict.

    Args:
        results: list of dicts, each a state snapshot from the emitter

    Returns:
        dict mapping path strings ('/top/A') to lists of values
    """
    timeseries = {}

    def _append(state, path=()):
        if isinstance(state, dict):
            if state.get('address'):
                return
            for key, value in state.items():
                _append(value, path + (key,))
        else:
            if path not in timeseries:
                timeseries[path] = []
            timeseries[path].append(state)

    for snapshot in results:
        _append(snapshot)

    return {render_path(k): v for k, v in timeseries.items()}


def flatten_timeseries(timeseries):
    """Expand array-valued timeseries entries into scalar sub-paths.

    Example:
        {'/fields/glucose': [array([[1],[2]]), ...]}
    becomes:
        {'/fields/glucose/0/0': [1, ...], '/fields/glucose/1/0': [2, ...]}
    """
    flat = {}
    for var, series in timeseries.items():
        if var in ('global_time', '/global_time'):
            flat[var] = series
            continue

        first = series[0]
        if isinstance(first, np.ndarray) and first.ndim >= 1:
            for idx in np.ndindex(first.shape):
                path = var + ''.join(f'/{i}' for i in idx)
                flat[path] = [arr[idx].item() for arr in series]
        else:
            flat[var] = series

    return flat


def extract_time(timeseries):
    """Pop and return the time vector from a timeseries dict.

    Returns:
        (time_list, timeseries_without_time)
    """
    ts = dict(timeseries)
    if 'global_time' in ts:
        time = ts.pop('global_time')
    elif '/global_time' in ts:
        time = ts.pop('/global_time')
    else:
        raise KeyError("No 'global_time' or '/global_time' found in timeseries.")
    return time, ts
