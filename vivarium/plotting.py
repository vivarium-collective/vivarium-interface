import io
import base64

import numpy as np
import matplotlib.pyplot as plt
import imageio

from vivarium.utils import pad_to_length
from vivarium.results import flatten_timeseries, extract_time


def plot_timeseries(timeseries, subplot_size=(10, 5), ncols=1, combined_vars=None):
    """Plot timeseries data with one subplot per variable.

    Args:
        timeseries: dict mapping path strings to value lists (must include global_time)
        subplot_size: (width, height) per subplot
        ncols: number of columns in the subplot grid
        combined_vars: list of lists of variable names to combine into shared subplots

    Returns:
        matplotlib Figure
    """
    timeseries = flatten_timeseries(timeseries)
    time, ts = extract_time(timeseries)

    combined_vars = combined_vars or []
    combined_flat = set(var for group in combined_vars for var in group)
    individual_vars = [var for var in ts if var not in combined_flat]

    total_plots = len(individual_vars) + len(combined_vars)
    if total_plots == 0:
        fig, _ = plt.subplots(1, 1, figsize=subplot_size)
        plt.close(fig)
        return fig

    nrows = (total_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(subplot_size[0] * ncols, subplot_size[1] * nrows))
    axes = axes.flatten() if total_plots > 1 else [axes]
    plot_idx = 0

    for var in individual_vars:
        ax = axes[plot_idx]
        data = pad_to_length(ts[var], len(time))
        ax.plot(time, data)
        ax.set_title(var)
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        plot_idx += 1

    for group in combined_vars:
        ax = axes[plot_idx]
        for var in group:
            if var not in ts:
                raise KeyError(f"Variable '{var}' not found in timeseries")
            data = pad_to_length(ts[var], len(time))
            ax.plot(time, data, label=var)
        ax.set_title(', '.join(group))
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plot_idx += 1

    # Hide unused axes
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.close(fig)
    return fig


def plot_snapshots(timeseries, times=None, n_snapshots=None):
    """Plot 2D snapshots of array fields at selected timepoints.

    Args:
        timeseries: dict with global_time and fields as lists of 2D arrays
        times: explicit timepoints to snapshot
        n_snapshots: number of evenly spaced snapshots (default 5)

    Returns:
        matplotlib Figure
    """
    if times is not None and n_snapshots is not None:
        raise ValueError("Specify either `times` or `n_snapshots`, not both.")
    if times is None and n_snapshots is None:
        n_snapshots = 5

    time, ts = extract_time(timeseries)

    # filter to only 2D array fields
    ts = {k: v for k, v in ts.items()
          if isinstance(v, list) and len(v) > 0
          and isinstance(v[0], np.ndarray) and v[0].ndim >= 2}

    if not ts:
        raise ValueError("No 2D array fields found in timeseries for snapshot plotting.")

    if times is not None:
        time_indices = [np.argmin(np.abs(np.array(time) - t)) for t in times]
    else:
        time_indices = np.linspace(0, len(time) - 1, n_snapshots, dtype=int)

    display_times = [time[i] for i in time_indices]
    field_names = list(ts.keys())
    num_rows = len(field_names)
    num_cols = len(display_times)

    fig, axes = plt.subplots(num_rows, num_cols,
                             figsize=(5 * num_cols, 5 * num_rows),
                             gridspec_kw={'wspace': 0.1})

    if num_rows == 1 and num_cols == 1:
        axes = np.array([[axes]])
    elif num_rows == 1:
        axes = np.array([axes])
    elif num_cols == 1:
        axes = np.array([[ax] for ax in axes])

    global_min_max = {}
    for field in field_names:
        flat = np.concatenate([arr.flatten() for arr in ts[field]])
        global_min_max[field] = (np.min(flat), np.max(flat))

    for row, field in enumerate(field_names):
        first_im = None
        for col, time_idx in enumerate(time_indices):
            ax = axes[row, col]
            snapshot = ts[field][time_idx]
            im = ax.imshow(snapshot, interpolation='nearest',
                           vmin=global_min_max[field][0],
                           vmax=global_min_max[field][1],
                           aspect='equal', cmap='viridis')
            ax.set_title(f"{field} at t={time[time_idx]:.2f}")
            if col == 0:
                first_im = im

        if first_im is not None:
            cbar_ax = fig.add_axes([
                axes[row, 0].get_position().x0 - 0.075,
                axes[row, 0].get_position().y0,
                0.015,
                axes[row, 0].get_position().height
            ])
            fig.colorbar(first_im, cax=cbar_ax)

    plt.close(fig)
    return fig


def make_video(timeseries, skip_frames=1, title=''):
    """Generate an HTML string displaying an animated GIF of 2D field evolution.

    Args:
        timeseries: dict with global_time and fields as lists of 2D arrays
        skip_frames: interval between frames
        title: title for the animation

    Returns:
        HTML string with embedded GIF
    """
    time, ts = extract_time(timeseries)

    field_names = list(ts.keys())
    n_fields = len(field_names)
    n_frames = len(time)

    global_min_max = {
        field: (
            np.min(np.concatenate([arr.flatten() for arr in ts[field]])),
            np.max(np.concatenate([arr.flatten() for arr in ts[field]]))
        )
        for field in field_names
    }

    images = []
    for i in range(0, n_frames, skip_frames):
        fig, axs = plt.subplots(1, n_fields, figsize=(5 * n_fields, 4))
        axs = [axs] if n_fields == 1 else axs

        for j, field in enumerate(field_names):
            ax = axs[j]
            vmin, vmax = global_min_max[field]
            img = ax.imshow(ts[field][i], interpolation='nearest',
                            vmin=vmin, vmax=vmax, cmap='viridis', aspect='equal')
            ax.set_title(f'{field} at t = {time[i]:.2f}')
            plt.colorbar(img, ax=ax)

        fig.suptitle(title, fontsize=16)

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120)
        buf.seek(0)
        images.append(imageio.imread(buf))
        buf.close()
        plt.close(fig)

    buf = io.BytesIO()
    imageio.mimsave(buf, images, format='GIF', duration=0.5, loop=0)
    buf.seek(0)
    data_url = 'data:image/gif;base64,' + base64.b64encode(buf.read()).decode()
    return f'<img src="{data_url}" alt="{title}" style="max-width:100%;"/>'
