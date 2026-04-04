"""
Vivarium — A Pythonic interface for building, inspecting, running,
and visualizing multiscale simulations with process bigraphs.
"""
import os
import json

import numpy as np
import pandas as pd
from IPython.display import HTML, display

from bigraph_schema import allocate_core, get_path, set_path
from process_bigraph import Composite
from process_bigraph.emitter import (
    add_emitter_to_composite,
    gather_emitter_results,
)
from bigraph_viz import plot_bigraph, register_types as register_viz_types
from process_bigraph import register_types as register_pb_types

from vivarium.utils import parse_path, render_path, round_floats
from vivarium.results import emitter_to_timeseries
from vivarium.plotting import (
    plot_timeseries as _plot_timeseries,
    plot_snapshots as _plot_snapshots,
    make_video,
)


class _SizedGraph:
    """Wraps a ResponsiveGraph with constrained sizing for Jupyter display."""

    def __init__(self, graph, max_width=600):
        self._graph = graph
        self._max_width = max_width

    def __getattr__(self, name):
        return getattr(self._graph, name)

    def _repr_html_(self):
        import re
        svg = self._graph._graph.pipe(format='svg').decode()
        # strip XML/DOCTYPE
        svg = re.sub(r'<\?xml[^?]*\?>\s*', '', svg)
        svg = re.sub(r'<!DOCTYPE[^>]*>\s*', '', svg)
        # remove any transform scale so the viewBox controls sizing
        svg = re.sub(r'scale\([0-9.]+ [0-9.]+\) ', '', svg)
        # set width to max_width, let height scale from viewBox
        svg = re.sub(
            r'<svg width="[^"]*" height="[^"]*"',
            f'<svg style="max-width:{self._max_width}px; height:auto;"',
            svg, count=1,
        )
        return svg

    def __repr__(self):
        return repr(self._graph)


def _build_core(core=None, register=None):
    """Create and configure a Core instance."""
    if core is None:
        core = allocate_core()
        register_pb_types(core)
        register_viz_types(core)
    if register is not None:
        register(core)
    return core


def _load_document(document):
    """Normalize a document argument into a dict.

    Accepts:
        - None -> empty document
        - str path to .json file -> loaded dict
        - dict -> passed through (with backward compat for 'composition' key)
    """
    if document is None:
        return {'schema': {}, 'state': {}}

    if isinstance(document, str):
        with open(document, 'r') as f:
            document = json.load(f)

    if not isinstance(document, dict):
        raise ValueError("Document must be a dict or a path to a JSON file.")

    # backward compatibility: rename 'composition' -> 'schema'
    if 'composition' in document and 'schema' not in document:
        document['schema'] = document.pop('composition')

    return document


class Vivarium:
    """A controlled environment for composite process-bigraph simulations.

    Args:
        document: A dict ``{schema, state}``, a path to a JSON file, or None for empty.
        core: A pre-built Core instance.  If None, one is created with default types.
        register: A callable ``register(core)`` for custom type/link registration.
        emitter: Emitter mode — ``'all'``, ``'none'``, or ``{'paths': [...]}``.
    """

    def __init__(self, document=None, core=None, register=None, emitter='all'):
        self.core = _build_core(core, register)
        self._emitter_mode = emitter

        doc = _load_document(document)
        self.composite = Composite(doc, core=self.core)

        # add emitter
        if emitter != 'none':
            add_emitter_to_composite(
                self.composite, self.core,
                emitter_mode=emitter,
                address='local:RAMEmitter')

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path, core=None, register=None, emitter='all'):
        """Load a Vivarium from a JSON file.

        Args:
            path: Path to the JSON file.
            core: Optional pre-built Core instance.
            register: Optional callable for custom registration.
            emitter: Emitter mode.
        """
        return cls(document=path, core=core, register=register, emitter=emitter)

    def save(self, path='out/simulation.json'):
        """Save current schema and state to a JSON file."""
        outdir = os.path.dirname(path) or '.'
        os.makedirs(outdir, exist_ok=True)
        doc = self.to_dict()
        with open(path, 'w') as f:
            json.dump(doc, f, indent=2, default=str)

    def to_dict(self):
        """Return the current document as ``{schema, state}``."""
        return {
            'schema': self.composite.serialize_schema(),
            'state': self.composite.serialize_state(),
        }

    # ------------------------------------------------------------------
    # Model building
    # ------------------------------------------------------------------

    def merge(self, schema=None, state=None, path=None):
        """Merge schema and/or state into the composite at the given path.

        Args:
            schema: Schema dict to merge.
            state: State dict to merge.
            path: Target path (string ``'/a/b'``, list, or tuple).
        """
        schema = schema or {}
        state = state or {}
        path = parse_path(path) or None
        self.composite.merge(schema, state, path)
        self._refresh()

    def add_process(self, name, process_id, config=None, inputs=None, outputs=None, path=None):
        """Add a temporal process to the composite.

        Args:
            name: Name for this process instance.
            process_id: Registered process identifier (e.g. ``'increase float'``).
            config: Process configuration dict.
            inputs: Dict mapping port names to state paths.
            outputs: Dict mapping port names to state paths.
            path: Parent path to nest the process under.
        """
        self._add_edge('process', name, process_id, config, inputs, outputs, path)

    def add_step(self, name, step_id, config=None, inputs=None, outputs=None, path=None):
        """Add a dependency-triggered step to the composite.

        Args:
            name: Name for this step instance.
            step_id: Registered step identifier.
            config: Step configuration dict.
            inputs: Dict mapping port names to state paths.
            outputs: Dict mapping port names to state paths.
            path: Parent path to nest the step under.
        """
        self._add_edge('step', name, step_id, config, inputs, outputs, path)

    def _add_edge(self, edge_type, name, edge_id, config, inputs, outputs, path):
        # Validate that the process/step is registered
        available = list(self.core.link_registry.keys())
        if edge_id not in available:
            raise ValueError(
                f"'{edge_id}' is not a registered {edge_type}. "
                f"Available: {available}"
            )

        config = config or {}
        inputs = parse_path(inputs or {})
        outputs = parse_path(outputs or {})
        path = parse_path(path) or None

        state = {
            name: {
                '_type': edge_type,
                'address': f'local:{edge_id}',
                'config': config,
                'inputs': inputs,
                'outputs': outputs,
            }
        }
        self.composite.merge({}, state, path)
        self._refresh()

    def connect(self, name, inputs=None, outputs=None, path=None):
        """Connect or reconnect wires for an existing process or step.

        Args:
            name: The name of the process/step.
            inputs: Dict mapping port names to state paths.
            outputs: Dict mapping port names to state paths.
            path: Parent path where the process lives.
        """
        path = parse_path(path) or ()
        state = {name: {}}
        if inputs is not None:
            state[name]['inputs'] = parse_path(inputs)
        if outputs is not None:
            state[name]['outputs'] = parse_path(outputs)
        self.composite.merge({}, state, path)
        self._refresh()

    def set(self, path, value):
        """Set a value directly at the given path in the state."""
        path = parse_path(path)
        set_path(self.composite.state, path=path, value=value)

    def get(self, path=None):
        """Get the value at the given path in the state. Returns full state if path is None."""
        if path is None:
            return self.composite.state
        path = parse_path(path)
        return get_path(self.composite.state, path)

    def remove(self, path):
        """Remove a node at the given path from the state."""
        path = parse_path(path)
        if not path or path == ['']:
            raise ValueError("Cannot remove the root state. Provide a path to remove.")
        # walk to the parent and delete the key
        parent = self.composite.state
        for step in path[:-1]:
            if not isinstance(parent, dict) or step not in parent:
                raise KeyError(f"Path not found: {render_path(path)}")
            parent = parent[step]
        key = path[-1]
        if not isinstance(parent, dict) or key not in parent:
            raise KeyError(f"Path not found: {render_path(path)}")
        del parent[key]
        self._refresh()

    def _refresh(self):
        """Rebuild instance paths and step network after structural changes."""
        self.composite.find_instance_paths(self.composite.state)
        self.composite.build_step_network()

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    @property
    def schema(self):
        """The current composite schema."""
        return self.composite.schema

    @property
    def state(self):
        """The current composite state."""
        return self.composite.state

    @property
    def time(self):
        """The current global time."""
        return self.composite.state.get('global_time', 0.0)

    def types(self, as_dataframe=False):
        """List all registered types.

        Args:
            as_dataframe: Return as a pandas DataFrame.
        """
        type_list = list(self.core.registry.keys())
        if as_dataframe:
            return pd.DataFrame(type_list, columns=['Type'])
        return type_list

    def processes(self, as_dataframe=False):
        """List all registered process/step links.

        Args:
            as_dataframe: Return as a pandas DataFrame.
        """
        link_list = list(self.core.link_registry.keys())
        if as_dataframe:
            return pd.DataFrame(link_list, columns=['Process'])
        return link_list

    def process_config(self, process_id, default=False):
        """Get the config schema for a registered process.

        Args:
            process_id: The process identifier string.
            default: If True, return the default config values instead of the schema.
        """
        process_class = self.core.link_registry.get(process_id)
        if process_class is None:
            raise KeyError(f"Process '{process_id}' not found in registry.")
        if default:
            result = self.core.default(process_class.config_schema)
            # core.default may return (schema, state) tuple
            if isinstance(result, tuple):
                return result[1]
            return result
        return process_class.config_schema

    def process_interface(self, process_id, config=None):
        """Get the inputs/outputs interface for a process.

        Args:
            process_id: The process identifier string.
            config: Optional config dict to instantiate with.

        Returns:
            pandas DataFrame with inputs and outputs.
        """
        process_class = self.core.link_registry.get(process_id)
        if process_class is None:
            raise KeyError(f"Process '{process_id}' not found in registry.")
        instance = process_class(config or {}, self.core)
        interface = instance.interface()
        rows = []
        for section in ('inputs', 'outputs'):
            for port, port_type in interface.get(section, {}).items():
                rows.append({'section': section, 'port': port, 'type': str(port_type)})
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def run(self, interval):
        """Run the simulation for the given time interval."""
        if not isinstance(interval, (int, float)) or interval <= 0:
            raise ValueError(f"interval must be a positive number, got {interval!r}")
        if self._emitter_mode != 'none':
            self._rebuild_emitter()
        self.composite.run(interval)

    def _rebuild_emitter(self):
        """Remove and re-add the emitter so it captures the current state structure."""
        emitter_path = ('emitter',)
        # remove existing emitter if present
        if 'emitter' in self.composite.state:
            del self.composite.state['emitter']
        if emitter_path in getattr(self.composite, 'step_paths', {}):
            del self.composite.step_paths[emitter_path]
        add_emitter_to_composite(
            self.composite, self.core,
            emitter_mode=self._emitter_mode,
            address='local:RAMEmitter')

    def step(self):
        """Execute one round of step dependencies."""
        self.composite.update({}, 0)

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def results(self, query=None):
        """Return raw emitter results (dict of path -> list of snapshots)."""
        return gather_emitter_results(self.composite, queries=query)

    def timeseries(self, query=None, decimals=None, as_dataframe=False):
        """Get timeseries data from emitter results.

        Args:
            query: Optional query to filter paths.
            decimals: Round floats to this many decimal places.
            as_dataframe: Return as a pandas DataFrame.

        Returns:
            dict mapping path strings to value lists, or DataFrame.
        """
        raw = gather_emitter_results(self.composite, queries=query)
        # Combine results from all emitters
        all_snapshots = []
        for path, snapshots in raw.items():
            all_snapshots.extend(snapshots)
        ts = emitter_to_timeseries(all_snapshots)
        if decimals is not None:
            ts = round_floats(ts, decimals)
        if as_dataframe:
            df = pd.DataFrame.from_dict(ts, orient='index').transpose()
            return df
        return ts

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def diagram(self, filename=None, out_dir=None, remove_nodes=None, **kwargs):
        """Generate a bigraph-viz diagram.

        Displays inline in Jupyter. Optionally saves to file.

        Args:
            filename: Output filename (without extension).
            out_dir: Output directory.
            remove_nodes: List of node paths to exclude from the diagram.
            **kwargs: Additional arguments passed to bigraph-viz.

        Returns:
            A ResponsiveGraph object (displays as HTML in Jupyter).
        """
        remove_nodes = remove_nodes or []
        remove_nodes = [
            tuple(n.lstrip('/').split('/')) if isinstance(n, str) else tuple(n)
            for n in remove_nodes
        ]
        # remove emitter by default
        if ('emitter',) not in remove_nodes:
            remove_nodes.append(('emitter',))

        kwargs['remove_nodes'] = remove_nodes

        graph = plot_bigraph(
            state=self.composite.state,
            schema=self.composite.schema,
            core=self.core,
            filename=filename,
            out_dir=out_dir,
            **kwargs,
        )
        return _SizedGraph(graph)

    def plot(self, query=None, decimals=None, subplot_size=(10, 5), ncols=1, combined_vars=None):
        """Plot timeseries results.

        Args:
            query: Optional query to filter paths.
            decimals: Round floats to this many decimal places.
            subplot_size: (width, height) per subplot.
            ncols: Number of columns in the subplot grid.
            combined_vars: List of lists of variable names to combine into shared subplots.

        Returns:
            matplotlib Figure.
        """
        ts = self.timeseries(query=query, decimals=decimals)
        return _plot_timeseries(ts, subplot_size=subplot_size, ncols=ncols,
                                combined_vars=combined_vars)

    def plot_snapshots(self, times=None, n_snapshots=None, query=None):
        """Plot 2D field snapshots at selected timepoints.

        Args:
            times: List of times to snapshot.
            n_snapshots: Number of evenly spaced snapshots.
            query: Optional query to filter fields.

        Returns:
            matplotlib Figure.
        """
        ts = self.timeseries(query=query)
        return _plot_snapshots(ts, times=times, n_snapshots=n_snapshots)

    def show_video(self, query=None, skip_frames=1, title=''):
        """Display an animated GIF of 2D field evolution in Jupyter.

        Args:
            query: Optional query to filter fields.
            skip_frames: Interval between frames.
            title: Title for the animation.
        """
        ts = self.timeseries(query=query)
        html = make_video(ts, skip_frames=skip_frames, title=title)
        display(HTML(html))

    # ------------------------------------------------------------------
    # Jupyter integration
    # ------------------------------------------------------------------

    def _repr_html_(self):
        """Display the bigraph diagram inline when the object is the last expression in a cell."""
        try:
            return self.diagram()._repr_html_()
        except Exception as e:
            return f'<pre>{self.__repr__()}\n\n[diagram error: {e}]</pre>'

    def __repr__(self):
        state_keys = list(self.composite.state.keys()) if self.composite.state else []
        return f"Vivarium(time={self.time}, state_keys={state_keys})"
