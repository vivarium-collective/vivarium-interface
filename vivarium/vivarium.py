"""
Vivarium is a simulation environment that runs composites in the process bigraph.
"""
import os
import io
import base64
import imageio
import inspect
from IPython.display import HTML, display, Image
import json
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd

# process bigraph imports
from process_bigraph import ProcessTypes, Composite, pf, pp
from process_bigraph.processes import TOY_PROCESSES
from process_bigraph.processes.growth_division import grow_divide_agent
from bigraph_schema import is_schema_key, set_path, get_path
from bigraph_schema.utilities import remove_path
from bigraph_viz import VisualizeTypes
from bigraph_viz.visualize_types import get_graphviz_fig


def round_floats(data, decimal_places):
    if not decimal_places:
        return data
    if isinstance(data, dict):
        return {k: round_floats(v, decimal_places) for k, v in data.items()}
    elif isinstance(data, list):
        return [round_floats(i, decimal_places) for i in data]
    elif isinstance(data, float):
        return round(data, decimal_places)
    elif isinstance(data, np.ndarray):
        return np.round(data, decimals=decimal_places)
    else:
        return data


class VivariumTypes(ProcessTypes, VisualizeTypes):
    def __init__(self):
        super().__init__()


def parse_path(path):
    if isinstance(path, dict):
        return {
            key: parse_path(value)
            for key, value in path.items()}
    elif isinstance(path, str):
        if path.startswith('/'):
            return path.split('/')[1:]
        else:
            return [path]
    else:
        return path


def render_path(path):
    if isinstance(path, dict):
        return {
            key: render_path(value)
            for key, value in path.items()}
    elif isinstance(path, str):
        if path.startswith('/'):
            return path
        else:
            return '/' + path
    elif isinstance(path, (tuple, list)):
        strpath = [str(x) for x in path]
        return '/' + '/'.join(strpath)
    else:
        return path


def render_type(type, core):
    type = core.access(type)
    rendered = {}
    for key, value in type.items():
        if is_schema_key(key):
            if key == '_description':
                rendered['description'] = value
            elif key == '_default':
                rendered['default'] = core.default(type)
        elif isinstance(value, dict):
            rendered[key] = render_type(value, core)
    return rendered

def get_type_label(value):
    if isinstance(value, int):
        return 'number'
    elif isinstance(value, float):
        return 'float'
    elif isinstance(value, str):
        return 'string'
    elif isinstance(value, list):
        return 'list'
    elif isinstance(value, np.ndarray):
        return 'array'
    else:
        print(f"Warning: {value} is not a recognized type, returning {type(value)}.")
        return type(value)

def pad_to_length(data, length):
    """Prepend zeros to data if it's shorter than length."""
    if len(data) < length:
        return [0] * (length - len(data)) + data
    return data

def flatten_timeseries_to_scalar_paths(timeseries):
    """
    Converts timeseries with multi-dimensional arrays into a flat dict where each key is a scalar path.

    Example:
        {'/fields/glucose': [array([[1],[2]]), array([[3],[4]])]}
    becomes:
        {'/fields/glucose/0/0': [1, 3], '/fields/glucose/1/0': [2, 4]}
    """
    flat_timeseries = {}
    for var, series in timeseries.items():
        # Skip time for now; handle it separately
        if var in ['global_time', '/global_time']:
            flat_timeseries[var] = series
            continue

        first = series[0]
        if isinstance(first, np.ndarray) and first.ndim >= 1:
            for idx in np.ndindex(first.shape):
                path = f"{var}" + "".join(f"/{i}" for i in idx)
                flat_timeseries[path] = [arr[idx].item() for arr in series]
        else:
            # Already scalar/1D
            flat_timeseries[var] = series

    return flat_timeseries


class Vivarium:
    """
    Vivarium is a controlled virtual environment for composite process-bigraph simulations.

    It manages packages and sets up the conditions for running simulations, and collects results through emitters.

    Args:
        document (dict, optional): The configuration document for the simulation, or path to the document.
        processes (dict, optional): Dictionary of processes to register.
        types (dict, optional): Dictionary of types to register.
        core (VivariumTypes, optional): The core type system.
        require (list, optional): List of required packages for the simulation.
        emitter_config (dict, optional): Configuration for the emitter.
    """

    def __init__(self,
                 document=None,
                 processes=None,
                 types=None,
                 core=None,
                 require=None,  # TODO -- implement this
                 emitter_config=None,
                 ):

        processes = processes or {}
        types = types or {}
        self.emitter_config = emitter_config or {"mode": "all",
                                                 "path": ("emitter",)}
        self.emitter_paths = {}

        # if no core is provided, create a new one
        self.core = VivariumTypes()

        # set the document
        if isinstance(document, str):
            # check if document is a json file path
            if not document.endswith(".json"):
                raise ValueError("Document path must be a JSON file.")

            # load the document from the file
            document_path = os.path.join(os.getcwd(), document)
            with open(document_path, "r") as json_file:
                document = json.load(json_file)
        elif document is None:
            document = {"composition": {}, "state": {}}
        elif not isinstance(document, dict):
            raise ValueError("Document must be a dictionary or a JSON file path.")

        # register processes
        self.register_processes(processes)

        # register types
        self.register_types(types)

        # TODO register other packages
        if require:
            pass
            # self.require = require
            # for package in self.require:
            #     package = self.find_package(package)
            #     self.core.register_types(package.get("types", {}))

        # make the composite
        self.composite = self.generate_composite_from_document(document)

    def __repr__(self):
        return (f"Vivarium( \n"
                f"{pf(self.composite.state)})")

    def reset_paths(self):
        self.composite.find_instance_paths(self.composite.state)
        self.composite.build_step_network()

    def get_state(self):
        return self.composite.state

    def get_schema(self):
        return self.composite.composition

    def get_dataclass(self, path=None):
        path = path or ()
        return self.core.dataclass(schema=self.composite.composition, path=path)

    def add_object(self,
                   name,
                   type=None,
                   path=None,
                   value=None
                   ):
        type = type or get_type_label(value)

        state = {}
        schema = {}
        if type == "array":
            shape = value.shape
            # get the datatype of the array
            element_type = get_type_label(value.flat[0])
            schema[name] = {
                '_type': type,
                '_shape': shape,
                '_data': element_type,
            }
        else:
            schema[name] = {
                '_type': type,
                # '_default': value,
            }


        state[name] = value

        # nest the process in the composite at the given path
        self.composite.merge(schema, state, path)
        self.composite.build_step_network()

    def merge_value(self,
                  path,
                  value
                  ):
        path = parse_path(path)
        self.composite.merge({}, value, path)

    def set_value(self,
                  path,
                  value
                  ):
        path = parse_path(path)
        
        # TODO -- make this set the value in the composite using core
        set_path(self.composite.state, path=path, value=value)
        # self.composite[path] = value
        # self.composite.set({}, value, path)

    def set_schema(self,
                   path,
                   schema
                   ):
        path = parse_path(path)
        # TODO -- make this set the value in the composite using core
        # set_path(self.composite.composition, path=path, value=schema)
        
        # TODO -- need to regenerate the composition
        self.composite.merge(schema, {}, path)


    def get_value(self, path, as_dataframe=False):
        if isinstance(path, str):
            path = parse_path(path)
        value = get_path(self.composite.state, path)
        if as_dataframe:
            if isinstance(value, dict):
                records = []
                for var, array in value.items():
                    if isinstance(array, np.ndarray) and array.ndim >= 1:
                        for idx, val in np.ndenumerate(array):
                            records.append({
                                'variable': var,
                                'index': idx,
                                'value': val
                            })
                return pd.DataFrame.from_records(records)
            else:
                return pd.DataFrame(value)  # fallback for 1D cases
        else:
            return value

    def add_process(self,
                    name,
                    process_id,
                    config=None,
                    inputs=None,
                    outputs=None,
                    path=None
                    ):
        edge_type = "process"  # TODO -- does it matter if this is a step or a process?
        config = config or {}
        inputs = inputs or {}
        outputs = outputs or {}
        path = path or ()

        # convert string paths to lists
        # TODO -- make this into a separate path-parsing function
        for ports in [inputs, outputs]:
            for port, port_path in ports.items():
                ports[port] = parse_path(port_path)

        # make the process spec
        state = {
            name: {
                "_type": edge_type,
                "address": f"local:{process_id}",  # TODO -- only support local right now?
                "config": config,
                "inputs": inputs,
                "outputs": outputs,
            }
        }

        # nest the process in the composite at the given path
        self.composite.merge({}, state, path)
        self.reset_emitters()
        self.reset_paths()

    def initialize_process(self,
                           # name,
                           path,
                           config=None
                           ):
        config = config or {}

        if isinstance(path, str):
            path = parse_path(path)

        # assert that the process is already in the composite at the path
        retrieved = get_path(self.composite.state, path)
        # assert self.core.inherits_from(retrieved, "edge"), f"Path {path} must contain an edge/process."
        # TODO -- assert that this is a proess
        
        state = retrieved['instance'].initial_state(config)
        
        # TODO - need to project this through the edge 
        # initial = self.core.initialize_edge_state(
        #     self.composite.composition,
        #     path,
        #     retrieved)
        
        # TODO -- this is a hack because path points to the process and we need to get relative to its projection
        top_path = ()
                
        # merge this into the composite state
        self.composite.merge({}, state, top_path)
        

    def connect_process(self,
                        name,
                        inputs=None,
                        outputs=None,
                        path=None
                        ):
        path = path or ()
        
        # assert that the process is already in the composite at the path
        retrieved = get_path(self.composite.state, path)
        assert name in retrieved, f"Process {name} not found at path {path}."
              
        # build the new state with the inputs and outputs
        state = {name: {}}
        if inputs is not None:
            assert isinstance(inputs, dict), "Inputs must be a dictionary."
            state[name]["inputs"] = inputs

        if outputs is not None:
            assert isinstance(outputs, dict), "Outputs must be a dictionary."
            state[name]["outputs"] = outputs

        # nest the process in the composite at the given path
        self.composite.merge({}, state, path)
        self.composite.build_step_network()

    def generate(self):
        """
        Generates a new composite.
        """
        document = self.make_document()
        composite = Composite(
            document,
            core=self.core)
        return composite

    def generate_composite_from_document(self, document):
        """
        Generates a new composite from a document.
        """
        # document["state"] = self.core.deserialize(document.get("composition", {}), document["state"])
        composite = Composite(
            document,
            core=self.core)

        return composite

    def register_processes(self, processes):
        """
        Register processes with the core.
        """
        if processes is None:
            pass
        elif isinstance(processes, dict):
            self.core.register_processes(processes)
        else:
            print("Warning: register_processes() should be called with a dictionary of processes.")

    def register_types(self, types):
        """
        Register types with the core.
        """
        if types is None:
            pass
        elif isinstance(types, dict):
            self.core.register_types(types)
        else:
            print("Warning: register_types() should be called with a dictionary of types.")

    # def process_schema(self, process_id, string_representation=False):
    #     warnings.warn(
    #         "process_schema() is deprecated and will be removed in a future release. "
    #         "Use process_config() instead.",
    #         category=DeprecationWarning,
    #         stacklevel=2
    #     )
        # self.process_config(process_id, string_representation=string_representation)
        
    def process_config(
            self, 
            process_id, 
            dataclass=False, 
            string_representation=False,
            default=False,
    ):
        """
        Get the config schema for a process.
        """
        assert isinstance(process_id, str), "process_id must be a string"
        assert sum([dataclass, string_representation, default]) <= 1, \
            "Only one of dataclass, string_representation, or default may be True"

        try:
            process = self.core.process_registry.access(process_id)
            if dataclass:
                return self.core.dataclass(process.config_schema)
            elif string_representation:
                return self.core.representation(process.config_schema)
            elif default:
                return self.core.default(process.config_schema)
            else:
                return process.config_schema
        except KeyError as e:
            print(f"Error finding process {process_id}: {e}")
            return None

    def process_interface(self, process_id, config=None):
        """
        Get the interface for a process.
        """
        config = config or {}
        process_class = self.core.process_registry.access(process_id)
        process_instance = process_class(config, self.core)
        interface = process_instance.interface()
        interface = self.core.access(interface)
        interface = render_type(interface, self.core)

        # Add function names to the inputs and outputs
        inputs_df = pd.DataFrame.from_dict(interface['inputs'], orient='index')
        outputs_df = pd.DataFrame.from_dict(interface['outputs'], orient='index', columns=['Type'])
        combined_df = pd.concat([inputs_df, outputs_df], keys=['Inputs', 'Outputs'])
        return combined_df

    def print_processes(self):
        """
        Print the list of registered processes.
        """
        print(self.core.process_registry.list())

    def get_processes(self, as_dataframe=False):
        """
        Print the list of registered processes.
        """
        processes = self.core.process_registry.list()
        if as_dataframe:
            return pd.DataFrame(processes, columns=['Process'])
        else:
            return processes

    def get_types(self, as_dataframe=False):
        types = self.core.list()
        if as_dataframe:
            return pd.DataFrame(types, columns=['Type'])
        else:
            return types

    def print_types(self):
        """
        Print the list of registered types.
        """
        print(self.core.list())

    def get_type(self, type_id):
        type_info = self.core.access(type_id)
        type_info = render_type(type_info, self.core)
        return pd.DataFrame(list(type_info.items()), columns=['Attribute', 'Value'])

    def make_document(self):
        serialized_state = self.composite.serialize_state()

        # TODO fix RecursionError
        # schema = self.core.representation(self.composite.composition)
        schema = self.composite.serialize_schema()
        # schema = self.composite.composition

        return {
            "state": serialized_state,
            "composition": schema,
        }

    def save(self,
             filename="simulation.json",
             outdir="out",
             ):
        # make the document
        document = self.make_document()

        # Convert outdir to an absolute path
        absoutdir = os.path.abspath(outdir)

        # save to JSON
        if not filename.endswith(".json"):
            filename = f"{filename}.json"
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        filepath = os.path.join(absoutdir, filename)
        with open(filepath, "w") as json_file:
            json.dump(document, json_file, indent=4)
            print(f"Saved file: {os.path.join(outdir, filename)}")

    def find_package(self, package):
        pass

    def run(self, interval):
        """
        Run the simulation for a given interval.
        """
        if not self.emitter_paths:
            self.add_emitter()

        self.composite.run(interval)

    def step(self):
        """
        Run the simulation for a single step.
        """
        self.composite.update({}, 0)

    def _read_emitter_config(self):
        address = self.emitter_config.get("address", "local:ram-emitter")
        config = self.emitter_config.get("config", {})
        mode = self.emitter_config.get("mode", "all")

        if mode == "all":
            inputs = {}
            for key in self.composite.state.keys():
                if is_schema_key(key):  # skip schema keys
                    continue
                if self.core.inherits_from(self.composite.composition[key], "edge"):  # skip edges
                    continue
                inputs[key] = [self.emitter_config.get("inputs", {}).get(key, key)]

        elif mode == "none":
            inputs = self.emitter_config.get("emit", {})

        elif mode == "bridge":
            print("Warning: emitter bridge mode not implemented.")
            inputs = {}

        elif mode == "ports":
            print("Warning: emitter ports mode not implemented.")
            inputs = {}

        if not "emit" in config:
            config["emit"] = {
                input: "any"
                for input in inputs}

        return {
            "_type": "step",
            "address": address,
            "config": config,
            "inputs": inputs}

    def add_emitter(self):
        """
        Add an emitter to the composite.
        """

        emitter_state = self._read_emitter_config()

        # set the emitter at the path
        path = tuple(self.emitter_config.get("path", ('emitter',)))
        emitter_state = set_path({}, path, emitter_state)

        self.composite.merge({}, emitter_state)

        # TODO -- this is a hack to get the emitter to show up in the state
        # TODO -- this should be done in the merge function
        _, instance = self.core.slice(
            self.composite.composition,
            self.composite.state,
            path)

        self.emitter_paths[path] = instance
        self.composite.step_paths[path] = instance

        # rebuild the step network
        self.composite.build_step_network()

    def reset_emitters(self):
        for path, emitter in self.emitter_paths.items():
            remove_path(self.composite.state, path)
            self.add_emitter()
#         for path, instance in self.composite.step_paths.items():
#             if "emitter" in path:
#                 remove_path(self.composite.state, path)
#                 self.add_emitter()

    def get_results(self,
                    query=None,
                    decimal_places=None
                    ):
        """
        retrieves results from the emitter
        """
        if query:
            # parse each query in the list from '/top/A' to ('top', 'A')
            query = [tuple(q.lstrip('/').split('/')) for q in query]
            if 'global_time' not in query:
                query.append(('global_time',))

        emitter_paths = list(self.emitter_paths.keys())
#         step_paths = list(self.composite.step_paths.keys())
        results = []
        for path in emitter_paths:
            if "emitter" in path:
                # emitter = get_path(self.composite.state, path)
                _, emitter = self.core.slice(
                    schema=self.composite.composition, 
                    state=self.composite.state,
                    path=path
                )
                schema = self.composite.composition  # TODO -- this only works if the emitter is at the root, make it more general
                results.extend(emitter['instance'].query(query, schema))

        return round_floats(results, decimal_places=decimal_places)

    def get_timeseries(self,
                       query=None,
                       decimal_places=None,
                       as_dataframe=False,
                       ):
        """
        Gets the results and converts them to timeseries format
        """

        emitter_results = self.get_results(query=query,
                                           decimal_places=decimal_places)

        def append_to_timeseries(timeseries, state, path=()):
            if isinstance(state, dict):
                if (state.get("address") in ["process", "step", "composite"]) or state.get("address"):
                    return
                for key, value in state.items():
                    append_to_timeseries(timeseries, value, path + (key,))
            else:
                if path not in timeseries:
                    # TODO -- what if entry appeared in the middle of the simulation? Fill with Nones?
                    timeseries[path] = []
                timeseries[path].append(state)

        # get the timeseries from the emitter results
        timeseries = {}
        for state in emitter_results:
            append_to_timeseries(timeseries, state)

        # Convert tuple keys to string keys for better readability
        timeseries = {render_path(key): value for key, value in timeseries.items()}

        # Convert the timeseries dictionary to a pandas DataFrame
        if as_dataframe:
            timeseries = pd.DataFrame.from_dict(timeseries, orient='index')
            timeseries = timeseries.transpose()
            
        return timeseries


    def plot_timeseries(self,
                        query=None,
                        decimal_places=None,
                        subplot_size=(10, 5),
                        ncols=1,
                        combined_vars=None
                        ):
        """
        Plots the timeseries data for all variables using matplotlib.
        Each variable (or group of variables) gets its own subplot.

        Args:
            query (dict, optional): Queries to retrieve specific data from the emitter.
            decimal_places (int, optional): Number of significant digits to round off floats. Default is None.
            subplot_size (tuple, optional): Size of each subplot. Default is (10, 5).
            ncols (int, optional): Number of columns in the subplot grid. Default is 1.
            combined_vars (list of lists, optional): Lists of variables to combine into the same subplot. Default is None.
        """
        timeseries = self.get_timeseries(query=query, decimal_places=decimal_places)
        timeseries = flatten_timeseries_to_scalar_paths(timeseries)

        # Extract time vector
        if 'global_time' in timeseries:
            time = timeseries.pop('global_time')
        elif '/global_time' in timeseries:
            time = timeseries.pop('/global_time')
        else:
            raise KeyError("Neither 'global_time' nor '/global_time' found in timeseries.")

        if combined_vars is None:
            combined_vars = []

        # Determine individual vars
        combined_vars_flat = set(var for group in combined_vars for var in group)
        individual_vars = [var for var in timeseries if var not in combined_vars_flat]

        total_plots = len(individual_vars) + len(combined_vars)
        nrows = (total_plots + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(subplot_size[0] * ncols, subplot_size[1] * nrows))
        axes = axes.flatten() if total_plots > 1 else [axes]
        plot_idx = 0

        # Plot individual variables
        for var in individual_vars:
            ax = axes[plot_idx]
            data = pad_to_length(timeseries[var], len(time))
            ax.plot(time, data)
            ax.set_title(var)
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            plot_idx += 1

        # Plot combined variable groups
        for group in combined_vars:
            ax = axes[plot_idx]
            for var in group:
                if var not in timeseries:
                    raise KeyError(f"Variable '{var}' not found in timeseries")
                data = pad_to_length(timeseries[var], len(time))
                ax.plot(time, data, label=var)
            ax.set_title(', '.join(group))
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plot_idx += 1

        plt.tight_layout()
        plt.close(fig)
        return fig

    def plot_snapshots(self, times=None, n_snapshots=None, query=None):
        """
        Plot 2D snapshots of specified fields at specified times.
        Rows = fields; Columns = timepoints.

        Args:
            times (list of float): List of times at which to take snapshots.
            n_snapshots (int, optional): Number of equally spaced snapshots. Mutually exclusive with `times`.
            query (dict, optional): Query to pass to `get_timeseries()` to filter data.
        """
        if (times is not None) and (n_snapshots is not None):
            raise ValueError("Specify either `times` or `n_snapshots`, not both.")
        if (times is None) and (n_snapshots is None):
            n_snapshots = 5

        timeseries = self.get_timeseries(query=query)

        # Extract time vector
        if 'global_time' in timeseries:
            time = timeseries.pop('global_time')
        elif '/global_time' in timeseries:
            time = timeseries.pop('/global_time')
        else:
            raise KeyError("Neither 'global_time' nor '/global_time' found in timeseries.")

        # Validate structure
        if not isinstance(timeseries, dict):
            raise TypeError("Timeseries must be a dict.")
        for key, series in timeseries.items():
            if not isinstance(series, list) or not all(isinstance(arr, np.ndarray) for arr in series):
                raise TypeError(f"Expected a list of NumPy arrays for field '{key}', got: {type(series)}")

        # Determine timepoints
        if times is not None:
            time_indices = [np.argmin(np.abs(np.array(time) - t)) for t in times]
            display_times = [time[i] for i in time_indices]
        elif n_snapshots is not None:
            total_steps = len(time)
            time_indices = np.linspace(0, total_steps - 1, n_snapshots, dtype=int)
            display_times = [time[i] for i in time_indices]
        else:
            raise ValueError("You must specify either `times` or `n_snapshots`.")

        field_names = list(timeseries.keys())
        num_rows = len(field_names)
        num_cols = len(display_times)
        fig, axes = plt.subplots(
            num_rows, num_cols,
            figsize=(5 * num_cols, 5 * num_rows),
            gridspec_kw={'wspace': 0.1}
        )

        # Normalize axes indexing
        if num_rows == 1 and num_cols == 1:
            axes = np.array([[axes]])
        elif num_rows == 1:
            axes = np.array([axes])
        elif num_cols == 1:
            axes = np.array([[ax] for ax in axes])

        # Compute global min/max per field
        global_min_max = {}
        for field in field_names:
            all_data = timeseries[field]
            flat = np.concatenate([arr.flatten() for arr in all_data])
            global_min_max[field] = (np.min(flat), np.max(flat))

        for row, field in enumerate(field_names):
            first_im = None
            for col, time_idx in enumerate(time_indices):
                ax = axes[row, col]
                if field not in timeseries:
                    ax.set_title(f"{field} not found")
                    continue

                snapshot = timeseries[field][time_idx]
                im = ax.imshow(snapshot,
                               interpolation='nearest',
                               vmin=global_min_max[field][0],
                               vmax=global_min_max[field][1],
                               aspect='equal',
                               cmap='viridis')

                ax.set_title(f"{field} at t={time[time_idx]:.2f}")
                if col == 0:
                    first_im = im

            # Add one colorbar on the left of the row
            cbar_ax = fig.add_axes([
                axes[row, 0].get_position().x0 - 0.075,  # x-position
                axes[row, 0].get_position().y0,  # y-position
                0.015,  # width
                axes[row, 0].get_position().height  # height
            ])
            fig.colorbar(first_im, cax=cbar_ax)

        # plt.tight_layout()
        plt.show()

    def show_video(self, query=None, skip_frames=1, title=''):
        """
        Generate and display a GIF showing 2D field evolution over time.

        Args:
            query (dict, optional): Filter to apply in get_timeseries().
            skip_frames (int, optional): Interval of timepoints to skip between frames. Default is 1.
            title (str, optional): Title displayed on top of the animation.
        """
        timeseries = self.get_timeseries(query=query)

        # Extract and remove time
        if 'global_time' in timeseries:
            time = timeseries.pop('global_time')
        elif '/global_time' in timeseries:
            time = timeseries.pop('/global_time')
        else:
            raise KeyError("Neither 'global_time' nor '/global_time' found in timeseries.")

        # Validate structure
        if not isinstance(timeseries, dict):
            raise TypeError("Expected a dict of field -> list of numpy arrays")
        for field, series in timeseries.items():
            if not isinstance(series, list) or not all(isinstance(arr, np.ndarray) for arr in series):
                raise TypeError(f"Field '{field}' must be a list of NumPy arrays")

        field_names = list(timeseries.keys())
        n_fields = len(field_names)
        n_frames = len(time)

        # Compute global min/max per field
        global_min_max = {
            field: (
                np.min(np.concatenate([arr.flatten() for arr in timeseries[field]])),
                np.max(np.concatenate([arr.flatten() for arr in timeseries[field]]))
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
                img = ax.imshow(timeseries[field][i],
                                interpolation='nearest',
                                vmin=vmin,
                                vmax=vmax,
                                cmap='viridis',
                                aspect='equal'
                                )
                ax.set_title(f'{field} at t = {time[i]:.2f}')
                plt.colorbar(img, ax=ax)

            fig.suptitle(title, fontsize=16)
            # plt.tight_layout(pad=0.3)

            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=120)
            buf.seek(0)
            images.append(imageio.imread(buf))
            buf.close()
            plt.close(fig)

        # Save GIF in memory and display
        buf = io.BytesIO()
        imageio.mimsave(buf, images, format='GIF', duration=0.5, loop=0)
        buf.seek(0)
        data_url = 'data:image/gif;base64,' + base64.b64encode(buf.read()).decode()
        display(HTML(f'<img src="{data_url}" alt="{title}" style="max-width:100%;"/>'))
        
    def diagram(self,
                filename=None,
                out_dir=None,
                remove_nodes=None,
                remove_emitter=False,
                **kwargs
                ):
        """
        Generate a bigraph-viz diagram of the composite.

        Args:
            filename (str, optional): Name of the file to save the diagram. Default is None.
            out_dir (str, optional): Directory to save the diagram. Default is None.
            remove_nodes (list, optional): List of nodes to remove from the diagram. Default is None.
            remove_emitter (bool, optional): Whether to remove the emitter from the diagram. Default is False.
            **kwargs: Additional keyword arguments for get_graphviz_fig.
        """
        # Get the signature of get_graphviz_fig
        get_graphviz_fig_signature = inspect.signature(get_graphviz_fig)

        # Filter kwargs to only include those accepted by get_graphviz_fig
        get_graphviz_kwargs = {
            k: v for k, v in kwargs.items()
            if k in get_graphviz_fig_signature.parameters}

        # get the remaining kwargs
        viztype_kwargs = {
            k: v for k, v in kwargs.items()
            if k not in get_graphviz_kwargs}

        # convert list of paths in 'top/A' format to tuple of paths in ('top', 'A') format
        remove_nodes = remove_nodes or []
        remove_nodes = [tuple(node.lstrip('/').split('/')) if isinstance(node, str) else node for node in remove_nodes]
        viztype_kwargs['remove_nodes'] = remove_nodes

        # generate the graph dict
        graph_dict = self.core.generate_graph_dict(
            self.composite.composition,
            self.composite.state,
            (),
            viztype_kwargs
            )

        # get the graphviz figure
        return self.core.plot_graph(
            graph_dict,
            filename=filename,
            out_dir=out_dir,
            options=get_graphviz_kwargs)


def test_vivarium():
    initial_mass = 1.0

    grow_divide = grow_divide_agent(
        {"grow": {"rate": 0.03}},
        {},
        ["environment", "0"])

    environment = {
        "environment": {
            "0": {
                "mass": initial_mass,
                "grow_divide": grow_divide}}}

    document = {
        "state": environment,
        'bridge': {
            'inputs': {
                'environment': ['environment']}}}

    sim = Vivarium(document=document, processes=TOY_PROCESSES)

    sim.add_emitter()

    # test navigating the state
    # assert sim.composite.state.environment["0"].mass == initial_mass
    sim.save("test_vivarium_pre_simulation.json")
    sim.diagram(filename="pre_simulation", out_dir="out")

    # run simulation
    sim.run(11)
    results = sim.get_timeseries()
    print(results)

    sim.save("test_vivarium_post_simulation.json")
    sim.diagram(filename="post_simulation", out_dir="out")


def test_build_vivarium():
    from vivarium.tests import DEMO_PROCESSES

    v = Vivarium(processes=DEMO_PROCESSES)
    # add an increase process called 'increase process'
    v.add_process(name='increase',
                  process_id='increase float',  # this is the process id
                  config={'rate': 1.1},  # set according to the config schema
                  )
    # connect the 'increase' process to the state through its inputs and outputs
    v.connect_process(
        name='increase',
        inputs={'amount': ['top', 'A']},
        outputs={'amount': ['top', 'A']}
    )
    # set value of 'top.A' to 100
    v.set_value(path=['top', 'A'], value=100.0)
    print(v.get_value(path=['top', 'A']))
    # add an emitter to save the history
    v.add_emitter()
    # run the simulation for 10 time units
    v.run(interval=10)

    # get the timeseries results
    timeseries = v.get_timeseries()
    print(timeseries)
    # plot the timeseries
    fig1 = v.plot_timeseries()

    # add another process and run again
    v.add_object(name='AA', path=['top'], value=1)

    # add another increase process
    v.add_process(name='increase2',
                  process_id='increase float',
                  config={'rate': 2.0},
                  inputs={'amount': ['top', 'AA']},
                  outputs={'amount': ['top', 'AA']}
                  )

    # # run the simulation for 10 time units
    # v.run(interval=10)
    #
    # # plot the timeseries results
    # timeseries = v.get_timeseries()
    # print(timeseries)
    # v.plot_timeseries()
    #
    # plot graph
    v.diagram(filename='test_build_vivarium', out_dir='out')

    # run the simulation for 10 time units
    v.set_value(path=['global_time'], value=0)
    v.run(interval=10)
    fig2 = v.plot_timeseries()


def test_load_vivarium():
    from vivarium.tests import DEMO_PROCESSES
    current_dir = os.path.dirname(__file__)
    document_path = os.path.join(current_dir, '../test_data/demo1.json')

    # make a new Vivarium object (v2) from the saved file
    v2 = Vivarium(document=document_path, processes=DEMO_PROCESSES)

    # add another object and process
    v2.add_object(name='C', path=['top'], value=1)
    v2.add_process(name='increase3',
                   process_id='increase float',
                   config={'rate': -0.1},
                   inputs={'amount': ['top', 'C']},
                   outputs={'amount': ['top', 'C']}
                   )

    # display the current state as a diagram
    v2.diagram(dpi='120',
               show_values=True,
               filename='test_load_vivarium',
               # show_types=True,
               )


if __name__ == "__main__":
    test_vivarium()
    test_build_vivarium()
    test_load_vivarium()
