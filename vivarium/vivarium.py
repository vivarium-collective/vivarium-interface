"""
Vivarium is a simulation environment that runs composites in the process bigraph.
"""
import os
import inspect
from IPython.display import display, Image
import json

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# process bigraph imports
from process_bigraph import ProcessTypes, Composite, pf, pp
from process_bigraph.processes import TOY_PROCESSES
from process_bigraph.processes.growth_division import grow_divide_agent
from bigraph_schema import is_schema_key, set_path, get_path
from bigraph_schema.utilities import remove_path
from bigraph_viz import VisualizeTypes
from bigraph_viz.visualize_types import get_graphviz_fig


def round_floats(data, significant_digits):
    if not significant_digits:
        return data
    if isinstance(data, dict):
        return {k: round_floats(v, significant_digits) for k, v in data.items()}
    elif isinstance(data, list):
        return [round_floats(i, significant_digits) for i in data]
    elif isinstance(data, float):
        return round(data, significant_digits)
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
        state = {}
        schema = {}
        if isinstance(value, np.ndarray):
            type = "array"
            shape = value.shape
            data = "float"  # TODO -- make this a parameter
            schema[name] = {
                '_type': type,
                '_shape': shape,
                '_data': data,
            }

        state[name] = value

        # nest the process in the composite at the given path
        self.composite.merge(schema, state, path)
        self.composite.build_step_network()

    def set_value(self,
                  path,
                  value
                  ):
        path = parse_path(path)
        self.composite.merge({}, value, path)

    def get_value(self, path):
        return get_path(self.composite.state, path)

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

    def connect_process(self,
                        process_name,
                        inputs=None,
                        outputs=None,
                        path=None
                        ):
        path = path or ()
        state = {process_name: {}}

        if inputs is not None:
            assert isinstance(inputs, dict), "Inputs must be a dictionary."
            state[process_name]["inputs"] = inputs

        if outputs is not None:
            assert isinstance(outputs, dict), "Outputs must be a dictionary."
            state[process_name]["outputs"] = outputs

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

    def process_schema(self, process_id):
        """
        Get the config schema for a process.
        """
        try:
            process = self.core.process_registry.access(process_id)
            return self.core.representation(process.config_schema)
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

    def get_processes(self):
        """
        Print the list of registered processes.
        """
        processes = self.core.process_registry.list()
        return pd.DataFrame(processes, columns=['Process'])

    def get_types(self):
        types = self.core.list()
        return pd.DataFrame(types, columns=['Type'])

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
        schema = self.core.representation(self.composite.composition)
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
                    significant_digits=None
                    ):
        """
        retrieves results from the emitter
        """
        if query:
            # parse each query in the list from 'top.A' to ('top', 'A')
            query = [tuple(q.split('.')) for q in query]
            if 'global_time' not in query:
                query.append(('global_time',))

        emitter_paths = list(self.emitter_paths.keys())
#         step_paths = list(self.composite.step_paths.keys())
        results = []
        for path in emitter_paths:
            if "emitter" in path:
                emitter = get_path(self.composite.state, path)
                results.extend(emitter['instance'].query(query))

        return round_floats(results, significant_digits=significant_digits)

    def get_timeseries(self,
                       query=None,
                       significant_digits=None
                       ):
        """
        Gets the results and converts them to timeseries format
        """

        emitter_results = self.get_results(query=query,
                                           significant_digits=significant_digits)

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
        results = pd.DataFrame.from_dict(timeseries, orient='index')
        results = results.transpose()
        return results
        # return pd.DataFrame(timeseries)


    def plot_timeseries(self,
                        query=None,
                        significant_digits=None,
                        subplot_size=(10, 5),
                        ncols=1,
                        combined_vars=None
                        ):
        """
        Plots the timeseries data for all variables using matplotlib, each variable in its own subplot.

        Args:
            query (dict, optional): Queries to retrieve specific data from the emitter.
            significant_digits (int, optional): Number of significant digits to round off floats. Default is None.
            subplot_size (tuple, optional): Size of each subplot. Default is (10, 5).
            ncols (int, optional): Number of columns in the subplot grid. Default is 1.
            combined_vars (list of lists, optional): Lists of variables to combine into the same subplot. Default is None.
        """
        timeseries = self.get_timeseries(query=query, significant_digits=significant_digits)
        # get either global_time or /global_time
        if 'global_time' in timeseries:
            time = timeseries.pop('global_time')
        elif '/global_time' in timeseries:
            time = timeseries.pop('/global_time')
        else:
            raise KeyError("Neither 'global_time' nor '/global_time' found in timeseries.")

        if combined_vars is None:
            combined_vars = []

        # Flatten combined_vars and remove duplicates
        combined_vars_flat = [var for sublist in combined_vars for var in sublist]
        combined_vars_flat = list(set(combined_vars_flat))

        # Variables not in combined_vars
        individual_vars = [var for var in timeseries if var not in combined_vars_flat]

        # Total number of subplots
        num_vars = len(individual_vars) + len(combined_vars)
        nrows = (num_vars + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(subplot_size[0] * ncols, subplot_size[1] * nrows))
        axes = axes.flatten() if num_vars > 1 else [axes]

        # Plot individual variables
        for ax, var in zip(axes, individual_vars):
            data = timeseries[var]
            if len(data) < len(time):
                data = [0] * (len(time) - len(data)) + data
            ax.plot(time, data)
            ax.set_title(var)
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')

        # Plot combined variables
        for ax, vars in zip(axes[len(individual_vars):], combined_vars):
            for var in vars:
                data = timeseries[var]
                if len(data) < len(time):
                    data = [0] * (len(time) - len(data)) + data
                ax.plot(time, data, label=var)
            ax.set_title(', '.join(vars))
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.tight_layout()
        plt.close(fig)
        return fig

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

        # convert list of paths in 'top.A' format to tuple of paths in ('top', 'A') format
        remove_nodes = remove_nodes or []
        remove_nodes = [tuple(node.split('.')) if isinstance(node, str) else node for node in remove_nodes]
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
        process_name='increase',
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
    # test_vivarium()
    test_build_vivarium()
    # test_load_vivarium()
