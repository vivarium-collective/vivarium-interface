"""
Vivarium is a simulation environment that runs composites in the process bigraph.
"""
import os
import inspect
from IPython.display import display, Image
import json
import matplotlib.pyplot as plt

# process bigraph imports
from process_bigraph import ProcessTypes, Composite, pf, pp
from process_bigraph.processes import TOY_PROCESSES
from process_bigraph.processes.growth_division import grow_divide_agent
from bigraph_schema import is_schema_key, set_path, get_path
from bigraph_schema.utilities import remove_path
from bigraph_viz import plot_bigraph
from bigraph_viz.visualize import VisualizeTypes

# from vivarium
from vivarium.dict_to_object import NestedDictToObject


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


class Vivarium:
    """
    Vivarium is a controlled virtual environment for composite process-bigraph simulations.

    It manages packages and sets up the conditions for running simulations, and collects results through emitters.

    Args:
        document (dict, optional): The configuration document for the simulation, or path to the document.
        processes (dict, optional): Dictionary of processes to register.
        types (dict, optional): Dictionary of types to register.
        core (ProcessTypes, optional): The core type system.
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

        # if no core is provided, create a new one
        self.core = core or ProcessTypes()
        self.viz_core = VisualizeTypes()  # TODO -- make this a part of the core?

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
            self.require = require
            for package in self.require:
                package = self.find_package(package)
                self.core.register_types(package.get("types", {}))

        # make the composite
        self.composite = self.generate_composite_from_document(document)

        # # add an emitter
        # # TODO -- make it so add_emitter does not have to be called by user at the right time
        # # support long-standing emitter rather than having it remade?
        # self.add_emitter()

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
        state = {
            name: {
                "_type": type or 'any',
                "_value": value
            }
        }

        # nest the process in the composite at the given path
        self.composite.merge({}, state, path)
        self.composite.build_step_network()

    def set_value(self,
                  path,
                  value
                  ):
        # TODO -- what's the correct way to do this?
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

        # make the process spec
        state = {
            name: {
                "_type": edge_type,
                "address": f"local:{process_id}",  # TODO -- only support local right now?
                "config": config,
                "inputs": inputs,
                "outputs": outputs,
                # "_inputs": {},
                # "_outputs": {},
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
        inputs = inputs or {}
        outputs = outputs or {}
        path = path or ()

        state = {
            process_name: {
                "inputs": inputs,
                "outputs": outputs,
            }
        }
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
        composite = Composite(
            document,
            core=self.core)

        # Wrap `state` for attribute-style access
        # composite.state = NestedDictToObject(composite.state)
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
        return process_instance.interface()

    def print_processes(self):
        """
        Print the list of registered processes.
        """
        print(self.core.process_registry.list())

    def print_types(self):
        """
        Print the list of registered types.
        """
        print(self.core.list())

    def access_type(self, type_id):
        return self.core.access(type_id)

    def make_document(self):
        serialized_state = self.composite.serialize_state()

        # TODO fix RecursionError
        # schema = self.core.representation(self.composite.composition)
        schema = self.composite.composition

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

    # def reset(self):
    #     document = self.make_document()
    #     self.composite = self.generate_composite_from_document(document)

    def run(self, interval):
        """
        Run the simulation for a given interval.
        """
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

        self.composite.emitter_paths[path] = instance
        self.composite.step_paths[path] = instance

        # rebuild the step network
        self.composite.build_step_network()

    def reset_emitters(self):
        for path, emitter in self.composite.emitter_paths.items():
            remove_path(self.composite.state, path)
            self.add_emitter()

    def get_results(self,
                    queries=None,
                    significant_digits=None
                    ):
        """
        retrieves results from the emitter
        """

        if queries is None:
            queries = {
                path: None
                for path in self.composite.emitter_paths.keys()}

        results = {}
        for path, query in queries.items():
            emitter = get_path(self.composite.state, path)
            results[path] = emitter['instance'].query(query)

        emitter_path = list(self.composite.emitter_paths.keys())
        if len(emitter_path) >= 1:
            emitter_path = emitter_path[0]
            results = results.get(emitter_path)
        return round_floats(results, significant_digits=significant_digits)

    def get_timeseries(self,
                       queries=None,
                       significant_digits=None
                       ):
        """
        Gets the results and converts them to timeseries format
        """

        emitter_results = self.get_results(queries=queries,
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
        timeseries = {".".join(key): value for key, value in timeseries.items()}

        return timeseries

    def plot_timeseries(self, queries=None, significant_digits=None):
        """
        Plots the timeseries data for all variables using matplotlib, each variable in its own subplot.

        Args:
            queries (dict, optional): Queries to retrieve specific data from the emitter.
            significant_digits (int, optional): Number of significant digits to round off floats. Default is None.
        """
        timeseries = self.get_timeseries(queries=queries, significant_digits=significant_digits)
        time = timeseries.pop('global_time')

        num_vars = len(timeseries)
        fig, axes = plt.subplots(num_vars, 1, figsize=(10, 5 * num_vars))

        if num_vars == 1:
            axes = [axes]

        for ax, (var, data) in zip(axes, timeseries.items()):

            # if data is less than time, pad with Nones
            if len(data) < len(time):
                data = [0] * (len(time) - len(data)) + data

            ax.plot(time, data)
            ax.set_title(var)
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')

        plt.tight_layout()
        plt.show()

    def diagram(self,
                filename="diagram",
                out_dir="out",
                options=None,
                show_emitter=False,
                **kwargs
                ):
        """
        Generate a bigraph-viz diagram of the composite.
        """
        # Get the signature of plot_bigraph
        plot_bigraph_signature = inspect.signature(plot_bigraph)

        # Filter kwargs to only include those accepted by plot_bigraph
        plot_bigraph_kwargs = {k: v for k, v in kwargs.items() if k in plot_bigraph_signature.parameters}

        # graphviz = self.viz_core.generate_graphviz(
        #     self.composite.composition,
        #     self.composite.state,
        #     (),
        #     options
        #     )
        #
        # self.viz_core.plot_graph(
        #     graphviz,
        #     filename=filename,
        #     out_dir=out_dir,
        #     **kwargs)

        state = self.composite.serialize_state()
        composition = self.composite.composition.copy()

        if not show_emitter:
            if 'emitter' in state:
                del state['emitter']
                del composition['emitter']
            if 'global_time' in state:
                del state['global_time']
                del composition['global_time']

        graph = plot_bigraph(
            state=state,
            schema=composition,
            core=self.core,
            # out_dir=out_dir,
            # filename=filename,
            **plot_bigraph_kwargs)

        # save and display the graph
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, filename)
        graph.render(output_path, format="png", cleanup=True)
        display(Image(filename=f"{output_path}.png"))


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
    }

    sim = Vivarium(document=document, processes=TOY_PROCESSES)
    sim.add_emitter()

    # test navigating the state
    # assert sim.composite.state.environment["0"].mass == initial_mass

    print(pf(sim.composite.state))

    # run simulation
    sim.run(interval=40.0)
    results = sim.get_timeseries()
    print(results)

    sim.save("test_vivarium_post_simulation.json")

    sim.diagram(filename="test_vivarium", out_dir="out")


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
    v.plot_timeseries()

    # add another process and run again
    v.add_object(name='AA', path=['top'], value=1)

    # add another increase process
    v.add_process(name='increase2',
                  process_id='increase float',
                  config={'rate': 2.0},
                  inputs={'amount': ['top', 'AA']},
                  outputs={'amount': ['top', 'AA']}
                  )

    # run the simulation for 10 time units
    v.run(interval=10)

    # plot the timeseries results
    timeseries = v.get_timeseries()
    print(timeseries)
    v.plot_timeseries()

    # plot graph
    v.diagram(filename='test_vivarium', out_dir='out')


if __name__ == "__main__":
    # test_vivarium()
    test_build_vivarium()
