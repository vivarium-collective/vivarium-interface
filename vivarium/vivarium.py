"""
Vivarium is a simulation environment that runs composites in the process bigraph.
"""
import os
import json

from IPython.display import display, Image
from process_bigraph.emitter import deep_merge
from process_bigraph import ProcessTypes, Composite, pf
from process_bigraph.processes import TOY_PROCESSES
from process_bigraph.processes.growth_division import grow_divide_agent
from bigraph_schema import is_schema_key, set_path, get_path
from bigraph_viz import plot_bigraph
from bigraph_viz.visualize import VisualizeTypes


class Vivarium:
    """
    Vivarium is a controlled virtual environment for composite process-bigraph simulations.

    It manages packages and sets up the conditions for running simulations, and collects results through emitters.

    Attributes:
        document (dict): The configuration document for the simulation.
        core (ProcessTypes): The core process types manager.
        composite (Composite): The composite object managing the simulation.
        require (list): List of required packages for the simulation.
    """

    def __init__(self,
                 document=None,
                 document_path=None,
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
        assert not (document and document_path), "Vivarium can be initialized with either a document or document_path, not both."
        document = document or {"composition": {}, "state": {}}
        if document_path:
            with open(document_path, "r") as json_file:
                document = json.load(json_file)

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
        self.composite = Composite(
            document,
            core=self.core)

        # # add an emitter
        # self.add_emitter()


    def __repr__(self):
        return (f"Vivarium( \n"
                f"{pf(self.composite.state)})")


    def get_state(self):
        return self.composite.state


    def get_schema(self):
        return self.composite.composition


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

    def generate(self):
        """
        Generates a new composite.
        """
        document = self.make_document()
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
        process = self.core.process_registry.access(process_id)
        return self.core.representation(process.config_schema)


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


    def make_document(self):

        # TODO -- why are wires not saved?
        serialized_state = self.composite.serialize_state()

        # TODO fix RecursionError
        # serialized_schema = self.core.representation(self.composite.composition)
        schema = self.composite.composition


        return {
            "state": serialized_state,
            "composition": schema,
        }


    def save(self,
             filename="simulation.json",
             outdir="out",
             ):
        # TODO: add in dependent packages and version
        # TODO: add in dependent types

        document = self.make_document()

        # save to JSON
        if not filename.endswith(".json"):
            filename = f"{filename}.json"
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        filename = os.path.join(outdir, filename)
        with open(filename, "w") as json_file:
            json.dump(document, json_file, indent=4)
            print(f"Created new file: {filename}")


    def find_package(self, package):
        pass


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


    def read_emitter_config(self, emitter_config):
        address = emitter_config.get("address", "local:ram-emitter")
        config = emitter_config.get("config", {})
        mode = emitter_config.get("mode", "none")

        if mode == "all":
            inputs = {}
            for key in self.composite.state.keys():
                if is_schema_key(key):  # skip schema keys
                    continue
                if self.core.inherits_from(self.composite.composition[key], "edge"):  # skip edges
                    continue
                inputs[key] = [emitter_config.get("inputs", {}).get(key, key)]

        elif mode == "none":
            inputs = emitter_config.get("emit", {})

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


    def add_emitter(self, emitter_config=None):
        """
        Add an emitter to the composite.
        """

        self.emitter_config = emitter_config or self.emitter_config
        if self.composite.emitter_paths:
            # TODO delete existing emitters
            pass

        emitter_state = self.read_emitter_config(self.emitter_config)

        # set the emitter at the path
        path = tuple(self.emitter_config.get("path", ('emitter',)))
        emitter_state = set_path({}, path, emitter_state)

        self.composite.merge({},emitter_state)

        _, instance = self.core.slice(
            self.composite.composition,
            self.composite.state,
            path)

        self.composite.emitter_paths[path] = instance
        self.composite.step_paths[path] = instance


    def get_results(self, queries=None):
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
            return results.get(emitter_path)
        return results


    def get_timeseries(self, queries=None):
        """
        Gets the results and converts them to timeseries format
        """

        emitter_results = self.get_results(queries=queries)

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


    def diagram(self, filename="diagram", out_dir="out", options=None, **kwargs):
        """
        Generate a bigraph-viz diagram of the composite.
        """

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

        graph = plot_bigraph(
            state=self.composite.state,
            schema=self.composite.composition,
            core=self.core,
            # out_dir=out_dir,
            # filename=filename,
            **kwargs)

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
    sim.run(interval=40.0)
    results = sim.get_timeseries()

    print(results)

    sim.save("test_vivarium.json")

    sim.diagram(filename="test_vivarium", out_dir="out")


if __name__ == "__main__":
    test_vivarium()
