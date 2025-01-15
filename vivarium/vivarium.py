"""
Vivarium is a simulation environment that runs composites in the process bigraph.
"""
import os
import json

from IPython.display import display, Image
from IPython.testing.tools import full_path
from process_bigraph.emitter import deep_merge
from process_bigraph import ProcessTypes, Composite, pf
from process_bigraph.processes import TOY_PROCESSES
from process_bigraph.processes.growth_division import grow_divide_agent
from bigraph_schema import is_schema_key, set_path, get_path
from bigraph_viz import plot_bigraph


# class BuilderNode:




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
                 processes=None,
                 types=None,
                 core=None,
                 require=None,  # TODO -- implement this
                 emitter=None,
                 ):

        processes = processes or {}
        types = types or {}
        emitter = emitter or "all"

        # if no core is provided, create a new one
        self.core = core or ProcessTypes()

        # set the document
        document = document or {"composition": {}, "state": {}}
        # TODO make this call self.add_emitter instead of using Composite"s method
        document["emitter"] = {"mode": emitter}

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


    def get_state(self):
        return self.composite.state


    def get_schema(self):
        return self.composite.composition


    def __repr__(self):
        return f"Vivarium({pf(self.composite.state)})"


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


    def generate(self):
        document = self.make_document()
        composite = Composite(
            document,
            core=self.core)
        return composite


    def register_processes(self, processes):
        if processes is None:
            pass
        elif isinstance(processes, dict):
            self.core.register_processes(processes)
        else:
            print("Warning: register_processes() should be called with a dictionary of processes.")


    def register_types(self, types):
        if types is None:
            pass
        elif isinstance(types, dict):
            self.core.register_types(types)
        else:
            print("Warning: register_types() should be called with a dictionary of types.")


    def process_schema(self, process_id):
        process = self.core.process_registry.access(process_id)
        return self.core.representation(process.config_schema)


    def process_interface(self, process_id, config=None):
        config = config or {}
        process_class = self.core.process_registry.access(process_id)
        process_instance = process_class(config, self.core)
        return process_instance.interface()


    def print_processes(self):
        print(self.core.process_registry.list())


    def print_types(self):
        print(self.core.list())


    def make_document(self):
        serialized_state = self.composite.serialize_state()

        # TODO fix RecursionError
        # serialized_schema = self.core.representation(self.composite.composition)
        return {
            "state": serialized_state,
            # "composition": serialized_schema,
        }


    def save(self,
             filename="simulation.json",
             outdir="out",
             ):
        # TODO: add in dependent packages and version
        # TODO: add in dependent types

        document = self.make_document()

        # save the dictionary to JSON
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        filename = os.path.join(outdir, filename)

        # write the new data to the file
        with open(filename, "w") as json_file:
            json.dump(document, json_file, indent=4)
            print(f"Created new file: {filename}")


    def find_package(self, package):
        pass


    def run(self, interval):
        self.composite.run(interval)


    def step(self):
        self.composite.update({}, 0)


    def read_emitter_config(self, emitter_config):
        address = emitter_config.get("address", "local:ram-emitter")
        config = emitter_config.get("config", {})
        mode = emitter_config.get("mode", "none")

        if mode == "all":
            inputs = {
                key: [emitter_config.get("inputs", {}).get(key, key)]
                for key in self.composite.state.keys()
                if not is_schema_key(key)}

        elif mode == "none":
            inputs = emitter_config.get("emit", {})

        elif mode == "bridge":
            inputs = {}

        elif mode == "ports":
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


    def add_emitter(self, emitter_config):
        # TODO -- this should add the emitter to the document, not directly to the composite

        path = tuple(emitter_config["path"])

        step_config = self.read_emitter_config(emitter_config)
        emitter = set_path(
            {}, path, step_config)

        self.composite.merge(
            {},
            emitter)

        _, instance = self.core.slice(
            self.composite.composition,
            self.composite.state,
            path)

        self.emitter_paths[path] = instance
        self.step_paths[path] = instance


    def get_results(self, queries=None):
        results = self.composite.gather_results(queries=queries)
        return results[("emitter",)]


    def get_timeseries(self, queries=None):
        results = self.composite.gather_results(queries=queries)
        emitter_results = results[("emitter",)]

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


    def diagram(self, filename="diagram", out_dir="out", **kwargs):
        kwargs["dpi"] = kwargs.get("dpi", "140")
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
