import os
import json
from dataclasses import is_dataclass, asdict

from bigraph_schema import Edge, get_path, set_path, deep_merge
from bigraph_schema.protocols import local_lookup_module
from process_bigraph import Process, Step, Composite, ProcessTypes, pf
from bigraph_viz.diagram import plot_bigraph


def node_from_tree(
        builder,
        schema,
        tree,
        path=()
):
    # TODO -- this might need to use core.fold()
    node = BuilderNode(builder, path)
    if isinstance(tree, dict):
        for key, subtree in tree.items():
            next_path = path + (key,)
            node.branches[key] = node_from_tree(
                builder=builder,
                schema=schema.get(key, schema) if schema else {},
                tree=subtree,
                path=next_path)

    return node


class BuilderNode:

    def __init__(self, builder, path):
        self.builder = builder
        self.path = path
        self.branches = {}

    def __repr__(self):
        tree = self.value()
        return f"BuilderNode({pf(tree)})"

    def __getitem__(self, keys):
        # Convert single key to tuple
        keys = (keys,) if isinstance(keys, (str, int)) else keys
        head = keys[0]
        if head not in self.branches:
            self.branches[head] = BuilderNode(
                builder=self.builder,
                path=self.path + (head,))

        tail = keys[1:]
        if len(tail) > 0:
            return self.branches[head].__getitem__(tail)
        else:
            return self.branches[head]

    def __setitem__(self, keys, value):
        # Convert single key to tuple
        keys = (keys,) if isinstance(keys, (str, int)) else keys
        head = keys[0]
        tail = keys[1:]
        path_here = self.path + (head,)

        if head not in self.branches:
            self.branches[head] = BuilderNode(
                builder=self.builder,
                path=path_here)

        if len(tail) > 0:
            self.branches[head].__setitem__(tail, value)
        elif isinstance(value, dict):
            if '_type' in value:
                set_path(
                    tree=self.builder.schema,
                    path=path_here,
                    value=value['_type'])

            if '_value' in value:
                set_path(
                    tree=self.builder.tree,
                    path=path_here,
                    value=value['_value'])

                self.branches[head] = BuilderNode(
                    builder=self.builder,
                    path=path_here)

            else:
                self.branches[head] = node_from_tree(
                    builder=self.builder,
                    schema=self.schema(),
                    tree=value,
                    path=path_here)
        else:
            # set the value
            set_path(tree=self.builder.tree, path=path_here, value=value)

    def update(self, state):
        self.builder.tree = deep_merge(self.builder.tree, state)
        self.builder.complete()

    def value(self):
        return get_path(self.builder.tree, self.path)

    def schema(self):
        return get_path(self.builder.schema, self.path)

    def top(self):
        return self.builder.node

    def add_process(
            self,
            name,
            config=None,
            inputs=None,
            outputs=None,
            **kwargs
    ):
        """ Add a process to the tree """
        # TODO -- assert this process is in the process_registry
        assert name, 'add_process requires a name as input'
        process_class = self.builder.core.process_registry.access(name)

        # Check if config is a dataclass and convert to dict if so
        if is_dataclass(config):
            config = asdict(config)
        else:
            config = config or {}
        config.update(kwargs)

        # what edge type is this? process or step
        edge_type = 'process'
        if issubclass(process_class, Step):
            edge_type = 'step'

        # make the process spec
        state = {
            '_type': edge_type,
            'address': f'local:{name}',  # TODO -- only support local right now?
            'config': config,
            'inputs': {} if inputs is None else inputs,
            'outputs': {} if outputs is None else outputs,
        }

        set_path(tree=self.builder.tree, path=self.path, value=state)
        self.builder.complete()

    def connect(self, port=None, target=None):
        value = self.value()
        schema = self.schema()
        assert self.builder.core.check('edge', value), "connect only works on edges"

        if port in schema['_inputs']:
            value['inputs'][port] = target
        if port in schema['_outputs']:
            value['outputs'][port] = target

    def connect_all(self, append_to_store_name='_store'):
        # Check if the current node is an edge and perform connections if it is
        value = self.value()
        if self.builder.core.check('edge', value):
            schema = self.schema()
            for port in schema.get('_inputs', {}).keys():
                if port not in value.get('inputs', {}):
                    value['inputs'][port] = [port + append_to_store_name]
            for port in schema.get('_outputs', {}).keys():
                if port not in value.get('outputs', {}):
                    value['outputs'][port] = [port + append_to_store_name]
            # Optionally, update the current node value here if necessary

        # Recursively apply connect_all to all child nodes
        for child in self.branches.values():
            child.connect_all(append_to_store_name=append_to_store_name)

    def interface(self, print_ports=False):
        value = self.value()
        schema = self.schema()
        if not self.builder.core.check('edge', value):
            warnings.warn(f"Expected '_type' to be in {EDGE_KEYS}, found '{tree_type}' instead.")
        elif self.builder.core.check('edge', value):
            process_ports = {}
            process_ports['_inputs'] = schema.get('_inputs', {})
            process_ports['_outputs'] = schema.get('_outputs', {})
            if not print_ports:
                return process_ports
            else:
                print(pf(process_ports))

    def emit(self, key=None, port=None):
        if key is None:
            key =  port
        value = self.value()
        schema = self.schema()
        if self.builder.core.check('edge', value):
            inputs = value.get('inputs', {})
            outputs = value.get('outputs', {})
            inputs_schema = schema.get('_inputs', {})
            outputs_schema = schema.get('_outputs', {})

            if not port:
                # connect to all ports
                for prt, val in inputs_schema.items():
                    self.builder.node['emitter', 'config', 'emit', prt] = val
                for prt, val in outputs_schema.items():
                    self.builder.node['emitter', 'config', 'emit', prt] = val
            elif port in inputs_schema:
                # update the emitter config
                self.builder.node['emitter', 'config', 'emit'].value().update({key: inputs_schema[port]})
                self.builder.node['emitter', 'inputs', key] = list(self.path[:-1]) + list(inputs[port])
                self.builder.node['emitter'].schema()['_inputs'].update({key: inputs_schema[port]})

            elif port in outputs_schema:
                self.builder.node['emitter', 'config', 'emit'].update({key: outputs_schema[port]})
                self.builder.node['emitter', 'inputs', key] = list(self.path[:-1]) + list(outputs[port])
                self.builder.node['emitter'].schema()['_inputs'].update({key: outputs_schema[port]})

            else:
                raise Exception(f"can not connect port {port}. available ports for this edge include {inputs_schema.keys()} {outputs_schema.keys()}")
        else:
            # emit the path
            self.builder.node['emitter', 'config', 'emit', key] = schema
            self.builder.node['emitter', 'inputs', key] = self.path


class Builder:

    def __init__(
            self,
            schema=None,
            tree=None,
            core=None,
            emitter='ram-emitter',
            file_path=None,
    ):
        schema = schema or {}
        tree = tree or {}

        if file_path:
            with open(file_path, 'r') as file:
                graph_data = json.load(file)
                tree = deep_merge(tree, graph_data)

        self.core = core or ProcessTypes()
        self.schema, self.tree = self.core.complete(schema, tree)
        self.node = node_from_tree(self, self.schema, self.tree)
        self.add_emitter(emitter=emitter)


    def __repr__(self):
        return f"Builder({pf(self.tree)})"

    def __getitem__(self, keys):
        return self.node[keys]

    def __setitem__(self, keys, value):
        self.node.__setitem__(keys, value)
        self.complete()


    def get_dataclass(self, process_name):
        """
        Fetches the dataclass for the given process name from the process_registry.

        Args:
            process_name (str): The name of the process whose dataclass is requested.

        Returns:
            dataclass for the requested process configuration schema.
        """
        if hasattr(self.core.process_registry, 'get_dataclass'):
            return self.core.process_registry.get_dataclass(process_name)
        else:
            raise NotImplementedError("Process registry does not support dataclass retrieval.")

    def update(self, state):
        self.node.update(state)

    # def list_types(self):
    #     return self.core.type_registry.list()
    #
    # def list_processes(self):
    #     return self.core.process_registry.list()

    def complete(self):
        self.schema, self.tree = self.core.complete(self.schema, self.tree)

    def connect_all(self, append_to_store_name='_store'):
        self.node.connect_all(append_to_store_name=append_to_store_name)

    def visualize(self, filename=None, out_dir=None, **kwargs):
        return plot_bigraph(
            state=self.tree,
            schema=self.schema,
            core=self.core,
            out_dir=out_dir,
            filename=filename,
            **kwargs)

    def generate(self):
        composite = Composite({
            'state': self.tree,
            'composition': self.schema},
            core=self.core)
        self.tree = composite.state
        self.schema =composite.composition
        return composite

    def document(self):
        return self.core.serialize(
            self.schema,
            self.tree)

    def write(self, filename, outdir='out'):
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        filepath = f"{outdir}/{filename}.json"
        document = self.document()

        # Writing the dictionary to a JSON file
        with open(filepath, 'w') as json_file:
            json.dump(document, json_file, indent=4)

        print(f"File '{filename}' successfully written in '{outdir}' directory.")

    def register_type(self, key, schema):
        self.core.type_registry.register(key, schema)

    def register_process(self, process_name, address=None):
        """
        Register processes into the local core type system
        """
        assert isinstance(process_name, str), f'Process name must be a string: {process_name}'

        if address is None:  # use as a decorator
            def decorator(cls):
                if not issubclass(cls, Edge):
                    raise TypeError(f"The class {cls.__name__} must be a subclass of Edge")
                self.core.process_registry.register(process_name, cls)
                return cls
            return decorator

        else:
            # Check if address is a string
            # TODO -- we want to also support remote processes with non local protocol
            if isinstance(address, str):
                process_class = local_lookup_module(address)
                self.core.process_registry.register(process_name, process_class)

            # Check if address is a class object
            elif issubclass(address, Edge):
                self.core.process_registry.register(process_name, address)
            else:
                raise TypeError(f"Unsupported address type for {process_name}: {type(address)}. Registration failed.")

    def add_process(self, process_id, name, config, path=None):
        path = path or []
        assert isinstance(process_id, str), f'Process id must be a string: {process_id}'
        path.append(process_id)
        self.node[path].add_process(name, config)
