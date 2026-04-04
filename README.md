# Vivarium

A Pythonic interface for building, inspecting, running, and visualizing multiscale simulations with [process bigraphs](https://github.com/vivarium-collective/process-bigraph).

Vivarium integrates:
- [**bigraph-schema**](https://github.com/vivarium-collective/bigraph-schema) — A serializable type system for compositional modeling
- [**process-bigraph**](https://github.com/vivarium-collective/process-bigraph) — Composite engine and process interface
- [**bigraph-viz**](https://github.com/vivarium-collective/bigraph-viz) — Visualization for bigraph diagrams

## Installation

```bash
pip install vivarium-interface
```

## Quick Start

```python
from vivarium import Vivarium
from process_bigraph import Process

# Define a process
class Grow(Process):
    config_schema = {'rate': {'_type': 'float', '_default': 0.01}}
    def inputs(self):
        return {'mass': 'float'}
    def outputs(self):
        return {'mass': 'float'}
    def update(self, state, interval):
        return {'mass': state['mass'] * self.config['rate'] * interval}

# Register and build
v = Vivarium(register=lambda core: core.register_link('grow', Grow) or core)
v.merge(schema={'mass': 'float'}, state={'mass': 1.0})
v.add_process('growth', 'grow', inputs={'mass': ['mass']}, outputs={'mass': ['mass']})

# Run and visualize
v.run(100)
v.plot()
v.diagram()
```

## Tutorials

| Notebook | Description |
|----------|-------------|
| [Getting Started](notebooks/01_getting_started.ipynb) | Core workflow: create, build, run, plot, save/load |
| [Spatial Modeling with Spatio-Flux](notebooks/02_spatio_flux.ipynb) | Load the spatio-flux core for diffusion, particles, and spatial FBA |
| [Whole-Cell E. coli with genEcoli](notebooks/03_genecoli.ipynb) | Load and simulate the genEcoli whole-cell model |

## API Overview

### Creating a Vivarium

```python
# Empty
v = Vivarium()

# With custom types/processes
v = Vivarium(register=my_register_function)

# With a pre-built core
v = Vivarium(core=my_core)

# From a saved file
v = Vivarium.load('simulation.json')
```

### Building Models

```python
v.merge(schema={'x': 'float'}, state={'x': 1.0})
v.add_process('name', 'process_id', config={...}, inputs={...}, outputs={...})
v.set('/path/to/value', 42.0)
v.get('/path/to/value')
```

### Running Simulations

```python
v.run(100)                          # Run for 100 time units
ts = v.timeseries(as_dataframe=True)  # Get results as DataFrame
```

### Visualization

```python
v.diagram()          # Bigraph diagram (inline in Jupyter)
v.plot()             # Timeseries plots
v.plot_snapshots()   # 2D field snapshots
v.show_video()       # Animated GIF of field evolution
```

### Inspection

```python
v.types(as_dataframe=True)       # List registered types
v.processes(as_dataframe=True)   # List registered processes
v.process_interface('grow')      # View inputs/outputs for a process
v.schema                         # Current composite schema
v.state                          # Current composite state
v.time                           # Current simulation time
```

### File I/O

```python
v.save('out/simulation.json')
v2 = Vivarium.load('out/simulation.json')
doc = v.to_dict()  # Returns {schema, state}
```

## Loading External Cores

Vivarium supports loading custom types and processes from external libraries:

```python
# Spatio-flux: spatial modeling
from spatio_flux import register_types
v = Vivarium(register=register_types)

# Chain multiple libraries
def my_setup(core):
    from spatio_flux import register_types as sf_reg
    sf_reg(core)
    core.register_link('my_process', MyProcess)
    return core

v = Vivarium(register=my_setup)
```

## License

MIT
