"""
Comprehensive tests for the Vivarium interface.

Covers:
- Core lifecycle (create, merge, run, results)
- Full tutorial workflow from 01_getting_started.ipynb
- Model building (add_process, add_step, connect, set/get, remove)
- Inspection (types, processes, process_config, process_interface)
- File I/O (save, load, to_dict, backward compat)
- Visualization (diagram, plot, timeseries)
- Error handling (invalid inputs, missing processes, bad paths)
"""

import os
import json
import tempfile

import numpy as np
import pytest
from process_bigraph import Process

from vivarium import Vivarium


# ---------------------------------------------------------------------------
# Test process definitions
# ---------------------------------------------------------------------------

class IncreaseFloat(Process):
    """Multiply input by rate each timestep."""
    config_schema = {
        'rate': {
            '_type': 'float',
            '_default': 0.1,
        }
    }

    def inputs(self):
        return {'amount': 'float'}

    def outputs(self):
        return {'amount': 'float'}

    def update(self, state, interval):
        return {'amount': state['amount'] * self.config['rate'] * interval}


def _register_test(core):
    core.register_link('increase_float', IncreaseFloat)
    return core


# ---------------------------------------------------------------------------
# Core lifecycle
# ---------------------------------------------------------------------------

class TestCoreLifecycle:
    """Basic create → build → run → results cycle."""

    def test_empty_vivarium(self):
        v = Vivarium()
        assert v.state is not None
        assert v.time == 0.0
        assert isinstance(repr(v), str)

    def test_create_with_document(self):
        doc = {'schema': {'x': 'float'}, 'state': {'x': 5.0}}
        v = Vivarium(document=doc)
        assert v.get('/x') == 5.0

    def test_create_with_register(self):
        v = Vivarium(register=_register_test)
        assert 'increase_float' in v.processes()

    def test_create_with_emitter_none(self):
        v = Vivarium(emitter='none')
        assert 'emitter' not in v.state


# ---------------------------------------------------------------------------
# Tutorial workflow (mirrors 01_getting_started.ipynb)
# ---------------------------------------------------------------------------

class TestTutorialWorkflow:
    """End-to-end workflow matching the Getting Started notebook."""

    def test_full_tutorial(self):
        # 1. Create empty Vivarium
        v = Vivarium(register=_register_test)

        # 2. Add state and schema
        v.merge(
            schema={'level': 'float'},
            state={'level': 100.0},
        )
        assert v.get('/level') == 100.0

        # 3. Add process and wire it
        v.add_process(
            name='growth',
            process_id='increase_float',
            config={'rate': 0.01},
            inputs={'amount': ['level']},
            outputs={'amount': ['level']},
        )

        # 4. Run the simulation
        v.run(10)
        assert v.time == 10.0

        # 5. Get timeseries results
        ts = v.timeseries()
        assert '/level' in ts
        assert '/global_time' in ts
        assert len(ts['/level']) > 1  # should have multiple snapshots

        # 6. Get as DataFrame
        df = v.timeseries(as_dataframe=True)
        assert '/level' in df.columns
        assert len(df) > 1

        # 7. Verify level increased (rate > 0)
        levels = ts['/level']
        assert levels[-1] > levels[0]

        # 8. Plot returns a figure
        fig = v.plot()
        assert fig is not None

        # 9. Inspect registry
        types_list = v.types()
        assert isinstance(types_list, list)
        assert len(types_list) > 0

        procs = v.processes()
        assert 'increase_float' in procs

        # 10. Process interface
        iface_df = v.process_interface('increase_float')
        assert 'port' in iface_df.columns
        ports = iface_df['port'].tolist()
        assert 'amount' in ports

    def test_save_and_reload(self):
        """Tutorial step 7: save and reload."""
        v = Vivarium(register=_register_test)
        v.merge(schema={'level': 'float'}, state={'level': 100.0})
        v.add_process(
            name='growth',
            process_id='increase_float',
            config={'rate': 0.01},
            inputs={'amount': ['level']},
            outputs={'amount': ['level']},
        )
        v.run(10)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'tutorial.json')
            v.save(path)

            assert os.path.exists(path)
            with open(path) as f:
                doc = json.load(f)
            assert 'schema' in doc
            assert 'state' in doc

            v2 = Vivarium.load(path, register=_register_test)
            assert v2.state is not None
            assert list(v2.state.keys())  # should have keys

    def test_extend_model(self):
        """Tutorial step 8: add a second variable and process."""
        v = Vivarium(register=_register_test)
        v.merge(schema={'level': 'float'}, state={'level': 100.0})
        v.add_process(
            name='growth',
            process_id='increase_float',
            config={'rate': 0.01},
            inputs={'amount': ['level']},
            outputs={'amount': ['level']},
        )
        v.run(10)

        # Add a second variable with a decay process
        v.merge(schema={'energy': 'float'}, state={'energy': 50.0})
        v.add_process(
            name='decay',
            process_id='increase_float',
            config={'rate': -0.02},
            inputs={'amount': ['energy']},
            outputs={'amount': ['energy']},
        )

        # Reset time and run again
        v.set('/global_time', 0.0)
        v.run(10)

        ts = v.timeseries()
        assert '/level' in ts
        assert '/energy' in ts

        # Energy should decrease (negative rate)
        energies = ts['/energy']
        assert energies[-1] < energies[0]


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------

class TestModelBuilding:

    def test_merge_schema_and_state(self):
        v = Vivarium()
        v.merge(schema={'x': 'float', 'y': 'float'}, state={'x': 1.0, 'y': 2.0})
        assert v.get('/x') == 1.0
        assert v.get('/y') == 2.0

    def test_merge_at_path(self):
        v = Vivarium()
        v.merge(schema={'inner': {'val': 'float'}}, state={'inner': {'val': 3.0}})
        assert v.get('/inner/val') == 3.0

    def test_set_and_get(self):
        v = Vivarium()
        v.merge(state={'x': 1.0}, schema={'x': 'float'})
        v.set('/x', 42.0)
        assert v.get('/x') == 42.0

    def test_get_full_state(self):
        v = Vivarium()
        v.merge(state={'a': 1.0}, schema={'a': 'float'})
        state = v.get()
        assert isinstance(state, dict)
        assert 'a' in state

    def test_remove(self):
        v = Vivarium()
        v.merge(state={'a': 1.0, 'b': 2.0}, schema={'a': 'float', 'b': 'float'})
        v.remove('/b')
        state = v.get()
        assert 'b' not in state
        assert 'a' in state

    def test_connect(self):
        v = Vivarium(register=_register_test)
        v.merge(schema={'x': 'float', 'y': 'float'}, state={'x': 1.0, 'y': 2.0})

        # Add process without wiring
        v.add_process(
            name='inc',
            process_id='increase_float',
            config={'rate': 0.1},
            inputs={'amount': ['x']},
            outputs={'amount': ['x']},
        )

        # Reconnect to y
        v.connect('inc', inputs={'amount': ['y']}, outputs={'amount': ['y']})

        v.run(5)
        ts = v.timeseries()
        # y should have changed since we wired to it
        assert '/y' in ts

    def test_add_step(self):
        """add_step uses the same _add_edge path as add_process."""
        v = Vivarium(register=_register_test)
        v.merge(schema={'x': 'float'}, state={'x': 1.0})
        # Steps use the same mechanism — just verifying it doesn't error
        v.add_step(
            name='my_step',
            step_id='increase_float',
            config={'rate': 0.5},
            inputs={'amount': ['x']},
            outputs={'amount': ['x']},
        )
        # Step should be in the state
        assert 'my_step' in v.state


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------

class TestInspection:

    def test_schema_property(self):
        v = Vivarium()
        v.merge(schema={'x': 'float'}, state={'x': 1.0})
        assert isinstance(v.schema, dict)

    def test_time_property(self):
        v = Vivarium(register=_register_test)
        v.merge(schema={'x': 'float'}, state={'x': 1.0})
        v.add_process('inc', 'increase_float', inputs={'amount': ['x']}, outputs={'amount': ['x']})
        assert v.time == 0.0
        v.run(5)
        assert v.time == 5.0

    def test_types_list(self):
        v = Vivarium()
        types_list = v.types()
        assert isinstance(types_list, list)
        assert 'float' in types_list

    def test_types_dataframe(self):
        v = Vivarium()
        df = v.types(as_dataframe=True)
        assert 'Type' in df.columns
        assert len(df) > 0

    def test_processes_list(self):
        v = Vivarium(register=_register_test)
        procs = v.processes()
        assert isinstance(procs, list)
        assert 'increase_float' in procs

    def test_processes_dataframe(self):
        v = Vivarium(register=_register_test)
        df = v.processes(as_dataframe=True)
        assert 'Process' in df.columns

    def test_process_config_schema(self):
        v = Vivarium(register=_register_test)
        config = v.process_config('increase_float')
        assert 'rate' in config

    def test_process_config_defaults(self):
        v = Vivarium(register=_register_test)
        defaults = v.process_config('increase_float', default=True)
        assert isinstance(defaults, dict)

    def test_process_interface(self):
        v = Vivarium(register=_register_test)
        df = v.process_interface('increase_float')
        assert 'port' in df.columns
        assert 'section' in df.columns
        sections = df['section'].unique()
        assert 'inputs' in sections
        assert 'outputs' in sections


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

class TestFileIO:

    def test_save_creates_file(self):
        v = Vivarium()
        v.merge(state={'val': 5.0}, schema={'val': 'float'})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.json')
            v.save(path)
            assert os.path.exists(path)

    def test_save_creates_subdirectory(self):
        v = Vivarium()
        v.merge(state={'val': 1.0}, schema={'val': 'float'})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'sub', 'deep', 'test.json')
            v.save(path)
            assert os.path.exists(path)

    def test_save_and_load_roundtrip(self):
        v = Vivarium(register=_register_test)
        v.merge(state={'val': 5.0}, schema={'val': 'float'})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.json')
            v.save(path)
            v2 = Vivarium.load(path, register=_register_test)
            assert v2.state is not None

    def test_to_dict(self):
        v = Vivarium()
        v.merge(state={'a': 1.0}, schema={'a': 'float'})
        doc = v.to_dict()
        assert 'schema' in doc
        assert 'state' in doc

    def test_backward_compat_composition_key(self):
        doc = {
            'composition': {'x': {'_type': 'float'}},
            'state': {'x': 1.0},
        }
        v = Vivarium(document=doc)
        assert v.state is not None

    def test_load_from_json_path(self):
        v = Vivarium()
        v.merge(state={'z': 9.0}, schema={'z': 'float'})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'doc.json')
            v.save(path)
            v2 = Vivarium(document=path)
            assert v2.get('/z') == 9.0


# ---------------------------------------------------------------------------
# Results and timeseries
# ---------------------------------------------------------------------------

class TestResults:

    @pytest.fixture
    def simulated_vivarium(self):
        v = Vivarium(register=_register_test)
        v.merge(schema={'x': 'float'}, state={'x': 1.0})
        v.add_process(
            name='inc',
            process_id='increase_float',
            config={'rate': 0.5},
            inputs={'amount': ['x']},
            outputs={'amount': ['x']},
        )
        v.run(5)
        return v

    def test_results_raw(self, simulated_vivarium):
        raw = simulated_vivarium.results()
        assert isinstance(raw, dict)

    def test_timeseries_dict(self, simulated_vivarium):
        ts = simulated_vivarium.timeseries()
        assert isinstance(ts, dict)
        assert '/global_time' in ts
        assert '/x' in ts

    def test_timeseries_dataframe(self, simulated_vivarium):
        df = simulated_vivarium.timeseries(as_dataframe=True)
        assert '/x' in df.columns
        assert len(df) > 1

    def test_timeseries_rounding(self, simulated_vivarium):
        ts = simulated_vivarium.timeseries(decimals=2)
        for val in ts['/x']:
            if isinstance(val, float):
                # Check that it's rounded to 2 decimal places
                assert val == round(val, 2)

    def test_timeseries_values_correct(self):
        """Verify the simulation produces correct exponential growth."""
        v = Vivarium(register=_register_test)
        v.merge(schema={'x': 'float'}, state={'x': 1.0})
        v.add_process(
            name='inc',
            process_id='increase_float',
            config={'rate': 1.0},  # doubles each step (amount * rate * interval=1)
            inputs={'amount': ['x']},
            outputs={'amount': ['x']},
        )
        v.run(3)

        ts = v.timeseries()
        times = ts['/global_time']
        values = ts['/x']

        # The emitter captures the initial state at t=0 *before* any update,
        # then once after each of the 3 steps -> 4 snapshots at times 0,1,2,3.
        # With rate=1.0 and interval=1: update = amount * 1.0 * 1.0 = amount,
        # so x doubles each step. Starting from x=1.0: 1, 2, 4, 8.
        assert times[:4] == pytest.approx([0.0, 1.0, 2.0, 3.0])
        assert len(values) >= 4
        assert values[0] == pytest.approx(1.0)  # initial state at t=0
        assert values[1] == pytest.approx(2.0)
        assert values[2] == pytest.approx(4.0)
        assert values[3] == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

class TestVisualization:

    def test_diagram(self):
        v = Vivarium()
        v.merge(state={'x': 1.0}, schema={'x': 'float'})
        graph = v.diagram()
        assert graph is not None
        html = graph._repr_html_()
        assert isinstance(html, str)
        assert 'svg' in html.lower()

    def test_diagram_with_process(self):
        v = Vivarium(register=_register_test)
        v.merge(schema={'x': 'float'}, state={'x': 1.0})
        v.add_process('inc', 'increase_float',
                       inputs={'amount': ['x']}, outputs={'amount': ['x']})
        graph = v.diagram()
        assert graph is not None

    def test_diagram_remove_nodes(self):
        v = Vivarium()
        v.merge(state={'a': 1.0, 'b': 2.0}, schema={'a': 'float', 'b': 'float'})
        graph = v.diagram(remove_nodes=['/a'])
        html = graph._repr_html_()
        assert isinstance(html, str)

    def test_plot(self):
        v = Vivarium(register=_register_test)
        v.merge(schema={'x': 'float'}, state={'x': 1.0})
        v.add_process('inc', 'increase_float',
                       inputs={'amount': ['x']}, outputs={'amount': ['x']})
        v.run(5)
        fig = v.plot()
        assert fig is not None

    def test_plot_combined_vars(self):
        v = Vivarium(register=_register_test)
        v.merge(schema={'a': 'float', 'b': 'float'}, state={'a': 1.0, 'b': 2.0})
        v.add_process('inc_a', 'increase_float',
                       inputs={'amount': ['a']}, outputs={'amount': ['a']})
        v.add_process('inc_b', 'increase_float',
                       inputs={'amount': ['b']}, outputs={'amount': ['b']})
        v.run(5)
        fig = v.plot(combined_vars=[['/a', '/b']])
        assert fig is not None

    def test_repr(self):
        v = Vivarium()
        r = repr(v)
        assert 'Vivarium' in r
        assert 'time=' in r

    def test_repr_html(self):
        v = Vivarium()
        v.merge(state={'x': 1.0}, schema={'x': 'float'})
        html = v._repr_html_()
        assert isinstance(html, str)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:

    def test_invalid_document_type(self):
        with pytest.raises(ValueError, match="must be a dict"):
            Vivarium(document=42)

    def test_add_process_unknown_id(self):
        v = Vivarium(register=_register_test)
        with pytest.raises(ValueError, match="not a registered process"):
            v.add_process('bad', 'nonexistent_process',
                          inputs={}, outputs={})

    def test_add_step_unknown_id(self):
        v = Vivarium(register=_register_test)
        with pytest.raises(ValueError, match="not a registered step"):
            v.add_step('bad', 'nonexistent_step',
                       inputs={}, outputs={})

    def test_process_config_not_found(self):
        v = Vivarium()
        with pytest.raises(KeyError, match="not found"):
            v.process_config('nonexistent')

    def test_process_interface_not_found(self):
        v = Vivarium()
        with pytest.raises(KeyError, match="not found"):
            v.process_interface('nonexistent')

    def test_remove_nonexistent_path(self):
        v = Vivarium()
        v.merge(state={'x': 1.0}, schema={'x': 'float'})
        with pytest.raises(KeyError, match="Path not found"):
            v.remove('/nonexistent')

    def test_remove_empty_path(self):
        v = Vivarium()
        with pytest.raises(ValueError, match="Cannot remove the root"):
            v.remove('')

    def test_run_invalid_interval(self):
        v = Vivarium()
        with pytest.raises(ValueError, match="positive number"):
            v.run(-1)
        with pytest.raises(ValueError, match="positive number"):
            v.run(0)
        with pytest.raises(ValueError, match="positive number"):
            v.run("ten")


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

class TestUtils:

    def test_parse_path_string(self):
        from vivarium.utils import parse_path
        assert parse_path('/top/A') == ['top', 'A']
        assert parse_path('top') == ['top']

    def test_parse_path_list(self):
        from vivarium.utils import parse_path
        assert parse_path(['a', 'b']) == ['a', 'b']

    def test_parse_path_none(self):
        from vivarium.utils import parse_path
        assert parse_path(None) == ()

    def test_parse_path_dict(self):
        from vivarium.utils import parse_path
        result = parse_path({'port': '/top/A'})
        assert result == {'port': ['top', 'A']}

    def test_render_path(self):
        from vivarium.utils import render_path
        assert render_path(('top', 'A')) == '/top/A'
        assert render_path(['top', 'A']) == '/top/A'

    def test_render_path_string(self):
        from vivarium.utils import render_path
        assert render_path('/top/A') == '/top/A'
        assert render_path('top') == '/top'

    def test_round_floats(self):
        from vivarium.utils import round_floats
        data = {'a': 1.23456, 'b': [2.34567, 3.45678]}
        result = round_floats(data, 2)
        assert result == {'a': 1.23, 'b': [2.35, 3.46]}

    def test_round_floats_none_decimals(self):
        from vivarium.utils import round_floats
        data = {'a': 1.23456}
        result = round_floats(data, None)
        assert result == data  # no rounding

    def test_round_floats_numpy(self):
        from vivarium.utils import round_floats
        arr = np.array([1.23456, 2.34567])
        result = round_floats(arr, 2)
        np.testing.assert_array_almost_equal(result, [1.23, 2.35])

    def test_pad_to_length(self):
        from vivarium.utils import pad_to_length
        assert pad_to_length([1, 2], 5) == [0, 0, 0, 1, 2]
        assert pad_to_length([1, 2, 3], 3) == [1, 2, 3]
        assert pad_to_length([1, 2, 3], 2) == [1, 2, 3]


# ---------------------------------------------------------------------------
# Results module
# ---------------------------------------------------------------------------

class TestResultsModule:

    def test_emitter_to_timeseries(self):
        from vivarium.results import emitter_to_timeseries
        snapshots = [
            {'global_time': 0.0, 'x': 1.0},
            {'global_time': 1.0, 'x': 2.0},
            {'global_time': 2.0, 'x': 4.0},
        ]
        ts = emitter_to_timeseries(snapshots)
        assert '/global_time' in ts
        assert '/x' in ts
        assert ts['/x'] == [1.0, 2.0, 4.0]

    def test_emitter_to_timeseries_skips_processes(self):
        from vivarium.results import emitter_to_timeseries
        snapshots = [
            {'global_time': 0.0, 'x': 1.0, 'proc': {'address': 'local:foo'}},
        ]
        ts = emitter_to_timeseries(snapshots)
        assert '/proc' not in ts  # processes should be skipped

    def test_flatten_timeseries(self):
        from vivarium.results import flatten_timeseries
        ts = {
            '/global_time': [0.0, 1.0],
            '/field': [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])],
        }
        flat = flatten_timeseries(ts)
        assert '/global_time' in flat
        assert '/field/0/0' in flat
        assert flat['/field/0/0'] == [1, 5]
        assert flat['/field/1/1'] == [4, 8]

    def test_extract_time(self):
        from vivarium.results import extract_time
        ts = {'/global_time': [0, 1, 2], '/x': [1, 2, 3]}
        time, rest = extract_time(ts)
        assert time == [0, 1, 2]
        assert '/global_time' not in rest
        assert '/x' in rest

    def test_extract_time_missing(self):
        from vivarium.results import extract_time
        with pytest.raises(KeyError, match="global_time"):
            extract_time({'/x': [1, 2, 3]})
