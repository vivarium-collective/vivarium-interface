import os
import json
import tempfile

import pytest
from process_bigraph import Process

from vivarium import Vivarium


# --- Test Process ---

class IncreaseFloat(Process):
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
        return {'amount': state['amount'] * self.config['rate']}


def _register_test(core):
    core.register_link('increase_float', IncreaseFloat)
    return core


# --- Tests ---

def test_empty_vivarium():
    v = Vivarium()
    assert v.state is not None
    assert v.time == 0.0
    assert isinstance(repr(v), str)


def test_with_register():
    v = Vivarium(register=_register_test)
    procs = v.processes()
    assert 'increase_float' in procs


def test_add_process_and_run():
    v = Vivarium(register=_register_test)

    v.merge(state={'level': 1.0}, schema={'level': 'float'})
    v.add_process(
        name='inc',
        process_id='increase_float',
        config={'rate': 0.5},
        inputs={'amount': ['level']},
        outputs={'amount': ['level']},
    )

    v.run(5)

    ts = v.timeseries()
    assert '/level' in ts or 'level' in ts or '/global_time' in ts


def test_set_and_get():
    v = Vivarium()
    v.merge(state={'x': 1.0}, schema={'x': 'float'})
    v.set('/x', 42.0)
    assert v.get('/x') == 42.0


def test_save_and_load():
    v = Vivarium(register=_register_test)
    v.merge(state={'val': 5.0}, schema={'val': 'float'})

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'test.json')
        v.save(path)

        assert os.path.exists(path)
        with open(path) as f:
            doc = json.load(f)
        assert 'schema' in doc
        assert 'state' in doc

        v2 = Vivarium.load(path, register=_register_test)
        assert v2.state is not None


def test_backward_compat_composition_key():
    doc = {
        'composition': {'x': {'_type': 'float'}},
        'state': {'x': 1.0},
    }
    v = Vivarium(document=doc)
    assert v.state is not None


def test_types_and_processes():
    v = Vivarium(register=_register_test)

    types_list = v.types()
    assert isinstance(types_list, list)
    assert len(types_list) > 0

    types_df = v.types(as_dataframe=True)
    assert 'Type' in types_df.columns

    procs_df = v.processes(as_dataframe=True)
    assert 'Process' in procs_df.columns


def test_process_config():
    v = Vivarium(register=_register_test)
    config = v.process_config('increase_float')
    assert 'rate' in config

    defaults = v.process_config('increase_float', default=True)
    assert isinstance(defaults, dict)


def test_process_interface():
    v = Vivarium(register=_register_test)
    df = v.process_interface('increase_float')
    assert 'port' in df.columns
    assert 'section' in df.columns


def test_to_dict():
    v = Vivarium()
    v.merge(state={'a': 1.0}, schema={'a': 'float'})
    doc = v.to_dict()
    assert 'schema' in doc
    assert 'state' in doc


def test_diagram():
    v = Vivarium()
    v.merge(state={'x': 1.0}, schema={'x': 'float'})
    graph = v.diagram()
    assert graph is not None
    html = graph._repr_html_()
    assert isinstance(html, str)
