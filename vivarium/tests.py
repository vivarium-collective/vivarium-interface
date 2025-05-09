
from bigraph_schema.units import units
from process_bigraph import Process
from process_bigraph.processes import TOY_PROCESSES

from vivarium import Vivarium


class IncreaseFloat(Process):
    config_schema = {
        'rate': {
            '_type': 'float',
            '_default': '0.1'
        }
    }

    def inputs(self):
        return {
            'amount': {
                '_type': 'float',
                '_default': 1.0
            }
        }

    def outputs(self):
        return {
            'amount': 'float'
        }

    def update(self, state, interval):
        return {
            'amount': state['amount'] * self.config['rate']
        }


class IncreaseMass(Process):
    config_schema = {
        'rate': {
            '_type': '/time',
            '_default': '0.1'
        }
    }

    def inputs(self):
        return {
            'mass': {
                '_type': 'mass',
                '_default': 1.0 * units.gram
            }
        }

    def outputs(self):
        return {
            'mass': 'mass'
        }

    def update(self, state, interval):
        time = interval * units.s
        return {
            'mass': state['mass'] * self.config['rate'] * time  # TODO - interval should be a time unit
        }



DEMO_PROCESSES = {
    'increase float': IncreaseFloat,
    'increase mass': IncreaseMass
}


def test_units():
    # make the vivarium
    v = Vivarium(processes=DEMO_PROCESSES)

    print(v.get_type('mass'))
    print(v.process_config('increase mass'))
    print(v.process_interface('increase mass'))

    # add a process
    v.add_process(name='increase',
                  process_id='increase mass',
                  config={'rate': 1/units.s},
                  inputs={'mass': '/units/mass'},   # TODO -- make slashes work
                  outputs={'mass': '/units/mass'},
                  )
    v.add_emitter()
    v.set_value(path='/units/mass', value=1*units.gram)  # TODO -- make this deserialize in vivarium

    # run the vivarium
    v.run(interval=10)

    timeseries = v.get_timeseries()
    print(f'TIMESERIES: {timeseries}')


def test_interface():

    # make the vivarium
    v = Vivarium(processes=DEMO_PROCESSES)

    # print('TYPES:')
    # v.print_types()
    # print('PROCESSES:')
    # v.print_processes()

    # print process schema and interface
    print(f'PROCESS SCHEMA: '
          f'{v.process_config("increase float")}')

    process_config = {'rate': 1}
    print(f'PROCESS INTERFACE: '
          f'{v.process_interface("increase float", process_config)}')

    # add a process
    v.add_process(name='increase',
                  process_id='increase float',
                  config=process_config,
                  inputs={'amount': ['top', 'level_in']},
                  outputs={'amount': ['top', 'level_out']},
                  # path=None
                  )
    # v.diagram(filename='A_STATE')
    v.set_value(path=['top', 'level_in'], value=1)

    # add an emitter
    v.add_emitter()
    # v.diagram(filename='B_STATE')


    # run the vivarium
    v.run(interval=10)

    timeseries = v.get_timeseries()
    print(f'TIMESERIES: {timeseries}')


    # add more states
    v.add_object(name='AA', path=['top'], value=1)
    v.diagram(filename='C_STATE')

    #
    v.save(filename='demo1', outdir='out')

    # reload the vivarium
    v2 = Vivarium(document='out/demo1.json', processes=DEMO_PROCESSES)
    v2.diagram(filename='D_STATE',
               dpi='140',
               show_values=True,
               # show_types=True
               )

    v2data = v2.get_dataclass()
    # breakpoint()


if __name__ == '__main__':
    # test_vivarium()
    # test_interface()
    test_units()
