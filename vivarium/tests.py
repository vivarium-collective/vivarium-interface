from fileinput import filename

from process_bigraph import Process
from process_bigraph.processes import TOY_PROCESSES

from vivarium import Vivarium


class IncreaseProcess(Process):
    config_schema = {
        'rate': {
            '_type': 'float',
            '_default': '0.1'}}

    def inputs(self):
        return {
            'level': 'float'}

    def outputs(self):
        return {
            'level': 'float'}

    def update(self, state, interval):
        return {
            'level': state['level'] * self.config['rate']}


DEMO_PROCESSES = {
    'increase': IncreaseProcess
}


def test_interface():

    # make the vivarium
    v = Vivarium(processes=DEMO_PROCESSES)

    print('TYPES:')
    v.print_types()
    print('PROCESSES:')
    v.print_processes()

    print('A STATE:')
    print(v)
    v.diagram(filename='A_STATE')

    # print process schema and interface
    print('PROCESS SCHEMA:')
    print(v.process_config_schema('increase'))

    process_config = {'rate': 1}
    print('PROCESS INTERFACE:')
    print(v.process_interface('increase', process_config))


    # add a process
    v.add_process(name='increase',
                  process_id='increase',
                  config=process_config,
                  inputs={'level': ['level in']},
                  outputs={'level': ['level out']},
                  # path=None
                  )

    v.add_emitter()
    print('B STATE:')
    print(v)
    v.diagram(filename='B_STATE')

    # run the vivarium
    v.run(interval=10)

    timeseries = v.get_timeseries()
    print('TIMESERIES:')
    print(timeseries)

    #
    # print('C STATE:')
    # print(v)
    # v.diagram(filename='C_STATE')



if __name__ == '__main__':
    # test_vivarium()
    test_interface()