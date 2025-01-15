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


TOY_PROCESSES = {
    'increase': IncreaseProcess
}


def test_interface():

    # make the vivarium
    v = Vivarium(processes=TOY_PROCESSES)


    print('TYPES:')
    v.print_types()
    print('PROCESSES:')
    v.print_processes()


    print('A STATE:')
    print(v)
    v.diagram(filename='A_STATE')

    # print process schema and interface
    print('PROCESS SCHEMA:')
    print(v.process_schema('increase'))

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


    # v.wire_processes('increase')

    print('B STATE:')
    print(v)
    v.diagram(filename='B_STATE')

    # generate to fill in graph
    # v.complete()
    # v.generate()

    print('C STATE:')
    print(v)
    v.diagram(filename='C_STATE')

    x=0



if __name__ == '__main__':
    # test_vivarium()
    test_interface()