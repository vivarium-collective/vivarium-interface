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
    v = Vivarium(processes=TOY_PROCESSES)
    print('TYPES:')
    v.print_types()
    print('PROCESSES:')
    v.print_processes()


    print('A STATE:')
    print(v)
    v.diagram(filename='A_STATE')

    # add a process
    v.add_process(name='increase',
                  process_id='increase',
                  config={'rate': 1},
                  # inputs=None,
                  # outputs=None,
                  # path=None
                  )

    print('B STATE:')
    print(v)
    v.diagram(filename='B_STATE')

    # generate to fill in graph
    v.fill()
    # v.generate()

    print('C STATE:')
    print(v)
    v.diagram(filename='C_STATE')

    x=0



if __name__ == '__main__':
    # test_vivarium()
    test_interface()