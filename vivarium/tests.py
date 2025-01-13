from process_bigraph import Process
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


def test_interface():
    v = Vivarium()






if __name__ == '__main__':
    # test_vivarium()
    test_interface()