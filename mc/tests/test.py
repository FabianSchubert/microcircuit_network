"""
unit tests
"""

import unittest

from mc.network import Network

from . import test_model


class TestNetwork(unittest.TestCase):

    def test_init_empty_net(self):
        """
        Test the instantiation of an empty default network.
        name: str
    model_def: types.ModuleType
    size_input: int
    size_hidden: list
    size_output: int
    t_inp_max: int
    spike_buffer_size: int
    spike_buffer_size_val: int
    spike_rec_pops: list = field(default_factory=list)
    spike_rec_pops_val: list = field(default_factory=list)
    dt: float = 0.1
    plastic: bool = True
    t_inp_static_max: int = 0
        """
        net = Network("testnet", test_model,
                      10, [10], 10,
                      10,
                      1000,
                      1000,
                      [],
                      [])

        self.assertIsInstance(net, Network)


if __name__ == '__main__':
    unittest.main()