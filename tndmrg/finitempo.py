import numpy as np
import tensornetwork as tn
from tensornetwork.network_components import Node, Edge

class MPO:
    """
    Class for finite matrix-product operator.
    MPO tensors are stored in with index order
    W_n = self.nodes[n]["bond n-1", "s'","s","bond n"]
                  s'
                  |
    bond n-1 --- W_n --- bond n
                  |
                  s
    """

    def __init__(self,tensors,backend=None):
        self.nodes = [Node(tensors[n],
                           name='node{}'.format(n),
                           axis_names= ['bond {}'.format(n), 'sp', 's', 'bond {}'.format(n+1)],
                           backend=backend)
                      for n in range(len(tensors))]
        self.backend = backend

    def __len__(self):
        return len(self.nodes)

