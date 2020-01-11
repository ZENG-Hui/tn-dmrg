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

    def _axis_names(self,n,L):
        if 0<n<L-1:
            return 
        elif n==0:
            return ['sp','s','bond {}'.format(n+1)]
        else:
            return ['bond {}'.format(n), 'sp','s']

    def __len__(self):
        return len(self.nodes)

def tfising_mpo(h,J,L,backend='numpy'):
    """
    generate a finite MPO representing the transverse field
    Ising model.
    """
    sz=np.array([[1,0],[0,-1]])
    sx=np.array([[0,1],[1,0]])
    Id=np.eye(2)

    h_bond=np.array([[Id,   np.zeros((2,2)), np.zeros((2,2))],
                     [sz,   np.zeros((2,2)), np.zeros((2,2))],
                     [h*sx, J*sz,            Id]])
    h_bond=h_bond.transpose((0,2,3,1))
    LH=np.array([0,0,1]).reshape(( 1,3 ))
    RH=np.array([1,0,0]).reshape(( 3,1 ))

    mpo_tensors = ( [np.tensordot(LH,h_bond,axes=[[-1],[0]])] +
                    [np.copy(h_bond) for i in range(1,L-1)] +
                    [ np.tensordot(h_bond,RH,axes=[[-1],[0]]) ] )

    return MPO(mpo_tensors,backend=backend)

