import tensornetwork as tn
import numpy as np
import tensornetwork.backends.backend_factory as backend_factory

#TODO: extend this to arbitrary cell size
class LocalHam():
    """
    Class representing a local Hamiltonian acting on a local wavefunction.
    Currently only 2-site local Hamiltonian is supported.
    """
    def __init__(self, H, psi, backend, pos=0):
        if len(H) != len(psi):
            raise ValueError('Expected MPO and MPS of equal length, got MPS of \
                              length {} and MPO of length {}'.format(len(psi),len(H)))
        self.H = H
        self.psi = psi
        self.pos = pos
        self.backend = backend
        self.renvs = self._build_right_envs(pos + 2)
        self.lenvs = self._build_left_envs(pos)


    def _build_right_envs(self,rpos):
        R = tn.Node(np.array([[[1]]]), backend=self.backend,
                    name='right_env_{}'.format(len(self.H)))
        renvs = [R]
        for i in reversed(range(rpos,len(self.H))):
            nodes = [R,self.psi.nodes[i], self.H.nodes[i], tn.conj(self.psi.nodes[i])]
            R = tn.ncon( nodes, [(1,3,5),(-1,2,1),(-2,2,4,3),(-3,4,5)],
                         backend=self.backend)
            R.set_name('right_env_{}'.format(i))
            renvs.append(R)

        return renvs


    def _build_left_envs(self,lpos):
        L = tn.Node(np.array([[[1]]]), backend=self.backend,
                    name='left_env_0')
        lenvs = [L]
        for i in range(lpos):
            nodes = [L,self.psi.nodes[i], self.H.nodes[i], tn.conj(self.psi.nodes[i])]
            L = tn.ncon(nodes, [(1,3,5),(1,2,-1),(3,2,4,-2),(5,4,-3)],
                        backend=self.backend)
            L.set_name('left_env_{}'.format(i+1))
            lenvs.append(L)

        return lenvs

    def __call__(self,v):
        be = backend_factory.get_backend(self.backend)
        shape = ([self.lenvs[-1].shape[0]] +
                  self.psi.physical_dimensions[self.pos:self.pos+2] +
                 [self.renvs[-1].shape[0]])
        vnode = tn.Node(be.reshape(v,shape))
        nodes = ( [self.lenvs[-1], vnode] +
                  self.H.nodes[self.pos:self.pos+2] +
                  [self.renvs[-1]])
        vout = tn.ncon(nodes,
                       [(1,3,-1), (1,2,4,6), (3,-2,2,5), (5,-3,4,7), (6,7,-4)],
                       backend=self.backend)
        return be.reshape(vout.tensor,be.shape(v))

