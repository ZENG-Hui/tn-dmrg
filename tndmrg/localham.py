import tensornetwork as tn
import numpy as np
import tensornetwork.backends.backend_factory as backend_factory

#TODO: extend this to arbitrary cell size (or at least one-site)
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
        self._len = len(psi)
        self.pos = pos
        self.backend = backend
        self.cell_len = 2
        self.renvs = self._build_right_envs(pos + self.cell_len)
        self.lenvs = self._build_left_envs(pos)

    def _build_right_envs(self,rpos):
        R = tn.Node(np.array([[[1]]]), backend=self.backend,
                    name='right_env_{}'.format(self._len))
        renvs = [R]
        for i in reversed(range(rpos,self._len)):
            self.psi.position(i-1)
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
            self.psi.postion(i+1)
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

    def _shift_position_left(self,pos):
        self.lenvs = self.lenvs[0:pos+1]
        while self.pos > pos:
            self.psi.position(self.pos+1)
            nodes = [self.renvs[-1],
                     self.psi.nodes[self.pos+1],
                     self.H.nodes[self.pos+1],
                     tn.conj(self.psi.nodes[self.pos+1])]
            R = tn.ncon( nodes, [(1,3,5),(-1,2,1),(-2,2,4,3),(-3,4,5)],
                         backend=self.backend)
            R.set_name('right_env_{}'.format(self.pos+1))
            self.renvs.append(R)
            self.pos -= 1

    def _shift_position_right(self,pos):
        self.renvs = self.renvs[:-(pos-self.pos)]
        while self.pos < pos:
            self.psi.position(self.pos+1)
            nodes = [self.lenvs[-1],
                     self.psi.nodes[self.pos],
                     self.H.nodes[self.pos],
                     tn.conj(self.psi.nodes[self.pos])]
            L = tn.ncon(nodes, [(1,3,5),(1,2,-1),(3,2,4,-2),(5,4,-3)],
                        backend=self.backend)
            L.set_name('left_env_{}'.format(self.pos+1))
            self.lenvs.append(L)
            self.pos +=1

    def position(self,pos):
        if not 0<=pos<=self._len-1:
            raise ValueError("Position `pos` must be between 0 and {}".format(self._len-2))

        if self.pos > pos:
            self._shift_position_left(pos)
        elif self.pos < pos:
            self._shift_position_right(pos)


    def energy(self):
        """
        Measure the energy expectation value by completing the network contraction.
        """
        E = self.renvs[-1]
        self.psi.position(self.pos)
        for i in [self.pos+1,self.pos]:
            nodes = [E,
                     self.psi.nodes[i],
                     self.H.nodes[i],
                     tn.conj(self.psi.nodes[i])]
            E = tn.ncon( nodes, [(1,3,5),(-1,2,1),(-2,2,4,3),(-3,4,5)],
                         backend=self.backend)

        E = tn.ncon([self.lenvs[-1],E], [(1,2,3),(1,2,3)])
        return E.tensor.item()
