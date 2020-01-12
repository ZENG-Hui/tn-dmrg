import tensornetwork.matrixproductstates.finite_mps as mps
from tndmrg.finitempo import MPO
from tndmrg.dmrg import dmrg
import numpy as np

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

def product_state_mps(d,site_states, backend=None, dtype=np.float64):
    tensors = [np.zeros((1,d,1), dtype=dtype) for i in range(len(site_states))]
    for i,s in enumerate(site_states):
        tensors[i][0,s,0]=1
    return mps.FiniteMPS(tensors, backend=backend)

if __name__ == "main":
    N=30
    J = -1
    h = -1
    num_sweeps = 10
    psi = product_state_mps(2,np.full(N,1))
    H = tfising_mpo(h,J,N)
    sweeps = [{'max_trunc_err' : 1e-12, 'max_bond_dim': None} for i in range(num_sweeps)]

    Es = dmrg(H,psi,sweeps)

    # compare to exact result at the critical point
    E_exact = -J*(1- 1/np.sin(np.pi/(4*N + 2)))
    print('Compare with analytical result for the ground-state energy:')
    print('|E_dmrg - E_exact| = {}'.format(np.abs(Es[-1] - E_exact)))
