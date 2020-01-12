import numpy as np
import tensornetwork as tn
from tensornetwork.matrixproductstates.finite_mps import FiniteMPS
from tndmrg.localham import LocalHam
from tndmrg.finitempo import MPO
from itertools import chain

def identity_mpo(N):
    return MPO([np.eye(2).reshape(1,2,2,1) for i in range(N)],backend="numpy")

def test_identity_mpo():
    N = 10
    psi = FiniteMPS.random([2 for i in range(10)],[4 for i in range(9)], dtype=np.float64)
    Id = identity_mpo(N)
    lh = LocalHam(Id,psi,"numpy")
    # test shifting right
    for b in range(len(psi)-1):
        lh.position(b)
        assert np.isclose(lh.energy(),1,atol=1e-10)

    # test shifting back left
    for b in range(len(psi)-2,-1,-1):
        lh.position(b)
        assert np.isclose(lh.energy(),1,atol=1e-10)
