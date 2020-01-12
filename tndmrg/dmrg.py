import tensornetwork as tn
from tensornetwork.backends import backend_factory
import numpy as np
from tndmrg.localham import LocalHam
from itertools import chain

def update_bond(psi, b, wf, ortho,
                normalize=True,
                max_truncation_error=None,
                max_bond_dim=None):
    """Update the MPS tensors at bond b using the two-site wave-function wf.
    If the MPS orthogonality center is at site b or b+1,the canonical form
    will be preserved with the new orthogonality center position depending on the value
    of ortho.

    Args:
      psi: The MPS for which the.
      b: The bond to update.
      wf: The two-site wave-function, it is assumed that wf is a tensornetwork.Node of the form
                    s_b      s_b+1
                     |       |
        bond b-1 ----   wf   ----- bond b+1

        where the edges order of wf is [bond b-1 , s_b, s_b+1, bond b+1]
      ortho: 'left' or 'right', on which side of the bond should the orthogonality center
        be located after the update.
      normalize: Whether to keep the wave-function normalized after update.
      max_truncation_error: The maximal allowed truncation error when discarding singular values.
      max_bond_dim: An upper bound on the number of kept singular values values.

    Returns:
      trunc_svals: A list of discarded singular values.
    """
    U,S,V,trunc_svals = tn.split_node_full_svd(wf,
                                             [wf[0],wf[1]],
                                             [wf[2],wf[3]],
                                             max_truncation_err=max_truncation_error,
                                             max_singular_values=max_bond_dim)
    S.set_tensor(S.tensor / tn.norm(S))
    if ortho=='left':
        U = U @ S
        if psi.center_position == b+1:
            psi.center_position -= 1
    elif ortho=='right':
        V = S @ V
        if psi.center_position == b:
            psi.center_position +=1
    else:
        raise ValueError("ortho must be 'left' or 'right'")

    tn.disconnect(U[-1])

    psi.nodes[b] = U
    psi.nodes[b+1] = V

    return trunc_svals

def dmrg(H,psi,sweeps,verbose=True):
    """Compute the ground state of H using the density-matrix
    renormalization group (DMRG) algorithm.

    Args:
      H: MPO.
      psi: initial MPS, changed in-place.
      sweeps: A list of dictionaries, where each dictionary contains
        the sweep parameters: 'max_bond_dim' (int or None), 'max_trunc_err' (float or None).
        The number of sweeps is the length of the sweeps list.
      verbose: If True, print information about energy and bond dimension after every sweep.

    Returns:
      Es: A list of ground-state energies obtained after each sweep.
    """
    psi.position(0)
    LH = LocalHam(H,psi,backend=psi.backend.name)
    be = backend_factory.get_backend(psi.backend.name)

    Es = []
    for (nsweep,sw) in enumerate(sweeps):
        dir = 'right'
        for b in chain(range(len(psi)-1),range(len(psi)-3,-1,-1)):
            psi.position(b)
            LH.position(b)
            wf = tn.ncon([psi.nodes[b],psi.nodes[b+1]],[(-1,-2,1),(1,-3,-4)])
            wf_shape = wf.shape
            v0 = be.reshape(wf.tensor,(-1,))
            E, v = be.eigsh_lanczos(LH,initial_state=v0)
            wf_new = tn.Node(be.reshape(v[0],wf_shape),backend=psi.backend)
            update_bond(psi,
                        b,
                        wf_new,
                        ortho=dir,
                        max_bond_dim=sw['max_bond_dim'],
                        max_truncation_error=sw['max_trunc_err'])

            if b==len(psi)-2:
                dir = 'left'
            if b==0 and dir=='right':
                Es.append(E[0])
        if verbose:
            print('Sweep {:d}: energy {}, max bond dim {}'.format(nsweep,
                                                                  Es[-1],
                                                                  max(psi.bond_dimensions)))
    return Es
