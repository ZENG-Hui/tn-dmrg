import tensornetwork as tn
from tensornetwork.backends import backend_factory
import numpy as np
from tndmrg.localham import LocalHam
from itertools import chain

def update_bond(psi,b,wf,ortho,**kwargs):
    U,S,V,trunc_svals = tn.split_node_full_svd(wf,
                                             [wf[0],wf[1]],
                                             [wf[2],wf[3]],
                                             max_truncation_err=kwargs.get('max_truncation_error',None),
                                             max_singular_values=kwargs.get('max_bond_dim',None))
    S.set_tensor(S.tensor / tn.norm(S))
    if ortho=='left':
        U = U @ S
    elif ortho=='right':
        V = S @ V
    else:
        raise ValueError("ortho must be 'left' or 'right'")

    tn.disconnect(U[-1])

    psi.nodes[b] = U
    psi.nodes[b+1] = V

    return trunc_svals

def dmrg(H,psi,sweeps,**kwargs):
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
                        max_trunc_err=sw['max_trunc_err'])

            if b==len(psi)-2:
                dir = 'left'
            if b==0 and dir=='right':
                Es.append(E[0])
        if kwargs.get('verbose',True):
            print('Sweep {:d}: energy {}, max bond dim {}'.format(nsweep,
                                                                  Es[-1],
                                                                  max(psi.bond_dimensions)))
    return Es
