This repository contains a toy DMRG code based on the [TensorNetwork](https://github.com/google/TensorNetwork) library.
I wrote this mainly to play around with TensorNetwork and get a feeling of how the library works. 

This can be useful for you if you are looking for examples of TensorNetwork usage and want to develop your own code based on it. 

However, if you are looking for a DMRG code for immediate use in your research I would suggest 
[ITensors.jl](https://github.com/ITensor/ITensors.jl) (Julia),
[ITensor](https://github.com/ITensor/ITensor) (C++) or [TenPy](https://github.com/tenpy/tenpy) (Python), among others. 

To install first install the master version of TensorNetwork, and then this package:
```
pip install git+https://github.com/google/TensorNetwork.git
pip install git+https://github.com/orialb/tn-dmrg.git
```

For usage see the [example script](examples/tfising.py) which calculates the ground state of the transverse-field Ising model.
