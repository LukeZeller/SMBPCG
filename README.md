# SMBPCG
We use LSTMs to improve the aesthetics of procedurally generated *Super Mario Bros.* levels. 
These levels are generated by applying CMA-ES to a GAN trained on the *Super Mario Bros.* level 1-1.

## Prerequisites
Installation instructions for PyTorch can be found here: https://pytorch.org/get-

We use pyjnius to wrap the Java emulator into Python. Cython is required to install pyjnius.
```
pip install cython
pip install pyjnius
pip install cma
```

## Authors
* **Luke Zeller** - [LukeZeller](https://github.com/LukeZeller)
* **Justin Mott** - [Justinmmott](https://github.com/justinmmott)
* **Saujas Nandi** - [s-nandi](https://github.com/s-nandi)