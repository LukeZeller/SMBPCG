# SMBPCG
We use LSTMs to improve the aesthetics of procedurally generated *Super Mario Bros.* levels.
These levels are generated by applying CMA-ES to a GAN trained on the *Super Mario Bros.* level 1-1.

## Prerequisites
Installation instructions for PyTorch can be found here: https://pytorch.org/get-started/locally/

We use pyjnius to wrap the Java emulator into Python. Cython is required to install pyjnius.
```
pip install cython
pip install pyjnius
pip install cma
```
Java 1.8 is required as it is used for the Mario AI compeition 2009 A* agent made by Robin Baumgarten, https://github.com/RobinB/mario-astar-robinbaumgarten.
## Authors
* **Luke Zeller** - [LukeZeller](https://github.com/LukeZeller)
* **Justin Mott** - [justinmmott](https://github.com/justinmmott)
* **Saujas Nandi** - [s-nandi](https://github.com/s-nandi)
