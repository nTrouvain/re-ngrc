# [Re] Next Generation Reservoir Computing

Replication of "Next Generation Reservoir Computing" from
Daniel J. Gauthier, Erik Bollt, Aaron Grifﬁth and Wendson A. S. Barbosa.

## Dependencies

All code was executed using [Python 3.7.9](https://www.python.org/downloads/release/python-379/), in conformity with the *Methods* section of the original paper.

### Basic model

Dependencies for basic model can be found in `requirements.txt`. They are based on the dependencies described in the *Methods* section of the original paper.

```
numpy==1.20.2
scipy==1.6.2
matplotlib==3.5.3
```

### ReservoirPy implementation

Dependencies for the [`reservoirpy`](https://github.com/reservoirpy/reservoirpy) based implementation of the NVAR can be found in `requirements-reservoirpy.txt`.

 `numpy` version had to be increased with regard to the original paper to support [`reservoirpy`](https://github.com/reservoirpy/reservoirpy) requirements.

```
numpy==1.21.6
scipy==1.6.2
matplotlib==3.5.3
reservoirpy==0.3.5
```
