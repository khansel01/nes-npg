# Reinforcement Learning
This project implements Natural Policy Gradients and Natural Evolution
Strategies algorithm for gym environments as well as quanser robot
environments. Various examples are provided applying these algorithms to
different platforms. However not all of them are solved yet. Using the main
module you can apply the algorithm to different platforms and adjust the
parameters to try to solve these platforms.

## Getting Started
### Prerequisites
This project is compatible with python3.6 and
[conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/ "Install miniconda")
. Further the
[quanser package](https://git.ias.informatik.tu-darmstadt.de/quanser/clients/tree/master "Install quanser package")
is needed.

##### Install python 3.6:
```bash
$ sudo add-apt-repository ppa:deadsnakes/ppa
$ sudo apt-get update
$ sudo apt-get install python3.6
```

### Install
##### Install virtual environment:
You can also change the environment name as necessary.
```bash
$ conda env create -f Skateboard_in_space.yaml
```

##### Install quanser pkg to environment
Navigate to quanser package *.../clients/*
```bash
$ source activate Skateboard_in_space
$ pip3 install -e .
```

## Usage
To run any of the given examples please simply execute the respective file.
The examples are set to run a benchmark test on a already trained policy.
Alternatively adjust the main.py to run the code on any platforms with
continuous action spaces. The algorithms are not implemented for discrete
action spaces.

##### A short example
```python
from nes import NES
from utilities.environment import Environment
from models.nn_policy import Policy
from agent import Agent

env = Environment('CartpoleStabShort-v0')
policy = Policy(env, hidden_dim=(8,))
algorithm = NES(policy.length)
agent = Agent(env, policy, algorithm)
agent.train_policy(episodes=200)
agent.run_benchmark()
```

## Project Structure
```sh
reinf/
    agent.py
    main.py
    nes.py
    npg.py
    __init__.py
    Readme.md
    models/
        __init__.py
        Readme.md
        baseline.py
        nn_policy.py
    utilities/
        __init__.py
        Readme.md
        conjugate_gradient.py
        environment.py
        estimations.py
        logger.py
        normalizer.py
    examples/
        ...
        trained_data/
            ...
```

## Developers
- Janosch Moos
- Kay Hansel
- Cedric Derstroff

## Bibliography
[1] Aravind Rajeswaran, Kendall Lowrey, Emanuel Todorov and
    Sham Kakade, Towards Generalization and Simplicity in Continuous
    Control, CoRR, 1703 (2017)

[2] Sham Kakade, A Natural Policy Gradient, NIPS, 01, 1531-1538
    (2001)

[3] John Schulman, Sergey Levine, Philipp Moritz, Michael I. Jordan
    and Pieter Abbeel, Trust Region Policy Optimization, CoRR, 1502
    (2015)

[4] Daan Wierstra, Tom Schaul, Tobias Glasmachers, Yi Sun, Jan Peters
    and Jürgen Schmidhuber, Natural Evolution Strategies, Journal of
    Machine Learning Research, 15, 949-980 (2014)
