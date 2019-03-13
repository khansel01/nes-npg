# Reinforcement Learning
This project implements Natural Policy Gradients and Natural Evolution
Strategies algorithm for gym environments as well as quanser robot
environments. Various examples are provided applying these algorithms to
different platforms, however, not all of them are solved yet. Using the main
module you can apply these algorithms to different platforms and adjust the
parameters to try to solve the platforms.

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
$ conda env create -f nes-npg.yaml
```

##### Install quanser pkg to environment
Navigate to quanser package *.../clients/*
```bash
$ source activate nes-npg
$ pip install -e .
```

##### Install this project to environment
Navigate to this package *.../reinf/* and call
```bash
$ pip install -e .
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
[1] Rajeswaran, A., Lowrey, K., Todorov, E., Kakade, S.: Towards
    Generalization and Sim-plicity  in  Continuous  Control  (Nips)
    (2017). DOI  10.1016/j.acra.2014.04.006.
    URL http://arxiv.org/abs/1703.02660

[2] Kakade, S.: A Natural Policy Gradient. In: NIPS’01 Proceedings
    of the 14th InternationalConference on Neural Information
    Processing Systems: Natural and Synthetic, pp. 1531–1538 (2001)

[3] Schulman, J., Levine, S., Moritz, P., Jordan, M.I.,
    Abbeel, P.: Trust Region Policy Optimization.
    CoRR abs/1502.0(2015). URL http://arxiv.org/abs/1502.0547711.

[4] Wierstra, D., Schaul, T., Glasmachers, T., Sun, Y., Peters, J.,
    Schmidhuber, J.: NaturalEvolution Strategies.  Journal of
    Machine Learning Research 15, 949–980 (2014).
    URL http://jmlr.org/papers/v15/wierstra14a.html

