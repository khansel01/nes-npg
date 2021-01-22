# Benchmarking the Natural Gradient in Policy Gradient Methods and Evolution Strategies
This repository is the official implementation of [Benchmarking the Natural Gradient in Policy Gradient Methods and Evolution Strategies](https://doi.org/10.1007/978-3-030-41188-6_7).

# Reinforcement Learning
This project implements Natural Policy Gradients and Natural Evolution
Strategies algorithm for gym environments as well as quanser robot
environments. Various examples are provided applying these algorithms to
different platforms, however, not all of them have been solved yet. Using the
main module you can apply these algorithms to different platforms and adjust
the parameters to try to solve the platforms.

## Getting Started
### Prerequisites
This project is compatible with Python 3.6 and
[conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/ "Install miniconda"). Further the
[quanser_robots package](https://git.ias.informatik.tu-darmstadt.de/quanser/clients/tree/master "Install quanser package")
is required.

##### Install Python 3.6:
```bash
$ sudo add-apt-repository ppa:deadsnakes/ppa
$ sudo apt-get update
$ sudo apt-get install python3.6
```

### Install
##### Install virtual environment:
You can also change the environment name if necessary.
```bash
$ conda env create -f nes-npg.yml
```

##### Install the quanser_robots package to environment
Navigate to quanser-robots package *.../clients/*
```bash
$ source activate nes-npg
$ pip install -e .
```
or
```bash
$ conda activate nes-npg
$ pip install -e .
```
##### Install this project to environment
Navigate to this package *.../reinf/* and call
```bash
$ pip install -e .
```

## Usage
To run any of the given examples please simply execute the respective file.
The examples are set to run a benchmark test on an already trained policy.
Alternatively, adjust the [main.py](./main.py) to run the code on any platforms
with continuous action spaces. The algorithms are not implemented for discrete
action spaces.
Please be aware that the hardware can have a huge impact on the convergence due
to slight mathematical errors between different hardware setups.

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
    nes-npg.yml
    npg.py
    Readme.md
    setup.py
    models/
        __init__.py
        baseline.py
        nn_policy.py
        Readme.md
    utilities/
        __init__.py
        conjugate_gradient.py
        environment.py
        estimations.py
        logger.py
        normalizer.py
        Readme.md
    examples/
        ...
        trained_data/
            ...
```

## TODO
The NES CartpoleSwingUp-v0 example is not optimally retrained for the new 
state space introduced recently in the Quanser package. The same goes for 
the real robot version. Both were initially trained on 0.9x. The new 
version works with 0.814x. We started retraining on the simulation but have 
not finished in time. The policy for the real system has not been retrained 
at all.


## Developers
- Janosch Moos
- Kay Hansel
- Cedric Derstroff

## Cite as
```bibtex
@Inbook{Hansel2021,
    author="Hansel, Kay and Moos, Janosch and Derstroff, Cedric",
    editor="Belousov, Boris and Abdulsamad, Hany and Klink, Pascal and Parisi, Simone and Peters, Jan",
    title="Benchmarking the Natural Gradient in Policy Gradient Methods and Evolution Strategies",
    bookTitle="Reinforcement Learning Algorithms: Analysis and Applications",
    year="2021",
    publisher="Springer International Publishing",
    pages="69--84",
    isbn="978-3-030-41188-6",
    doi="10.1007/978-3-030-41188-6_7",
    url="https://doi.org/10.1007/978-3-030-41188-6_7"
}
```

## Bibliography
[1] Rajeswaran, A., Lowrey, K., Todorov, E., Kakade, S.: Towards
    Generalization and Simplicity  in  Continuous  Control  (Nips)
    (2017). DOI  10.1016/j.acra.2014.04.006.
    URL http://arxiv.org/abs/1703.02660

[2] Kakade, S.: A Natural Policy Gradient. In: NIPS’01 Proceedings
    of the 14th International Conference on Neural Information
    Processing Systems: Natural and Synthetic, pp. 1531–1538 (2001)

[3] Schulman, J., Levine, S., Moritz, P., Jordan, M.I.,
    Abbeel, P.: Trust Region Policy Optimization.
    CoRR abs/1502.0 (2015). URL http://arxiv.org/abs/1502.0547711.

[4] Wierstra, D., Schaul, T., Glasmachers, T., Sun, Y., Peters, J.,
    Schmidhuber, J.: Natural Evolution Strategies.  Journal of
    Machine Learning Research 15, 949–980 (2014).
    URL http://jmlr.org/papers/v15/wierstra14a.html

