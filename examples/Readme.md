# Examples
Contains modules for various examples applying the NPG and NES to different
platforms. Each example is set to load an existing policy and run a benchmark
test as well as render a single simulation for visual evaluation. \
However, not all policies managed to converge.

## Structure
```sh
examples/
    __init__.py
    nes_ball_balancer.py
    nes_cartpolesswingup.py
    nes_cartpolesswingup_rr.py
    nes_doublependulum.py
    nes_pendulum.py
    nes_qube.py
    nes_qube_rr.py
    npg_ball_balancer.py
    npg_cartpolesswingup.py
    npg_doublependulum.py
    npg_pendulum.py
    npg_qube.py
    npg_qube_rr.py
    Readme.md
    trained_data/
        ...
```