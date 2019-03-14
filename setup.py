from setuptools import setup

setup(name='nes-npg',
      version='0.0.1',
      description='NES and NPG implementation for the reinforcement learning'
                  'lecture at TU Darmstadt',
      author='Cedric Derstroff, Kay Hansel, Janosch Moos',
      author_email='group14%reinforcement.learning@gmx.de',
      packages=['utilities', 'models', 'examples'],
      py_modules=['nes', 'agent', 'npg'],
      zip_safe=False,
      install_requires=['quanser_robots', 'gym>=0.10.5', 'numpy', 'torch',
                        'matplotlib'])
