from setuptools import setup

setup(name='nes-npg',
      version='0.0.1',
      description='NES and NPG',
      author='Cedric Derstroff, Kay Hansel, Janosch Moos',
      author_email='group14%reinforcement.learning@gmx.de',
      packages=['utilities', 'models'],
      py_modules=['nes', 'agent', 'npg'],
      zip_safe=False,
      install_requires=['quanser-robots'])
