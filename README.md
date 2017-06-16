# Auto Calibration Simulation for Hangprinter
Hangprinter is a parallel line driven RepRap with four translational degrees of freedom.
It's repo is [here](https://github.com/tobbelobb/hangprinter).

The Hangprinter Project has a goal of auto-calibration.
That requires locating anchor points by sampling relative line lengths with tight lines at unknown positions, starting from an unknown position.
This code tries to optimize the anchor positions to fit the samples of relative line lengths.
It uses [mystic](https://github.com/uqfoundation/mystic) for the non-convex optimization algorithms.
Relies heavily on [numpy](https://github.com/numpy/numpy).

Here's the commands I used to get Python libs up and running on my machine running Ubuntu 14.04:
```bash
sudo apt-get install build-essential python-dev python-pip python-tk
sudo pip install matplotlib
sudo pip install --upgrade matplotlib
git clone https://github.com/uqfoundation/mystic.git
cd mystic
python setup.py build
sudo python setup.py install
```

Once dependencies are in place, the simulation runs with
```bash
python ./simulation.py
```
