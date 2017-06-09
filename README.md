# Auto Calibration Simulation for Hangprinter
Hangprinter is a parallel line driven RepRap with four translational degrees of freedom.
It's repo is [here](https://github.com/tobbelobb/hangprinter).

The Hangprinter Project has a goal of auto-calibration.
That requires locating anchor points by sampling relative line lengths with tight lines at unknown positions, starting from an unknown position.
This code tries to optimize the anchor positions to fit the samples of relative line lengths.
It uses [mystic](https://github.com/uqfoundation/mystic) for the non-convex optimization algorithms.
Relies heavily on [numpy](https://github.com/numpy/numpy).
