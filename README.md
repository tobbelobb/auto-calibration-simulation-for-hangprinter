# This Branch Is Part Of HP4 Development. To Be Used Together With RepRapFirmware.

HP4 Prototype/RepRapFirmware uses `M114 S2` to collect auto calibration data in the form of motor/encoder positions in degrees. HP3 worked differently. This branch of this repo auto calibrates based on encoder position data. That is why this branch of this repo only works with HP4, and not HP3.

# Auto Calibration Simulation for Hangprinter v4

Hangprinter is a parallel line driven RepRap 3D Printer.
It's repo is [here](https://gitlab.com/tobben/hangprinter).

The Hangprinter Project has a goal of auto-calibration.
That requires locating anchor points by sampling relative line lengths with tight lines at unknown positions.
This code tries to optimize the anchor positions to fit the samples of relative line lengths.

Note that this code assumes that the B-anchor has a positive x-coordinate.
If your machine's B-anchor is negative then your output anchors Bx and Cx will have the wrong signs.

## Dependencies
Relies heavily on [numpy](https://github.com/numpy/numpy).
It uses either [scipy](https://scipy.org/) or [mystic](https://github.com/uqfoundation/mystic) for the optimization algorithms.

Here's the commands I used to get Python2.7 libs up and running on my machine running Ubuntu 14.04:
```bash
sudo apt-get install build-essential python-dev python-pip python-tk python-scipy python-numpy
```

To install mystic, try
```bash
sudo pip install mystic
```

Once dependencies and data are in place, take the time to do
```bash
python ./simulation.py --help
```

The default optimization runs with
```bash
python ./simulation.py
```

Its output is quite noisy.
If it works (should finish in a few seconds/minutes), then the bottom part of your output looks similar to
```
number of samples: 15
input xyz coords:  42
total cost:        1.811659e+02
cost per sample:   1.207772e+01

M669 A0.0:-1604.54:-114.08 B1312.51:1270.88:-162.19 C-1440.27:741.63:-161.23 D2345.00
M666 Q0.035620 R65.239:65.135:65.296:64.673
Spool buildup factor: 0.03561954933157288
Spool radii: [65.2393924  65.13533978 65.29587164 64.67297058]
```
Note that these values are only test data and does not correspond to your Hangprinter setup (yet).

## How to Collect Data Points?
Data collection depends on motor encoders (Mechaduinos, Smart Steppers, or ODrives).
As of Mar 15, 2021, this is the procedure:
 - Go into torque mode on all motors: `G95 A15 B15 C15 D15`.
   Adjust torque magnitude as fits your particular machine.
 - Drag mover to the origin and zero the counters: `G92 X0 Y0 Z0`
 - Mark reference point for all encoders: `G96`
 - Repeat 13 - ca 20 times:
   - Drag mover to position of data point collection.
   - Collect data point: `M114 S2`

Mar 15, 2021 note about `M114 S2`: Stock ODrive Firmware has switched from returning encoder counts to returning radians. The RepRapFirmware binary in the hangprinter
repo has not been updated to match this change yet. Check it's date stamp right [here](https://gitlab.com/tobben/hangprinter/-/tree/version_4_dev/firmware/RepRapFirmware)
before you use it. If it's later than March 15, 2021, then you're good. Otherwise, `M114 S2` won't work as expected with recent versions of ODrive Firmware.

## How to Insert Data Points?
Before you run the simulation, open `simulation.py` and modify the main function, near the bottom of the file.
Replace `??` with data points collected with your Hangprinter.
```python
    ...
    # Replace this with your collected data
    samp = np.array([
[??, ??, ??, ??],
[??, ??, ??, ??]
        ])
    ...
```
When values are inserted, run again with
```bash
python ./simulation.py
```

## Output Explanation
The first block give some stats trying to describe the quality of the parameters that were found
```
number of samples: 15
input xyz coords:  42
total cost:        1.811659e+02
cost per sample:   1.207772e+01
```
It's recommended that the sum of `number of samples + (input xyz coords)/3` should be above 12.
Using fewer samples makes it probable that the solver finds bogus anchor positions that still minimizes cost.

Ideal data points collected on an ideal machine would give `total cost: 0.000000` for any sample size above 10.
In real life this does not happen.
The `cost per sample` value let you compare results from your different data sets of unequal size.

The second block contains the anchor positions that the script found.
They are formatted so they can be pasted directly into RepRapFirmware's configuration.
```
M669 A0.0:-1604.54:-114.08 B1312.51:1270.88:-162.19 C-1440.27:741.63:-161.23 D2345.00
M666 Q0.035620 R65.239:65.135:65.296:64.673
```

## Alternative Optimization Algorithms
The script accepts a `-m` or `--method` argument.
Try for example
```bash
python ./simulation.py --method L-BFGS-B -d
```
... the default [`SLSQP`](https://en.wikipedia.org/wiki/Sequential_quadratic_programming), and the [`L-BFGS-B`](https://en.wikipedia.org/wiki/Limited-memory_BFGS) methods
have slightly different characteristics.

If you want to use the `PowellDirectionalSolver`, you also need Mystic:
```
git clone https://github.com/uqfoundation/mystic.git
cd mystic
python setup.py build
sudo python setup.py install
```

For more on usage, try
```bash
python ./simulation.py --help
```
