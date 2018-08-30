# Auto Calibration Simulation for Hangprinter
Hangprinter is a parallel line driven RepRap 3D Printer.
It's repo is [here](https://github.com/tobbelobb/hangprinter).

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

Once dependencies and data are in place, the simulation runs with
```bash
python ./simulation.py
```

If it works (should finish in a few seconds/minutes), then your output looks similar to
```
samples:         11
total cost:      0.254898
cost per sample: 0.023173

Warning: Sample count below 13 detected.
         Do not trust the below values.
         Collect more samples.

#define ANCHOR_A_Y -1164
#define ANCHOR_A_Z  -144
#define ANCHOR_B_X   999
#define ANCHOR_B_Y   585
#define ANCHOR_B_Z  -115
#define ANCHOR_C_X  -977
#define ANCHOR_C_Y   519
#define ANCHOR_C_Z  -106
#define ANCHOR_D_Z  2875

M665 W-1164.26 E-143.50 R998.88 T585.38 Y-115.04 U-977.06 I518.88 O-105.57 P2874.93
```
Note that these values are only test data and does not correspond to your Hangprinter setup (yet).

## How to Collect Data Points?
Data collection depends on Mechaduinos and well calibrated line buildup compensation.
As of Jan 31, 2018, this is the procedure:
 - Go into torque mode on all motors: `G95 A35 B35 C35 D35`.
   Adjust torque magnitude as you prefer.
 - Drag mover to the origin and zero counters: `G92 X0 Y0 Z0`
 - Mark reference point for all encoders: `G96 A B C D` (Stock Marlin accepts `G96` as a short hand for `G96 A B C D`)
 - Repeat 13 - ca 20 times:
   - Drag mover to position of data point collection.
   - Collect data point: `M114 S1` (Old firmwares, before Feb 6, 2018 used: `G97 A B C D`)

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
samples:         11
total cost:      0.254896
cost per sample: 0.023172
```
It's recommended to use 13 samples or more.
Using fewer samples makes it probable that the solver finds bogus anchor positions that still minimizes cost.

Ideal data points collected on an ideal machine would give `total cost: 0.000000` for any sample size above 10.
In real life this does not happen.
The `cost per sample` value let you compare results from your different data sets of unequal size.

The second block contains the anchor positions that the script found.
They are formatted so they can be pasted directly into Marlin's `Configuration.h`.
```
#define ANCHOR_A_Y -1164
#define ANCHOR_A_Z  -144
#define ANCHOR_B_X   999
#define ANCHOR_B_Y   585
#define ANCHOR_B_Z  -115
#define ANCHOR_C_X  -977
#define ANCHOR_C_Y   519
#define ANCHOR_C_Z  -106
#define ANCHOR_D_Z  2875
```

The gcode line that is the third block can be used to set anchor calibration values on a running Hangprinter without re-uploading firmware.
```
M665 W-1164.31 E-143.53 R998.78 T585.33 Y-114.98 U-977.11 I518.88 O-105.60 P2874.87
```
If you have `EEPROM_SETTINGS` enabled you can save these values with `M500`.
If you don't save them they will be forgotten when you power down your machine.


## Debug
The script accepts a `-d` or `--debug` flag.
It calculates the difference between the output anchor positions and your manually measured ones:
```
Err_A_Y:   -52.307
Err_A_Z:   -28.532
Err_B_X:    28.781
Err_B_Y:    35.332
Err_B_Z:     0.022
Err_C_X:    -7.113
Err_C_Y:   -31.124
Err_C_Z:     9.399
Err_D_Z:     9.869
Method: L-BFGS-B
RUN TIME : 1.43998289108
```

For the `Err_ABCD_XYZ`-values to be meaningful you must have inserted your manually measured values into the `anchors` array in the script:
```python
    # Rough approximations from manual measuring.
    # Does not affect optimization result. Only used for manual sanity check.
    anchors = np.array([[  0.0,   ay?,   az?],
                        [  bx?,   by?,   bz?],
                        [  cx?,   cy?,   cz?],
                        [  0.0,   0.0,   dz?]])
```
The debug check is only relevant if you suspect that the script outputs bogus values.
Error larger than ca 100 mm is generally a sign that something's up.

## Alternative Optimization Algorithms
The script accepts a `-m` or `--method` argument.
Try for example
```bash
python ./simulation.py --method SLSQP -d
```
... for the [`SLSQP`](https://en.wikipedia.org/wiki/Sequential_quadratic_programming) method that is faster, but requires more data points than the default [`L-BFGS-B`](https://en.wikipedia.org/wiki/Limited-memory_BFGS) method.

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
