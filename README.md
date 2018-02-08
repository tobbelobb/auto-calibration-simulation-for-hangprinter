# Auto Calibration Simulation for Hangprinter
Hangprinter is a parallel line driven RepRap with four translational degrees of freedom.
It's repo is [here](https://github.com/tobbelobb/hangprinter).

The Hangprinter Project has a goal of auto-calibration.
That requires locating anchor points by sampling relative line lengths with tight lines at unknown positions, starting from an unknown position.
This code tries to optimize the anchor positions to fit the samples of relative line lengths.

Note that this code assumes that the B-anchor has a positive x-coordinate.
If your machine's B-anchor is negative then your output anchors Bx and Cx will have the wrong signs.

## Dependencies
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

Once dependencies and data are in place, the simulation runs with
```bash
python ./simulation.py
```

If it works, then your output looks similar to
```
samples:         11
total cost:      0.254896
cost per sample: 0.023172

#define ANCHOR_A_Y -1164
#define ANCHOR_A_Z  -144
#define ANCHOR_B_X   999
#define ANCHOR_B_Y   585
#define ANCHOR_B_Z  -115
#define ANCHOR_C_X  -977
#define ANCHOR_C_Y   519
#define ANCHOR_C_Z  -106
#define ANCHOR_D_Z  2875

M665 W-1164.31 E-143.53 R998.78 T585.33 Y-114.98 U-977.11 I518.88 O-105.60 P2874.87
```
Note that these values are only test data and does not correspond to your Hangprinter setup (yet).

## How to Collect Data Points?
Data collection depends on Mechaduinos and well calibrated line buildup compensation.
As of Jan 31, 2018, this is the procedure:
 - Go into torque mode on all motors: `G95 A35 B35 C35 D35`.
   Adjust torque magnitude as you prefer.
 - Drag mover to the origin and zero counters: `G92 X0 Y0 Z0`
 - Mark reference point for all encoders: `G96 A B C D` (Stock Marlin accepts `G96` as a short hand for `G96 A B C D`)
 - Repeat 12 - ca 20 times:
   - Drag mover to position of data point collection.
   - Collect data point: `M114 S1` (Old firmwares, before Feb 6, 2018 used: `G97 A B C D`)

## How to Insert Data Points?
Before you run the simulation, open `simulation.py` and modify the main function.
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
The first block give some stats trying to describe the quality of the output parameters
```
samples:         11
total cost:      0.254896
cost per sample: 0.023172
```
It's recommended to use 12 samples or more.
Using fewer samples makes it probable that the solver finds bogus anchor positions that still minimizes cost.

Ideal data points collected on an ideal machine would give `total cost: 0.000000` for any sample size.
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
If you don't save them they will be reset when you restart your machine.


## Debug
The script accepts a `-d` or `--debug` flag.
It calculates the difference between your manually measured anchor positions, and the anchor positions calculated by the script:
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
```

For these values to be meaningful you must have inserted your manually measured values into the `anchors` array in the script:
```python
    # Rough approximations from manual measuring.
    # Does not affect optimization result. Only used for manual sanity check.
    anchors = np.array([[   0.0,     ay?,     az?],
                        [   bx?,     by?,     bz?],
                        [   cx?,     cy?,     cz?],
                        [   0.0,     0.0,     dz?]])
```
