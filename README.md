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
```python
cost: 0.254896
Output Anchors:
[[    0.         -1164.30732212  -143.53179478]
 [  998.78058643   585.33185503  -114.97805252]
 [ -977.11333826   518.87558222  -105.60105899]
 [    0.             0.          2874.86861506]]
Errors:
[[  0.00000000e+00  -5.23073221e+01  -2.85317948e+01]
 [  2.87805864e+01   3.53318550e+01   2.19474805e-02]
 [ -7.11333826e+00  -3.11244178e+01   9.39894101e+00]
 [  0.00000000e+00   0.00000000e+00   9.86861506e+00]]
```
Note that these values are only test data and does not correspond to your Hangprinter setup (yet).

## How to Collect Data Points?
Data collection depends on Mechaduinos and well calibrated line buildup compensation.
As of Jan 31, 2018, this is the procedure (using HangprinterMarlin, not stock Marlin yet).
 - Go into torque mode on all motors: `G95 A35 B35 C35 D35`.
   Adjust torque magnitude as you prefer.
 - Drag mover to the origin and zero counters: `G92 X0 Y0 Z0`
 - Mark reference point for all encoders: `G96 A B C D`
 - Repeat 10 - 20 times:
   - Drag mover to position of data point collection.
   - Collect data point: `G97 A B C D`

## How to Insert Data Points?
Before you run the simulation, open `simulation.py` and modify the main function.
Replace `?` with your approximated values (not mandatory but useful).
Replace `??` with data points collected with your Hangprinter.
At least 10 data points at a mean distance ~1m from the origin are recommended.
```python
if __name__ == "__main__":
    # Rough approximations from manual measuring.
    # Does not affect optimization result. Only used for manual sanity check.
    az = ?
    bz = ?
    cz = ?
    anchors = np.array([[   0.0,       ?,     az],
                        [     ?,       ?,     bz],
                        [     ?,       ?,     cz],
                        [   0.0,     0.0,      ?]])

    # Replace this with your collected data
    samp = np.array([
[??, ??, ??, ??],
[??, ??, ??, ??]
        ])
```
When values are inserted, run again with
```bash
python ./simulation.py
```

## Output Explanation
```python
cost: 0.254896
Output Anchors:
[[      0.       ANCHOR_A_Y  ANCHOR_A_Z]
 [  ANCHOR_B_X   ANCHOR_B_Y  ANCHOR_B_Z]
 [  ANCHOR_C_X   ANCHOR_C_Y  ANCHOR_C_Z]
 [      0.           0.      ANCHOR_D_Z]]
Errors:
[[     0.    err_A_y   err_A_z]
 [  err_B_x  err_B_y   err_B_z]
 [  err_C_x  err_C_y   err_C_z]
 [     0.       0.     err_D_z]]
```

Ideal data points collected on an ideal machine would give `cost: 0.0`.
In real life this does not happen.
The cost value is there to let you compare your different data sets of equal size.
The `Output Anchors`-set associated with the lowest cost is generally your best one.
The `Errors` is your manual sanity check.
`Errors` are calculated by differentiating with your approximations denoted `?` above.
