# This Branch Is Part Of HP4 Development. To Be Used Together With RepRapFirmware.

HP4 Prototype/RepRapFirmware uses `M569.3` and `M569.4` to collect auto calibration data in the form of motor/encoder positions in degrees. HP3 worked differently. This branch of this repo auto calibrates based on encoder position data. That is why this branch of this repo only works with HP4, and not HP3.

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
SLSQP
Hit Ctrl+C and wait a bit to stop solver and get current best solution.
number of samples: 26
input xyz coords:  15
total cost:        6.955602e+00
cost per sample:   2.675232e-01

M669 A15.81:-1591.97:-125.62 B1294.58:1231.78:-169.71 C-1397.45:727.84:-147.24 D23.90:1.86:2354.88
M666 Q0.050000 R75.849:75.831:75.428:75.142

```
Note that these values are only test data and does not correspond to your Hangprinter setup (yet).

## How to Collect Data Points?

There's a script called `get_auto_calibration_data_automatically.sh` in the hp-mark repo that semi-automates the data collection.
See https://gitlab.com/tobben/hp-mark/-/tree/master/use.

I use the following flags and settings:
```
DATA_SERIES_NAME="anchor-calibration-0" ./get_auto_calibration_data_automatically.sh --try-hard --bed-reference --reprojection-error-limit 0.5 --show result
```

When using hp-mark, we get measured xyz-positions in addition to motor positions for each data sample.
This guides the `simulation.py` script when searching for better calibration values.

### What if I can't run hp-mark?

Then `simulation.py` can still be used without xyz-position data, but it's harder to do.

Data collection depends on motor encoders (ODrives).
As of July 28, 2021, this is the procedure:
 - Go into torque mode on all motors: `M569.4 P40.0:41.0:42.0:43.0 T0.001`.
   Adjust torque magnitude as fits your particular machine.
 - Drag mover to the origin and zero the counters: `G92 X0 Y0 Z0`
 - Mark reference point for all encoders: `M569.3 P40.0:41.0:42.0:43.0 S`
 - Repeat 15 - ca 20 times:
   - Drag mover to position of data point collection.
   - Collect data point: `M569.3 P40.0:41.0:42.0:43.0`

## Line Length Data
The program wants to find line lengths that match your physical setup.
Hand measure your four line lengths when your nozzle is at the origin,
and input the four (space separated) values through the `-l` or `--line_lengths` argument,
or you can edit the `line_lengths_origin` values in the source file directly.


## How to Insert Data Points In The Source File Directly?
Open `simulation.py` and modify the main function, near the bottom of the file.
Replace `??` with data points collected with your Hangprinter.
```python
...
# Replace this with your collected data
motor_pos_samp = np.array(
    [
        [??, ??, ??, ??],
        [??, ??, ??, ??]
    ])
xyz_of_samp = np.array(
    [
        [??, ??, ??],
        [??, ??, ??]
    ])
line_lengths_origin = np.array([??, ??, ??, ??])
...
```

The first `motor_pos_samp` quadruplet corresponds to the first `xyz_of_samp` triplet and so on.
It's ok to have more `motor_pos_samp` quadruplets than there are `xyz_of_samp` triplets.

When values are inserted, you can run with no `-x`/`-s`/`-l` flags
```bash
python ./simulation.py
```

## Output Explanation
The first two lines describe which numerical optimizer is beeing used, and how you can stop it should you want to
```
SLSQP
Hit Ctrl+C and wait a bit to stop solver and get current best solution.
```

The rest for the first block is printed after the optimization finished.
It describes the quality of the parameters that were found
```
number of samples: 26
input xyz coords:  15
total cost:        6.955602e+00
cost per sample:   2.675232e-01
```
It's recommended that the sum of `number of samples + (input xyz coords)/3` should be above 12.
Using fewer samples makes it probable that the solver finds bogus anchor positions that still minimizes cost.
The program will print a warning if you have too few data points.

Ideal data points collected on an ideal machine would give `total cost: 0.000000` for any sample size above 10.
In real life this does not happen.
The `cost per sample` value let you compare results from your different data sets of unequal size.
In my experience, a `cost per sample` below 1.0, in combination with a large number of samples (>20) and
a >12 number of input xyz coords, generally means you've found yourself a very good config.

The second block contains the anchor positions and spool radii that the script found to work best.
They are formatted so they can be pasted directly into RepRapFirmware's configuration.
```
M669 A15.81:-1591.97:-125.62 B1294.58:1231.78:-169.71 C-1397.45:727.84:-147.24 D23.90:1.86:2354.88
M666 Q0.050000 R75.849:75.831:75.428:75.142
```


## Alternative Optimization Algorithms
I've never met a situation that required alternative optimization algorithms, but they exist.

The script accepts a `-m` or `--method` argument.
Try for example (warning, will print a lot of debug output):
```bash
python ./simulation.py --method L-BFGS-B --debug
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
