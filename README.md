# This Branch Is Part Of HP5 Development. To Be Used Together With RepRapFirmware.

HP5 uses four low anchors and one high anchor. The code in this branch will not work for any other anchor configuration.

# Auto Calibration Simulation for Hangprinter v5

Hangprinter is a parallel line driven RepRap 3D Printer.
It's repo is [here](https://gitlab.com/tobben/hangprinter).

The Hangprinter Project has a goal of auto-calibration.
That requires locating anchor points by sampling relative line lengths with straight lines at unknown positions.
This code tries to optimize the anchor positions to fit the samples of relative line lengths.

Note that this code assumes:
 - A-anchor has a near-zero x-coordinate and negative y-coordinate
 - B-anchor has a positive x-coordinate and near-zero y-coordinate
 - C-anchor has a near-zero x-coordinate and positive y-coordinate
 - D-anchor has a negative x-coordinate and near-zero y-coordinate
 - I-anchor has a near-zero x-coordinate and near-zero y-coordinate


# Get started
Here are the approximate names of packages you need if you're on Ubuntu:
```bash
sudo apt-get install build-essential python-dev python-pip python-tk
```

To avoid cluttering your python environment, it is recommended to always work in a python virtual environment. Installation is only done once, and activation is done everytime you start a new shell. the venv is installed in auto-calibration-simulation-for-hangprinter/.venv

Install venv on MacOS / Linux

    python3 -m venv .venv
Install venv on Windows

    python -m venv .venv

Activate venv on MacOS / Linux

    source .venv/bin/activate
Activate venv on Windows

    source .venv/Scripts/activate
Install packages in the requirement list

    pip install -r requirements.txt

Verify the installed packages

    pip list

To exit the python environment

    deactivate
To completely remove the venv (deactivate it first)

    rm -r .venv

Once dependencies and data are in place, take the time to do
```bash
./simulation.py --help
```

The default optimization runs with
```bash
./simulation.py
```

If it works it should finish in a few seconds.
The output should look similar to
```
Assuming zero flex
Not forcing hand measured line lengths
Number of samples: 27 (is above 26? True)
Input xyz coords:  81 (is above 18? True)
Cost per sample:   2.527060e-12 (is below 10.0? True)
Line length error: 1.764959e-05 (is below 50.0? True)
All quality conditions met? (True)

M669 A-0.00:-2000.00:-120.00 B2000.00:0.00:-120.00 C-0.00:2000.00:-120.00 D-2000.00:0.00:-120.00 I0.00:-0.00:2000.00
M666 R75.000:75.000:75.000:75.000:75.000
;Here follows constants that are set in the script
M666 Q0.068750 W2.00 S20000.00 U2:2:2:2:4 O1:1:1:1:1 L20:20:20:20:20 H255:255:255:255:255
```
Note that these values are calculated from test data.
They don't correspond to your Hangprinter setup (yet).

## How to Collect Data Points?

There's a script called `get_auto_calibration_data_automatically.sh` in the hp-mark repo that semi-automates the data collection.
See https://gitlab.com/tobben/hp-mark/-/tree/master/use.

I use the following flags and settings:
```
DATA_SERIES_NAME="anchor-calibration-0" ./get_auto_calibration_data_automatically.sh --try-hard --bed-reference --reprojection-error-limit 0.5 --show result
```

When using hp-mark, we get measured xyz-positions in addition to motor positions for each data sample.
This guides the `simulation.py` script when searching for better calibration values.

However, it's recommended to only use 9 known xyz-positions.
First take 8 samples at z=0 (on your print bed), and only trust hp-marks x- and y-values.
Set the z-values to 0 manually.

Then, take one sample high up, like above z=1000.0 mm.
For this value trust all of hp-marks xyz-values, or use a measurement tape to double-check the z-value.

For all the ~40 following data collection points, it's recommended to discard your hp-mark measurements, and only use the encoder readings (`motor_pos_samp`).
This is because hp-mark at the time of writing only uses a single camera, and it's difficult to make its measurements accurate enough to be useful for simulation.py,
especially along the z-direction.

For the ~40 last data collection points, make them as random and spread out as you can. Go close to the anchors and far away from them, and in all directions.

The `get_auto_calibration_data_automatically.sh` script will spit out your calibration data for you in the end, in the right order and with the right names to paste
directly into `simulation.py`. Remember to delete the ~40 last `xyz_of_samp` though (unless you reeeeally trust your hp-mark setup a lot).

### What if I can't run hp-mark?

Then `simulation.py` can still be used without `get_auto_calibration_data_automatically.sh`, but it takes a bit more of manual labour to collect the data.

Data collection depends on motor encoders (ODrives).
As of July 28, 2021, this is the procedure:
 - Go into torque mode on all motors: `M569.4 P40.0:41.0:42.0:43.0 T0.001`.
   Adjust torque magnitude as fits your particular machine.
 - Drag mover to the origin and zero the counters: `G92 X0 Y0 Z0`
 - Mark reference point for all encoders: `M569.3 P40.0:41.0:42.0:43.0 S`
 - Repeat ca 40 times:
   - Drag mover to position of data point collection.
   - Collect data point: `M569.3 P40.0:41.0:42.0:43.0`

Note that even if you don't run hp-mark, you still need some known xyz-positions for the algorithm to work.
You can hand measure 8 xy-positions on your print bed (z=0), and then collect data points with known xyz-positions there.

You also need one data point collected at a known high-up position.
Use for example an aiming plumb to confirm that your D-lines are vertical and your nozzle is directly above the origin to set x=y=0, and hand-measure the z-value or
something.

You will need to type your known positions into `data.py` by hand, in the right order.

Your hand measurements of xy-positions need a pair of XY-axes to measure along.
Draw a line from your origin point through the middle of your A-anchor.
(No need for perfection, but roughly.)
That's the negative direction of your y-axis.

Your B-anchor is on the positive x-side.
Your C-anchor is on the positive y-side.
Your D-anchor is on the negative x-side.

Your I-anchor should be roughly vertically above your origin.
(Again, no need for perfection, but roughly.)

## Line Length Data
The program wants to find line lengths that match your physical setup.
Hand measure your four line lengths when your nozzle is at the origin,
and edit the `line_lengths_when_at_origin` values in `data.py`.

The hand measured line lengths will be used to verify that `simulation.py` found a good set of
config values or not.

You can also force `simulation.py` to find config values that match perfectly with your hand-measurements,
but this is an advanced feature since it sacrefices the ability to double-check the resulting config.


## How to Insert Data Points In The Source File Directly?
Open `data.py` and modify the main function, near the top of the file.
Replace `??` with data points collected with your Hangprinter.
```python
...
# Replace this with your collected data

line_lengths_when_at_origin = np.array([??, ??, ??, ??, ??])

motor_pos_samp = np.array(
    [
        [??, ??, ??, ??, ??],
        [??, ??, ??, ??, ??]
    ])
xyz_of_samp = np.array(
    [
        [??, ??, ??],
        [??, ??, ??]
    ])
...
```

The first `motor_pos_samp` row corresponds to the first `xyz_of_samp` row and so on.
After inserting your data you run the simulation as usual with:

```bash
./simulation.py
```

## Output Explanation
The first two lines describe mode you're running in. Example:
```
Assuming zero flex
Not forcing hand measured line lengths
```

The rest for the first block is printed after the optimization finished.
It describes the quality of the parameters that were found. Example:
```
Number of samples: 27 (is above 26? True)
Input xyz coords:  81 (is above 18? True)
Cost per sample:   2.527060e-12 (is below 10.0? True)
Line length error: 1.764959e-05 (is below 50.0? True)
All quality conditions met? (True)
```
Ideal data points collected on an ideal machine would give `Cost per sample: 0.000000` and `Line length error: 0.000000` for any sample size above 10.
In real life this does not happen.
If all quality conditions are met it generally means you've found yourself a good config.

The higher the quality of the data set, the lower the cost and error, and the better the config values.

The second block contains the config values that go into your Hangprinter's `config.g` file. Example:
```
M669 A-0.00:-2000.00:-120.00 B2000.00:0.00:-120.00 C-0.00:2000.00:-120.00 D-2000.00:0.00:-120.00 I0.00:-0.00:2000.00
M666 R75.000:75.000:75.000:75.000:75.000
;Here follows constants that are set in the script
M666 Q0.068750 W2.00 S20000.00 U2:2:2:2:4 O1:1:1:1:1 L20:20:20:20:20 H255:255:255:255:255
```

The first line contains the anchor positions that the script found to work best.
These values are the most important ones to get right.

Then follows the spool radii (`M666 R...`), including any line buildup your spools might carry when your nozzle is at the origin.

And lastly comes a line with configuration constants that were set in the `simulation.py` script.
They describe the spool buildup factor, the mover weight, the line stiffness, the mechanical advantage, lines per spool, motor gear teeth and spool gear teeth.

All verified and ready to be copy/pasted into `config.g`.

## Alternative Cost Functions
The script accepts the argument `-a` or `--advanced`.
If given, the script will try to account for
 - line flex,
 - line lengths at the origin,
 - in the future possibly also line forces,

... all found in your `data.py`.

This will cause the script to run slower and sometimes print a few runtime warnings along the way.

It will fail more often than the standard cost function, but it also has the potential to find better config values.

For example a data set that gives ~23 mm line length error with the standard method might get a ~21 mm line length error with the advanced method.

For more on usage, try
```bash
python ./simulation.py --help
```

# Set up git pre-commit hook
Make sure you have activated the venv, and everything on `requirement.txt` is installed
Standing at the repo root level, run:

    pre-commit install

Now pre-commit will run automatically on git commit.

If you for some reason don't want to run pre-commit during a commit, add `--no-verify` like:

    git commit --no-verify

# Open Project in VSCode
Click the file `auto-calibration-simulation-for-hangprinter.code-workspace`. Or from VSCode, click File -> `Open Workspace from file...`. The settings in this file harmonizes with the pre-commit hook.

# How to use black
black will auto-format python code to a specific style, which makes the code more consistent.

    black --line-length 120 <file or directory>
# How to use python profiler
Make sure you have activated the venv, and everything on `requirement.txt` is installed

## Use cProfile to generate a profile of the python script of interest

    python3 -m cProfile -o <dir/to/the/output/profile> <my_python_script.py>
For example:

    python3 -m cProfile -o ./tmp.prof ./python_simulator/main.py
## Use snakeviz to visualize the generated profile on browser

    snakeviz tmp.prof
