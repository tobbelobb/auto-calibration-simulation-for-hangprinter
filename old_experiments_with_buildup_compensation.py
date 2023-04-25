import numpy as np
from simulation import (
    distance_samples_relative_to_origin,
    pos_to_motor_pos_samples,
    motor_pos_samples_to_distances_relative_to_origin,
)

# Approximate anchor placements in my office
anchors = np.array(
    [
        [0, -1620, -10],
        [1800 * np.cos(np.pi / 5), 1800 * np.sin(np.pi / 5), -10],
        [-1620 * np.cos(np.pi / 6), 1620 * np.sin(np.pi / 6), -10],
        [0, 0, 2350],
    ]
)


def how_big_spool_r_difference_will_static_buildup_compensation_make(line_on_spool, spool_buildup_factor, spool_r):
    """How much will the spool radii increase if we wind an amount of line around it?

    Derivation: Imagine that the spool is made up of wound line all the way to the core.

    Spool volume before adding new line = h*pi*r^2 = v*x0,
    <==> r = sqrt(v*x0/(h*pi))

    where h is spool height, v is volume per mm of line, x0 is the amount of line needed
    to create a wound core of radius r

    Spool volume after adding new line = h*pi*R^2 = v*x1,
    <==> R = sqrt(v*x1/(h*pi))
    so
    R - r = sqrt(v/(h*pi)) * (sqrt(x1) - sqrt(x0))
    """
    line_to_build_up_spool_r = (spool_r ** 2) / spool_buildup_factor
    return np.sqrt(spool_buildup_factor) * (
        np.sqrt(line_on_spool + line_to_build_up_spool_r) - np.sqrt(line_to_build_up_spool_r)
    )


def how_big_difference_will_static_buildup_compensation_make(
    anchors,
    pos,
    spool_buildup_factor,
    mech_adv,
    spool_r,
    line_on_spool_when_at_the_origin,
):
    samp = distance_samples_relative_to_origin(anchors, pos)
    compensated_spool_r = spool_r + how_big_spool_r_difference_will_static_buildup_compensation_make(
        line_on_spool_when_at_the_origin, spool_buildup_factor, spool_r
    )
    buildup_comp = mech_adv * samp * 360 / (2 * np.pi * compensated_spool_r)
    no_buildup_comp = mech_adv * samp * 360 / (2 * np.pi * spool_r)
    # Return the difference in mm actually travelled with these two alternatives
    return np.abs(
        motor_pos_samples_to_distances_relative_to_origin(
            buildup_comp,
            spool_buildup_factor,
            compensated_spool_r,
            1,
            mech_adv,
            np.array([1, 1, 1, 1]),
        )
    ) - np.abs(
        motor_pos_samples_to_distances_relative_to_origin(
            no_buildup_comp,
            spool_buildup_factor,
            compensated_spool_r,
            1,
            mech_adv,
            np.array([1, 1, 1, 1]),
        )
    )


def buildup_compensation(
    anchors,
    pos,
    spool_buildup_factor,
    mech_adv,
    spool_r_before_static_compensation,
    spool_r_after_static_compensation,
):
    samp = distance_samples_relative_to_origin(anchors, pos)
    no_buildup_comp = mech_adv * samp * 360 / (2 * np.pi * spool_r_before_static_compensation)
    # Gear ratio is always 1 since we're only considering spool rotations
    buildup_comp = pos_to_motor_pos_samples(
        anchors,
        pos,
        spool_buildup_factor,
        spool_r_after_static_compensation,
        np.ones(4),
        mech_adv,
    )
    # Return the difference in mm actually travelled with these two alternatives
    return np.abs(
        motor_pos_samples_to_distances_relative_to_origin(
            buildup_comp,
            spool_buildup_factor,
            spool_r_after_static_compensation,
            1,
            mech_adv,
            np.array([1, 1, 1, 1]),
        )
    ) - np.abs(
        motor_pos_samples_to_distances_relative_to_origin(
            no_buildup_comp,
            spool_buildup_factor,
            spool_r_after_static_compensation,
            1,
            mech_adv,
            np.array([1, 1, 1, 1]),
        )
    )


def how_big_difference_will_dynamic_buildup_compensation_make(anchors, pos, spool_buildup_factor, mech_adv, spool_r):
    """For each position in pos, compute how large effect the compensation will have.
    Returned unit is "mm of effector movement towards or away from anchor".
    """
    return buildup_compensation(
        anchors,
        pos,
        spool_buildup_factor,
        mech_adv,
        spool_r,
        spool_r,  # Use same spool_r. No static compensation in this function.
    )


def how_big_difference_will_buildup_compensation_make(
    anchors,
    pos,
    spool_buildup_factor,
    mech_adv,
    spool_r,
    line_on_spool_when_at_the_origin,
):
    """For each position in pos, compute how large effect the compensation will have.
    Returned unit is "mm of effector movement towards or away from anchor".
    """
    return buildup_compensation(
        anchors,
        pos,
        spool_buildup_factor,
        mech_adv,
        spool_r,
        spool_r
        + how_big_spool_r_difference_will_static_buildup_compensation_make(
            line_on_spool_when_at_the_origin, spool_buildup_factor, spool_r
        ),
    )


################################################################################################
######## Dynamic Compensation ##################################################################
################################################################################################

# How much maximum dyamic compensation we'd get on a HP2
# In the most extreme positions
hp2_dyn_comp = how_big_difference_will_dynamic_buildup_compensation_make(
    anchors, anchors, 0.008 / 0.7, np.array([2, 2, 2, 6]), 33
)
# array([[-13.77148104,   8.1767179 ,   7.37944913,   4.1349553 ],
#        [ 10.70476621, -17.00170537,   8.19752565,   6.0140723 ],
#        [  7.37944913,   6.00649088, -13.77148104,   4.1349553 ],
#        [  8.10059161,   7.15931149,   8.10059161, -86.93427784]])
#  A negative number means the motor will rotate less (in any direction) than it would have without compensation.
#  A positive number means the motor will rotate more than it would without compensation.
#  When motor rotates less, the line length changes less
#  The unit of this and all following arrays is mm.
#  Numbers represent effector movement added or subtracted by compensation algorithm.
#
#  Note on choice of args for HP2:
#  The mechanical advantage was really [1, 1, 1, 2], but we had [2, 2, 2, 3] lines per spool.
#  In total, this causes same buildup effect as a [2, 2, 2, 6] mechanical advantage
#  The spool buildup factor was larger because the spool was narrower by a factor of 1/0.7

# How much maximum dyamic compensation we'd get on a HP3
hp3_dyn_comp = how_big_difference_will_dynamic_buildup_compensation_make(
    anchors, anchors, 0.008, np.array([2, 2, 2, 3]), 55
)
# array([[-3.47041322,  2.06053291,  1.85962118,   0.52100437],
#        [ 2.69760108, -4.28442975,  2.06577646,   0.75777311],
#        [ 1.85962118,  1.5136357 , -3.47041322,   0.52100437],
#        [ 2.04134909,  1.80414649,  2.04134909, -10.95371901]])
# On the HP3, wider spools  with larger radius reduced the dynamic spool buildup a lot
# HP3 also removed the mechanical advantage on the D-axis

# How much maximum dyamic compensation we'd get on a HP4
hp4_dyn_comp = how_big_difference_will_dynamic_buildup_compensation_make(
    anchors, anchors, 0.008, np.array([2, 2, 2, 2]), 65
)
# array([[-2.48473373,  1.47529279,  1.33144475,  0.24868453],
#       [ 1.93141853, -3.0675503 ,  1.47904705,  0.36169841],
#       [ 1.33144475,  1.08372734, -2.48473373,  0.24868453],
#       [ 1.46155763,  1.29172619,  1.46155763, -5.22840237]])
# The HP4 further increased the spool radius, and gave each line its own spool
# Increasing the total number of spools from 4 to 9.
# HP4 re-introduced mechanical advantage on the D-axis, and also introduced
# mechanical advantage on the A, B, and C-axes.
# It did so while at the same time cutting the dynamic buildup in ~half.

# Observe that these are upper bounds on errors, in the hypothetical scenario where we move the effector out to the anchor.
# If you move max ca 1/3 of the way to an anchor you get a max dynamic compenstion of only 0.58 mm.
# Most practical print moves will require only ~0.1 mm of dynamic compensation on all axes.

# The amount of dynamic buildup that is left since HP2 is shown by this calculation
fraction_of_dyn_comp_needed_on_hp4_vs_hp2 = hp4_dyn_comp / hp2_dyn_comp
# array([[0.18042604, 0.18042604, 0.18042604, 0.06014201],
#        [0.18042604, 0.18042604, 0.18042604, 0.06014201],
#        [0.18042604, 0.18042604, 0.18042604, 0.06014201],
#        [0.18042604, 0.18042604, 0.18042604, 0.06014201]])
# We see that less than 6 - 18% of the dynamic compensation is needed on HP4 compared to HP2
# In other words: Dynamic spool buildup is reduced by ca 82% on ABC, and 94% on D, since HP2
# Similar computation shows dynamic buildup was reduced by 28% (ABC) and 52% (D) between HP3 and HP4

################################################################################################
######## Static Compensation ###################################################################
################################################################################################

# All those calculations assumed that spool radii at the origin were correct, and compensated for
# the buildup that happened when moving away from the origin.
# Let's instead set the correct spool radii at the origin, and not compensate for spool radii changes at all
hp2_static_comp = how_big_difference_will_static_buildup_compensation_make(
    anchors,
    anchors,
    0.008 / 0.7,
    np.array([2, 2, 2, 6]),
    33,
    np.array([5000, 5000, 5000, 12000]),
)
# array([[ -42.64695469,  -31.92476516,  -30.34781217,  -30.85261284],
#        [ -36.46081797,  -47.47009576,  -31.96484068,  -37.09348638],
#        [ -30.34781217,  -27.4119855 ,  -42.64695469,  -30.85261284],
#        [ -31.77769909,  -29.89717571,  -31.77769909, -153.31013063]])
# There were 2*2500 mm origin-idle lines on ABC. Two lines, no mech adv.
# There were 2*3*2000 mm origin-idle lines on D. Three lines, mech adv of two.

hp3_static_comp = how_big_difference_will_static_buildup_compensation_make(
    anchors,
    anchors,
    0.008,
    np.array([2, 2, 2, 3]),
    55,
    np.array([5000, 5000, 5000, 6000]),
)
# array([[-10.72106283,  -8.19929385,  -7.79058608,  -4.04210374],
#        [ -9.37713369, -11.91784016,  -8.20968558,  -4.8727733 ],
#        [ -7.79058608,  -7.03073739, -10.72106283,  -4.04210374],
#        [ -8.16116125,  -7.67386438,  -8.16116125, -18.74234315]])
# There were 2*2500 mm origin-idle lines on ABC. Two lines, no mech adv.
# There were 3*2000 mm origin-idle lines on D. Three lines, no mech adv.

hp4_static_comp = how_big_difference_will_static_buildup_compensation_make(
    anchors,
    anchors,
    0.008,
    np.array([2, 2, 2, 2]),
    65,
    np.array([5000, 5000, 5000, 4000]),
)
# array([[-7.67402308, -5.88140343, -5.58797583, -1.93535984],
#        [-6.72717476, -8.52953831, -5.88886442, -2.33358816],
#        [-5.58797583, -5.04252281, -7.67402308, -1.93535984],
#        [-5.85402535, -5.50418169, -5.85402535, -8.92192291]])
# There are 2*2500 mm origin-idle lines on ABC. One line per spool, mech adv of two.
# There are 2*2000 mm origin-idle lines on D. One line per spool, mech adv of two.

################################################################################################
######## Both Static and Dynamic Compensation ##################################################
################################################################################################

# Now, let's compensate for both static and dynamic buildup
hp2_comp = how_big_difference_will_buildup_compensation_make(
    anchors,
    anchors,
    0.008 / 0.7,
    np.array([2, 2, 2, 6]),
    33,
    np.array([5000, 5000, 5000, 12000]),
)
# array([[ -55.73183566,  -24.15571112,  -23.33627775,  -27.18014818],
#        [ -26.28975568,  -63.62415285,  -24.1760163 ,  -31.75208208],
#        [ -23.33627775,  -21.70495821,  -55.73183566,  -27.18014818],
#        [ -24.08097593,  -23.0948036 ,  -24.08097593, -230.5208961 ]])

hp3_comp = how_big_difference_will_buildup_compensation_make(
    anchors,
    anchors,
    0.008,
    np.array([2, 2, 2, 3]),
    55,
    np.array([5000, 5000, 5000, 6000]),
)
# array([[-14.14618518,  -6.16565207, -5.95523402,  -3.52923742],
#        [ -6.71473784, -16.14635565, -6.17086868,  -4.12683654],
#        [ -5.95523402,  -5.5368555 ,-14.14618518,  -3.52923742],
#        [ -6.14645293,  -5.89326303, -6.14645293, -29.524966  ]])

hp4_comp = how_big_difference_will_buildup_compensation_make(
    anchors,
    anchors,
    0.008,
    np.array([2, 2, 2, 2]),
    65,
    np.array([5000, 5000, 5000, 4000]),
)
# array([[-10.13545333,  -4.41994691,  -4.26901825,  -1.68854468],
#        [ -4.81387036, -11.56831908,  -4.42368885,  -1.97460865],
#        [ -4.26901825,  -3.96895939, -10.13545333,  -1.68854468],
#        [ -4.40617518,  -4.22457017,  -4.40617518, -14.11102321]])


################################################################################################
######## Comparing Significance of Static and Dynamic Compensation #############################
################################################################################################
# hp4_static_comp
# array([[-7.67402308, -5.88140343, -5.58797583, -1.93535984],
#        [-6.72717476, -8.52953831, -5.88886442, -2.33358816],
#        [-5.58797583, -5.04252281, -7.67402308, -1.93535984],
#        [-5.85402535, -5.50418169, -5.85402535, -8.92192291]])
# hp4_dyn_comp
# array([[-2.48473373,  1.47529279,  1.33144475,  0.24868453],
#        [ 1.93141853, -3.0675503 ,  1.47904705,  0.36169841],
#        [ 1.33144475,  1.08372734, -2.48473373,  0.24868453],
#        [ 1.46155763,  1.29172619,  1.46155763, -5.22840237]])
# hp4_comp
# array([[-10.13545333,  -4.41994691,  -4.26901825,  -1.68854468],
#        [ -4.81387036, -11.56831908,  -4.42368885,  -1.97460865],
#        [ -4.26901825,  -3.96895939, -10.13545333,  -1.68854468],
#        [ -4.40617518,  -4.22457017,  -4.40617518, -14.11102321]])
dynamic_adjustment_hp4 = hp4_dyn_comp / hp4_static_comp
# array([[ 0.32378502, -0.25084026, -0.2382696 , -0.12849524],
#        [-0.28710694,  0.35963849, -0.25115998, -0.15499668],
#        [-0.2382696 , -0.21491769,  0.32378502, -0.12849524],
#        [-0.24966712, -0.23468088, -0.24966712,  0.58601743]])

# We see that static compensation avoids a lot of unneccesary motor rotation.
# The dynamic compensation adjusts (adds or subtracts to) the static compensation by a smaller amount.
# If we would only compensate the spool radius statically, it would get us to +-37% of the total compensation that we want along each axis.
# (37% is the maximum of dynamic adjustment for HP4 in my office.)
# We can conclude that static compensation is more important to get right than dynamic compensation in my office.

# Sidenote:
# This 37% maximum is fairly constant for all Hangprinter setups, given that the user always ties in
# just enough origin-idle line on all spools to be able to move to the positions of all anchors without emptying any of the spools.

# Another sidenote:
# While printing, the fraction of dynamic compensation is much less than 37%.
# I'd consider > 10% of dynamic compensation to be unusual during print.

# Why Dynamic Compensation At All Then?
# Notice in the data how static compensation only ever makes the motors rotate less than they would have done with no compensation.
# Dynamic compensation always adjust in both directions (at least one motor always gets positive adjustment, at least one will get negative adjustment, in any move).
# Even after applying a perfect static compensation, we will still have one, two, or three motors rotate too much,
# and their antagonist motors rotate too little, in every move that goes away from the origin (and cancel this error back out when moving back towards origin).
# That is, without dynamic compensation we're limited in how far we can move away from the origin withough over-tightening.
# Static compensation does not mitigate over-tightening whenever the effector moves too far away from the origin (in any direction).

# Shorter answer:
# We want to move to the ceiling during the homing procedure.
# We need that move to be controlled an accurate.
