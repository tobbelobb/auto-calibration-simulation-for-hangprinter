#!/usr/bin/python3

from simulation import *


class tester:
    "Hold some state that is common to several tests."
    anchors = symmetric_anchors(4000, 0, 0, 0, 0)
    pos = positions(3, 1000)
    spool_r = 75 * np.ones(5)
    gear_factor = 255.0 / 20.0
    mech_adv = np.array([2, 2, 2, 2, 4])
    samp = distance_samples_relative_to_origin(anchors, pos)
    samp_in_degrees_without_buildup_comp = gear_factor * mech_adv * samp * 360 / (2 * np.pi * spool_r)

    def motor_pos_samples_with_defaults(self, spool_buildup_factor):
        return pos_to_motor_pos_samples(
            self.anchors,
            self.pos,
            0,
            False,
            spool_buildup_factor,
            self.spool_r,
            self.gear_factor,
            self.mech_adv,
        )

    def test_motor_pos_spool_buildup_compensation_sign(self):
        """When line length has increased, the buildup compensated data samples should show more rotation than un-compensated samples, and vice versa."""
        spool_buildup_factor = 0.008
        samp_in_degrees_with_buildup_comp = self.motor_pos_samples_with_defaults(spool_buildup_factor)
        indices_with_error = (
            np.sign(np.abs(samp_in_degrees_with_buildup_comp) - np.abs(self.samp_in_degrees_without_buildup_comp))
            == np.sign(self.samp)
        ) == False
        epsilon = 1e-8
        if np.any(indices_with_error):
            # Don't care about sign errors if values are very small
            if np.any(np.abs(samp_in_degrees_with_buildup_comp[indices_with_error]) > epsilon) or np.any(
                np.abs(self.samp_in_degrees_without_buildup_comp[indices_with_error] > epsilon)
            ):
                print("Signs of buildup compensation goes the wrong way")
                return False
        return True

    def test_samp_with_low_buildup_factor(self):
        """When the buildup factor is very small, buildup compensated values should converge towards un-compensated values"""
        spool_buildup_factor = 0.00001
        samp_in_degrees_with_buildup_comp = self.motor_pos_samples_with_defaults(spool_buildup_factor)
        diffs = np.abs(samp_in_degrees_with_buildup_comp - self.samp_in_degrees_without_buildup_comp)
        what_is_considered_a_small_amount_of_degrees_here = 5
        if not (np.all(diffs < what_is_considered_a_small_amount_of_degrees_here)):
            print("Lack of convergence problem 1")
            return False
        samp_in_degrees_with_buildup_comp2 = self.motor_pos_samples_with_defaults(spool_buildup_factor / 2)
        hopefully_smaller_diffs = np.abs(samp_in_degrees_with_buildup_comp2 - self.samp_in_degrees_without_buildup_comp)
        very_small_float = 1e-16
        if not (np.all(hopefully_smaller_diffs - very_small_float < diffs)):
            print("Lack of convergence problem 2")
            return False
        very_small_buildup_factor = 1e-8
        very_small_angle = 0.01
        samp_in_degrees_with_buildup_comp3 = self.motor_pos_samples_with_defaults(very_small_buildup_factor)
        if np.any(
            np.abs(samp_in_degrees_with_buildup_comp3 - self.samp_in_degrees_without_buildup_comp) > very_small_angle
        ):
            print("Lack of convergence problem 3")
            return False
        return True

    def test_cost_sq_for_pos_samp_forward_transform(self):
        if np.linalg.norm(forward_transform(self.anchors, self.samp[1]) - self.pos[1]) > 0.0001:
            print("Forward transform doesn't work")
            print("Got this: %s" % forward_transform(self.anchors, self.samp[1]))
            print("Expected this: %s " % self.pos[1])
            return False

        return True

    def test_distance_samples(self):
        pos_local = np.r_[
            positions(2, 100),
            [
                [
                    0,
                    0,
                    0,
                ],
                [0, 0, 100],
            ],
        ]
        dist_samp = distance_samples_relative_to_origin(self.anchors, pos_local)
        if abs(np.linalg.norm(dist_samp[-2])) > 1e-4:
            print("distance_samples_relative_to_origin seems broken 0")
            return False
        if dist_samp[-1][A] < 0:
            print("distance_samples_relative_to_origin seems broken 1")
            return False
        if dist_samp[-1][B] < 0:
            print("distance_samples_relative_to_origin seems broken 2")
            return False
        if dist_samp[-1][C] < 0:
            print("distance_samples_relative_to_origin seems broken 3")
            return False
        if abs(dist_samp[-1][D] + 100) > 1e-4:
            print("distance_samples_relative_to_origin seems broken 4")
            return False


def run():
    # Create tester objects
    testerObj = tester()
    # Specify which tests to run
    tests = np.array(
        [
            testerObj.test_motor_pos_spool_buildup_compensation_sign,
            testerObj.test_samp_with_low_buildup_factor,
            testerObj.test_cost_sq_for_pos_samp_forward_transform,
            testerObj.test_distance_samples,
        ]
    )
    # Run tests one by one and collect restults
    results = np.array([test() for test in tests])
    # Print some helpful information
    if np.any(results == False):
        print("")
        for test in tests[results == False]:
            print(test.__name__, "failed")
        num_fails = np.sum(results == False)
        if num_fails != 1:
            print("A total of", num_fails, "tests failed.")
    else:
        print("All", np.size(results), "tests passed.")
    return results


if __name__ == "__main__":
    run()
    exit(0)
