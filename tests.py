from simulation import *

class tester:
    "Hold some state that is common to several tests."
    anchors = symmetric_anchors(4000, 0, 0, 0)
    pos = positions(3, 2000, 0)
    spool_r = 65*np.ones(4)
    gear_factor = 12.75
    mech_adv = 2*np.ones(4)
    samp = samples_relative_to_origin_no_fuzz(anchors, pos)
    samp_in_degrees_without_buildup_comp = gear_factor*mech_adv*samp*360/(2*np.pi*spool_r)

    def motor_pos_samples_with_defaults(self, spool_buildup_factor):
        return motor_pos_samples_with_spool_buildup_compensation(self.anchors,
                self.pos,
                spool_buildup_factor,
                self.spool_r,
                self.gear_factor,
                self.mech_adv)

    def test_motor_pos_spool_buildup_compensation_sign(self):
        """When line length has increased, the buildup compensated data samples should show more rotation than un-compensated samples, and vice versa.
        """
        spool_buildup_factor = 0.008
        samp_in_degrees_with_buildup_comp =  self.motor_pos_samples_with_defaults(spool_buildup_factor)
        indices_with_error = (np.sign(np.abs(samp_in_degrees_with_buildup_comp) - np.abs(self.samp_in_degrees_without_buildup_comp)) == np.sign(self.samp)) == False
        epsilon = 1e-8
        if np.any(indices_with_error):
            # Don't care about sign errors if values are very small
            if np.any(np.abs(samp_in_degrees_with_buildup_comp[indices_with_error]) > epsilon) or np.any(np.abs(self.samp_in_degrees_without_buildup_comp[indices_with_error] > epsilon)):
                print("Signs of buildup compensation goes the wrong way")
                return False
        return True

        return np.all(np.sign(np.abs(samp_in_degrees_with_buildup_comp) - np.abs(self.samp_in_degrees_without_buildup_comp)) == np.sign(self.samp))
    def test_samp_with_low_buildup_factor(self):
        """When the buildup factor is very small, buildup compensated values should converge towards un-compensated values
        """
        spool_buildup_factor = 0.00001
        samp_in_degrees_with_buildup_comp = self.motor_pos_samples_with_defaults(spool_buildup_factor)
        diffs = np.abs(samp_in_degrees_with_buildup_comp - self.samp_in_degrees_without_buildup_comp)
        what_is_considered_a_small_amount_of_degrees_here = 5
        if not(np.all(diffs < what_is_considered_a_small_amount_of_degrees_here)):
            print("Lack of convergence problem 1")
            return False
        samp_in_degrees_with_buildup_comp2 = self.motor_pos_samples_with_defaults(spool_buildup_factor/2)
        hopefully_smaller_diffs = np.abs(samp_in_degrees_with_buildup_comp2 - self.samp_in_degrees_without_buildup_comp)
        very_small_float = 1e-16
        if (not(np.all(hopefully_smaller_diffs-very_small_float < diffs))):
            print("Lack of convergence problem 2")
            return False
        very_small_buildup_factor = 1e-8
        very_small_angle = 0.01
        samp_in_degrees_with_buildup_comp3 = self.motor_pos_samples_with_defaults(very_small_buildup_factor)
        if np.any(np.abs(samp_in_degrees_with_buildup_comp3 - self.samp_in_degrees_without_buildup_comp) > very_small_angle):
            print("Lack of convergence problem 3")
            return False
        return True

def run():
    # Create tester objects
    testerObj = tester()
    # Specify which tests to run
    tests = np.array([
        testerObj.test_motor_pos_spool_buildup_compensation_sign,
        testerObj.test_samp_with_low_buildup_factor
        ])
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
