from simulation import *

#a = symmetric_anchors(1500)

# From measurment
a = np.array([[0.0, -2163.0, -75.5],
              [-1841.0, 741.0, -75.5],
              [1639.0, 1404.0, -75.5],
              [0.0, 0.0, 3250.5]])

pos = positions(3, 300, 0)
samp_exp = samples(a, pos, 0)
# Keep the zero measurement as the first row
samp_exp_view = samp_exp[1:]
# Make index 2 = C motor the sensor motor
samp_exp_view = samp_exp_view[samp_exp_view[:,1].argsort()]
samp_exp_view = samp_exp_view[samp_exp_view[:,2].argsort(kind='mergesort')]
samp_exp[1:] = samp_exp_view

# Array samp is gotten from measurements
u = np.shape(samp)[0]
diffs = samp_exp[0:u] - samp
x = np.arange(u)
A = np.vstack([x, np.ones(len(x))]).T # A = [[x 1]]
m, c = np.linalg.lstsq(A, diffs[:,2])[0] # Linear regression. diffs[:,2] ~= mx + c

# Results:
# np.mean(diffs[:,2]) = -1.9
# np.std(diffs[:,2]) = 7.95
# m = -0.64
# c = 3.85

# After removing linear error:
# np.mean(diffs[:,2]) = 0.2
# np.std(diffs[:,2]) = 7.1
