from simulation import *

a = symmetric_anchors(1500)
pos = positions(3, 400, 0)
samp = samples(a, pos, 0)
samp = samp[samp[:,2].argsort()]
samp = samp[samp[:,3].argsort(kind='mergesort')]
samp_rel = np.zeros(np.shape(samp))

for i in range(1, 27):
    samp_rel[i] = samp[i] - samp[i-1]

last_movement = -samp[26] # Go back
