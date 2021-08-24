#!/usr/bin/octave

format shortG
% These are found as output of the script
% /path/to/auto-calibration-simulation-for-hangprinter/simulation.py
anchs = [[114.20, -1611.56, -47.65];
         [1225.42, 1349.40, -164.69];
         [-1458.59, 599.47, -182.37];
         [-47.13, 116.45, 2296.38];]';

R = rotmatrix(fminsearch(@(angs) minimize_me(anchs, angs), [0 0 0]));
rotated_anchs = (R*anchs)'

% These are found as output from the script
% /home/torbjorn/repos/hp-mark/use/get_auto_calibration_data_automatically.sh
% It is probably also stored in the file
% /path/to/auto-calibration-simulation-for-hangprinter/simulation.py
xyz_of_samps = [[120.598, -47.4539, 10.3903];
[-373.229, -128.299, 33.3251];
[-227.145, 207.431, 4.89257];
[76.8445, -71.1445, 706.312];
[78.4024, -71.8729, 706.238];
[-59.3678, -78.8227, 702.561];
[-70.1781, 13.4975, 695.123];
[-78.4726, 1.36425, 719.559];
[-75.7426, 1.94881, 719.184];
[-146.141, -11.464, 723.303];
[-117.092, 107.432, 717.002];
[-167.541, 72.8723, 825.57];
[-16.9196, 60.6574, 822.055];
[-113.799, 91.6077, 762.606];]';

% I put the rotated xyz values back into simulation.py, and reruns simulation.py,
% just to get more copy/paste friendly output, even though the rotated_anchs
% printed earlier in this script contains all the information we need to write
% a correct M669 A...:..:... B...:...:... C...:...:... D...:...:... command manually
rotated_xyz_of_samps = (R*xyz_of_samps)'

% These are found in
% /path/to/hp-mark/hpm/hpm/example-marker-params/my-marker-params.xml
provided_marker_positions = [[ -48.34,    -123.23,     164.33];
                            [  71.473,    -117.94,     160.98];
                            [  158.16,    -16.707,     159.47];
                            [  44.551,     167.61,     164.43];
                            [ -52.246,     164.12,     167.15];
                            [ -166.99,    -50.646,     168.39];]';


% Put these in my-marker-params.xml to make your hp-mark rotate coords
% in the same way that RepRapFirmware does.
% And redo the camera position calibration:
% $ ./hpm -c <and-some-more-options>
rotated_provided_marker_positions = (R*provided_marker_positions)'
