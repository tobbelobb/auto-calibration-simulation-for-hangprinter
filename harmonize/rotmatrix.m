function R = rotmatrix(angs)

  alpha = angs(1);
  beta = angs(2);
  gamma = angs(3);

  R01 = -cos(alpha)*cos(beta)*sin(gamma) - sin(alpha)*cos(gamma);
  R10 = sin(alpha)*cos(beta)*cos(gamma) + cos(alpha)*sin(gamma);
  R02 = cos(alpha)*sin(beta);
  R12 = sin(alpha)*sin(beta);
  R20 = -sin(beta)*cos(gamma);
  R21 = sin(beta)*sin(gamma);

  R = [
        [cos(alpha) * cos(beta) * cos(gamma) - sin(alpha) * sin(gamma),                                                            R01,       R02];
        [                                                          R10, -sin(alpha) * cos(beta) * sin(gamma) + cos(alpha) * cos(gamma),       R12];
        [                                                          R20,                                                            R21, cos(beta)];
      ];
end
