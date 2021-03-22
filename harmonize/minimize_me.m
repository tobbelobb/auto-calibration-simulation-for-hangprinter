function y = minimize_me(anchs, angs)

  R = rotmatrix(angs);
  %anchs = [[-48.71, -1640.65, -144.91]; [1345.96, 1206.50, -125.41]; [-1399.17, 780.88, -148.90]; [-20.17, -40.76, 2555.49]]';
  res = R*anchs;
  y = res(1,1)*res(1,1) + res(1,4)*res(1,4) + res(2,4)*res(2,4);
end
