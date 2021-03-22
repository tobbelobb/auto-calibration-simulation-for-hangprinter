function y = minimize_me(anchs, angs)

  R = rotmatrix(angs);
  res = R*anchs;
  y = res(1,1)*res(1,1) + res(1,4)*res(1,4) + res(2,4)*res(2,4);
end
