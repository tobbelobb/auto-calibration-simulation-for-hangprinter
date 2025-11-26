#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

#include <Matrix.h>
#include <flex.hpp>
#include <util.hpp>

// Interval-analysis inspired forward transform following 2015NICE4018.
using Point = std::array<float, 3>;
constexpr float stepsPerDegree = 25.0F / 360.0F;

// Borrowed from hangprinter-flex-compensation tests
constexpr std::array<Point, 4> fourAnchors = {
    {{16.4F, -1610.98F, -131.53F}, {1314.22F, 1268.14F, -121.28F}, {-1415.73F, 707.61F, -121.82F}, {0.0F, 0.0F, 2299.83F}}};

constexpr std::array<Point, 5> fiveAnchors = {{
    {0.0F, -2000.0F, -120.0F},
    {2000.0F, 0.0F, -120.0F},
    {0.0F, 2000.0F, -120.0F},
    {-2000.0F, 0.0F, -120.0F},
    {0.0F, 0.0F, 2000.0F},
}};

static std::array<Point, HANGPRINTER_MAX_ANCHORS> anchorSet(size_t numAnchors) {
  std::array<Point, HANGPRINTER_MAX_ANCHORS> anchors{};
  anchors.fill(Point{0.0F, 0.0F, 0.0F});
  if (numAnchors == 5) {
    for (size_t i = 0; i < 5; ++i) {
      anchors[i] = fiveAnchors[i];
    }
  } else {
    for (size_t i = 0; i < 4 && i < numAnchors; ++i) {
      anchors[i] = fourAnchors[i];
    }
  }
  return anchors;
}

static std::array<float, HANGPRINTER_MAX_ANCHORS> originLengths(size_t numAnchors) {
  std::array<float, HANGPRINTER_MAX_ANCHORS> lengths{};
  lengths.fill(0.0F);
  if (numAnchors == 5) {
    for (size_t i = 0; i < 5; ++i) {
      lengths[i] = hyp3(fiveAnchors[i], Point{0.0F, 0.0F, 0.0F});
    }
  } else {
    for (size_t i = 0; i < 4 && i < numAnchors; ++i) {
      lengths[i] = hyp3(fourAnchors[i], Point{0.0F, 0.0F, 0.0F});
    }
  }
  return lengths;
}

static Flex makeFlex(size_t numAnchors) {
  if (numAnchors == 5) {
    return Flex{fiveAnchors};
  }
  if (numAnchors == 4) {
    return Flex{fourAnchors};
  }
  return Flex{numAnchors};
}

struct Interval {
  float lo{0.0F};
  float hi{0.0F};

  float width() const { return hi - lo; }
  float mid() const { return 0.5F * (lo + hi); }
};

struct Box {
  Interval x{};
  Interval y{};
  Interval z{};

  float maxWidth() const { return std::max({x.width(), y.width(), z.width()}); }
  Point mid() const { return {x.mid(), y.mid(), z.mid()}; }
};

static bool empty(const Interval &i) { return i.lo > i.hi; }
static bool empty(const Box &b) { return empty(b.x) || empty(b.y) || empty(b.z); }

static Interval intersect(Interval a, Interval b) {
  return {std::max(a.lo, b.lo), std::min(a.hi, b.hi)};
}

static void intersectInto(Box &target, const Box &other) {
  target.x = intersect(target.x, other.x);
  target.y = intersect(target.y, other.y);
  target.z = intersect(target.z, other.z);
}

static std::pair<float, float> distanceSquaredBounds(const Box &b, const Point &anchor) {
  auto axisBounds = [](Interval const &axis, float anchorCoord) {
    Interval delta{axis.lo - anchorCoord, axis.hi - anchorCoord};
    float minVal = 0.0F;
    if (delta.lo > 0.0F) {
      minVal = delta.lo * delta.lo;
    } else if (delta.hi < 0.0F) {
      minVal = delta.hi * delta.hi;
    }
    float maxVal = std::max(delta.lo * delta.lo, delta.hi * delta.hi);
    return std::pair<float, float>{minVal, maxVal};
  };

  auto [minX, maxX] = axisBounds(b.x, anchor[0]);
  auto [minY, maxY] = axisBounds(b.y, anchor[1]);
  auto [minZ, maxZ] = axisBounds(b.z, anchor[2]);
  return {minX + minY + minZ, maxX + maxY + maxZ};
}

static void applySphereBounds(Box &b, const Point &anchor, float length, float margin) {
  Interval bx{anchor[0] - length - margin, anchor[0] + length + margin};
  Interval by{anchor[1] - length - margin, anchor[1] + length + margin};
  Interval bz{anchor[2] - length - margin, anchor[2] + length + margin};
  b.x = intersect(b.x, bx);
  b.y = intersect(b.y, by);
  b.z = intersect(b.z, bz);
}

static Box initialSearchBox(const std::vector<Point> &anchors, const std::vector<float> &lengths, float margin) {
  Box box{
      Interval{anchors[0][0] - lengths[0] - margin, anchors[0][0] + lengths[0] + margin},
      Interval{anchors[0][1] - lengths[0] - margin, anchors[0][1] + lengths[0] + margin},
      Interval{anchors[0][2] - lengths[0] - margin, anchors[0][2] + lengths[0] + margin},
  };

  for (size_t i = 1; i < anchors.size(); ++i) {
    Box bound{
        Interval{anchors[i][0] - lengths[i] - margin, anchors[i][0] + lengths[i] + margin},
        Interval{anchors[i][1] - lengths[i] - margin, anchors[i][1] + lengths[i] + margin},
        Interval{anchors[i][2] - lengths[i] - margin, anchors[i][2] + lengths[i] + margin},
    };
    intersectInto(box, bound);
  }
  return box;
}

struct EvalResult {
  bool feasible{false};
  bool solved{false};
  Box box{};
};

static EvalResult evaluateBox(Box box, const std::vector<Point> &anchors, const std::vector<float> &lengths,
                              float lengthTol, float widthTol) {
  for (size_t i = 0; i < anchors.size(); ++i) {
    applySphereBounds(box, anchors[i], lengths[i], lengthTol);
    if (empty(box)) {
      return {};
    }

    auto [minSq, maxSq] = distanceSquaredBounds(box, anchors[i]);
    float lenHi = lengths[i] + lengthTol;
    float lenLo = std::max(0.0F, lengths[i] - lengthTol);
    if (minSq > lenHi * lenHi || maxSq < lenLo * lenLo) {
      return {};
    }
  }

  EvalResult res{};
  res.feasible = true;
  res.solved = (box.maxWidth() <= widthTol);
  res.box = box;
  return res;
}

static std::array<Box, 2> bisectBox(const Box &box) {
  float wx = box.x.width();
  float wy = box.y.width();
  float wz = box.z.width();

  Box left = box;
  Box right = box;

  if (wx >= wy && wx >= wz) {
    float mid = box.x.mid();
    left.x.hi = mid;
    right.x.lo = mid;
  } else if (wy >= wx && wy >= wz) {
    float mid = box.y.mid();
    left.y.hi = mid;
    right.y.lo = mid;
  } else {
    float mid = box.z.mid();
    left.z.hi = mid;
    right.z.lo = mid;
  }
  return {left, right};
}

static bool intervalSolve(const std::vector<Point> &anchors, const std::vector<float> &lengths, Point &solution) {
  constexpr float margin = 5.0F;
  constexpr float lengthTol = 1.0F;
  constexpr float widthTol = 5.0F;
  constexpr size_t maxBoxes = 50000;

  Box root = initialSearchBox(anchors, lengths, margin);
  if (empty(root)) {
    return false;
  }

  std::vector<Box> stack;
  stack.reserve(256);
  stack.push_back(root);
  std::vector<Box> solutions;
  solutions.reserve(16);
  Box bestBox{};
  bool hasBestBox = false;
  float bestBoxCost = std::numeric_limits<float>::infinity();

  size_t iterations = 0;
  while (!stack.empty() && iterations < maxBoxes) {
    Box box = stack.back();
    stack.pop_back();
    ++iterations;

    EvalResult eval = evaluateBox(box, anchors, lengths, lengthTol, widthTol);
    if (!eval.feasible) {
      continue;
    }
    Point mid = eval.box.mid();
    float midCost = 0.0F;
    for (size_t i = 0; i < anchors.size(); ++i) {
      float dist = hyp3(mid, anchors[i]);
      midCost += std::abs(dist - lengths[i]);
    }
    if (midCost < bestBoxCost) {
      bestBoxCost = midCost;
      bestBox = eval.box;
      hasBestBox = true;
    }

    if (eval.solved) {
      solutions.push_back(eval.box);
      continue;
    }

    auto children = bisectBox(eval.box);
    stack.push_back(children[0]);
    stack.push_back(children[1]);
  }

  if (solutions.empty()) {
    if (!hasBestBox) {
      return false;
    }
    solution = bestBox.mid();
    return true;
  }

  float bestCost = std::numeric_limits<float>::infinity();
  Point best{0.0F, 0.0F, 0.0F};
  for (const auto &b : solutions) {
    Point mid = b.mid();
    float cost = 0.0F;
    for (size_t i = 0; i < anchors.size(); ++i) {
      float dist = hyp3(mid, anchors[i]);
      float err = dist - lengths[i];
      cost += std::abs(err);
    }
    if (cost < bestCost) {
      bestCost = cost;
      best = mid;
    }
  }

  solution = best;
  return true;
}

struct LmResult {
  bool converged{false};
  size_t iterations{0};
  float cost{0.0F};
};

static float evaluateCost(const Flex &flex, const Point &pos,
                          const std::array<float, HANGPRINTER_MAX_ANCHORS> &targetSteps, size_t numAnchors,
                          std::array<float, HANGPRINTER_MAX_ANCHORS> &predicted,
                          std::array<float, HANGPRINTER_MAX_ANCHORS> &residual) {
  predicted.fill(0.0F);
  residual.fill(0.0F);
  flex.CartesianToMotorStepsMatrix(pos, predicted);
  float cost = 0.0F;
  for (size_t i = 0; i < numAnchors; ++i) {
    residual[i] = predicted[i] - targetSteps[i];
    cost += residual[i] * residual[i];
  }
  return 0.5F * cost;
}

static LmResult refineWithLevenbergMarquardt(Point &pos, const Flex &flex,
                                             const std::array<float, HANGPRINTER_MAX_ANCHORS> &targetSteps,
                                             size_t numAnchors) {
  constexpr float jacStep = 0.5F;
  constexpr float lambdaInit = 1e-2F;
  constexpr float lambdaUp = 8.0F;
  constexpr float lambdaDown = 0.35F;
  constexpr float gradTol = 1e-3F;
  constexpr float stepTol = 1e-3F;
  constexpr float costTol = 1e-4F;
  constexpr size_t maxIters = 50;

  std::array<float, HANGPRINTER_MAX_ANCHORS> predicted{};
  std::array<float, HANGPRINTER_MAX_ANCHORS> residual{};
  LmResult result{};

  float cost = evaluateCost(flex, pos, targetSteps, numAnchors, predicted, residual);
  result.cost = cost;
  float lambda = lambdaInit;

  for (size_t iter{0}; iter < maxIters; ++iter) {
    ++result.iterations;
    float J[HANGPRINTER_MAX_ANCHORS][3] = {{0.0F}};

    for (size_t axis{0}; axis < 3; ++axis) {
      Point shifted = pos;
      shifted[axis] += jacStep;
      std::array<float, HANGPRINTER_MAX_ANCHORS> predShift{};
      std::array<float, HANGPRINTER_MAX_ANCHORS> resShift{};
      flex.CartesianToMotorStepsMatrix(shifted, predShift);
      for (size_t i{0}; i < numAnchors; ++i) {
        J[i][axis] = (predShift[i] - predicted[i]) / jacStep;
      }
    }

    float g[3] = {0.0F, 0.0F, 0.0F};
    float H[3][3] = {{0.0F, 0.0F, 0.0F}, {0.0F, 0.0F, 0.0F}, {0.0F, 0.0F, 0.0F}};
    for (size_t i{0}; i < numAnchors; ++i) {
      g[0] += J[i][0] * residual[i];
      g[1] += J[i][1] * residual[i];
      g[2] += J[i][2] * residual[i];

      H[0][0] += J[i][0] * J[i][0];
      H[0][1] += J[i][0] * J[i][1];
      H[0][2] += J[i][0] * J[i][2];
      H[1][0] += J[i][1] * J[i][0];
      H[1][1] += J[i][1] * J[i][1];
      H[1][2] += J[i][1] * J[i][2];
      H[2][0] += J[i][2] * J[i][0];
      H[2][1] += J[i][2] * J[i][1];
      H[2][2] += J[i][2] * J[i][2];
    }

    FixedMatrix<float, 3, 4> system;
    for (size_t r{0}; r < 3; ++r) {
      for (size_t c{0}; c < 3; ++c) {
        system(r, c) = H[r][c];
      }
      system(r, r) += lambda;
    }
    system(0, 3) = -g[0];
    system(1, 3) = -g[1];
    system(2, 3) = -g[2];

    if (!system.GaussJordan(3, 4)) {
      lambda *= lambdaUp;
      continue;
    }

    Point delta{system(0, 3), system(1, 3), system(2, 3)};
    float gradNorm = sqrtf(fsquare(g[0]) + fsquare(g[1]) + fsquare(g[2]));
    if (hyp3(delta, Point{0.0F, 0.0F, 0.0F}) < stepTol && gradNorm < gradTol) {
      result.converged = true;
      break;
    }

    Point candidate{pos[0] + delta[0], pos[1] + delta[1], pos[2] + delta[2]};
    std::array<float, HANGPRINTER_MAX_ANCHORS> predCandidate{};
    std::array<float, HANGPRINTER_MAX_ANCHORS> resCandidate{};
    float candidateCost = evaluateCost(flex, candidate, targetSteps, numAnchors, predCandidate, resCandidate);

    if (candidateCost < cost) {
      float improvement = cost - candidateCost;
      pos = candidate;
      predicted = predCandidate;
      residual = resCandidate;
      cost = candidateCost;
      lambda = std::max(lambda * lambdaDown, 1e-6F);
      result.cost = cost;
      if (improvement < costTol && hyp3(delta, Point{0.0F, 0.0F, 0.0F}) < stepTol) {
        result.converged = true;
        break;
      }
    } else {
      lambda *= lambdaUp;
    }
  }

  return result;
}

static bool MotorStepsToCartesianNice(const std::array<float, HANGPRINTER_MAX_ANCHORS> &motorPos, Point &machinePos,
                                      size_t numAnchors) {
  Flex const flex = makeFlex(numAnchors);
  auto const origins = originLengths(numAnchors);
  auto const anchorsArr = anchorSet(numAnchors);

  std::vector<Point> anchors;
  std::vector<float> lengths;
  anchors.reserve(numAnchors);
  lengths.reserve(numAnchors);

  std::array<float, HANGPRINTER_MAX_ANCHORS> targetSteps{};
  for (size_t i{0}; i < numAnchors; ++i) {
    anchors.push_back(anchorsArr[i]);
    targetSteps[i] = motorPos[i] * stepsPerDegree;
    float linePos = flex.MotorPosToLinePos(targetSteps[i], i);
    lengths.push_back(linePos + origins[i]);
  }

  bool intervalOk = intervalSolve(anchors, lengths, machinePos);
  if (!intervalOk) {
    machinePos = {0.0F, 0.0F, 0.0F};
  }

  Point const startPos = machinePos;
  std::array<float, HANGPRINTER_MAX_ANCHORS> predicted{};
  std::array<float, HANGPRINTER_MAX_ANCHORS> residual{};
  float const preLmCost = evaluateCost(flex, machinePos, targetSteps, numAnchors, predicted, residual);

  LmResult const lm = refineWithLevenbergMarquardt(machinePos, flex, targetSteps, numAnchors);
  if (lm.cost >= preLmCost) {
    machinePos = startPos;
  }

  return intervalOk;
}

int main() {
  Point machinePos{0.0F, 0.0F, 0.0F};
  size_t constexpr numAnchors = 5;
  std::vector<std::array<float, numAnchors>> motorPoss = {
      {-369.47366890F, -375.18362935F, 318.27641438F, 323.88876891F, 1076.68163402F},
      {-5980.29607895F, 12630.56774248F, 12288.67350302F, -6506.18962476F, -13348.06053879F},
      {-2353.64278337F, 15870.99892165F, 15755.46225500F, -2522.71380922F, -27477.49741579F},
      {1433.28912224F, 10685.68516553F, 1454.31086905F, -10542.99803343F, 1813.47411829F},
      {3112.53568603F, 11779.51141393F, 3231.55807349F, -7609.55533435F, -17364.20990461F},
      {7221.24317502F, 14331.47424095F, 6657.95387927F, -1853.21287105F, -36135.10874008F},
      {10106.78583366F, 10693.29200030F, -7583.66231627F, -8505.26876850F, 6010.84493879F},
      {11680.14400927F, 12421.57492733F, -5209.71780789F, -6326.70127134F, -14780.55685279F},
      {16309.54520647F, 16482.81831770F, -1424.01387322F, -1674.11174073F, -29909.99659944F},
      {-10145.44532568F, 1094.96136691F, 10156.51947369F, 1434.73615456F, 3572.60726310F},
      {-7640.99960502F, 3134.81886277F, 11567.99443696F, 2933.32488052F, -16735.05779004F},
      {-2825.34894707F, 7163.91339055F, 14266.90144144F, 5840.37464568F, -33865.11200386F},
      {-7448.08057312F, 10796.70751776F, 10151.74389637F, -8459.78905687F, 3271.18757879F},
      {2211.41905092F, 2111.20882241F, 858.18022603F, 961.42598207F, -17727.66189890F},
      {6585.49993026F, 5349.29768666F, 5599.61645498F, 6829.22955353F, -40179.35611831F},
      {9827.26906637F, 973.41612678F, -9847.89764725F, 1374.65365736F, 4088.32777321F},
      {11672.37320821F, 3419.76976053F, -7745.30493425F, 2691.62201517F, -16638.39998295F},
      {13421.03929287F, 6834.73758758F, -2414.14469240F, 5523.60605573F, -33907.52745615F},
      {-8153.62357389F, -7386.54442994F, 10370.98726659F, 9875.86705528F, 3214.03835550F},
      {-6347.85480184F, -5493.48884529F, 12625.31736324F, 12063.42984255F, -14568.06462464F},
      {-2585.49492187F, -1676.70121477F, 15851.52670557F, 15223.11520330F, -28367.04177068F},
      {730.23171944F, -9759.54189457F, 1580.93148691F, 9745.76583762F, 3967.14372473F},
      {2429.55959648F, -7123.34974121F, 3539.66195357F, 11159.52316296F, -17372.69974680F},
      {6079.30436810F, -2972.78187516F, 7172.81870678F, 14578.78537233F, -33851.97153806F},
      {11195.81863308F, -7791.69855043F, -8587.42819506F, 10696.72883673F, 3279.60983059F},
      {12162.75651477F, -6552.47720614F, -6177.61954587F, 12406.22670936F, -12538.57963324F},
      {14653.37178962F, -2366.53557957F, -1021.48237225F, 15599.48126071F, -29623.47054342F},
  };

  std::vector<Point> expectedPoss = {
      {17.9255679F, -17.6352632F, -27.497151F},
      {-523.885148F, -501.061599F, 507.741532F},
      {-538.35251F, -530.255594F, 953.040004F},
      {-545.189816F, -0.55803779F, 26.4689901F},
      {-520.999273F, -3.27341154F, 537.019049F},
      {-470.799477F, 16.5498517F, 1047.17018F},
      {-505.362595F, 467.529134F, -42.5453256F},
      {-513.234032F, 464.122399F, 533.480869F},
      {-538.29312F, 526.060302F, 1030.39159F},
      {8.98405788F, -520.642383F, -26.6148823F},
      {-5.52678502F, -514.840431F, 517.628213F},
      {-38.6000122F, -492.617793F, 991.936222F},
      {-507.629503F, -465.97604F, 31.7537695F},
      {-30.5601452F, 35.9665477F, 458.719776F},
      {43.2898928F, 28.8400594F, 1046.72968F},
      {10.5874156F, 504.407891F, -44.2588518F},
      {-19.9804722F, 520.41744F, 517.004705F},
      {-37.9958866F, 454.266612F, 975.222737F},
      {455.930483F, -487.688259F, 26.0515823F},
      {482.712486F, -520.125685F, 535.481389F},
      {495.517745F, -539.35609F, 965.228753F},
      {500.096373F, -22.437457F, -42.0139801F},
      {490.657689F, -30.4215669F, 527.319817F},
      {506.742907F, -31.9689121F, 998.360926F},
      {490.590662F, 523.118645F, 41.4092948F},
      {517.848381F, 501.639033F, 482.878157F},
      {525.399857F, 459.863258F, 981.665044F},
  };

  float totalAbsErr = 0.0F;
  for (size_t i{0}; i < motorPoss.size(); ++i) {
    bool const ok = MotorStepsToCartesianNice(motorPoss[i], machinePos, numAnchors);
    Point err = expectedPoss[i] - machinePos;
    totalAbsErr += std::abs(err[0]) + std::abs(err[1]) + std::abs(err[2]);
    std::cout << err << '\n';
    if (!ok) {
      std::cout << "Interval search fell back on sample " << i << '\n';
    }
  }

  std::cout << "Mean absolute error: " << totalAbsErr / static_cast<float>(motorPoss.size()) << '\n';
  return 0;
}
