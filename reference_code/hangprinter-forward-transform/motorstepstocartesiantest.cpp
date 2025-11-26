#include <algorithm>
#include <iostream>
#include <array>
#include <vector>
#include <cmath>
#include <Matrix.h>
#include <flex.hpp>

using Point = std::array<float, 3>;
constexpr float stepsPerDegree = 25.0F / 360.0F;

// Borrowed from hangprinter-flex-compensation
constexpr std::array<Point, 4> fourAnchors = {
      {{16.4, -1610.98, -131.53}, {1314.22, 1268.14, -121.28}, {-1415.73, 707.61, -121.82}, {0.0, 0.0, 2299.83}}};

constexpr std::array<Point, 5> fiveAnchors = {
      //{{16.4, -1610.98, -131.53}, {1314.22, 128.14, -121.28}, {-15.73, 1415.61, -121.82}, {-1211.62, 18.14, -111.18}, {10.0, -10.0, 2299.83}}};
      {{    0., -2000.,  -120.},
       { 2000.,     0.,  -120.},
       {    0.,  2000.,  -120.},
       {-2000.,     0.,  -120.},
       {    0.,     0.,  2000.}}};


static inline float hyp3(Point const &a, Point const &b) {
  return sqrtf(fsquare(a[2] - b[2]) + fsquare(a[1] - b[1]) + fsquare(a[0] - b[0]));
}
static inline float hyp3(Point const &a, float const b[HANGPRINTER_MAX_ANCHORS]) {
  return sqrtf(fsquare(a[2] - b[2]) + fsquare(a[1] - b[1]) + fsquare(a[0] - b[0]));
}

std::array<float, 5> fiveLineLengthsOrigin = {
    sqrtf(fsquare(fiveAnchors[A_AXIS][0]) + fsquare(fiveAnchors[A_AXIS][1]) + fsquare(fiveAnchors[A_AXIS][2])),
    sqrtf(fsquare(fiveAnchors[B_AXIS][0]) + fsquare(fiveAnchors[B_AXIS][1]) + fsquare(fiveAnchors[B_AXIS][2])),
    sqrtf(fsquare(fiveAnchors[C_AXIS][0]) + fsquare(fiveAnchors[C_AXIS][1]) + fsquare(fiveAnchors[C_AXIS][2])),
    sqrtf(fsquare(fiveAnchors[D_AXIS][0]) + fsquare(fiveAnchors[D_AXIS][1]) + fsquare(fiveAnchors[D_AXIS][2])),
    sqrtf(fsquare(fiveAnchors[I_AXIS][0]) + fsquare(fiveAnchors[I_AXIS][1]) + fsquare(fiveAnchors[I_AXIS][2]))};

std::array<float, 4> fourLineLengthsOrigin = {
    sqrtf(fsquare(fourAnchors[A_AXIS][0]) + fsquare(fourAnchors[A_AXIS][1]) + fsquare(fourAnchors[A_AXIS][2])),
    sqrtf(fsquare(fourAnchors[B_AXIS][0]) + fsquare(fourAnchors[B_AXIS][1]) + fsquare(fourAnchors[B_AXIS][2])),
    sqrtf(fsquare(fourAnchors[C_AXIS][0]) + fsquare(fourAnchors[C_AXIS][1]) + fsquare(fourAnchors[C_AXIS][2])),
    sqrtf(fsquare(fourAnchors[D_AXIS][0]) + fsquare(fourAnchors[D_AXIS][1]) + fsquare(fourAnchors[D_AXIS][2]))};

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
      lengths[i] = fiveLineLengthsOrigin[i];
    }
  } else {
    for (size_t i = 0; i < 4 && i < numAnchors; ++i) {
      lengths[i] = fourLineLengthsOrigin[i];
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

static float norm3(Point const &p) {
  return sqrtf(fsquare(p[0]) + fsquare(p[1]) + fsquare(p[2]));
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

    // Numerical Jacobian of the residuals wrt xyz
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
    if (norm3(delta) < stepTol && gradNorm < gradTol) {
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
      if (improvement < costTol && norm3(delta) < stepTol) {
        result.converged = true;
        break;
      }
    } else {
      lambda *= lambdaUp;
    }
  }

  return result;
}



std::ostream &operator<<(std::ostream &o, std::array<auto, 4> &a){
  return o << "[ " << a[0] << ", " << a[1] << ", " << a[2] << ", " << a[3] << " ], ";
}

std::ostream &operator<<(std::ostream &o, std::array<auto, 5> &a){
  return o << "[ " << a[0] << ", " << a[1] << ", " << a[2] << ", " << a[3] << ", " << a[4] << " ], ";
}

inline bool det(FixedMatrix<float, 3, 4> M){
  return M(0, 0) * (M(1, 1) * M(2, 2) - M(1, 2) * M(2, 1)) + M(0, 1) * (M(1, 2) * M(2, 0) - M(2, 2) * M(1, 0)) + M(0, 2) * (M(1, 0) * M(2, 1) - M(1, 1) * M(2, 0));
}

inline bool singular_3x3(FixedMatrix<float, 3, 4> M){
  float const threshold = 1e-1;
  return fabsf(det(M)) < threshold;
}

void ForwardTransform5(float const l[HANGPRINTER_MAX_ANCHORS], Point &machinePos){
  float anch_prim[4][3]{{0.0}};
  float lineLengthsOriginSq[5]{0.0};
  float l_sq[5]{0.0};
  float k[4]{0.0};
  FixedMatrix<float, 3, 4> M[4];
  float machinePos_tmp[3]{0.0};

  for (size_t i{0}; i < 4; ++i) {
    for (size_t j{0}; j < 3; ++j) {
        anch_prim[i][j] = fiveAnchors[i][j] - fiveAnchors[4][j];
    }
  }

  for (size_t i{0}; i < 5; ++i) {
    lineLengthsOriginSq[i] = fiveLineLengthsOrigin[i]*fiveLineLengthsOrigin[i];
    l_sq[i] = l[i]*l[i];
  }

  for (size_t i{0}; i < 4; ++i) {
    k[i] = ((lineLengthsOriginSq[i] - lineLengthsOriginSq[4]) - (l_sq[i] - l_sq[4])) / 2.0;
  }

  for (size_t matrix_num{0}; matrix_num < 4; matrix_num++){
    for (size_t row{0}; row < 3; ++row) {
      size_t r = (row + matrix_num) % 4;
      for (size_t col{0}; col < 3; ++col) {
          M[matrix_num](row, col) = anch_prim[r][col];
      }
      M[matrix_num](row, 3) = k[r];
    }
  }

  // Solve the four systems
  size_t used{0};
  for (int k = 0; k < 4; ++k) {
    if (not singular_3x3(M[k])) {
      used++;
      M[k].GaussJordan(3, 4);
      for (size_t i{0}; i < 3; ++i) {
        machinePos_tmp[i] += M[k](i, 3);
      }
    }
  }
  if (used != 0) {
    for (size_t i{0}; i < 3; ++i) {
      machinePos[i] = machinePos_tmp[i]/static_cast<float>(used);
    }
  }
}


void ForwardTransform4(float const a, float const b, float const c, float const d, Point &machinePos)
{
	// Anchor location norms
//	if (fabsf(anchors[D_AXIS][X_AXIS]) > 0.1F ||
//		fabsf(anchors[D_AXIS][Y_AXIS]) > 0.1F ||
//		fabsf(anchors[A_AXIS][X_AXIS]) > 0.1F ||
//		fabsf(anchors[B_AXIS][X_AXIS]) < 0.1F ||
//		fabsf(anchors[C_AXIS][X_AXIS]) < 0.1F ||
//		fabsf(anchors[A_AXIS][Y_AXIS]) < 0.1F) {
//		return;
//	}

  // Force the anchor location norms Ax=0, Dx=0, Dy=0
  // through a series of rotations.
  float bestError = 99999999999999999999999999.0;
  for (float perturb{0.0}; perturb < 0.1; perturb+=0.5) {
    float const La = a + perturb;
    float const Lb = b + perturb;
    float const Lc = c + perturb;
    float const Ld = d + perturb;
    float const x_angle = atan(fourAnchors[D_AXIS][Y_AXIS]/fourAnchors[D_AXIS][Z_AXIS]);
    float const rxt[3][3] = {{1, 0, 0}, {0, cosf(x_angle), sinf(x_angle)}, {0, -sinf(x_angle), cosf(x_angle)}};
    float anchors_tmp0[4][3] = { 0 };
    for (size_t row{0}; row < 4; ++row) {
      for (size_t col{0}; col < 3; ++col) {
        anchors_tmp0[row][col] = rxt[0][col]*fourAnchors[row][0] + rxt[1][col]*fourAnchors[row][1] + rxt[2][col]*fourAnchors[row][2];
      }
    }
    float const y_angle = atan(-anchors_tmp0[D_AXIS][X_AXIS]/anchors_tmp0[D_AXIS][Z_AXIS]);
    float const ryt[3][3] = {{cosf(y_angle), 0, -sinf(y_angle)}, {0, 1, 0}, {sinf(y_angle), 0, cosf(y_angle)}};
    float anchors_tmp1[4][3] = { 0 };
    for (size_t row{0}; row < 4; ++row) {
      for (size_t col{0}; col < 3; ++col) {
        anchors_tmp1[row][col] = ryt[0][col]*anchors_tmp0[row][0] + ryt[1][col]*anchors_tmp0[row][1] + ryt[2][col]*anchors_tmp0[row][2];
      }
    }
    float const z_angle = atan(anchors_tmp1[A_AXIS][X_AXIS]/anchors_tmp1[A_AXIS][Y_AXIS]);
    float const rzt[3][3] = {{cosf(z_angle), sinf(z_angle), 0}, {-sinf(z_angle), cosf(z_angle), 0}, {0, 0, 1}};
    for (size_t row{0}; row < 4; ++row) {
      for (size_t col{0}; col < 3; ++col) {
        anchors_tmp0[row][col] = rzt[0][col]*anchors_tmp1[row][0] + rzt[1][col]*anchors_tmp1[row][1] + rzt[2][col]*anchors_tmp1[row][2];
      }
    }

    const float Asq = fsquare(fourLineLengthsOrigin[A_AXIS]);
    const float Bsq = fsquare(fourLineLengthsOrigin[B_AXIS]);
    const float Csq = fsquare(fourLineLengthsOrigin[C_AXIS]);
    const float Dsq = fsquare(fourLineLengthsOrigin[D_AXIS]);
    const float aa = fsquare(La);
    const float dd = fsquare(Ld);
    const float k0b = (-fsquare(Lb) + Bsq - Dsq + dd) / (2.0 * anchors_tmp0[B_AXIS][X_AXIS]) + (anchors_tmp0[B_AXIS][Y_AXIS] / (2.0 * anchors_tmp0[A_AXIS][Y_AXIS] * anchors_tmp0[B_AXIS][X_AXIS])) * (Dsq - Asq + aa - dd);
    const float k0c = (-fsquare(Lc) + Csq - Dsq + dd) / (2.0 * anchors_tmp0[C_AXIS][X_AXIS]) + (anchors_tmp0[C_AXIS][Y_AXIS] / (2.0 * anchors_tmp0[A_AXIS][Y_AXIS] * anchors_tmp0[C_AXIS][X_AXIS])) * (Dsq - Asq + aa - dd);
    const float k1b = (anchors_tmp0[B_AXIS][Y_AXIS] * (anchors_tmp0[A_AXIS][Z_AXIS] - anchors_tmp0[D_AXIS][Z_AXIS])) / (anchors_tmp0[A_AXIS][Y_AXIS] * anchors_tmp0[B_AXIS][X_AXIS]) + (anchors_tmp0[D_AXIS][Z_AXIS] - anchors_tmp0[B_AXIS][Z_AXIS]) / anchors_tmp0[B_AXIS][X_AXIS];
    const float k1c = (anchors_tmp0[C_AXIS][Y_AXIS] * (anchors_tmp0[A_AXIS][Z_AXIS] - anchors_tmp0[D_AXIS][Z_AXIS])) / (anchors_tmp0[A_AXIS][Y_AXIS] * anchors_tmp0[C_AXIS][X_AXIS]) + (anchors_tmp0[D_AXIS][Z_AXIS] - anchors_tmp0[C_AXIS][Z_AXIS]) / anchors_tmp0[C_AXIS][X_AXIS];

    float machinePos_tmp0[3];
    machinePos_tmp0[Z_AXIS] = (k0b - k0c) / (k1c - k1b);
    machinePos_tmp0[X_AXIS] = k0c + k1c * machinePos_tmp0[Z_AXIS];
    machinePos_tmp0[Y_AXIS] = (Asq - Dsq - aa + dd) / (2.0 * anchors_tmp0[A_AXIS][Y_AXIS]) + ((anchors_tmp0[D_AXIS][Z_AXIS] - anchors_tmp0[A_AXIS][Z_AXIS]) / anchors_tmp0[A_AXIS][Y_AXIS]) * machinePos_tmp0[Z_AXIS];

    //// Rotate machinePos_tmp back to original coordinate system
    float machinePos_tmp1[3];
    for (size_t row{0}; row < 3; ++row) {
      machinePos_tmp1[row] = rzt[row][0]*machinePos_tmp0[0] + rzt[row][1]*machinePos_tmp0[1] + rzt[row][2]*machinePos_tmp0[2];
    }
    for (size_t row{0}; row < 3; ++row) {
      machinePos_tmp0[row] = ryt[row][0]*machinePos_tmp1[0] + ryt[row][1]*machinePos_tmp1[1] + ryt[row][2]*machinePos_tmp1[2];
    }
    for (size_t row{0}; row < 3; ++row) {
      machinePos_tmp1[row] = rxt[row][0]*machinePos_tmp0[0] + rxt[row][1]*machinePos_tmp0[1] + rxt[row][2]*machinePos_tmp0[2];
    }

    float lineLengthA = sqrtf(fsquare(fourAnchors[A_AXIS][0] - machinePos_tmp1[X_AXIS]) + fsquare(fourAnchors[A_AXIS][1] - machinePos_tmp1[Y_AXIS]) + fsquare(fourAnchors[A_AXIS][2] - machinePos_tmp1[Z_AXIS]));
    float lineLengthB = sqrtf(fsquare(fourAnchors[B_AXIS][0] - machinePos_tmp1[X_AXIS]) + fsquare(fourAnchors[B_AXIS][1] - machinePos_tmp1[Y_AXIS]) + fsquare(fourAnchors[B_AXIS][2] - machinePos_tmp1[Z_AXIS]));
    float lineLengthC = sqrtf(fsquare(fourAnchors[C_AXIS][0] - machinePos_tmp1[X_AXIS]) + fsquare(fourAnchors[C_AXIS][1] - machinePos_tmp1[Y_AXIS]) + fsquare(fourAnchors[C_AXIS][2] - machinePos_tmp1[Z_AXIS]));
    float lineLengthD = sqrtf(fsquare(fourAnchors[D_AXIS][0] - machinePos_tmp1[X_AXIS]) + fsquare(fourAnchors[D_AXIS][1] - machinePos_tmp1[Y_AXIS]) + fsquare(fourAnchors[D_AXIS][2] - machinePos_tmp1[Z_AXIS]));
    float error = fsquare(lineLengthA - a) + fsquare(lineLengthB - b) +fsquare(lineLengthC - c) + fsquare(lineLengthD - d);
    if (error < bestError) {
      bestError = error;
      machinePos[X_AXIS] = machinePos_tmp1[X_AXIS];
      machinePos[Y_AXIS] = machinePos_tmp1[Y_AXIS];
      machinePos[Z_AXIS] = machinePos_tmp1[Z_AXIS];
      // std::cout << "using perturb: " << perturb << "  error is: " << bestError <<'\n';
    }
  }
}

void ForwardTransform(float distances[HANGPRINTER_MAX_ANCHORS], Point &machinePos, size_t numAnchors){
  if (numAnchors == 4) {
    ForwardTransform4(distances[0], distances[1], distances[2], distances[3], machinePos);
  } else {
    ForwardTransform5(distances, machinePos);
    //std::cout << "[" << distances[0]
    //          << ", " << distances[1]
    //          << ", " << distances[2]
    //          << ", " << distances[3]
    //          << ", " << distances[4] << "]\n";
  }
}


void MotorStepsToCartesian(std::array<float, HANGPRINTER_MAX_ANCHORS> const motorPos, Point &machinePos, size_t numAnchors)
{
  Flex const flex = makeFlex(numAnchors);
  auto const origins = originLengths(numAnchors);

  std::array<float, HANGPRINTER_MAX_ANCHORS> targetSteps{};
  float distances[HANGPRINTER_MAX_ANCHORS]{0.0};

  for (size_t i{0}; i < numAnchors; ++i) {
    targetSteps[i] = motorPos[i] * stepsPerDegree;
    distances[i] = flex.MotorPosToLinePos(targetSteps[i], i) + origins[i];
  }

  // First pass: rigid assumption to get a decent seed
  ForwardTransform(distances, machinePos, numAnchors);

  // Second pass: invert the flex-aware motor model with a damped LSQ solve
  refineWithLevenbergMarquardt(machinePos, flex, targetSteps, numAnchors);
}

void MotorStepsToCartesianNoFlex(std::array<float, HANGPRINTER_MAX_ANCHORS> const motorPos, Point &machinePos, size_t const numAnchors)
{
  float distances[HANGPRINTER_MAX_ANCHORS]{0.0};
  Flex const flex = makeFlex(numAnchors);
  auto const origins = originLengths(numAnchors);

  for(size_t i{0}; i < numAnchors; ++i){
    distances[i] = flex.MotorPosToLinePos(motorPos[i]*stepsPerDegree, i);
    distances[i] += origins[i];
  }
  ForwardTransform(distances, machinePos, numAnchors);
}

int main(){
  Point machinePos = { 0 };
  size_t constexpr numAnchors = 5;
  std::vector<std::array<float, numAnchors>> motorPoss = {
       // No flex float
       //{  -369.97133086,   -375.66959393,    316.34560633,    321.94609589,   1077.75975713},
       //{ -5759.01057516,  12615.08424755,  12275.59850843,  -6281.97726753, -13229.98337407},
       //{ -2060.97219543,  15716.33224264,  15602.69034285,  -2227.35643322, -27381.62276038},
       //{  1442.4251883 ,  10668.27508761,   1463.3685043 , -10512.72933158,   1849.83037262},
       //{  3159.98245383,  11763.18458403,   3277.83725587,  -7486.81252308, -17322.05908512},
       //{  7291.94468887,  14266.33816963,   6743.79551879,  -1553.68867173, -36100.64439146},
       //{ 10090.57203732,  10675.889441  ,  -7542.39424108,  -8463.82172607,   6068.07267818},
       //{ 11669.77976994,  12405.8842912 ,  -5005.52677327,  -6115.64295821, -14676.39535396},
       //{ 16129.59499481,  16299.63450814,  -1127.9130486 ,  -1373.47262378, -29820.36571367},
       //{-10122.43960565,   1100.72500198,  10138.58559936,   1439.38883774,   3605.756548  },
       //{ -7527.74534992,   3176.81909782,  11551.45807825,   2977.18307984, -16695.41664699},
       //{ -2531.87450143,   7228.3147109 ,  14207.41418159,   5938.05560439, -33823.45808762},
       //{ -7398.16588201,  10779.40735575,  10135.92434965,  -8409.43737118,   3330.02537493},
       //{  2207.56498303,   2107.80296987,    860.35188277,    963.14232699, -17745.30959353},
       //{  6582.41640758,   5361.04628015,   5608.31555101,   6823.30727184, -40219.08056667},
       //{  9809.00939173,    977.4905076 ,  -9827.85176463,   1377.48954933,   4119.45515998},
       //{ 11655.94023884,   3460.42828979,  -7629.63832939,   2739.08083977, -16597.3181957 },
       //{ 13385.27733777,   6915.40099512,  -2118.11173865,   5637.00997696, -33863.99419124},
       //{ -8108.90080311,  -7342.20860536,  10353.26565632,   9859.27771014,   3267.94097714},
       //{ -6116.93384181,  -5267.96158535,  12609.82866003,  12052.1408033 , -14452.27937288},
       //{ -2290.52818859,  -1396.80907196,  15701.50047006,  15083.33018867, -28278.36387567},
       //{   734.77431785,  -9739.7352902 ,   1582.85973463,   9727.42550404,   3997.73874831},
       //{  2473.96096152,  -7017.4007458 ,   3574.18124601,  11142.58224923, -17338.278132  },
       //{  6168.36875155,  -2680.22457923,   7234.30238782,  14510.73989763, -33810.94743147},
       //{ 11178.91365375,  -7734.00545613,  -8529.47143446,  10680.97899928,   3344.9573659 },
       //{ 12148.6236308 ,  -6352.25541301,  -5979.22868207,  12390.52086778, -12429.90917227},
       //{ 14529.08862584,  -2070.20634574,   -749.1973495 ,  15459.74674515, -29542.51506619}
       // With flex max 120 target 20
       // This flex data is not completely perfect because it was generated by
       // auto-calibration-simulation-for-hangprinter, which has a slightly altered flex function,
       // given that it tries to predict flex in a dataset that was presumably collected while in torque mode,
       // which is different to planning flex compensation in a motion planner.
       // It should however, be very close, and we should see the flex based solution perform better
       // than the NoFlex solution.
       {  -369.47366890,   -375.18362935,    318.27641438,    323.88876891,   1076.68163402},
       { -5980.29607895,  12630.56774248,  12288.67350302,  -6506.18962476, -13348.06053879},
       { -2353.64278337,  15870.99892165,  15755.46225500,  -2522.71380922, -27477.49741579},
       {  1433.28912224,  10685.68516553,   1454.31086905, -10542.99803343,   1813.47411829},
       {  3112.53568603,  11779.51141393,   3231.55807349,  -7609.55533435, -17364.20990461},
       {  7221.24317502,  14331.47424095,   6657.95387927,  -1853.21287105, -36135.10874008},
       { 10106.78583366,  10693.29200030,  -7583.66231627,  -8505.26876850,   6010.84493879},
       { 11680.14400927,  12421.57492733,  -5209.71780789,  -6326.70127134, -14780.55685279},
       { 16309.54520647,  16482.81831770,  -1424.01387322,  -1674.11174073, -29909.99659944},
       {-10145.44532568,   1094.96136691,  10156.51947369,   1434.73615456,   3572.60726310},
       { -7640.99960502,   3134.81886277,  11567.99443696,   2933.32488052, -16735.05779004},
       { -2825.34894707,   7163.91339055,  14266.90144144,   5840.37464568, -33865.11200386},
       { -7448.08057312,  10796.70751776,  10151.74389637,  -8459.78905687,   3271.18757879},
       {  2211.41905092,   2111.20882241,    858.18022603,    961.42598207, -17727.66189890},
       {  6585.49993026,   5349.29768666,   5599.61645498,   6829.22955353, -40179.35611831},
       {  9827.26906637,    973.41612678,  -9847.89764725,   1374.65365736,   4088.32777321},
       { 11672.37320821,   3419.76976053,  -7745.30493425,   2691.62201517, -16638.39998295},
       { 13421.03929287,   6834.73758758,  -2414.14469240,   5523.60605573, -33907.52745615},
       { -8153.62357389,  -7386.54442994,  10370.98726659,   9875.86705528,   3214.03835550},
       { -6347.85480184,  -5493.48884529,  12625.31736324,  12063.42984255, -14568.06462464},
       { -2585.49492187,  -1676.70121477,  15851.52670557,  15223.11520330, -28367.04177068},
       {   730.23171944,  -9759.54189457,   1580.93148691,   9745.76583762,   3967.14372473},
       {  2429.55959648,  -7123.34974121,   3539.66195357,  11159.52316296, -17372.69974680},
       {  6079.30436810,  -2972.78187516,   7172.81870678,  14578.78537233, -33851.97153806},
       { 11195.81863308,  -7791.69855043,  -8587.42819506,  10696.72883673,   3279.60983059},
       { 12162.75651477,  -6552.47720614,  -6177.61954587,  12406.22670936, -12538.57963324},
       { 14653.37178962,  -2366.53557957,  -1021.48237225,  15599.48126071, -29623.47054342}
  };

  std::vector<Point> expectedPoss = {
       {  17.9255679 ,  -17.6352632 ,  -27.497151  },
       {-523.885148  , -501.061599  ,  507.741532  },
       {-538.35251   , -530.255594  ,  953.040004  },
       {-545.189816  ,   -0.55803779,   26.4689901 },
       {-520.999273  ,   -3.27341154,  537.019049  },
       {-470.799477  ,   16.5498517 , 1047.17018   },
       {-505.362595  ,  467.529134  ,  -42.5453256 },
       {-513.234032  ,  464.122399  ,  533.480869  },
       {-538.29312   ,  526.060302  , 1030.39159   },
       {   8.98405788, -520.642383  ,  -26.6148823 },
       {  -5.52678502, -514.840431  ,  517.628213  },
       { -38.6000122 , -492.617793  ,  991.936222  },
       {-507.629503  , -465.97604   ,   31.7537695 },
       { -30.5601452 ,   35.9665477 ,  458.719776  },
       {  43.2898928 ,   28.8400594 , 1046.72968   },
       {  10.5874156 ,  504.407891  ,  -44.2588518 },
       { -19.9804722 ,  520.41744   ,  517.004705  },
       { -37.9958866 ,  454.266612  ,  975.222737  },
       { 455.930483  , -487.688259  ,   26.0515823 },
       { 482.712486  , -520.125685  ,  535.481389  },
       { 495.517745  , -539.35609   ,  965.228753  },
       { 500.096373  ,  -22.437457  ,  -42.0139801 },
       { 490.657689  ,  -30.4215669 ,  527.319817  },
       { 506.742907  ,  -31.9689121 ,  998.360926  },
       { 490.590662  ,  523.118645  ,   41.4092948 },
       { 517.848381  ,  501.639033  ,  482.878157  },
       { 525.399857  ,  459.863258  ,  981.665044  }
  };

  std::vector<Point> errsWithFlex{};
  std::vector<Point> errsNoFlex{};
  for (size_t i{0}; i < motorPoss.size(); ++i) {
    std::array<float, numAnchors> const motorPos = motorPoss[i];
    MotorStepsToCartesian(motorPos, machinePos, numAnchors);
    errsWithFlex.emplace_back(expectedPoss[i] - machinePos);
    MotorStepsToCartesianNoFlex(motorPos, machinePos, numAnchors);
    errsNoFlex.emplace_back(expectedPoss[i] - machinePos);
    //std::cout << machinePos << '\n';
  }


  float totErr = 0.0F;
  float totErrNoFlex = 0.0F;
  for (size_t i{0}; i < motorPoss.size(); ++i) {
    totErr += std::abs(errsWithFlex[i][0]);
    totErr += std::abs(errsWithFlex[i][1]);
    totErr += std::abs(errsWithFlex[i][2]);
    totErrNoFlex += std::abs(errsNoFlex[i][0]);
    totErrNoFlex += std::abs(errsNoFlex[i][1]);
    totErrNoFlex += std::abs(errsNoFlex[i][2]);
    //std::cout << std::abs(errsWithFlex[i][0]) - std::abs(errsNoFlex[i][0])
    //          << ", "
    //          << std::abs(errsWithFlex[i][1]) - std::abs(errsNoFlex[i][1])
    //          << ", "
    //          << std::abs(errsWithFlex[i][2]) - std::abs(errsNoFlex[i][2])
    //          << "\n";
    std::cout << errsWithFlex[i] << '\n';
  }
  std::cout << "With flex: " << totErr/motorPoss.size() << '\n';
  std::cout << "No flex: " << totErrNoFlex/motorPoss.size() << '\n';
  return 0;
}
