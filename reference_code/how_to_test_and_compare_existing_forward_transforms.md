• Quick take: the three solvers differ a lot in generality and robustness. motorstepstocartesiantest_nice.cpp (interval + LM) is the most broadly applicable and noise-tolerant; motorstepstocartesiantest_quadratic.cpp (Halley/LM) is the fastest but needs
  a good seed and clean data; motorstepstocartesiantest.cpp (Pott 2015 linear solve + LM) sits in the middle but is hard-wired for 4–5 anchors and assumes a convenient anchor frame.

  Key observations (code refs)

  - motorstepstocartesiantest.cpp: Closed-form sphere intersection for 4 anchors via axis rotations and for 5 by averaging 4-anchor solves; then a damped LM flex-aware refine. Limited to 4/5 anchors and assumes the geometry can be rotated to Ax=Dx=Dy=0,
    so special-casing is baked in. No explicit outlier handling; least-squares happens only in the LM refine. Should be reasonably fast; robustness depends on good geometry and small noise.
  - motorstepstocartesiantest_nice.cpp: Interval/bisection search (intervalSolve) shrinks a box with length tolerances, then optional LM refine. Handles any anchor count; global search helps with multimodal or inconsistent data and gives graceful
    fallbacks when lines are “too long/short.” Cost is higher (box splits up to 50k) but deterministic and tolerant.
  - motorstepstocartesiantest_quadratic.cpp: Halley/Newton variants (solveHalley, solveHybrid, solveLm) on the squared-length residual with analytic J/H. Very small iteration counts and no finite-diff Jacobian, so fastest. Needs a decent initial guess
    and well-conditioned anchors; no damping or interval guard means it can diverge or settle in a local minimum when data is inconsistent. Over-/under-actuation is handled only via least squares; no explicit feasibility checks.

  Suggested comparison test suite (automatable)

  1. Clean baseline: HP5 anchor set with synthetic motor steps from ground-truth poses (use Flex to generate lengths + steps, no noise). Metrics: positional MAE, iterations, and runtime. Expect: quadratic ≈ LM-fast with top accuracy; Pott OK if geometry
     fits; interval slower but solid.
  2. Randomized anchors (4–8) in a bounded cube, random poses: checks generality. Expect Pott to fail >5 anchors and be fragile when axes can’t be normalized; quadratic and interval still solve; interval wins on success rate.
  3. Gaussian noise on lengths (e.g., σ = 1–5 mm) and motor quantization: robustness. Expect interval to keep returning feasible boxes with small drift; quadratic may wander if seed is poor; Pott’s analytic seed may degrade, LM may partially recover.
  4. Systematic bias per line (+/- constant per anchor): mimics spool diameter or pretension error. Measure pose bias and residual norm. Expect interval to partially absorb via tolerance; quadratic to return biased LSQ solution; Pott to bias unless LM
     compensates.
  5. Infeasible mixes (one anchor length off by 2–5%): count how many runs report/flag infeasibility vs return garbage. Expect interval to reject or fall back cleanly; quadratic/Pott may return a pose with high residual.
  6. Degenerate geometry: anchors nearly coplanar (Slideprinter) and nearly co-linear (edge of workspace). Check stability and variance over noise. Expect interval to still bound; quadratic may be ill-conditioned (JTJ singular); Pott’s rotation-based
     solve likely poor.
  7. Underactuated Spidercam-like setup (four high anchors, gravity-held): feed consistent lengths plus small noise. Measure spread of returned Z. Expect all to show ambiguity; interval should produce a wide box (good signal of underconstraint);
     quadratic may oscillate; Pott may average inconsistent systems.
  8. Overconstrained 6–8 anchors: generate lengths from a pose, optionally add noise. Compare residual sums. Expect interval to give near-LSQ; quadratic to give fast LSQ; Pott not applicable.
  9. Flex inversion stress: generate with flex enabled and recover pose with/without flex compensation in each solver. Compare error improvements to see who benefits from the refine step.
  10. Performance microbench: run each solver over 1k poses (mixed cases above) and time total CPU plus mean iterations. Expect quadratic fastest, Pott next, interval slowest but predictable.

  What to look for in results

  - Success rate on noisy/infeasible cases (not just error when it converges).
  - Residual cost vs ground-truth pose error (to spot “good residual, bad pose” cases).
  - Sensitivity to seed: try origin vs a random offset vs averaging anchors.
  - Conditioning flags (e.g., JTJ singular in quadratic, empty box in interval, determinant check in Pott).

  Likely outcomes

  - Generality/robustness: interval + LM (nice) wins; it’s the best candidate for broad robot classes and bad data, at the cost of speed.
  - Performance: quadratic (Halley) wins when it converges; could be default if paired with a guard/fallback.
  - Simplicity: quadratic code is shortest and cleanest; Pott has special-case geometry logic and averaging tricks; interval adds stack management but is conceptually simple.

  Natural next steps

  1. Implement the test harness above (shared dataset generator + metrics logger) to gather hard numbers.
  2. Decide on a two-tier strategy: fast quadratic by default with interval fallback, or interval as primary for safety-critical moves.
