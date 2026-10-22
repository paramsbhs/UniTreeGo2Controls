# Failure log

What broke, how it was diagnosed, and what changed. Add an entry every time
something surprises you: the QP going infeasible, the estimator diverging, a
gain that oscillates. Include the log file and figure that show it.

---

## F-001 · Startup sag from ramping PD gains

**Phase:** 1 · **Date:** 2026-10-22

**Symptom.** After reset the base dropped from 27.0 cm to 22.3 cm within 0.3 s
before recovering, with a 2.7° pitch transient.

**Cause.** Smooth startup was implemented by scaling kp/kd from 0 → 1 over 0.5 s.
The robot starts from the `home` keyframe already standing, so for the first few
hundred milliseconds the legs had almost no stiffness under 150 N of body weight.

**Fix.** Keep full gains from t = 0 and instead blend the *target* pose from the
measured pose with a min-jerk profile (`StanceController.update`). Dip reduced to
~2 cm.

## F-002 · Steady-state PD sag under body weight

**Phase:** 1 · **Date:** 2026-10-22

**Symptom.** A remaining ~2 cm dip at startup and ~5 mm under/overshoot at each
height step, removed only slowly (≈1 s) by the height integrator.

**Diagnosis.** Logged calf torque at steady state was ≈6.5 N·m. Predicted static
knee torque for an equal share of the weight with the foot under the hip:
F · l₁ · sin θ_thigh = (15.2 · 9.81 / 4) · 0.213 · sin(0.95) ≈ 6.5 N·m. Match → the
PD was spending its position error to generate the weight-bearing torque.

**Fix.** Added the static-load knee feedforward τ = Jᵀf (`static_load_ff`). Startup
dip reduced to ~1 cm and per-foot loads became more even. The remaining transient
is the integrator settling; a proper fix is the WBC's contact-force distribution
(Phase 4).

**Takeaway for Phase 4.** Even the simplest controller needed a dynamics term to
behave. A pure joint-space PD is not a fair baseline unless it gets this
feedforward too.
