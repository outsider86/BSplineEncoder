"""Fit B-spline control points on an explicitly supplied knot vector."""

import numpy as np

from spline_encoder import fit_bspline, make_clamped_knots


timestamps = np.linspace(0.0, 2.0, 41)
actions = np.stack((
    np.sin(timestamps),
    np.cos(timestamps),
    0.25 * timestamps,
), axis=1)

# Breakpoints describe span boundaries. The helper repeats both endpoints to
# produce the full knot vector expected by SciPy and fit_bspline.
knots = make_clamped_knots([0.0, 0.5, 1.0, 1.5, 2.0], degree=3)
fit = fit_bspline(actions, timestamps, knots, degree=3)

print("controls:", fit.control_points.shape)
print("rank:", fit.design_rank, "/", fit.num_basis)
print("RMSE:", fit.rmse)
query_times, reconstructed_actions = fit.sample(101)
print(query_times.shape, reconstructed_actions.shape)
