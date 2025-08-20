# Bezierv: Bézier Distributions for Bounded Data

Bezierv provides utilities to **fit, analyze, sample, and convolve** Bézier-based
random variables from empirical data. It includes multiple fitting algorithms
(projected gradient, projected subgradient, nonlinear (Pyomo/IPOPT), and Nelder–Mead),
plus helpers for plotting and Monte Carlo convolution.

> **Tip:** See the [API reference](reference.md) for auto-generated docs from the code.

---

## Quickstart

Install your package (editable mode recommended while developing):

```bash
pip install -e .
```

Fit a Bézier random variable to data with `DistFit` and sample from it:

```python
import numpy as np
from distfit import DistFit

# Synthetic bounded data
rng = np.random.default_rng(42)
data = rng.beta(2, 5, 1000)  # replace with your data

fitter = DistFit(data, n=5)            # n = degree (n control segments, n+1 control points)
bz, mse = fitter.fit(method="projgrad")  # or: 'nonlinear', 'projsubgrad', 'neldermead'
print("MSE:", mse)

samples = bz.random(10_000, rng=42)     # draw samples via inverse CDF
q90 = bz.quantile(0.90)                 # 90% quantile
```

Plot empirical vs. fitted CDF (optional):

```python
import matplotlib.pyplot as plt
bz.plot_cdf(data)       # overlays ECDF and Bézier CDF
plt.show()
```

---

## Convolution (sum of independent Bézier RVs)

Use `Convolver` to approximate the **sum** of several fitted distributions via
Monte Carlo, then fit another Bézier RV to the result:

```python
from convolver import Convolver
from distfit import DistFit

# Fit two Bezier RVs separately (bz1, bz2) ... then:
conv = Convolver([bz, bz])          # example: sum with itself
bz_sum = conv.convolve(n_sims=50_000, rng=123, method="projgrad", n=7)
```

---

## Fitting methods at a glance

- **Projected Gradient (`projgrad`)** – fast and simple; optimizes *z* controls with projection.
- **Projected Subgradient (`projsubgrad`)** – updates both *x* and *z* with projection.
- **Nonlinear (`nonlinear`)** – solves a constrained model via Pyomo (e.g., IPOPT).
- **Nelder–Mead (`neldermead`)** – derivative-free simplex search with penalties.

---

## Next steps

- Browse the **[API reference](reference.md)** for the full class and function docs.
- Configure `mkdocstrings` in `mkdocs.yml` (see the top of *reference.md* for a minimal snippet).