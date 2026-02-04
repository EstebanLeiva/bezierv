# Bezierv: Flexible Bézier Random Variables

<div align="center">
  <img src="assets/logo.png" alt="bezierv logo" width="200"/>
</div>

**bezierv** is a Python package for fitting, analyzing, and sampling from Bézier-based random variables. Bézier random variables can adapt to virtually any continuous distribution shape.

!!! tip "New to Bézier distributions?"
    Start with our [Quick Start Guide](#quick-start) for a hands-on introduction, or explore the [Interactive Demo](#interactive-visualization) to see Bézier curves in action.

---

## ✨ Key Features

- **🎯 Flexible Fitting**: Adapt to any continuous distribution shape
- **⚡ Multiple Algorithms**: Choose from 4 optimization methods
- **🔄 Convolution Support**: Compute sums of random variables exactly or via Monte Carlo  
- **🎮 Interactive Tools**: Browser-based curve editor with real-time updates
- **📊 Rich Visualization**: Built-in plotting for CDFs, PDFs, and control points
- **🔢 Statistical Functions**: Moments, quantiles, sampling, and probability calculations

---

## Quick Start

### Installation

Install bezierv using pip:

```bash
pip install bezierv
```

### Basic Example

Fit a Bézier distribution to your data in just a few lines:

```python
import numpy as np
from bezierv.classes.distfit import DistFit

# Generate sample data (replace with your own)
np.random.seed(42)
data = np.random.beta(2, 5, 1000)  # Skewed distribution

# Fit Bézier distribution
fitter = DistFit(data, n=5)  # 5 control segments (6 control points)
bezier_rv, mse = fitter.fit(method="projgrad")

print(f"Fit completed with MSE: {mse:.6f}")

# Use the fitted distribution
samples = bezier_rv.random(100)      # Generate new samples
mean = bezier_rv.get_mean()            # Compute mean
q90 = bezier_rv.quantile(0.90)         # 90th percentile
cdf_val = bezier_rv.cdf_x(0.5)         # P(X ≤ 0.5)

print(f"Mean: {mean:.3f}, 90% quantile: {q90:.3f}")
```

### Visualization

Compare your fitted distribution with the empirical data:

```python
import matplotlib.pyplot as plt

# Create side-by-side plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot CDF comparison
bezier_rv.plot_cdf(data, ax=ax1)
ax1.set_title("Cumulative Distribution Function")

# Plot PDF
bezier_rv.plot_pdf(ax=ax2)
ax2.set_title("Probability Density Function")

plt.tight_layout()
plt.show()
```

---

## 🎮 Interactive Visualization

Launch an interactive Bézier curve editor to explore how control points affect distribution shape:

```python
from bezierv.classes.bezierv import InteractiveBezierv
from bokeh.plotting import curdoc

# Define initial control points
controls_x = [0.0, 0.25, 0.75, 1.0]  # X-coordinates (domain)
controls_z = [0.0, 0.1, 0.9, 1.0]    # Z-coordinates (CDF values)

# Create interactive editor
editor = InteractiveBezierv(controls_x, controls_z)

# Launch in Bokeh server
curdoc().add_root(editor.layout)
curdoc().title = "Bézier Distribution Editor"
```

Save as `bezier_app.py` and run:
```bash
python -m bokeh serve --show bezier_app.py
```

This opens an interactive tool in your browser where you can:
- ✏️ **Edit control points** by clicking and dragging with the Point Draw Tool
- ➕ **Add/remove points** to change complexity (add with a click with the Point Draw Tool, delete with button)
- 📊 **View real-time updates** of both CDF and PDF
- 💾 **Export control points** as CSV

---

## 🔄 Convolution: Sums of Random Variables

### Monte Carlo Convolution (Fast)

```python
from bezierv.classes.convolver import Convolver

# Fit two separate distributions
data1 = np.random.gamma(2, 2, 1000)
data2 = np.random.exponential(1, 1000)

rv1 = DistFit(data1, n=4).fit()[0]
rv2 = DistFit(data2, n=4).fit()[0]

# Compute their sum via Monte Carlo
convolver = Convolver([rv1, rv2])
sum_rv = convolver.convolve(n_sims=10000, rng=42)

print(f"Sum mean: {sum_rv.get_mean():.3f}")
```

---

## 🔧 Fitting Algorithms

Choose the best algorithm for your use case:

| Algorithm | Method Call | 
|-----------|-------------|
| **Projected Gradient** | `method="projgrad"` | 
| **Projected Subgradient** | `method="projsubgrad"` | 
| **Nonlinear Optimization** | `method="nonlinear"` |
| **Nelder-Mead** | `method="neldermead"` |

### Algorithm Comparison Example

```python
methods = ["projgrad", "projsubgrad", "nonlinear", "neldermead"]
results = {}

for method in methods:
    fitter = DistFit(data, n=5)
    bz, mse = fitter.fit(method=method, max_iter_PG=1000)
    results[method] = {"mse": mse, "mean": bz.get_mean()}
    print(f"{method:12s}: MSE = {mse:.6f}, Mean = {bz.get_mean():.6f}")
```

---

## 📊 Advanced Examples

### Multi-Modal Distributions

Fit complex, multi-modal distributions:

```python
# Create bimodal data
data_bimodal = np.concatenate([
    np.random.normal(2, 0.5, 500),    # First mode
    np.random.normal(8, 0.8, 500)     # Second mode
])

# Use more control points for complex shapes
fitter = DistFit(data_bimodal, n=10)
bimodal_rv, mse = fitter.fit(method="nonlinear")

# Visualize the complex fit
bimodal_rv.plot_pdf()
```

---

## 🎯 Best Practices

### Choosing the Number of Control Points

- **Simple data**: n=3-5 (few parameters, fast fitting)
- **Complex shapes**: n=6-10 (more flexibility)  
- **Multi-modal**: n=8-15 (capture multiple peaks)

!!! warning "Overfitting"
    More control points ≠ always better. Start simple and increase complexity only if needed.

### Algorithm Selection Guide

1. **Start with `projgrad`** - fastest and works well for most cases
2. **Try `nonlinear`** if you need highest accuracy and can afford to fail

### Performance Tips

```python
# For large datasets, consider subsampling for initial fit
if len(data) > 10000:
    subset = np.random.choice(data, 5000, replace=False)
    fitter = DistFit(subset, n=5)
    quick_fit, _ = fitter.fit(method="projgrad")
```

---

## 📚 Next Steps

- **[🔧 API Reference](reference.md)** - Complete function documentation
- **[📖 Tutorials](tutorials.md)** - Step-by-step learning with examples
- **[🐛 Issues](https://github.com/EstebanLeiva/bezierv/issues)** - Report bugs or request features

---