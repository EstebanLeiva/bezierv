<p align="center">
  <img src="docs/assets/logo.png" alt="bezierv logo" width="260"/>
</p>

<h1 align="center">bezierv</h1>
<p align="center">
  <em>Fit smooth Bézier random variables to empirical data</em>
</p>

<p align="center">
  <img alt="PyPI" src="https://img.shields.io/pypi/v/bezierv?style=flat-square">
  <img alt="Python" src="https://img.shields.io/pypi/pyversions/bezierv?style=flat-square">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-informational?style=flat-square">
  <a href="https://estebanleiva.github.io/bezierv/"><img alt="Docs" src="https://img.shields.io/badge/docs-online-brightgreen?style=flat-square"></a>
  <img alt="CI" src="https://img.shields.io/github/actions/workflow/status/EstebanLeiva/bezierv/ci.yml?style=flat-square">
</p>

---

## Why Bézier Random Variables?

Traditional parametric distributions (normal, exponential, etc.) can be too rigid for real-world data. **bezierv** bridges the gap between non-parametric and parametric approaches by using Bézier curves to create smooth, flexible distributions that can fit virtually any shape.

**Key advantages:**
- 📈 **Flexible**: Fit any continuous distribution shape
- 🎛️ **Controllable**: Intuitive control points for fine-tuning
- 🔄 **Composable**: Built-in convolution for sums of random variables
- ⚡ **Fast**: Multiple efficient fitting algorithms
- 🎨 **Visual**: Interactive tools for exploration

---

## Quick Start

### Installation

```bash
pip install bezierv
```

### Basic Usage

```python
import numpy as np
from bezierv import DistFit

# Generate or load your data
rng = np.random.default_rng(42)
data = rng.beta(2, 5, 1000)  # Example: skewed data

# Fit a Bézier distribution (MSE objective, projected-gradient solver)
fitter = DistFit(data, n=5)  # 5 control segments
bezier_rv, mse = fitter.fit(method='mse', algorithm='projgrad')

# Use the fitted distribution
samples = bezier_rv.random(10000)      # Generate new samples
q90 = bezier_rv.quantile(0.90)         # 90th percentile
mean = bezier_rv.mean()            # Distribution mean
prob = bezier_rv.cdf_x(0.5)            # P(X <= 0.5)

# Visualize the fit
bezier_rv.plot_cdf(data)  # Compare with empirical CDF
bezier_rv.plot_pdf()      # Show probability density
```

### Advanced: Convolution of Random Variables

```python
from bezierv import DistFit, Convolver

# Fit two distributions
rv1, _ = DistFit(data1, n=4).fit(method='mse', algorithm='projgrad')
rv2, _ = DistFit(data2, n=4).fit(method='mse', algorithm='projgrad')

# Compute their sum: Z = X + Y
convolver = Convolver([rv1, rv2])
sum_mc, _ = convolver.convolve(n_sims=10000, rng=42)
```

---

## Documentation

| Resource | Description |
|----------|-------------|
| [**📖 User Guide**](https://estebanleiva.github.io/bezierv/) | Complete tutorials and examples |
| [**🔧 API Reference**](https://estebanleiva.github.io/bezierv/reference/) | Detailed function documentation |
| [**🎮 Interactive Demo**](#interactive-tool) | Browser-based curve editor |
| [**📚 Tutorials**](https://estebanleiva.github.io/bezierv/tutorials/) | Step-by-step examples and guides |

---

## Interactive Tool

Launch an interactive Bézier curve editor in your browser:

```python
from bezierv.classes.bezierv import InteractiveBezierv
from bokeh.plotting import curdoc

# Create interactive editor
editor = InteractiveBezierv(
    controls_x=[0.0, 0.25, 0.75, 1.0],
    controls_z=[0.0, 0.1, 0.9, 1.0]
)

curdoc().add_root(editor.layout)
```

Then run: `bokeh serve --show your_app.py`

---

## Algorithms

bezierv includes multiple fitting algorithms optimized for different scenarios:

| Objective | Algorithm | Call |
|-----------|-----------|------|
| MSE | Projected Gradient | `method='mse', algorithm='projgrad'` |
| MSE | Nonlinear (IPOPT) | `method='mse', algorithm='nonlinear'` |
| MSE | Nelder-Mead | `method='mse', algorithm='neldermead'` |
| MLE | Primal Gradient | `method='mle'` |

`method='mse'` returns `(bezierv, mse)`. `method='mle'` returns `(bezierv, nll)`.



## Citation

If you use **bezierv** in your research, please cite the accompanying paper (forthcoming on arXiv):

```bibtex
@article{leiva2026bezierv,
  title   = {Computational Framework for {B\'{e}zier} Distributions},
  author  = {Leiva, Esteban and Medaglia, Andr\'{e}s L. and Zuluaga, Luis F.},
  year    = {2026},
  note    = {Forthcoming on arXiv}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
