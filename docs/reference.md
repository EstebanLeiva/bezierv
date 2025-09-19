# API Reference

This page provides comprehensive documentation for all bezierv classes, functions, and algorithms.

---

## Core Classes

### Bezierv - Main Distribution Class

The primary class for representing and working with Bézier random variables.

::: bezierv.classes.bezierv.Bezierv
    handler: python
    options:
      heading_level: 4
      show_root_heading: true
      show_source: false
      members_order: source
      group_by_category: true
      show_category_heading: true
      show_symbol_type_heading: true
      docstring_style: numpy
      separate_signature: true

??? example "Quick Example"
    ```python
    from bezierv.classes.bezierv import Bezierv
    import numpy as np
    
    # Create a simple Bézier distribution
    controls_x = [0, 1, 2, 3]
    controls_z = [0, 0.3, 0.8, 1]
    
    bezier_rv = Bezierv(3, controls_x, controls_z)
    
    # Evaluate PDF and CDF
    pdf_value = bezier_rv.pdf_x(1.5)
    cdf_value = bezier_rv.cdf_x(1.5)
    
    # Generate random samples
    samples = bezier_rv.random(100, rng=42)
    ```

---

### InteractiveBezierv - Interactive Editor

Interactive Bokeh-based editor for hands-on exploration of Bézier distributions.

::: bezierv.classes.bezierv.InteractiveBezierv
    handler: python
    options:
      heading_level: 4
      show_root_heading: true
      show_source: false
      members_order: source
      group_by_category: true
      show_category_heading: true
      docstring_style: numpy
      separate_signature: true

??? example "Interactive Session"
    ```python
    from bezierv.classes.bezierv import InteractiveBezierv
    from bokeh.plotting import curdoc
    
    # Start with uniform distribution
    controls_x = [0, 1, 2, 3]
    controls_z = [0, 0.33, 0.67, 1]
    
    editor = InteractiveBezierv(controls_x, controls_z)
    curdoc().add_root(editor.layout)
    
    # Run with: bokeh serve --show script.py
    ```

---

## Distribution Fitting

### DistFit - Fit Distributions to Data

Fits Bézier distributions to empirical data using various optimization algorithms.

::: bezierv.classes.distfit.DistFit
    handler: python
    options:
      heading_level: 4
      show_root_heading: true
      show_source: false
      members_order: source
      group_by_category: true
      show_category_heading: true
      docstring_style: numpy
      separate_signature: true

??? example "Fitting Example"
    ```python
    from bezierv.classes.distfit import DistFit
    import numpy as np
    
    # Generate sample data
    data = np.random.gamma(2, 3, 500)
    
    # Fit Bézier distribution
    fitter = DistFit(data, n=5)
    bezier_rv, mse = fitter.fit(method="projgrad")
    
    print(f"Fit quality (MSE): {mse:.6f}")
    
    # Visualize results
    bezier_rv.plot_cdf(data)
    bezier_rv.plot_pdf()
    ```

---

## Convolution Operations

### Convolver - Sum of Random Variables

Computes convolutions (sums) of independent Bézier random variables using Monte Carlo simulation or exact numerical integration.

::: bezierv.classes.convolver.Convolver
    handler: python
    options:
      heading_level: 4
      show_root_heading: true
      show_source: false
      members_order: source
      group_by_category: true
      show_category_heading: true
      docstring_style: numpy
      separate_signature: true

??? info "Method Comparison"
    We recommend using monte carlo convolution.

??? example "Convolution Example"
    ```python
    from bezierv.classes.convolver import Convolver
    from bezierv.classes.distfit import DistFit
    import numpy as np
    
    # Create two distributions
    data1 = np.random.gamma(2, 2, 1000)  # Task 1 duration
    data2 = np.random.lognormal(1, 0.5, 1000)  # Task 2 duration
    
    # Fit distributions
    rv1, _ = DistFit(data1, n=5).fit()
    rv2, _ = DistFit(data2, n=5).fit()
    
    # Compute convolution
    convolver = Convolver([rv1, rv2])
    
    total_mc = convolver.convolve(n_sims=1000, rng=42)
    
    print(f"Expected total time (MC): {total_mc.get_mean():.2f}")
    ```

---

## Optimization Algorithms

These algorithms are used internally by `DistFit` to optimize the fitting process. You typically don't need to call these directly.

### Projected Gradient Method

Fast and stable gradient-based optimization with projection onto the feasible region.

::: bezierv.algorithms.proj_grad
    handler: python
    options:
      heading_level: 4
      show_root_heading: true
      show_source: false
      members_order: source
      docstring_style: numpy
      separate_signature: true

!!! tip "When to Use"
    **Default choice** for most fitting problems. Provides good balance of speed and accuracy.

---

### Projected Subgradient Method

Memory-efficient optimization suitable for large datasets.

::: bezierv.algorithms.proj_subgrad
    handler: python
    options:
      heading_level: 4
      show_root_heading: true
      show_source: false
      members_order: source
      docstring_style: numpy
      separate_signature: true

---

### Nonlinear Optimization

Robust scipy-based nonlinear solver for complex fitting problems.

::: bezierv.algorithms.non_linear
    handler: python
    options:
      heading_level: 4
      show_root_heading: true
      show_source: false
      members_order: source
      docstring_style: numpy
      separate_signature: true

!!! tip "When to Use"
    Use when accuracy is more important.

---

### Nelder-Mead

Derivative-free optimization method that doesn't require gradients.

::: bezierv.algorithms.nelder_mead
    handler: python
    options:
      heading_level: 4
      show_root_heading: true
      show_source: false
      members_order: source
      docstring_style: numpy
      separate_signature: true

---

### Algorithm Utilities

Helper functions and utilities used by the optimization algorithms.

::: bezierv.algorithms.utils
    handler: python
    options:
      heading_level: 4
      show_root_heading: true
      show_source: false
      members_order: source
      docstring_style: numpy
      separate_signature: true

---

## See Also

- **[Quick Start Guide](index.md#quick-start)** - Get started in 5 minutes
- **[Tutorials](tutorials.md)** - Step-by-step learning
- **[FAQ](faq.md)** - Common questions and troubleshooting