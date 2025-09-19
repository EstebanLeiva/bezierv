# Tutorials

This section provides step-by-step tutorials for common use cases with bezierv.

## Tutorial 1: Fitting Your First Bézier Distribution

### Prerequisites
- Basic Python knowledge
- Familiarity with numpy and matplotlib
- Understanding of probability distributions

### Objective
Learn how to fit a Bézier distribution to real data and interpret the results.

### Step 1: Prepare Your Data

```python
import numpy as np
import matplotlib.pyplot as plt
from bezierv.classes.distfit import DistFit

# Example: Customer service wait times (right-skewed data)
np.random.seed(42)
wait_times = np.random.gamma(2, 3, 1000)  # Shape=2, Scale=3

# Visualize raw data
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(wait_times, bins=30, alpha=0.7, density=True)
plt.title("Raw Data Histogram")
plt.xlabel("Wait Time (minutes)")
plt.ylabel("Density")

plt.subplot(1, 2, 2)
sorted_data = np.sort(wait_times)
empirical_cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
plt.plot(sorted_data, empirical_cdf, 'o-', markersize=2)
plt.title("Empirical CDF")
plt.xlabel("Wait Time (minutes)")
plt.ylabel("Cumulative Probability")
plt.tight_layout()
plt.show()
```

### Step 2: Choose Number of Control Points

The number of control points (n+1) determines the flexibility of your distribution:

```python
# Test different complexities
n_values = [3, 5, 7, 10]
mse_values = []

for n in n_values:
    fitter = DistFit(wait_times, n=n)
    _, mse = fitter.fit(method="projgrad")
    mse_values.append(mse)
    print(f"n={n}: MSE = {mse:.6f}")

# Plot MSE vs complexity
plt.figure(figsize=(8, 5))
plt.plot(n_values, mse_values, 'o-')
plt.xlabel("Number of Control Points (n)")
plt.ylabel("Mean Squared Error")
plt.title("Model Complexity vs Fit Quality")
plt.grid(True, alpha=0.3)
plt.show()

# Choose optimal n (balance between fit and complexity)
optimal_n = n_values[np.argmin(mse_values)]
print(f"Optimal n: {optimal_n}")
```

### Step 3: Fit the Distribution

```python
# Fit with optimal parameters
fitter = DistFit(wait_times, n=optimal_n)
bezier_rv, mse = fitter.fit(method="projgrad")

print(f"Final MSE: {mse:.6f}")
print(f"Control points (x): {bezier_rv.controls_x}")
print(f"Control points (z): {bezier_rv.controls_z}")
```

### Step 4: Validate the Fit

```python
# Compare fitted vs empirical distribution
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# CDF comparison
bezier_rv.plot_cdf(wait_times, ax=axes[0,0])
axes[0,0].set_title("CDF Comparison")

# PDF comparison
bezier_rv.plot_pdf(ax=axes[0,1])
axes[0,1].hist(wait_times, bins=30, alpha=0.5, density=True, 
               label="Empirical")
axes[0,1].set_title("PDF Comparison")
axes[0,1].legend()

# Q-Q plot
from scipy import stats
theoretical_quantiles = np.linspace(0.01, 0.99, 100)
empirical_quantiles = np.percentile(wait_times, theoretical_quantiles * 100)
bezier_quantiles = [bezier_rv.quantile(q) for q in theoretical_quantiles]

axes[1,0].plot(empirical_quantiles, bezier_quantiles, 'o', alpha=0.6)
axes[1,0].plot([min(empirical_quantiles), max(empirical_quantiles)], 
               [min(empirical_quantiles), max(empirical_quantiles)], 'r--')
axes[1,0].set_xlabel("Empirical Quantiles")
axes[1,0].set_ylabel("Bézier Quantiles")
axes[1,0].set_title("Q-Q Plot")

# Residuals
x_test = np.linspace(min(wait_times), max(wait_times), 100)
empirical_cdf_interp = np.interp(x_test, sorted_data, empirical_cdf)
bezier_cdf = [bezier_rv.cdf_x(x) for x in x_test]
residuals = np.array(bezier_cdf) - empirical_cdf_interp

axes[1,1].plot(x_test, residuals, 'o-', markersize=3)
axes[1,1].axhline(y=0, color='r', linestyle='--')
axes[1,1].set_xlabel("Wait Time")
axes[1,1].set_ylabel("CDF Residuals")
axes[1,1].set_title("Fit Residuals")

plt.tight_layout()
plt.show()
```

### Step 5: Use the Fitted Distribution

```python
# Calculate statistics
mean_wait = bezier_rv.get_mean()
q50 = bezier_rv.quantile(0.5)   # Median
q90 = bezier_rv.quantile(0.9)   # 90th percentile
q95 = bezier_rv.quantile(0.95)  # 95th percentile

print(f"Expected wait time: {mean_wait:.2f} minutes")
print(f"Median wait time: {q50:.2f} minutes")
print(f"90% of customers wait less than: {q90:.2f} minutes")
print(f"95% of customers wait less than: {q95:.2f} minutes")

# Probability calculations
prob_long_wait = 1 - bezier_rv.cdf_x(10)  # P(wait > 10 minutes)
prob_short_wait = bezier_rv.cdf_x(2)       # P(wait ≤ 2 minutes)

print(f"Probability of waiting > 10 minutes: {prob_long_wait:.3f}")
print(f"Probability of waiting ≤ 2 minutes: {prob_short_wait:.3f}")

# Generate synthetic data
synthetic_wait_times = bezier_rv.random(10000, rng=123)
print(f"Generated {len(synthetic_wait_times)} synthetic wait times")
```

---

## Tutorial 2: Convolution - Modeling Sums of Random Variables

### Objective
Learn how to model the sum of two independent random variables using convolution.

### Scenario: Total Project Time
Imagine a project with two phases, each with uncertain durations.

```python
import numpy as np
from bezierv.classes.distfit import DistFit
# Phase 1: Development time (Gamma distribution)
dev_times = np.random.gamma(3, 2, 1000)  # Mean ≈ 6 days

# Phase 2: Testing time (Log-normal distribution)  
test_times = np.random.lognormal(1, 0.5, 1000)  # Right-skewed

# Fit Bézier distributions to each phase
dev_fitter = DistFit(dev_times, n=5)
dev_rv, _ = dev_fitter.fit(method="projgrad")

test_fitter = DistFit(test_times, n=5)
test_rv, _ = test_fitter.fit(method="projgrad")

print("Phase distributions fitted successfully")
```

### Monte Carlo Convolution

```python
from bezierv.classes.convolver import Convolver

# Create convolver
convolver = Convolver([dev_rv, test_rv])

# Method 1: Monte Carlo (fast)
total_time_mc = convolver.convolve(n_sims=100, rng=42, n=6)

print(f"Monte Carlo convolution completed")
print(f"Expected total time: {total_time_mc.get_mean():.2f} days")
```

## Tutorial 3: Interactive Bézier Editor

### Objective
Learn to use the interactive tool for hands-on exploration of Bézier distributions.

### Step 1: Basic Interactive Session

Create a file called `interactive_demo.py`:

```python
from bezierv.classes.bezierv import InteractiveBezierv
from bokeh.plotting import curdoc

# Start with a simple uniform distribution
controls_x = [0, 1, 2, 3]
controls_z = [0, 0.33, 0.67, 1]

# Create the interactive editor
editor = InteractiveBezierv(controls_x, controls_z)

# Add to Bokeh document
curdoc().add_root(editor.layout)
curdoc().title = "Bézier Distribution Explorer"
```

Run with: `bokeh serve --show interactive_demo.py`

### Step 2: Interactive Features

In the browser interface, you can:

1. **Drag control points** to see real-time changes
2. **Add points** by clicking in empty areas
3. **Delete points** by dragging them off the plot
4. **Edit coordinates** precisely in the data table
5. **Download configurations** as CSV files

## Next Steps

- Explore the [API Reference](reference.md) for detailed function documentation  

---

!!! tip "Pro Tips"
    - Always visualize your data before fitting
    - Start with simple models (low n) and increase complexity as needed