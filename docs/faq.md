# Frequently Asked Questions (FAQ)

## General Questions

### What are Bézier random variables?

Bézier random variables are a flexible class of probability distributions defined using Bézier curves. They allow you to model complex probability distributions by specifying control points that define the shape of the cumulative distribution function (CDF).

**Key advantages:**
- **Flexibility**: Can approximate many common distributions
- **Computational efficiency**: Fast evaluation and sampling

### When should I use bezierv instead of standard distributions?

Use bezierv when:

✅ **Your data doesn't fit standard distributions well**
- Complex multimodal data
- Unusual skewness or tail behavior
- Domain-specific constraints

✅ **You need flexible modeling**
- Scenario analysis with varying shapes
- Fitting distributions to expert knowledge
- Custom risk modeling

❌ **Don't use bezierv when:**
- Simple distributions (normal, exponential) fit well
- You need well-established statistical properties
- Computational resources are extremely limited

## Technical Questions

### How many control points should I use?

The number of control points (n+1) controls the flexibility of your distribution:

- **n=2-3**: Simple unimodal distributions
- **n=4-6**: Most real-world applications 
- **n=7-10**: Complex multimodal distributions
- **n>10**: Very complex shapes (risk of overfitting)

**Rule of thumb**: Start with n=5 and adjust based on fit quality and complexity needs.

```python
# Test different complexities
for n in [3, 5, 7, 10]:
    fitter = DistFit(data, n=n)
    _, mse = fitter.fit()
    print(f"n={n}: MSE = {mse:.6f}")
```

### Which optimization algorithm should I choose?

**Recommendation**: Start with `projgrad`. Switch to `nonlinear` if you need better fits.

### How do I know if my fit is good?

Check multiple criteria:

**1. Visual Inspection:**
```python
# Plot CDF and PDF comparison
bezier_rv.plot_cdf(data)
bezier_rv.plot_pdf()
```

**2. Quantitative Metrics:**
```python
# Mean Squared Error (lower is better)
_, mse = fitter.fit()
print(f"MSE: {mse:.6f}")
```

### How do I generate synthetic data?

```python
# Generate samples from fitted distribution
synthetic_data = bezier_rv.random(n_samples=1000, rng=42)
```

## Performance Questions

### How can I speed up fitting?

**1. Reduce data size:**
```python
# Use a representative sample
if len(data) > 10000:
    sample_data = np.random.choice(data, 10000, replace=False)
    fitter = DistFit(sample_data, n=5)
```

**2. Use faster algorithms:**
```python
# projgrad is usually fastest
fitter.fit(method="projgrad")
```

## Still have questions?

- 💬 [GitHub Discussions](https://github.com/EstebanLeiva/bezierv/discussions) - Ask the community
- 🐛 [Issues](https://github.com/EstebanLeiva/bezierv/issues) - Report bugs or request features  
- 📧 Email: esteban.leiva@uniandes.edu.co
- 📚 [Documentation](https://estebanleiva.github.io/bezierv/) - Comprehensive guides and API reference
