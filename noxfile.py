# noxfile.py
import sys
import nox

# Which Python interpreters to exercise
PY_VERSIONS = ["3.10", "3.11", "3.12", "3.13"]

# Exact/pinned specs per Python (from your compatibility matrix)
PINNED = {
    "3.10": [
        "numpy>=1.26,<2",
        "scipy>=1.13,<1.16",
        "matplotlib>=3.9,<3.11",
        "statsmodels>=0.14.2,<0.15",
        "pyomo>=6.8,<7",
    ],
    "3.11": [
        "numpy>=1.26,<2.3",
        "scipy>=1.13,<1.17",
        "matplotlib>=3.9,<3.11",
        "statsmodels>=0.14.2,<0.15",
        "pyomo>=6.8,<7",
    ],
    "3.12": [
        "numpy>=2.1,<2.3",
        "scipy>=1.14.1,<1.17",
        "matplotlib>=3.9,<3.11",
        "statsmodels>=0.14.2,<0.15",
        "pyomo>=6.8,<7",
    ],
    "3.13": [
        "numpy>=2.2,<2.4",
        "scipy>=1.14.1,<1.17",
        "matplotlib>=3.9,<3.11",
        "statsmodels>=0.14.2,<0.15",
        "pyomo>=6.8,<7",
    ],
}

# Prefer conda if present; otherwise uv or virtualenv (last one is guaranteed).
# You can also override on the CLI:  nox -db conda   (see notes below)
@nox.session(python=PY_VERSIONS, venv_backend="conda|uv|virtualenv", reuse_venv=True)
def tests(session: nox.Session) -> None:
    """Install per-Python pins, then run pytest (or a smoke run if no tests)."""
    # Determine current interpreter key like '3.12'
    pykey = f"{sys.version_info.major}.{sys.version_info.minor}"
    pins = PINNED[pykey]

    # 1) Install your pins FIRST to force-resolve exact versions,
    # 2) then install your package (respects pyproject markers),
    # 3) then pytest.
    session.install(*pins)
    session.install("-e", ".")
    session.install("pytest")

    # Show what actually got installed
    session.run(
        "python",
        "-c",
        "import numpy, scipy, matplotlib, statsmodels, pyomo;"
        "import sys;"
        "print('Python', sys.version.split()[0]);"
        "print('numpy', numpy.__version__);"
        "print('scipy', scipy.__version__);"
        "print('matplotlib', matplotlib.__version__);"
        "print('statsmodels', statsmodels.__version__);"
        "print('pyomo', pyomo.__version__);",
    )

    # Run pytest; treat 'no tests collected' (exit code 5) as success
    session.run("pytest", "-q", success_codes=[0, 5])


# Make these the defaults if you just run `nox`
nox.options.sessions = [f"tests-{v}" for v in PY_VERSIONS]
# Fail fast if a requested interpreter is missing
nox.options.error_on_missing_interpreters = True