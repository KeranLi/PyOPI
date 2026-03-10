# Legacy Examples

This directory contains older example scripts from earlier versions of the OPI package.

## Status

These scripts are kept for reference but may not work with the current package structure.

## Recommended Examples

For up-to-date examples, see the parent directory:
- `examples/simple_example.py` - Basic usage
- `examples/comprehensive_example.py` - Full features

## Legacy Files

| File | Original Purpose |
|------|------------------|
| `functionality_comparison.py` | Compare MATLAB vs Python |
| `plot_results.py` | Plotting utilities |
| `reproduce_with_real_data.py` | Real data processing |
| `run_demo_simulation.py` | Demo simulation |
| `run_simple_simulation.py` | Simple simulation |

## Migration

These old scripts used the flat module structure. To migrate:

```python
# OLD (legacy)
from base_state import base_state
from precipitation_grid import precipitation_grid

# NEW (current)
from opi.core import base_state, precipitation_grid
```

See `examples/simple_example.py` for the correct modern usage.
