# OPI Utility Scripts

This directory contains utility scripts for OPI package maintenance and development.

## Available Scripts

### verify_structure.py
Verifies that the OPI package structure is correct and all modules are importable.

**Usage**:
```bash
python scripts/verify_structure.py
```

**Checks**:
- All core modules exist and are importable
- All submodules (models, solvers, io, utils, visualization, parallel) work
- Package can be imported without errors

### cleanup_old_files.py
Cleans up deprecated files that have been replaced by the new modular structure.

**Usage**:
```bash
python scripts/cleanup_old_files.py
```

**WARNING**: Only run this after confirming the new structure works correctly!
This removes old files that are no longer needed.

## For Developers

These scripts are primarily for development and maintenance:
- Use `verify_structure.py` after making changes to ensure nothing is broken
- Use `cleanup_old_files.py` once when transitioning from old to new structure
