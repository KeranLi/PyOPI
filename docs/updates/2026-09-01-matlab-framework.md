# 2026/09/01 MATLAB Framework Update

The MATLAB portion of PyOPI was reorganized and expanded into a combined OPI
simulation and proxy data-assimilation framework.

## What changed

- Updated the OPI one-wind and two-wind simulation, fitting, mapping and
  private numerical functions under `OPI_matlab/OPI_programs/`.
- Added oxygen-only and oxygen-plus-clumped-temperature workflows.
- Added carbonate d18O, clumped-temperature and paleosol observation
  operators, chronology handling, likelihood evaluation and posterior
  weighting.
- Added assimilation configuration tables under `OPI_matlab/data/assimilation/`
  and observation inputs under `OPI_matlab/data/observations/`.
- Added MATLAB tests under `OPI_matlab/tests/`.
- Added the generic `OPI_matlab/run_assimilation.m` entry point. It accepts an
  external experiment directory containing `design/case_manifest.csv` and
  `calc_only/<case_id>/` OPI result files; no geological scenario is bundled.

## Apple Silicon parallel fitting

Parallel fitting uses MATLAB Parallel Computing Toolbox and local process
workers. The `.run` file's second non-comment line controls the mode:

```text
0   serial
1   parallel
```

The helper `OPI_matlab/OPI_programs/opiStartParallelPool.m` starts or resizes a
local pool. On Apple Silicon, four to eight workers are a reasonable starting
point:

```matlab
addpath('OPI_programs')
pool = opiStartParallelPool(6);
opiFit_TwoWinds('runs/run033.run')
delete(pool)
```

The bundled `runs/run033.run` example is configured with parallel mode enabled.
Serial OPI simulation remains available without the toolbox; only parallel
fitting requires it.

## Repository hygiene

Generated scenarios, result MAT files, logs, figures and MATLAB temporary files
are ignored by the repository `.gitignore`. Run definitions, source code,
configuration tables and tests remain version controlled.
