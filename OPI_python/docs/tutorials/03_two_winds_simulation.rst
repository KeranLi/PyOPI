Tutorial 3: Two Winds Simulation
================================

This tutorial covers the two-winds model for complex flow patterns.

When to Use Two Winds
---------------------

The two-winds model is useful when:

- Wind direction varies spatially
- Multiple wind regimes need to be combined
- Simple single-wind assumptions are insufficient

Using TwoWindsModel
-------------------

.. code-block:: python

    from opi.models import TwoWindModel, TwoWindsParameters
    
    model = TwoWindModel(topography)
    
    params = TwoWindsParameters(
        U1=10.0, azimuth1=270.0,  # First wind component
        U2=5.0, azimuth2=180.0,   # Second wind component
        T0=288.0,
        M=0.8
    )
    
    result = model.run(params)

Weighting Components
--------------------

The model combines results from both wind components based on 
their relative strengths.
