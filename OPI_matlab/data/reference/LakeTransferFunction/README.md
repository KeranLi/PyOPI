# LakeTransferFunction Reference Data

Source repository:
https://github.com/alexaterrazas/LakeTransferFunction

Local source paper in this project:
`ref/The Depositional Record - 2025 - Terrazas - Seasonal lake‐to‐air temperature transfer functions derived from an analysis of.pdf`

## Files

- `ERA5_LakeTemp.csv`: modern lake-surface-water-temperature and ERA5 climate dataset.

## Dataset Scope

The CSV contains 1395 lakes and 195 columns. Important variable groups:

- Lake metadata: `Hylak_id`, `center_long`, `center_lat`, `Lake_name`, `Country`, `Continent`, `Lake_type`
- Lake setting: `Lake_area`, `Depth_avg`, `Elevation`, `abs_lat`, `elevation_km`
- Lake surface water temperature: `lswt_ann_avg`, `lswt_ao_avg`, `lswt_amj_avg`, `lswt_jja_avg`, `lswt_warmest_avg`
- ERA5 air temperature: `tas_ann_avg`, `tas_ao_avg`, `tas_amj_avg`, `tas_jja_avg`, `tas_warmest_month`
- ERA5 climate fields: monthly `nearest_temp_*`, `nearest_tcc_*`, `nearest_rh_*`, `nearest_ssr_*`, `nearest_u10_*`

## Transfer Function Notation

The notebooks define transfer functions that estimate mean annual air temperature (`MAAT`, represented by `tas_ann_avg`) from lake surface water temperature (`Tw`).

Seasonal lake-temperature inputs:

- annual: `lswt_ann_avg`
- spring-through-summer AO: `lswt_ao_avg`
- spring AMJ: `lswt_amj_avg`
- summer JJA: `lswt_jja_avg`
- warmest month: `lswt_warmest_avg`

Latitude is absolute latitude in degrees. Elevation is in kilometers for the formula terms below.

## TF3: Tw, Tw^2, Latitude

```text
annual:
MAAT = -0.0402*Tw^2 + 2.5413*Tw - 0.0024*lat - 14.256

AO:
MAAT = -0.0141*Tw^2 + 1.6034*Tw - 0.1099*lat - 8.7553

AMJ:
MAAT = -0.0167*Tw^2 + 1.4745*Tw - 0.1068*lat - 3.0564

JJA:
MAAT = -0.0015*Tw^2 + 0.9189*Tw - 0.2738*lat + 1.4836

warmest month:
MAAT = -0.0067*Tw^2 + 1.1595*Tw - 0.3147*lat - 0.9535
```

## TF4: Tw, Tw^2, Latitude, Elevation

```text
annual:
MAAT = -0.0403*Tw^2 + 2.3890*Tw - 0.0767*lat - 0.7038*elev_km - 8.8227

AO:
MAAT = -0.0162*Tw^2 + 1.4369*Tw - 0.2282*lat - 1.5486*elev_km + 0.496

AMJ:
MAAT = -0.0172*Tw^2 + 1.2746*Tw - 0.2331*lat - 1.6609*elev_km + 6.1134

JJA:
MAAT = -0.0043*Tw^2 + 0.7775*Tw - 0.3937*lat - 2.3094*elev_km + 12.0188

warmest month:
MAAT = -0.0055*Tw^2 + 0.8336*Tw - 0.4307*lat - 2.4042*elev_km + 11.8427
```

## Relevance to OPI

This dataset does not directly convert carbonate clumped-isotope temperature to air temperature. It provides a lake-water-to-air-temperature transfer layer. For OPI assimilation, a conservative proxy chain is:

```text
carbonate clumped isotope temperature
  -> lake-water or carbonate-formation temperature
  -> lake-to-air transfer function
  -> air-temperature / lapse-rate constraint in OPI
```

Use these transfer functions as proxy-forward-model components, not as a hard assumption that clumped temperature equals mean annual air temperature.

## Toolbox-Free Warmest-Season Model

The project-specific MATLAB model is trained with:

```matlab
addpath('OPI_programs')
opiTrain_TerrazasWarmestML
```

Derived model and spatial-block validation outputs are written under
`data/derived/LakeTransferFunction/TerrazasWarmestML/`; the reference CSV is
never modified. The primary model is
`tas_warmest_month -> lswt_warmest_avg`, with absolute latitude, log lake
area, and log depth as additional predictors. Elevation is excluded so that
the warm-season lapse-rate calculation remains an explicit and independent
part of the paleoelevation workflow.
