function ...
   [T0Record, d2H0Record, ageRecord, T0RecordSmooth, d2H0RecordSmooth] = ... 
   climateRecords(matFileMEBM, matFileBenthicForamClimate, lat, ...
   T0Pres, d2H0Pres, dD2H0_dD18O0, dD2H0_dT0, spanAge)
% Calculate time series for T0 and d2H0 at a specified latitude.
% T0 is the sea level surface air temperature, and d2H0 is the
% isotopic composition of base precipitation.
% The latitude is usually set to the centroid location for a study area,
% and is assumed here to be fixed over time.
% T0 is estimated using a moist energy balance model (MEBM) and a
% calibration from a benthic-foram temperature record.
% d2H0 is estimated using
% d2H0Pres + dD2H0_dT0*(T0Past - T0Pres) + (d2HSWPast - d2HSWPres),
% where d2H0Pres is the present d2H0 value at the specified latitude,
% dD2H0_dT0 is an empirical scaling that relates d2H0 to T0, and d2HSW
% refers to the isotopic composition of seawater.
% There are benthic-foram temperature records can be set to
% Cramer et al, 2011, or Miller et al., 2020.
%
% Input Arguments:
% matFileMEBM: full path and filename for MEBM data (string)
% matFileBenticForamClimate: full path and file name for
%   benthic-foram temperature record (string)
% lat: latitude for study area (assumed fixed with time) (degree) (scalar)
% T0Pres: present sea-level air temperature (K) at latitude (scalar)
% d2H0Pres: present d2H0 at specified latitude (per unit) (scalar)
% dD2H0_dD18O0: slope of local meteoric water line (dimensionless) (scalar)
% dD2H0_dT0: temperature scaling for d2H0 (per unit/C) (scalar)
% spanAge: age range (Ma) for smoothing records
%
% Output Arguments (all column vectors of equal length)
% T0Record: time series for T0 (K) at specified latitude
% d2H0Record: time series for d2H0 (per unit) at specified latitude
% ageRecord: age (Ma) for record values
% T0RecordSmooth: T0 time series, smoothed using Loess using spanAge
% d2H0RecordSmooth: d2H0 time series, smoothed using Loess using spanAge
%
% Mark Brandon, Yale University, 2022

%% Initialize variables
%... Convert from Celsius to kelvin
TC2K = 273.15;

%% Initialize climate records
%... Load variables from the selected benthic foram climate dataset:
% ageRecord: AgeClimate (Ma)
% TdwRecord: deep-water temperature (K)
% d18OswRecord: d18O for sea water (per mil, VSMOW)
load(matFileBenthicForamClimate, ...
    'ageRecord', 'TdwRecord', 'd18OswRecord');

%... Load estimated temperature field as a function of variations
% in outgoing long radiation (OLR), as determined using a 1D moist
% energy-balance model (MEBM code provided by Gerard Roe, 2019).
% dA_MEBM = change in the intercept term for radiative heat loss
%    in the MEBM (dA = 0 gives the "preindustrial" condition) (W/m^2)
% lat_MEBM = latitude (degree)
% T_MEBM = zonal-annual-mean surface-air temperature
%    for a water world (no land), as estimated by MEBM.
load(matFileMEBM, 'dA_MEBM', 'lat_MEBM', 'T_MEBM');
T_MEBM = T_MEBM + TC2K;

%... Estimated latitude for deep-water production
% This value below was determined by finding the latitude in the
% MEBM where T_MEBM for the modern Earth (dA = 0) is equal to the
% deep-water temperature in the modern oceans.
Tdw_MEBMPres = interp2(dA_MEBM, lat_MEBM, T_MEBM, 0, lat_MEBM);
lat_dw = interp1(Tdw_MEBMPres, lat_MEBM, TdwRecord(1));

%... Calculate the variation of dA with age, using the MEBM, lat_dw,
% and the deep-water temperature record Tdw.
% The relationship between Tdw and dA_MEBM is difficult to calculate
% at the highest values for Tdw, but the relationship is otherwise
% linear in that highest range. Thus, I have set griddedInterpolant
% to extrapolate linearly for values outside of the interpolant range.
Tdw_MEBM = interp1(lat_MEBM, T_MEBM, lat_dw);
F = griddedInterpolant(Tdw_MEBM(end:-1:1), dA_MEBM(end:-1:1), ...
    'linear', 'linear');
dARecord = F(TdwRecord);

%% Compute
%... Estimate T0Record at specified latitude
TLat_MEBM = interp1(lat_MEBM, T_MEBM, abs(lat))';
F = griddedInterpolant(dA_MEBM, TLat_MEBM', 'linear', 'linear');
T_MEBMPres = F(0);
T_MEBMPast = F(dARecord);
T0Record = T0Pres + (T_MEBMPast - T_MEBMPres);

%... Estimate d2H0Record at specified latitude
d18OswPres = d18OswRecord(1);
d2H0Record = d2H0Pres ...
    + dD2H0_dD18O0*(d18OswRecord - d18OswPres) ...
    + dD2H0_dT0*(T0Record - T0Pres);

%... Calculate smoothed versions for T0Record and d2H0Record
nRecord = length(ageRecord);
nSpan = ceil(nRecord*spanAge/(max(ageRecord) - min(ageRecord)));
T0RecordSmooth = smooth(T0Record, nSpan, 'loess');
d2H0RecordSmooth = smooth(d2H0Record, nSpan, 'loess');
T0RecordSmooth(ageRecord==0) = T0Pres;
d2H0RecordSmooth(ageRecord==0) = d2H0Pres;

end
