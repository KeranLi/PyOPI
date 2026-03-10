function [d2HGrid, d18OGrid, ....
    evapD2HGrid, uEvapD2HGrid, evapD18OGrid, uEvapD18OGrid] = ...
    isotopeGrid( ...
    s, t, Sxy, Txy, lat, lat0, hWind, fMWind, rHWind, fPWind, ...
    z223Wind, z258Wind, tauF, ...
    U, T, gammaSat, hS, hR, d2H0, d18O0, dDH0dLat, dD18O0_dLat, isFit)
% Calculate isotope grid as a function of temperature, and the dlnFM_dS 
% grid, which is the log derivative of the precipitation field in 
% the wind direction.
% The calculation includes three steps:
% 1) The fractionation associated with the creation and fall of the 
% precipitation is determined by a weighted vertical average of the
% fractionation factor along the fall path, where the weighting accounts
% for the production rate of precipitation as a function of height. 
% This vertical averaging is done along a fall path, and starts
% with calculation the fractionation factor as a function of temperature and 
% phase (ice, water) along the fall path. This initial fractionation factor
% is adjusted to account for isotopic resetting of rain drops with vapor 
% as they fall from the freezing surface to the ground.
% 2) A fraction of the fallen precipitation, equal to (1-fP), is returned
% back to the atmosphere as water vaport produced by evaporation, where
% fP fraction of the precipitation that leaves through the base of the
% model. fP is assumed to a constant that applies across the model domain.
% 3) The vertical averaged fractionation factors are then integrated
% along wind path to get the isotopic composition of the precipitation
% and the evaporated vapor at the base of the model.
% The calculation accounts for ice and water, as determined by the 
% freezing surface is set to 258 K, which is the midpoint of 
% the 268 - 248 K range used by Cias and Jouzel (1994) to represent 
% the Bergeron-Findeisen zone. Temperatures are calculated using the
% LTOP solution. 
% The regional variation in isotopic composition is defined by a
% "regional" composition defined by a linear relationship in latitude,
% relative to the sample centroid. Thus, the final calculated 
% precipitation isotope fields are a sum of regional and orographic 
% contributions. 

% Mark Brandon, Yale University, 2016-2020

%% Initialize system
warning('off', 'MATLAB:griddedInterpolant:MeshgridEval2DWarnId');

%% Initialize variables
%... Shear for bringing fall path to vertical, where the shear ratio
% is the horizontal over vertical distances for fall of precipitation.
shear = U*tauF/hS;
%... Constants for fractionation due to evaporation
% Diffusivity ratios from Merlivat, 1978, with the rare isotopologue
% in the numerator
DRatio2H = 0.9755;
DRatio18O = 0.9723;
% Exponent, n, for fractionation due to evaporation.
% Recommended value: n = 1
% This exponent has a potential range from about 0.5 to 1. 
% My experience with OPI is that specific values in this range 
% have little influence on the best-fit solution. 
% Consider as well the following published estimates:
% Stewart, 1975 estimates n = 0.58 for evaporation of falling water drops. 
% Criss, 1999, p. 175 recommends n = 1 for evaporation from soils.
n = 1;

%% Set up wind-direction grid
%... Parameters for wind grid for topographic data
dS = s(2)-s(1);
[~, nT] = size(hWind);

% Horizontal shear needed to transform to vertical paths to land surface
sSurfaceShearWind = s + shear*hWind;

%... Construct grid for temperature at land surface
TLSWind = T(1) - gammaSat(1)*hWind;

% Horizontal shear of isothermal surface, to make fall paths vertical.
sShearWind = s + shear*z223Wind;
for j = 1:nT
    % Use first crossing where surface is steeper than fall path
    iMonotonic = sShearWind(:,j) ...
        >cummax([sShearWind(1,j)-1; sShearWind(1:end-1,j)]);
    z223Wind(:,j) = interp1(sShearWind(iMonotonic,j), ...
        z223Wind(iMonotonic,j), sSurfaceShearWind(:,j), ...
        'linear', z223Wind(1,j));
end
% Finalize by calculating height of isothermal surface above 
% land surface along fall path. 
zBar223Wind = z223Wind - hWind;
clear z223Wind

% Horizontal shear of isothermal surface, to make fall paths vertical.
sShearWind = s + shear*z258Wind;
for j = 1:nT
    % Use first crossing where surface is steeper than fall path
    iMonotonic = sShearWind(:,j) ...
        >cummax([sShearWind(1,j)-1; sShearWind(1:end-1,j)]);
    z258Wind(:,j) = interp1(sShearWind(iMonotonic,j), ...
        z258Wind(iMonotonic,j), sSurfaceShearWind(:,j), ...
        'linear', z258Wind(1,j));
end
clear sShearWind
% Finalize by calculating height of isothermal surface above 
% land surface along fall path. 
zBar258Wind = z258Wind - hWind;
clear z258Wind

% Calculate zBarFSWind, the height of freezing surface above land surface.
% Set to zero where freezing surface is below land surface. 
zBarFSWind = zBar258Wind;
zBarFSWind(zBarFSWind<0) = 0;

% Differentiate ln(fM) in wind direction
[~, dLnFM_dSWind] = gradient(log(fMWind), dS);

% Calculate hydrogen-isotope grid
% Get specific equilibrium factors as required for averaging calculation.
aLSWind = fractionationHydrogen(TLSWind);
a258 = fractionationHydrogen(258);
a223 = fractionationHydrogen(223);
% Calculate fractionation factors by vertical averaging.
% Subscripts A and B refer to above and below 258 K point.
bA = (a223 - a258)./(zBar223Wind - zBar258Wind);
bA(isnan(bA)) = 0;
bB = (a258 - aLSWind)./zBar258Wind;
bB(isnan(bB)) = 0;
% Fractionation factor for precipitation, equal to R_prec/R_vapor.
aPrecWind = ...
    ( (a258 + bA.*(hS + zBarFSWind - zBar258Wind)).*exp(-zBarFSWind./hR) ...
    + aLSWind.*(1 - exp(-zBarFSWind./hR)) ).*exp(-zBarFSWind./hS) ...
    + bB.*(hR^2*hS/(hS + hR)^2) ...
    .*(1 - (1 + (1/hS + 1/hR).*zBarFSWind) ...
    .*exp(-(1/(hS + 1/hR).*zBarFSWind))) + aLSWind.*(1 - exp(-zBarFSWind./hS));
clear bA bB cA cB

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %... Initial water vapor composition for d2H (no evaporative recycling)
% aPrec0 = mean(aPrecWind(1,:));
% d2H0_Vapor = (1 + d2H0)/aPrec0 - 1;
% fprintf('\n\nd2H0 for initial precipitation (per mil): %.1f\n', d2H0*1e3)
% fprintf('d2H0 for initial water vapor (per mil): %.1f\n', d2H0_Vapor*1e3)
% fprintf('alpha for initial precipitation: %.7f\n\n', aPrec0)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Next calculate fractionation due to evaporative recycling,
% using method from Criss, 1999 (p. 154-155, and 175).
% Get equilibrium fractionation factor at the evaporation temperature,
% which is defined as the temperature at the surface of the evaporating 
% water. We follow the general practice, which is to approximate 
% this temperature using the surface air temperature. Note this
% is the only place where temperature affects isotope fractionation 
% associated with evaporation.
a1EvapWind = fractionationHydrogen(TLSWind);
% Calculate fractionation factor for evaporation at rH = 0.
a0EvapWind = a1EvapWind.*DRatio2H^-n;
% Calculate fractionation factor for evaporation process (R_evap/R_vapor).
aEvapWind = a1EvapWind.*rHWind./(1 - a0EvapWind.*(1 - rHWind));
% Calculate exponent for integration of evaporative fractionation.
uEvap_d2HWind = 1./(a0EvapWind.*(1-rHWind)) - 1;
uEvap_d2HWind(rHWind==1) = 0;
% Combine to get fractionation factor for residual precipitation
% relative to atmospheric water vapor (R_residual/R_vapor).
aResidualVaporWind = fPWind.^uEvap_d2HWind.*aPrecWind ...
    + (1 - fPWind.^uEvap_d2HWind).*aEvapWind;
clear a0Evap a1Evap

% If opiFit, then remove unneeded arrays
if isFit==true, clear aEvap_d2HWind uEvap_d2HWind, end

% Integrate fractionation along the wind direction (down the columns)
% Result is the isotope ratio for the precipitation.
R_PrecWind = aPrecWind./aPrecWind(1,:) ...
    .*exp(cumtrapz((aResidualVaporWind - 1).*dLnFM_dSWind).*dS);
clear aPrec aResidual_Vapor

% If opiCalc, then calculate d2HEvapGrid and uEvap_d2HGrid
evapD2HGrid = [];
uEvapD2HGrid = [];
if isFit==false
    % Calculate d2HEvapWind and the convert to geographic grid
    d2HEvapWind = (aEvapWind./ aPrecWind).*R_PrecWind - 1;
    F = griddedInterpolant({s, t}, d2HEvapWind, 'linear', 'none');
    clear d2HEvapWind
    evapD2HGrid = F(Sxy, Txy);
    % Convert uEvap_d2HWind back to geographic grid
    F = griddedInterpolant({s, t}, uEvap_d2HWind, 'linear', 'none');
    clear uEvap_d2HWind
    uEvapD2HGrid = F(Sxy, Txy);
    clear F
end

%... Finalize d2H calculation
% 1) Transform d2H back to geographic grid
% 2) Converts from isotope ratio, RP, to to delta, d, representation.
% 3) Account for in regional isotopic composition of precipitation, 
% with d2H0 for value at centroid, and dD2H0dLat as the latitudinal
% gradient, which is relative to the absolute value of latitude. 
F = griddedInterpolant({s, t}, R_PrecWind, 'linear', 'none');
clear R_PrecWind
d2HGrid = ...
    (1 + d2H0 + dDH0dLat*(abs(lat) - abs(lat0))).*F(Sxy, Txy) - 1;

%% Calculate oxygen-isotope grid
% Get specific equilibrium factors as required for averaging calculation.
aLSWind = fractionationOxygen(TLSWind);
a258 = fractionationOxygen(258);
a223 = fractionationOxygen(223);
% Calculate fractionation factors by vertical averaging.
% Subscripts A and B refer to above and below freezing surface.
bA = (a223 - a258)./(zBar223Wind - zBar258Wind);
bA(isnan(bA)) = 0;
bB = (a258 - aLSWind)./zBar258Wind;
bB(isnan(bB)) = 0;
clear TLS zBar223Wind
% Fractionation factor for precipitation, equal to R_prec/R_vapor.
aPrecWind = ...
    ( (a258 + bA.*(hS + zBarFSWind - zBar258Wind)).*exp(-zBarFSWind./hR) ...
    + aLSWind.*(1 - exp(-zBarFSWind./hR)) ).*exp(-zBarFSWind./hS) ...
    + bB.*(hR^2*hS/(hS + hR)^2) ...
    .*(1 - (1 + (1/hS + 1/hR).*zBarFSWind) ...
    .*exp(-(1/(hS + 1/hR).*zBarFSWind))) + aLSWind.*(1 - exp(-zBarFSWind./hS));
clear bA bB cA cB zBar258Wind

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %... Initial water vapor composition for d2H (no evaporative recycling)
% aPrec0 = mean(aPrecWind(1,:));
% d18O0_Vapor = (1 + d18O0)/aPrec0 - 1;
% fprintf('\n\nd18O0 for initial precipitation (per mil): %.1f\n', d18O0*1e3)
% fprintf('d18O0 for initial water vapor (per mil): %.1f\n', d18O0_Vapor*1e3)
% fprintf('alpha for initial precipitation: %.7f\n\n', aPrec0)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Next calculate fractionation due to evaporative recycling,
% using method from Criss, 1999 (p. 154-155, and 175).
% Get equilibrium fractionation factor at evaporation temperature.
% (See note above about approximation of this temperature.)
a1EvapWind = fractionationOxygen(TLSWind);
% Calculate fractionation factor for evaporation at rH = 0.
a0EvapWind = a1EvapWind.*DRatio18O^-n;
% Calculate fractionation factor for evaporation process (R_evap/R_vapor).
aEvapWind = a1EvapWind.*rHWind./(1 - a0EvapWind.*(1 - rHWind));
% Calculate exponent for integration of evaporative fractionation.
uEvap_d18OWind = 1./(a0EvapWind.*(1-rHWind)) - 1;
uEvap_d18OWind(rHWind==1) = 0;
% Combine to get fractionation factor for residual precipitation
% relative to atmospheric water vapor (R_residual/R_vapor).
aResidualVaporWind = fPWind.^uEvap_d18OWind.*aPrecWind ...
    + (1 - fPWind.^uEvap_d18OWind).*aEvapWind;                                                                                     
% relative to atmospheric water vapor (R_residual/R_vapor).
clear a0Evap a1Evap
% If opiFit, then remove unneeded arrays
if isFit==true, clear aEvap_d18OWind uEvap_d18OWind, end

% Integrate fractionation along the wind direction (down the columns)
% Result is the isotope ratio for the precipitation.
R_PrecWind = aPrecWind./aPrecWind(1,:) ...
    .*exp(cumtrapz((aResidualVaporWind-1).*dLnFM_dSWind).*dS);
clear aPrec aResidual_Vapor dLnFM_dSWind

% If opiCalc, then calculate d18OEvapGrid and uEvap_d18OGrid
evapD18OGrid = [];
uEvapD18OGrid = [];
if isFit==false
    % Calculate d18OEvapWind and the convert to geographic grid
    d18OEvapWind = (aEvapWind./ aPrecWind).*R_PrecWind - 1;
    F = griddedInterpolant({s, t}, d18OEvapWind, 'linear', 'none');
    clear d18OEvapWind
    evapD18OGrid = F(Sxy, Txy);
    % Convert uEvap_d18OWind back to geographic grid
    F = griddedInterpolant({s, t}, uEvap_d18OWind, 'linear', 'none');
    clear uEvap_d18OWind
    uEvapD18OGrid = F(Sxy, Txy);
    clear F
end

%... Finalize d18O precipitation calculation
% 1) Transform d18O back to geographic grid
% 2) Converts from isotope ratio, RP, to to delta, d, representation.
% 3) Account for in regional isotopic composition of precipitation, 
% with d18O0 for value at centroid, and dd18O0dLat as the latitudinal
% gradient, which is relative to the absolute value of latitude. 
F = griddedInterpolant({s, t}, R_PrecWind, 'linear', 'none');
clear R_PrecWind
d18OGrid = ...
    (1 + d18O0 + dD18O0_dLat*(abs(lat) - abs(lat0))).*F(Sxy, Txy) - 1;

end