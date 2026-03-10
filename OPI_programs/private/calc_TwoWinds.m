function [chiR2, nu, stdResiduals, ...
    zBar_1, T_1, gammaEnv_1, gammaSat_1, gammaRatio_1, ...
    rhoS0_1, hS_1, rho0_1, hRho_1, ...
    d18O0_1, dD18O0_dLat_1, tauF_1, pGrid_1, fMGrid_1, rHGrid_1, ...
    d2HGrid_1, d18OGrid_1, pSumPred_1, d2HPred_1, d18OPred_1, ...
    zBar_2, T_2, gammaEnv_2, gammaSat_2, gammaRatio_2, ...
    rhoS0_2, hS_2, rho0_2, hRho_2, ...
    d18O0_2, dD18O0_dLat_2, tauF_2, pGrid_2, fMGrid_2, rHGrid_2, ....
    d2HGrid_2, d18OGrid_2, pSumPred_2, d2HPred_2, d18OPred_2, ...
    pGrid, d2HGrid, d18OGrid, pSumPred, d2HPred, d18OPred, fractionPGrid] = ...
    calc_TwoWinds(beta, fC, hR, ...
    x, y, lat, lat0, hGrid, bMWLSample, ijCatch, ptrCatch, ...
    sampleD2H, sampleD18O, cov, nParametersFree, isFit)
% opiCalc_TwoWind function provides a full calculation of misfit, climate
% and isotope values for a specified solution vector, beta, for two
% moisture sources (TwoWind).
% The function can be used with with an optimization function
% (e.g., fminCRS), where the first two output arguments,
% reduced chi-square and nu (degrees of freedom), are used to search for
% a best-fit solution. The full set of output arguments are also provided
% so that the function can be used to characterize a specific solution.
% The current version of the function includes the following features:
% 1) Predicted values for d2H and d18O are determined using a
%    precipitation-weighted average for the upslope catchment, as
%    specified by ijCatch and ptrCatch. The catchment can also be limited
%    to one grid node for samples that are designed as "local".
% 2) Predicted values fo d2H and d18O are set to NaN when the sample
%    catchment is "dry" (i.e. the predicted precipitation is zero for all
%    nodes in the catchment). Reduced chi square is also adjusted to
%    account for the reduced degrees of freedom due to dry catchments.
%
% v. 3.5 Sept - Dec, 2020: Adopted wind-oriented grids for all calculation,
%        and converted to the anelastic, saturated Euler equations of 
%        Durran and Klemp, 1982. Also includes options for regional or
%        leeside evaporation.
%        Jan - Feb, 2021: Introduced relative-humidity based evaporation. 
%        Corrected bugs associated with evaporation calculation. 
%        Improved representation of temperature field.
%        Replaced parameter NM, the buoyancy frequency, with M, the mountain-height index.
%        April 29, 2021: Converted calculations so that the solutions are
%        now in terms of d2H, not d18O.
%
% v. 3.6 July, 2021 
%        1) Set getInput so that all isotope samples are input from data file,
%        and the selection between primary and altered samples is done internal to 
%        program. The altered isotope samples are also saved as separate variables. 
%        2) Removed regional evaporation, and set so that evaporation 
%        options are automatically accounted for. 
%        3) Added a plot to show how dExcess varies as a function of the elevation
%        of liftMax relative to the sample elevation. 
%        4) Added a map showing relative humidity at the ground. 
%        5) Removed penalty option for dry samples. 

% Mark Brandon, Yale University, 2016-2021.

%% User-defined variables
% Precipitation state #1
U_1 = beta(1);          % wind speed (m/s)
azimuth_1 = beta(2);    % azimuth (degrees)
T0_1 = beta(3);         % sea-level temperature (K)
M_1 = beta(4);          % mountain-height number (dimensionless)
kappa_1 = beta(5);      % eddy diffusion (m/s^2)
tauC_1 = beta(6);       % condensation time (s)
d2H0_1 = beta(7);       % d2H of base precipitation (per unit)
dD2H0dLat_1 = beta(8);  % latitudinal gradient of base-prec d2H (1/deg lat)
fP0_1 = beta(9);        % residual precipitation after evaporation (fraction) 
fraction = beta(10);    % fractional size of precipitation state 1
% Precipitation state #2
U_2 = beta(11);         % wind speed (m/s)
azimuth_2 = beta(12);   % azimuth (degrees)
T0_2 = beta(13);        % sea-level temperature (K)
M_2 = beta(14);         % mountain-height number (dimensionless)
kappa_2 = beta(15);     % eddy diffusion (m/s^2)
tauC_2 = beta(16);      % condensation time (s)
d2H0_2 = beta(17);      % d2H of base precipitation (per unit)
dD2H0dLat_2 = beta(18); % latitudinal gradient of base-prec d2H (1/deg lat)
fP0_2 = beta(19);       % residual precipitation after evaporation (fraction) 
%... Convert mountain-height index to buoyancy frequency (rad/s)
hMax = max(hGrid, [], 'all');
NM_1 = M_1*U_1/hMax;
NM_2 = M_2*U_2/hMax;

%... Number of locations for predictions
if ~isempty(ijCatch) && ~isempty(ptrCatch)
    nSamples = length(ptrCatch); 
end

%% Calculation precipitation state #1
%... Calculate environmental profile for atmosphere upwind of topography.
% Stable atmosphere has gammaSat > gammaEnv
[zBar_1, T_1, gammaEnv_1, gammaSat_1, gammaRatio_1, ...
    rhoS0_1, hS_1, rho0_1, hRho_1] = baseState(NM_1, T0_1);

%... Set d18O isotopic composition for base precipitation
d18O0_1 = (d2H0_1 - bMWLSample(1))/bMWLSample(2);
dD18O0_dLat_1 = dD2H0dLat_1/bMWLSample(2);

%... Calculate precipitation rate (kg/(m^2 s))
[s, t, Sxy, Txy, pGrid_1, hWind, fMWind_1, rHWind_1, fPWind_1, ...
    z223Wind_1, z258Wind_1, tauF_1] = ...
    precipitationGrid ...
    (x, y, hGrid, U_1, azimuth_1, NM_1, fC, kappa_1, tauC_1, hRho_1, zBar_1, ...
    T_1, gammaEnv_1, gammaSat_1, gammaRatio_1, rhoS0_1, hS_1, fP0_1);
%... Scale pGrid
pGrid_1 = fraction*pGrid_1;

%... Calculate grids for precipitation isotopes and moisture ratio
[d2HGrid_1, d18OGrid_1] = isotopeGrid( ...
    s, t, Sxy, Txy, lat, lat0, hWind, fMWind_1, rHWind_1, fPWind_1, ...
    z223Wind_1, z258Wind_1, tauF_1, ...
    U_1, T_1, gammaSat_1, hS_1, hR, ...
    d2H0_1, d18O0_1, dD2H0dLat_1, dD18O0_dLat_1, isFit);
clear hWind fPWind_1

%... Transform fMWind_1 back to geographic grid
F = griddedInterpolant({s, t}, fMWind_1, 'linear', 'none');
clear fMWind_1
fMGrid_1 = F(Sxy, Txy);

%... Calculate rHGrid_1, if needed
rHGrid_1 = [];
if ~isFit==true
    if numel(rHWind_1)>1
        %... Using evaportion recycling, so calculate rHGrid_1
        F = griddedInterpolant({s, t}, rHWind_1, 'linear', 'none');
        clear rHWind_1
        rHGrid_1 = F(Sxy, Txy);
    else
        %.... No evaporation, so rHGrid_1 = 1;
        rHGrid_1 = 1;
        clear rHWind_1
    end
end

%... Precipitation isotopes d2HPred and d18OPred predicted for 
% sample catchments where wet locations. Dry locations are set to 
% nan values in a later calculation.
d2HPred_1 = zeros(nSamples,1);
d18OPred_1 = zeros(nSamples,1);
pSumPred_1 = zeros(nSamples,1);
for k = 1:nSamples
    % Extract indices for sample catchment
    if k~=nSamples
        ij = ijCatch(ptrCatch(k):ptrCatch(k+1)-1);
    else
        ij = ijCatch(ptrCatch(k):end);
    end
    % Calculate predicted precipitation for sample catchment
    pSumPred_1(k) = sum(pGrid_1(ij));
    % Calculated precipitation-weighted isotope compositions for catchment
    if pSumPred_1(k)>0
        % Normalize weights
        wtPrec = pGrid_1(ij)./pSumPred_1(k);
        % Calculate catchment-weighted composition of water isotopes
        d2HPred_1(k) = sum(wtPrec.*d2HGrid_1(ij));
        d18OPred_1(k) = sum(wtPrec.*d18OGrid_1(ij));
    end
end

%% Calculation precipitation state #2
%... Calculate environmental profile for atmosphere upwind of topography.
% Stable atmosphere has gammaSat > gammaEnv
[zBar_2, T_2, gammaEnv_2, gammaSat_2, gammaRatio_2, ...
    rhoS0_2, hS_2, rho0_2, hRho_2] = baseState(NM_2, T0_2);
 
%... Set d2H isotopic composition for base precipitation
d18O0_2 = (d2H0_2 - bMWLSample(1))/bMWLSample(2);
dD18O0_dLat_2 = dD2H0dLat_2/bMWLSample(2);

%... Calculate precipitation rate (kg/(m^2 s))
[s, t, Sxy, Txy, pGrid_2, hWind, fMWind_2, rHWind_2, fPWind_2, ...
    z223Wind_2, z258Wind_2, tauF_2] = ...
    precipitationGrid ...
    (x, y, hGrid, U_2, azimuth_2, NM_2, fC, kappa_2, tauC_2, hRho_2, zBar_2, ...
    T_2, gammaEnv_2, gammaSat_2, gammaRatio_2, rhoS0_2, hS_2, fP0_2);
%... Scale pGrid
pGrid_2 = (1 - fraction)*pGrid_2;

%... Calculate grids for precipitation isotopes and moisture ratio
[d2HGrid_2, d18OGrid_2] = isotopeGrid( ...
    s, t, Sxy, Txy, lat, lat0, hWind, fMWind_2, rHWind_2, fPWind_2, ...
    z223Wind_2, z258Wind_2, tauF_2, ...
    U_2, T_2, gammaSat_2, hS_2, hR, ...
    d2H0_2, d18O0_2, dD2H0dLat_2, dD18O0_dLat_2, isFit);
clear hWind fPWind_2

%... Transform fMWind_2 back to geographic grid
F = griddedInterpolant({s, t}, fMWind_2, 'linear', 'none');
clear fMWind_2
fMGrid_2 = F(Sxy, Txy);

%... Calculate rHGrid_2, if needed
rHGrid_2 = [];
if ~isFit==true
    if numel(rHWind_2)>1
        %... Using evaportion recycling, so calculate rHGrid_2
        F = griddedInterpolant({s, t}, rHWind_2, 'linear', 'none');
        clear rHWind_2
        rHGrid_2 = F(Sxy, Txy);
    else
        %.... No evaporation, so rHGrid_2 = 1;
        rHGrid_2 = 1;
        clear rHWind_2
    end
end

%... Precipitation isotopes d2HPred and d18OPred predicted for 
% sample catchments where wet locations. Dry locations are set to 
% nan values in a later calculation.
d2HPred_2 = zeros(nSamples,1);
d18OPred_2 = zeros(nSamples,1);
pSumPred_2 = zeros(nSamples,1);
for k = 1:nSamples
    % Extract indices for sample catchment
    if k~=nSamples
        ij = ijCatch(ptrCatch(k):ptrCatch(k+1)-1);
    else
        ij = ijCatch(ptrCatch(k):end);
    end
    % Calculated precipitation-weighted isotope compositions for catchment
    pSumPred_2(k) = sum(pGrid_2(ij));
    if pSumPred_2(k)>0
        % Normalize weights
        wtPrec = pGrid_2(ij)./pSumPred_2(k);
        % Calculate catchment-weighted composition of water isotopes
        d2HPred_2(k) = sum(wtPrec.*d2HGrid_2(ij));
        d18OPred_2(k) = sum(wtPrec.*d18OGrid_2(ij));
    end
end

%% Combine precipitation states and final estimates for water isotopes
%... Calculate combined precipitation grid
pGrid = pGrid_1 + pGrid_2;
%... Calculate water isotope grids. Locations with pGrid==0 are set to nans.
d2HGrid = (pGrid_1.*d2HGrid_1 + pGrid_2.*d2HGrid_2)./pGrid;
d18OGrid = (pGrid_1.*d18OGrid_1 + pGrid_2.*d18OGrid_2)./pGrid;
d2HGrid(isinf(d2HGrid)) = nan;
d18OGrid(isinf(d18OGrid)) = nan;
%... Calculate grid of fraction of total precipitation from state #1
fractionPGrid = pGrid_1./pGrid;

%... Calculate predicted water isotopes for sample locations, and
% set to nans where precipitation rate is zero.
pSumPred = pSumPred_1 + pSumPred_2;
iWet = (pSumPred>0);
d2HPred = (pSumPred_1.*d2HPred_1 + pSumPred_2.*d2HPred_2)./pSumPred;
d18OPred = (pSumPred_1.*d18OPred_1 + pSumPred_2.*d18OPred_2)./pSumPred;
d2HPred(~iWet) = nan;
d18OPred(~iWet) = nan;
d2HPred_1(pSumPred_1==0) = nan;
d18OPred_1(pSumPred_1==0) = nan;
d2HPred_2(pSumPred_2==0) = nan;
d18OPred_2(pSumPred_2==0) = nan;

%... Calculate reduced chi square and standardized residuals
% Sum number of precipitation-present samples
nSamplesWet = sum(iWet);
% Calculate degrees of freedom
nu = nSamplesWet - nParametersFree;
if nu > 0
    %... Calculate chiR2 using samples that come from wet locations.
    % The residual matrix, r, has a column for each sample.
    r = [(sampleD2H(iWet) - d2HPred(iWet))'; ...
         (sampleD18O(iWet) - d18OPred(iWet))';];
    % Reduced chi square, with correction for small-sample bias 
    % using the t value for the probability at +1 sigma for the 
    % normal distribution.
    chiR2 = tinv(0.84134, nu)*sum(dot(r, cov\r))/nu;
    %... Calculate standardized residuals for all samples.
    r = [(sampleD2H - d2HPred)'; ...
        (sampleD18O - d18OPred)'];
    % Calculate standardized residuals
    stdResiduals = sqrt(dot(r, cov\r))';
else
    % Set to nan for cases where the degrees of freedom < 1
    chiR2 = nan;
    stdResiduals = nan(nSamples,1);
end
end
