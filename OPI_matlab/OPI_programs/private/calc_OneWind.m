function [chiR2, nu, stdResiduals, ...
    zBar, T, gammaEnv, gammaSat, gammaRatio, ...
    rhoS0, hS, rho0, hRho, ...
    d18O0, dD18O0_dLat, tauF, pGrid, fMGrid, rHGrid, ...
    evapD2HGrid, uEvapD2HGrid, evapD18OGrid, uEvapD18OGrid, ...
    d2HGrid, d18OGrid, iWet, d2HPred, d18OPred] = ...
    calc_OneWind(beta, fC, hR, x, y, lat, lat0, ...
    hGrid, bMWLSample, ijCatch, ptrCatch, ...
    sampleD2H, sampleD18O, cov, nParametersFree, isFit)
% calc_OneWind provides a full calculation of misfit, climate
% and isotope values for a specified solution vector, beta, for one
% moisture source (OneWind).
%
% The function can be used with with an optimization function
% (e.g., fminCRS), where the first two output arguments,
% reduced chi-square and nu (degrees of freedom), are used to search for
% a best-fit solution. The full set of output arguments are also provided
% so that the function can be used to characterize a specific solution.
% The current version of the function includes the following features:
% 1) Predicted values for d2H and d18O are determined using a
%    precipitation-weighted average for the upslope catchment, as
%    specified by ijCatch and ptrCatch. The catchment can also be limited
%    to a one grid node for samples that are designed as "local".
% 2) Predicted values for d2H and d18O are set to NaN when the sample
%    catchment is "dry" (i.e. the predicted precipitation is zero for all
%    nodes in the catchment). Reduced chi square is also adjusted to
%    account for the reduced degrees of freedom due to dry catchments.
%
% v. 3.6 July, 2021
%  1) Set getInput so that all isotope samples are input from data file,
%  and the selection between primary and altered samples is done internal to
%  program. The altered isotope samples are also saved as separate variables.
%  2) Removed regional evaporation, and set so that evaporation
%  options are automatically accounted for.
%  3) Added a plot to show how dExcess varies as a function of the elevation
%  of liftMax relative to the sample elevation.
%  4) Added a map showing relative humidity at the ground.
%  5) Removed penalty option for dry samples.
%
%  July 24, 2022: Modified the termination criterion so 
%  mSet = mu*(nParametersFree + 1), mu0 = mu, and 
%  epsilon = stdev(F_searchSet).

% Mark Brandon, Yale University, 2016-2021.

%% Initialize system
%... Turn off warning for cases where arrays are changed in size
% with each loop iteration.
%#ok<*AGROW>

%% Initialize variables
%... Set following variables to empty, in case they are not assigned.
chiR2 = []; nu = []; stdResiduals = [];
iWet = []; d2HPred = []; d18OPred = []; 

%% User-defined variables
U = beta(1);           % wind speed (m/s)
azimuth = beta(2);     % azimuth (degrees)
T0 = beta(3);          % sea-level temperature (K)
M =  beta(4);          % mountain-height number (dimensionless)
kappa = beta(5);       % eddy diffusion (m/s^2)
tauC = beta(6);        % condensation time (s)
d2H0 = beta(7);        % d2H of base precipitation (per unit)
dD2H0_dLat = beta(8);  % latitudinal gradient of base-prec d2H (1/deg lat)
fP0 = beta(9);         % residual precipitation after evaporation (fraction)
%... Convert mountain-height index to buoyancy frequency (rad/s)
NM = M*U/max(hGrid, [], 'all');

%... Number of locations for predictions
if ~isempty(ijCatch) && ~isempty(ptrCatch)
    nSamples = length(ptrCatch); 
end

%% Start calculation
%... Calculate environmental profile for atmosphere upwind of topography.
% The atmosphere is assumed to be stable (NM > 0), and baseState includes
% a check to ensure that gammaEnv > 0.
[zBar, T, gammaEnv, gammaSat, gammaRatio, ...
    rhoS0, hS, rho0, hRho] = baseState(NM, T0);

%... Set d180 isotopic composition for base precipitation
d18O0 = (d2H0 - bMWLSample(1))/bMWLSample(2);
dD18O0_dLat = dD2H0_dLat/bMWLSample(2);

%... Calculate precipitation rate (kg/(m^2 s)) and related grids
[s, t, Sxy, Txy, pGrid, hWind, fMWind, rHWind, fPWind, ...
    z223Wind, z258Wind, tauF] = ...
    precipitationGrid ...
    (x, y,  hGrid, U, azimuth, NM, fC, kappa, tauC, hRho, zBar, ...
    T, gammaEnv, gammaSat, gammaRatio, rhoS0, hS, fP0);

%... Calculate grids for precipitation isotopes and moisture ratio
[d2HGrid, d18OGrid, ....
    evapD2HGrid, uEvapD2HGrid, evapD18OGrid, uEvapD18OGrid] = ...
    isotopeGrid( ...
    s, t, Sxy, Txy, lat, lat0, hWind, fMWind, rHWind, fPWind, ...
    z223Wind, z258Wind, tauF, ...
    U, T, gammaSat, hS, hR, d2H0, d18O0, dD2H0_dLat, dD18O0_dLat, isFit);
clear hWind fPWind

%... Transform fMWind back to geographic grid
F = griddedInterpolant({s, t}, fMWind, 'linear', 'none');
clear fMWind
fMGrid = F(Sxy, Txy);

%... Calculate rHGrid, if needed
rHGrid = [];
if ~isFit==true
    if numel(rHWind)>1
        %... Using evaportion recycling, so calculate rHGrid
        F = griddedInterpolant({s, t}, rHWind, 'linear', 'none');
        clear rHWind
        rHGrid = F(Sxy, Txy);
    else
        %.... No evaporation, so rHGrid = 1;
        rHGrid = 1;
        clear rHWind
    end
end

%... Calculations for sample locations
if ~isempty(ijCatch) && ~isempty(ptrCatch)
    %... Precipitation isotopes d2HPred and d18OPred predicted for
    % sample locations, and accounting for catchment area if selected.
    d2HPred = nan(nSamples,1);
    d18OPred = nan(nSamples,1);
    iWet = false(nSamples,1);
    for k = 1:nSamples
        % Extract indices for sample catchment
        ij = catchmentIndices(k, ijCatch, ptrCatch);
        % Calculate weights using predicted precipitation
        pSum = sum(pGrid(ij));
        % Calculated precipitation-weighted isotope compositions for catchment
        if pSum>0
            % Normalize weights
            wtPrec = pGrid(ij)./pSum;
            % Calculate catchment-weighted composition of water isotopes
            iWet(k) = true;
            d2HPred(k) = sum(wtPrec.*d2HGrid(ij));
            d18OPred(k) = sum(wtPrec.*d18OGrid(ij));
        else
            % Use simple mean for dry sites
            d2HPred(k) = mean(d2HGrid(ij));
            d18OPred(k) = mean(d18OGrid(ij));
        end
    end
end

% Skip chiR2 calculation for simulation case, as indicated by
% where sample isotopes set to nans.
if all(isnan([sampleD18O, sampleD2H])), return, end

%... Calculate reduced chi square and standardized residuals
% Sum number of precipitation-present samples
nSamplesWet = sum(iWet);
% Calculate degrees of freedom
nu = nSamplesWet - nParametersFree;
if nu > 0
    %... Calculate chiR2 using samples that come from wet locations.
    % The residual matrix, r, has a column for each sample.
    r = [(sampleD2H(iWet) - d2HPred(iWet))'; ...
         (sampleD18O(iWet) - d18OPred(iWet))'];
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
