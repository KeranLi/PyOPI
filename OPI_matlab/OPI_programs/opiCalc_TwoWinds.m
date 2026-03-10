function opiCalc_TwoWinds
% opiCalc_TwoWind takes input from a run file with a "two-winds" solution,
% and then calculates the full range of results for that solution.
% The variables are saved in a mat file, which can then be used for
% producing plots and maps, and for other analysis.
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

% Mark Brandon, Yale University, 2016-2022.

%% Initialize system
close all
clc
dbstop if error

%% Constants
%... Start time for calculation
startTimeOpiCalc = datetime;
%... Convert from Celsius to kelvin
TC2K = 273.15;
%... Mean radius of the Earth (m)
radiusEarth = 6371e3;
%... Meters per arc degree for the Earth's surface
mPerDegree = pi*radiusEarth/180;
%... Characteristic distance (m) for isotopic exchange (Lee et al., 2007)
% Based on the average velocity of falling rain, 6 m/s, and the
% average characteristic time for isotopic exchange, 90 second.
hR = 540;
%... Estimated standard-deviation ratio for isotopic variation at 
% a location due to seaonal variation, where the ratio is equal to
% the maximum sd/minimum sd.
% The estimate here is based on monthly measurements of 
% precipitation isotopes at the GNIP station at Coyhaique, Chile.
sdResRatio = 28.3;

%% Get run file information
% function [runPath, runFile, runTitle, isParallel, dataPath, ...
%     topoFile, rTukey, sampleFile, contDivideFile, restartFile, ...
%     mapLimits, sectionLon0, sectionLat0, mu, epsilon0, ...
%     parameterLabels, exponents, lB, uB, beta] ...
%     = getRunFile(runFile)
[runPath, runFile, runTitle, ~, dataPath, ...
    topoFile, rTukey, sampleFile, contDivideFile, ~, ...
    mapLimits, sectionLon0, sectionLat0, ~, ~, ...
    ~, ~, lB, uB, beta] ...
    = getRunFile;
if isempty(beta)
    error('opiCalc requires that the run file include an OPI solution at the end of the file.')
end
if length(beta)~=19
    error('Number of parameters is incorrect for this program')
end
%... Number of free parameters
nParametersFree = sum(lB~=uB);

%% Get input data
% [lon, lat, x, y, hGrid, lon0, lat0, ...
%     sampleLineNum, sampleLon, sampleLat, sampleX, sampleY, ...
%     sampleD2H, sampleD18O, sampleDExcess, sampleLC, ...
%     sampleLineAlt, sampleLonAlt, sampleLatAlt, sampleXAlt, sampleYAlt, ...
%     sampleD2HAlt, sampleD18OAlt, sampleDExcessAlt, sampleLCAlt, ...
%     bMWLSample, sdDataMin, sdDataMax, cov, fC] ...
%     = getInput(dataPath, topoFile, rTukey, sampleFile, sdResRatio)
[lon, lat, x, y, hGrid, lon0, lat0, ...
    sampleLine, sampleLon, sampleLat, sampleX, sampleY, ...
    sampleD2H, sampleD18O, ~, sampleLC, ...
    sampleLineAlt, sampleLonAlt, sampleLatAlt, sampleXAlt, sampleYAlt, ...
    sampleD2HAlt, sampleD18OAlt, ~, sampleLCAlt, ...
    bMWLSample, sdDataMin, sdDataMax, cov, fC] ...
    = getInput(dataPath, topoFile, rTukey, sampleFile, sdResRatio);

%... Number of samples
nSamples = length(sampleLon);
nSamplesAlt = length(sampleLonAlt);

% Unpack beta, the solution vector
% Precipitation state #1
U_1 = beta(1);          % wind speed (m/s)
azimuth_1 = beta(2);    % azimuth (degrees)
T0_1 = beta(3);         % sea-level temperature (K)
M_1 = beta(4);          % mountain-height number (dimensionless)
kappa_1 = beta(5);      % eddy diffusion (m/s^2)
tauC_1 = beta(6);       % condensation time (s)
d2H0_1 = beta(7);       % d2H of base precipitation (per unit)
dD2H0_dLat_1 = beta(8); % latitudinal gradient of base-prec d2H (1/deg lat)  
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
dD2H0_dLat_2 = beta(18);% latitudinal gradient of base-prec d2H (1/deg lat)  
fP0_2 = beta(19);       % residual precipitation after evaporation (fraction)
%... Convert mountain-height number to buoyancy frequency (rad/s)
hMax = max(hGrid, [], 'all');
NM_1 = M_1*U_1/hMax;
NM_2 = M_2*U_2/hMax;

%... If empty, set section origin to map origin 
if isempty([sectionLon0, sectionLat0])
    sectionLon0 = lon0;
    sectionLat0 = lat0;
end

%... Test for samples
if isempty([sampleLon, sampleLat])
    error('Samples must be present for opiCalc_TwoWind.')
end
  
% Find grid nodes, for local and catchment samples ("L" and "C" in
% sampleLC).
[ijCatch, ptrCatch] = ...
    catchmentNodes(sampleX, sampleY, sampleLC, x, y, hGrid);

%% Determine side for each sample relative to continental-divide polyline
%... Load continental-divide data
if isempty(contDivideFile)
    error('Continental-divide file required for two-winds calculation.');
end
load([dataPath, '/', contDivideFile], 'contDivideLon', 'contDivideLat');
[contDivideX, contDivideY] = ...
    lonlat2xy(contDivideLon, contDivideLat, lon0, lat0);
%... Outline of hGrid domain
regionX = [x(1) x(end) x(end) x(1) x(1)]';
regionY = [y(1) y(1) y(end) y(end) y(1)]';
%... Use continential divide polyline to cut hGrid domain into two sides
[pointX, pointY, ij] = ...
    polyxpoly(regionX, regionY, contDivideX, contDivideY);
%... Check if continental-divide polyline crosses the full region
if isempty(pointX) || length(pointX)<2
    error('Continental divide polyline has less than 2 intersections with the region boundary.');
end
%... Limit intersections to two points on the region boundary
if ~isempty(pointX) && length(pointX)>2
    warning('Continental divide has more than two intersections with region.');
    fprintf('Side masks for the region is set to the first and last interections.\n\n');
    pointX = [pointX(1); pointX(end)];
    pointY = [pointY(1); pointY(end)];
    ij = [ij(1), ij(end)];
end
%... Calculate mask to distinguish sides of the region relative to
% the continental divide.
% Define functions to find indices on region boundary (ipoly)
% and continental divide (iLine). Input arguments are: i0, i1, n are 
% the start and stop indices, and length of the polygon or line.
% The output is a sequence of indices for the polygon or line that
% organized in the correct direction relative to the start and stop 
% indices, and for the closed form of the polygon. 
iPoly = @(i0, i1, n) mod(((i0-1) : (i1-1+(i0>i1)*n)), n) + 1;
iLine = @(i0, i1) (i0 : 2*(i1>=i0)-1 : i1);
regionSideX = ...
    [pointX(1); regionX(iPoly(ij(1,1)+1, ij(2,1), length(regionX))); ...
    pointX(2); contDivideX(iLine(ij(2,2)+1, ij(1,2))); ...
    pointX(1)];
regionSideY = ...
    [pointY(1); regionY(iPoly(ij(1,1)+1, ij(2,1), length(regionX))); ...
    pointY(2); contDivideY(iLine(ij(2,2)+1, ij(1,2))); ...
    pointY(1)];
%... Identify region side in terms of moisture source 1 or 2
meanX = mean(x);
meanY = mean(y);
r = min([x(end), y(end)])/2;
windEntryX = meanX + r*sind(-azimuth_1);
windEntryY = meanY + r*cosd(-azimuth_1);
if inpolygon(windEntryX, windEntryY, regionSideX, regionSideY)
    regionSideState = 1;
else
    regionSideState = 2;
end
%... Identify sample locations relative to the continental divide
% isSampleSideState01 is a logical variable, with true and false indicating
% if a sample location is on the state #1 or state #2 side of the range,
% relative to the continental-divide polyline.
isSampleSide01 = inpolygon(sampleX, sampleY, regionSideX, regionSideY);
if regionSideState~=1
    isSampleSide01 = ~isSampleSide01;
end

%% Report starting conditions
%... Start diary file
logFilename=[runPath, '/', mfilename, '_Log.txt'];
if isfile(logFilename), delete (logFilename); end
diary(logFilename);
fprintf (['Program: ', mfilename, '\n'])
fprintf('Start time: %s\n', startTimeOpiCalc)
fprintf('Run file path:\n%s\n', runPath)
fprintf('Run file name:\n%s\n', runFile)
fprintf('Run title:\n');
fprintf('%s\n', runTitle);
fprintf('Path name for data directory:\n')
fprintf('%s\n', dataPath)
fprintf('\n------------------ Topography File ------------------\n')
fprintf('Topography file: %s\n', topoFile)
fprintf('Maximum elevation: %.0f m\n', hMax);
[nY, nX] = size(hGrid);
fprintf('Grid size, nx and ny: %d, %d\n', nX, nY)
fprintf('Minimum and maximum for longitude: %.5f, %.5f degrees\n', lon(1), lon(end))
fprintf('Minimum and maximum for latitude: %.5f, %.5f degrees\n', lat(1), lat(end))
dLon = lon(2) - lon(1);
dLat = lat(2) - lat(1);
fprintf('Grid spacing, dLon and dLat: %.5f, %.5f degrees\n', dLon, dLat)
fprintf('Grid spacing, dx and dy: %.2f, %.2f km\n', ...
    dLon*mPerDegree*1e-3*cosd(lat0), dLat*mPerDegree*1e-3)
fprintf('Lon, lat for map origin: %.5f, %.5f degrees\n', lon0, lat0);
if lon0==mean(sampleLon) && lat0==mean(sampleLat)
    fprintf('Map origin is set to sample centroid.\n')
else
    fprintf('Map origin is set center of the topographic grid.\n')
end
fprintf('Size of cosine window as fraction of grid size: %g (dimensionless)\n', rTukey)
fprintf('Coriolis frequency at map-origin latitude (mrad/s): %.3f\n', ...
    fC*1e3)
fprintf('Lon, lat for section origin (degrees): %.5f, %.5f\n', sectionLon0, sectionLat0);
fprintf('\n-------------------- Sample File --------------------\n')
fprintf('Sample file: %s\n', sampleFile)
fprintf('Number of all samples: %d\n', nSamples + nSamplesAlt)    
fprintf('Number of primary samples: %d\n', nSamples)
fprintf('Number of local primary samples: %d\n', sum(sampleLC=='L'))
fprintf('Number of catchment primary samples: %d\n', sum(sampleLC=='C'))
fprintf('Number of altered samples: %d\n', nSamplesAlt)
fprintf('Centroid for primary samples, longitude, latitude: %.5f, %.5f degrees\n', ...
    mean(sampleLon), mean(sampleLat))
fprintf('Minimum and maximum for longitude: %.5f, %.5f\n', ...
    min(sampleLon), max(sampleLon))
fprintf('Minimum and maximum for latitude: %.5f, %.5f\n', ...
    min(sampleLat), max(sampleLat))
fprintf('\n---------------------- Constants --------------------\n')
fprintf('Characteristic distance for isotopic exchange: %.0f m\n', hR)
fprintf('Standard-deviation ratio for data residuals: %.2f (dimensionless)\n', sdResRatio)
fprintf('\n--------- Constraints for Best-Fit Solution ---------\n')
fprintf('Precipitation State #1:\n')
fprintf('Wind speed: %g, %g m/s\n', lB(1), uB(1))
fprintf('Wind azimuth: %g, %g degrees\n', lB(2), uB(2))
fprintf('Sea-level surface-air temperature: %g, %g K  (%.1f, %.1f °C)\n', ...
    lB(3), uB(3), lB(3) - TC2K, uB(3) - TC2K)
fprintf('Mountain-height number: %g, %g (dimensionless)\n', lB(4), uB(4))
fprintf('Horizontal eddy diffusivity: %g, %g m^2/s\n', lB(5), uB(5))
fprintf('Average condensation time (s): %g, %g s\n', lB(6), uB(6))
fprintf('d2H for base precipitation: %g, %g per mil\n', [lB(7), uB(7)]*1e3)
fprintf('d2H latitude gradient for base precipitation: %g, %g per mil/deg lat\n', [lB(8), uB(8)]*1e3)
fprintf('Residual precipitation after evaporation: %g, %g (fraction)\n', lB(9), uB(9))
fprintf('Fraction for precipitation state 1: %g, %g\n', lB(10), uB(10))
fprintf('\nPrecipitation State #2:\n')
fprintf('Wind speed: %g, %g m/s\n', lB(11), uB(11))
fprintf('Wind azimuth: %g, %g degrees\n', lB(12), uB(12))
fprintf('Sea-level surface-air temperature: %g, %g K  (%.1f, %.1f °C)\n', ...
    lB(13), uB(13), lB(13) - TC2K, uB(13) - TC2K)
fprintf('Mountain-height number (dimensionless): %g, %g\n', lB(14), uB(14))
fprintf('Horizontal eddy diffusivity: %g, %g m^2/s\n', lB(15), uB(15))
fprintf('Average condensation time: %g, %g s\n', lB(16), uB(16))
fprintf('d2H for base precipitation: %g, %g per mil\n', [lB(17), uB(17)]*1e3)
fprintf('d2H latitude gradient for base precipitation: %g, %g per mil/deg lat\n', [lB(18), uB(18)]*1e3)
fprintf('Residual precipitation after evaporation: %g, %g (fraction)\n', lB(19), uB(19))
fprintf('\n----------------- Evaporation Option ----------------\n')
if lB(9)==1 && uB(9)==1 && lB(19)==1 && uB(19)==1
    fprintf('No Evaporative recycling active for both precipitation states.\n')
else
    fprintf('Evaporative recycling active for both precipitation states.\n')
end
%... Force diary to write
diary off; diary on

%% Calculate full result for best-fit solution
isFit = false;
[chiR2, nu, stdResiduals, ...
    zBar_1, T_1, gammaEnv_1, gammaSat_1, gammaRatio_1, ...
    rhoS0_1, hS_1, rho0_1, hRho_1, ...
    d18O0_1, dD18O0_dLat_1, tauF_1, pGrid_1, fMGrid_1, rHGrid_1, ...
    d2HGrid_1, d18OGrid_1, pSumPred_1, d2HPred_1, d18OPred_1, ...
    zBar_2, T_2, gammaEnv_2, gammaSat_2, gammaRatio_2, ...
    rhoS0_2, hS_2, rho0_2, hRho_2, ...
    d18O0_2, dD18O0_dLat_2, tauF_2, pGrid_2, fMGrid_2, rHGrid_2, ...
    d2HGrid_2, d18OGrid_2, pSumPred_2, d2HPred_2, d18OPred_2, ...
    pGrid, d2HGrid, d18OGrid, pSumPred, d2HPred, d18OPred, fractionPGrid] = ...
    calc_TwoWinds(beta, fC, hR, ...
    x, y, lat, lat0, hGrid, bMWLSample, ijCatch, ptrCatch, ...
    sampleD2H, sampleD18O, cov, nParametersFree, isFit);

%... Calculate the precipitation-weighted catchment average
% for maximum lifting and elevation for each sample point,
% while accounting for both type L and type C sample locations,
% and also for primary and alterated samples.
[liftMaxPred_1, elevationPred_1] = ...
    lifting(x, y, hGrid, pGrid_1, azimuth_1, ...
    U_1, tauF_1, ijCatch, ptrCatch);
[liftMaxPred_2, elevationPred_2] = ...
    lifting(x, y, hGrid, pGrid_2, azimuth_2, ...
    U_2, tauF_2, ijCatch, ptrCatch);
liftMaxPred = ...
    (pSumPred_1.*liftMaxPred_1 + pSumPred_2.*liftMaxPred_2)./pSumPred;
elevationPred = ...
    (pSumPred_1.*elevationPred_1 + pSumPred_2.*elevationPred_2)./pSumPred;

%... Save results in mat file
save([runPath, '/', mfilename, '_Results.mat'], ...
    'startTimeOpiCalc', 'runPath', 'runFile', 'runTitle', ...
    'dataPath', 'topoFile', 'rTukey', 'sampleFile', 'contDivideFile', ...
    'mapLimits', 'sectionLon0', 'sectionLat0', ...
    'lon', 'lat', 'x', 'y', 'lon0', 'lat0', ...
    'sampleLine', 'sampleLon', 'sampleLat', 'sampleX', 'sampleY', ...
    'sampleD2H', 'sampleD18O', 'sampleLC', ...
    'sampleLineAlt', 'sampleLonAlt', 'sampleLatAlt', 'sampleXAlt', 'sampleYAlt', ...
    'sampleD2HAlt', 'sampleD18OAlt', 'sampleLCAlt', ...
    'bMWLSample', 'ijCatch', 'ptrCatch', 'isSampleSide01', ...
    'hR', 'sdResRatio', 'sdDataMin', 'sdDataMax','cov', 'fC', ...
    'lB', 'uB', 'nParametersFree', ...
    'beta', 'chiR2', 'nu', 'stdResiduals', ...
    'zBar_1', 'T_1', 'gammaEnv_1', 'gammaSat_1', 'gammaRatio_1', ...
    'rhoS0_1', 'hS_1', 'rho0_1', 'hRho_1', ...    
    'd18O0_1', 'dD18O0_dLat_1', 'tauF_1', 'pGrid_1', 'fMGrid_1', 'rHGrid_1', ...
    'd2HGrid_1', 'd18OGrid_1', 'pSumPred_1', 'd2HPred_1', 'd18OPred_1', ...
    'liftMaxPred_1', 'elevationPred_1', ...
    'zBar_2', 'T_2', 'gammaEnv_2', 'gammaSat_2', 'gammaRatio_2', ...
    'rhoS0_2', 'hS_2', 'rho0_2', 'hRho_2', ...    
    'd18O0_2', 'dD18O0_dLat_2', 'tauF_2', 'pGrid_2', 'fMGrid_2', 'rHGrid_2', ...
    'd2HGrid_2', 'd18OGrid_2', 'pSumPred_2', 'd2HPred_2', 'd18OPred_2', ...
    'liftMaxPred_2', 'elevationPred_2', ...
    'pGrid', 'd2HGrid', 'd18OGrid', 'pSumPred', 'd2HPred', 'd18OPred', ...
    'liftMaxPred', 'elevationPred', 'fractionPGrid', '-v7.3');

%... Sum number of precipitation-present samples
iWet = (pSumPred>0);
nSamplesWet = sum(iWet);
%... Calculate statistics for predicted water isotopes
[bMWLPred, sdPredMin, sdPredMax] = ...
    estimateMWL(d18OPred(iWet), d2HPred(iWet), sdResRatio);
%... Calculate standard deviation of residuals
sdRes_d2H = sqrt(sum((sampleD2H(iWet) - d2HPred(iWet)).^2) ...
    /(nSamplesWet - nParametersFree));
sdRes_d18O = sqrt(sum((sampleD18O(iWet) - d18OPred(iWet)).^2) ...
    /(nSamplesWet - nParametersFree));
%... Calculate approximate standard error for predicted d2H and d18O
% These standard errors assume a location close to the sample centroid.
se_d2HPred = sdRes_d2H/sqrt(nSamplesWet - nParametersFree);
se_d18OPred = sdRes_d18O/sqrt(nSamplesWet - nParametersFree);

%% Report best-fit results
fprintf('\n---------------------- Solution ---------------------\n')
fprintf('Precipitation State #1:\n')
fprintf('Wind speed (m/s): %.1f m/s\n', U_1)
fprintf('Azimuth: %.1f degrees\n', azimuth_1)
fprintf('Sea-level temperature: %.1f K (%.1f °C)\n', T0_1, T0_1 - TC2K)
fprintf('Mountain-height number: %.3f (dimensionless)\n', M_1)
fprintf('Horizontal eddy diffusivity: %.0f m^2/s\n', kappa_1)
fprintf('Average residence time for cloud water: %.0f s\n', tauC_1)
fprintf('d2H for base precipitation: %.1f per mil\n', d2H0_1*1e3)
fprintf('d2H latitude gradient for base precipitation: %.3f per mil/deg lat\n', dD2H0_dLat_1*1e3)
fprintf('Residual precipitation after evaporation: %.2f (fraction)\n', fP0_1)
fprintf('Fraction for precipitation state #1: %.2f\n', fraction);
fprintf('\nPrecipitation State #2:\n')
fprintf('Wind speed (m/s): %.1f m/s\n', U_2)
fprintf('Azimuth: %.1f degrees\n', azimuth_2)
fprintf('Sea-level temperature: %.1f K (%.1f °C)\n', T0_2, T0_2 - TC2K)
fprintf('Mountain-height number: %.3f (dimensionless)\n', M_2)
fprintf('Horizontal eddy diffusivity: %.0f m^2/s\n', kappa_2)
fprintf('Average residence time for cloud water: %.0f s\n', tauC_2)
fprintf('d2H for base precipitation: %.1f per mil\n', d2H0_2*1e3)
fprintf('d2H latitude gradient for base precipitation: %.3f per mil/deg lat\n', dD2H0_dLat_2*1e3)
fprintf('Residual precipitation after evaporation: %.2f (fraction)\n', fP0_2)
fprintf('\n---- Other Variables Related to Best-Fit Solution ---\n')
fprintf('Precipitation State #1:\n')
fprintf('Moist buoyancy frequency: %.3f mrad/s\n', NM_1*1e3)
fprintf('d18O for base precipitation: %.1f per mil\n', d18O0_1*1e3)
fprintf('d18O latitude gradient for base precipitation: %.3f per mil/deg lat\n', dD18O0_dLat_1*1e3)
fprintf('Average residence time for falling precipitation: %.0f s\n', tauF_1)
fprintf('Water-vapor density at sea level: %.2f g/m^3\n', rhoS0_1*1e3)
fprintf('Scale height for water vapor: %.0f m\n', hS_1)
fprintf('Average velocity for falling precipitation: %.1f\n', hS_1/tauF_1)
fprintf('Total density at sea level: %.2f g/m^3\n', rho0_1);
fprintf('Scale height for total density: %.0f m\n', hRho_1)
fprintf('Average lapse-rate ratio, gammaSat/gammaEnv: %.2f\n', gammaRatio_1)
fprintf('\nPrecipitation State #2:\n')
fprintf('Moist buoyancy frequency: %.3f mrad/s\n', NM_2*1e3)
fprintf('d18O for base precipitation: %.1f per mil\n', d18O0_2*1e3)
fprintf('d18O latitude gradient for base precipitation: %.3f per mil/deg lat\n', dD18O0_dLat_2*1e3)
fprintf('Average residence time for falling precipitation: %.0f s\n', tauF_2)
fprintf('Water-vapor density at sea level: %.2f g/m^3\n', rhoS0_2*1e3)
fprintf('Scale height for water vapor: %.0f m\n', hS_2)
fprintf('Average velocity for falling precipitation: %.1f m/s\n', hS_2/tauF_2)
fprintf('Total density at sea level: %.2f g/m^3\n', rho0_2);
fprintf('Scale height for total density: %.0f m\n', hRho_2)
fprintf('Average lapse-rate ratio, gammaSat/gammaEnv: %.2f\n', gammaRatio_2)
fprintf('\n------------ Observed Meteoric Water Line -----------\n')
fprintf('Principal standard deviations (per mil): %.2f, %.2f\n', ...
    sdDataMin*1e3, sdDataMax*1e3);
fprintf('Intercept and slope (per mil): %.2f, %.2f\n', ...
    bMWLSample(1)*1e3, bMWLSample(2))
fprintf('\n----------- Predicted Meteoric Water Line -----------\n')
fprintf('Principal standard deviations (per mil): %.2f, %.2f\n', ...
    sdPredMin*1e3, sdPredMax*1e3);
fprintf('Intercept and slope (per mil): %.1f, %.2f\n', ...
    bMWLPred(1)*1e3, bMWLPred(2))
fprintf('\n------------------ Quality of Fit -------------------\n')
fprintf('Reduced chi-square: %.4f\n', chiR2)
fprintf('Degrees of freedom: %d\n', nu);
fprintf('Number of primary samples in wet locations: %d\n', nSamplesWet)
fprintf('Number of primary samples: %d\n', nSamples);
fprintf('Standard deviation of wet residuals for d2H: %.1f per mil\n', ...
    sdRes_d2H*1e3)
fprintf('Standard deviation of wet residuals for d18O: %.1f per mil\n', ...
    sdRes_d18O*1e3)
fprintf('Approximate standard error for predicted d2H: %.1f per mil\n', ...
    se_d2HPred*1e3)
fprintf('Approximate standard error for predicted d18O: %.1f per mil\n', ...
    se_d18OPred*1e3)
fprintf('\n----------------- Computation Time ------------------\n')
fprintf('Duration for computation: %.2f minutes\n', ...
    minutes(datetime - startTimeOpiCalc));
fprintf('\n--------------------- Mat File ----------------------\n')
fprintf('Results saved in the run directory as mat file:\n%s\n', ...
    [mfilename, '_Results.mat']);
%... Close the diary
diary off;

end
