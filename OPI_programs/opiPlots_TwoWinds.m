function opiPlots_TwoWinds
% opiPlots_TwoWinds takes as input a run file and an associated mat file
% with a "two-winds" solution, and creates plot figures of the results.
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
startTime = datetime;
%... Convert from Celsius to kelvin
TC2K = 273.15;
%... Mean radius of the Earth (m)
radiusEarth = 6371e3;
%... Meters per arc degree for the Earth's surface
mPerDegree = pi*radiusEarth/180;
%... Colors for plotting
blue = [57 106 177]./255;
red = [204 37 41]./255;
% black = [83 81 84]./255;
% green = [62 150 81]./255;

%% Load results from opiCalc matfile with best-fit solution
%... Load results from opiCalc matfile
matPathResults = fnPersistentPath;
[matFile_Results, matPathResults] = uigetfile([matPathResults,'/opiCalc*.mat']);
if matFile_Results==0, error('opiCalc matfile not found'), end
%... Remove terminal slash, if present
if matPathResults(end)=='/' || matPathResults(end)=='\'
    matPathResults = matPathResults(1:end-1);
end
fnPersistentPath(matPathResults);

%... Mat file content for two-winds solution
% 'startTimeOpiCalc', 'runPath', 'runFile', 'runTitle', ...
% 'dataPath', 'topoFile', 'rTukey', 'sampleFile', ...
% 'sectionLon0', 'sectionLat0', ...
% 'lon', 'lat', 'x', 'y', 'lon0', 'lat0', ...
% 'sampleLine', 'sampleLon', 'sampleLat', 'sampleX', 'sampleY', ...
% 'sampleD2H', 'sampleD18O', 'sampleLC', ...
% 'sampleLineAlt', 'sampleLonAlt', 'sampleLatAlt', 'sampleXAlt', 'sampleYAlt', ...
% 'sampleD2HAlt', 'sampleD18OAlt', 'sampleLCAlt', ...
% 'bMWLSample', 'ijCatch', 'ptrCatch', 'isSampleSide01', ...
% 'hR', 'sdResRatio', 'sdDataMin', 'sdDataMax','cov', 'fC', ...
% 'lB', 'uB', 'nParametersFree', ...
% 'beta', 'chiR2', 'nu', 'stdResiduals', ...
% 'zBar_1', 'T_1', 'gammaEnv_1', 'gammaSat_1', 'gammaRatio_1', ...
% 'rhoS0_1', 'hS_1', 'rho0_1', 'hRho_1', ...    
% 'd18O0_1', 'dD18O0_dLat_1', 'tauF_1', 'pGrid_1', 'fMGrid_1', 'rHGrid_1', ...
% 'd2HGrid_1', 'd18OGrid_1', 'pSumPred_1', 'd2HPred_1', 'd18OPred_1', ...
% 'liftMaxPred_1', 'elevationPred_1', ...
% 'zBar_2', 'T_2', 'gammaEnv_2', 'gammaSat_2', 'gammaRatio_2', ...
% 'rhoS0_2', 'hS_2', 'rho0_2', 'hRho_2', ...    
% 'd18O0_2', 'dD18O0_dLat_2', 'tauF_2', 'pGrid_2', 'fMGrid_2', 'rHGrid_2', ...
% 'd2HGrid_2', 'd18OGrid_2', 'pSumPred_2', 'd2HPred_2', 'd18OPred_2', ...
% 'liftMaxPred_2', 'elevationPred_2', ...
% 'pGrid', 'd2HGrid', 'd18OGrid', 'pSumPred', 'd2HPred', 'd18OPred', ...
% 'liftMaxPred', 'elevationPred', 'fractionPGrid', '-v7.3');
load([matPathResults, '/', matFile_Results], 'beta')
if length(beta)==9, error('Mat file is for a one-wind solution.'), end
load([matPathResults, '/', matFile_Results], ...
    'runPath', 'runFile', 'runTitle', ...
    'dataPath', 'topoFile', 'rTukey', 'sampleFile', ...
    'sectionLon0', 'sectionLat0', ...
    'lon', 'lat', 'lon0', 'lat0', ...
    'sampleLon', 'sampleLat', 'sampleX', 'sampleY', ...
    'sampleD2H', 'sampleD18O', 'sampleLC', 'sampleLonAlt', 'bMWLSample', ...
    'isSampleSide01', ...
    'lB', 'uB', ...
    'hR', 'sdResRatio', 'sdDataMin', 'sdDataMax','cov', 'fC', ...
    'chiR2', 'nu', 'stdResiduals', ...
    'gammaRatio_1', ...
    'rhoS0_1', 'hS_1', 'rho0_1', 'hRho_1', ...
    'd18O0_1', 'dD18O0_dLat_1', 'tauF_1', ...
    'pSumPred_1', 'd2HPred_1', 'd18OPred_1', ...
    'liftMaxPred_1', 'elevationPred_1', ...
    'gammaRatio_2', ...
    'rhoS0_2', 'hS_2', 'rho0_2', 'hRho_2', ...
    'd18O0_2', 'dD18O0_dLat_2', 'tauF_2', ...
    'pSumPred_2', 'd2HPred_2', 'd18OPred_2', ...
    'liftMaxPred_2', 'elevationPred_2', ...
    'pSumPred', 'd2HPred', 'd18OPred');

%... Test for samples
if isempty([sampleLon, sampleLat])
    error('Samples must be present for opiPlots_TwoWinds.')
end

%... If empty, set section origin to map origin 
if isempty([sectionLon0, sectionLat0])
    sectionLon0 = lon0;
    sectionLat0 = lat0;
end

%... Read topographic data
[~, ~, hGrid] = gridRead([dataPath, '/', topoFile]);
if any(isnan(hGrid(:))), error('Elevation grid contains nans.'), end
%... Set elevations below sea level to 0 meters
hGrid(hGrid<0) = 0;

% Create 2D cosine-taper window, with
% fractional width = rTukey/2 on each margin of the grid.
[nY, nX] = size(hGrid);
window = tukeywin(nY, rTukey)*tukeywin(nX, rTukey)';
hGrid = window.*hGrid;
clear window

%... Unpack beta, the solution vector
% Precipitation state #1
U_1 = beta(1);           % wind speed (m/s)
azimuth_1 = beta(2);     % azimuth (degrees)
T0_1 = beta(3);          % sea-level temperature (K)
M_1 = beta(4);           % mountain-height number (dimensionless)
kappa_1 = beta(5);       % eddy diffusion (m/s^2)
tauC_1 = beta(6);        % condensation time (s)
d2H0_1 = beta(7);        % d2H of base precipitation (per unit)
dD2H0_dLat_1 = beta(8);  % latitudinal gradient of base-prec d2H (1/deg lat)
fP0_1 = beta(9);         % residual precipitation after evaporation (fraction)
fraction = beta(10);     % fractional size of precipitation state 1
% Precipitation state #2
U_2 = beta(11);          % wind speed (m/s)
azimuth_2 = beta(12);    % azimuth (degrees)
T0_2 = beta(13);         % sea-level temperature (K)
M_2 = beta(14);          % mountain-height number (dimensionless)
kappa_2 = beta(15);      % eddy diffusion (m/s^2)
tauC_2 = beta(16);       % condensation time (s)
d2H0_2 = beta(17);       % d2H of base precipitation (per unit)
dD2H0_dLat_2 = beta(18); % latitudinal gradient of base-prec d2H (1/deg lat)
fP0_2 = beta(19);        % residual precipitation after evaporation (fraction)
%... Convert mountain-height number to buoyancy frequency (rad/s)
hMax = max(hGrid, [], 'all');
NM_1 = M_1*U_1/hMax;
NM_2 = M_2*U_2/hMax;

%... Number of samples
nSamples = length(sampleLon);
nSamplesAlt = length(sampleLonAlt);

%... Sum number of precipitation-present samples
iWet_1 = (pSumPred_1>0);
iWet_2 = (pSumPred_2>0);
iWet = (pSumPred>0);
nSamplesWet = sum(iWet);

%... Calculate statistics for predicted water isotopes
[bMWLPred, sdPredMin, sdPredMax] = ...
    estimateMWL(d18OPred(iWet), d2HPred(iWet), sdResRatio);

%% Caculate best-fit lines for isotopes vs lifting. 
% The precipitation isotopes are represented using their predicted 
% values from the best-fit OPI solution, either as point estimates if
% the sample location is designated as "local" (type L), or as the 
% precipitated-weighted value for the upstream catchment if the sample is 
% designated as "catchment" (type C). 
% The lifting is represented either by elevation or by the maximum
% lifting along the upwind path. The local elevation and maximum lifting are 
% calculated as either "local" or "catchment" values depending on the
% designation of the sample (type L or C).  
% The isotopes are not adjusted for latitudinal gradients 
% (dD2H0_dLat, dD18O0_dLat), but this effect is will only contribute small
% symmetric errors. The slope of the lines are estimated by least squares
% and the intercept is held fixed so that it matches the estimated 
% base isotope values (d2H0, d18O).
% Note: the terms sample, pred, and grid refer to 
% sample observations, predicted values for sample locations (which can be
% either "local" or "catchment", and local estimates for nodes in a grid.

%... Predicted isotopes vs maximum lifting, state #1
x = liftMaxPred_1(iWet_1);
y = d2HPred_1(iWet_1) - d2H0_1;
bLiftMaxLine_d2HPred_1(1) = sum(y.*x)./sum(x.^2);
bLiftMaxLine_d2HPred_1(2) = d2H0_1;
y = d18OPred_1(iWet_1) - d18O0_1;
bLiftMaxLine_d18OPred_1(1) = sum(y.*x)./sum(x.^2);
bLiftMaxLine_d18OPred_1(2) = d18O0_1;

%... Predicted isotopes vs maximum lifting, state #2
x = liftMaxPred_2(iWet_2);
y = d2HPred_2(iWet_2) - d2H0_2;
bLiftMaxLine_d2HPred_2(1) = sum(y.*x)./sum(x.^2);
bLiftMaxLine_d2HPred_2(2) = d2H0_2;
y = d18OPred_2(iWet_2) - d18O0_2;
bLiftMaxLine_d18OPred_2(1) = sum(y.*x)./sum(x.^2);
bLiftMaxLine_d18OPred_2(2) = d18O0_2;

%... Predicted isotopes vs elevation, state #1
x = elevationPred_1(iWet_1);
y = d2HPred_1(iWet_1) - d2H0_1;
bLineElev_d2HPred_1(1) = sum(y.*x)./sum(x.^2);
bLineElev_d2HPred_1(2) = d2H0_1;
y = d18OPred_1(iWet_1) - d18O0_1;
bLineElev_d18OPred_1(1) = sum(y.*x)./sum(x.^2);
bLineElev_d18OPred_1(2) = d18O0_1;

%... Predicted isotopes vs elevation, state #2
x = elevationPred_2(iWet_2);
y = d2HPred_2(iWet_2) - d2H0_2;
bLineElev_d2HPred_2(1) = sum(y.*x)./sum(x.^2);
bLineElev_d2HPred_2(2) = d2H0_2;
y = d18OPred_2(iWet_2) - d18O0_2;
bLineElev_d18OPred_2(1) = sum(y.*x)./sum(x.^2);
bLineElev_d18OPred_2(2) = d18O0_2;

%% Report results
%... Start diary file
logFilename=[matPathResults, '/', mfilename, '_Log.txt'];
if isfile(logFilename), delete (logFilename); end
diary(logFilename);
fprintf (['Program: ', mfilename, '\n'])
fprintf('Start time: %s\n', startTime)
fprintf('Run file path:\n%s\n', runPath)
fprintf('Run file name:\n%s\n', runFile)
fprintf('Run title:\n');
fprintf('%s\n', runTitle);
fprintf('Path name for data directory:\n')
fprintf('%s\n', dataPath)
fprintf('\n------------------ Topography File ------------------\n')
fprintf('Topography file: %s\n', topoFile)
fprintf('Maximum elevation (m): %.0f\n', hMax);
[nY, nX] = size(hGrid);
fprintf('Grid size, nx and ny: %d, %d\n', nX, nY)
fprintf('Minimum and maximum for longitude: %.5f, %.5f degrees\n', lon(1), lon(end))
fprintf('Minimum and maximum for latitude: %.5f, %.5f degrees\n', lat(1), lat(end))
dLon = lon(2) - lon(1);
dLat = lat(2) - lat(1);
fprintf('Grid spacing, dLon and dLat (degrees): %.5f, %.5f\n', dLon, dLat)
fprintf('Grid spacing, dx and dy (km): %.2f, %.2f\n', ...
    dLon*mPerDegree*1e-3*cosd(lat0), dLat*mPerDegree*1e-3)
fprintf('Lon, lat for map origin (degrees): %.5f, %.5f\n', lon0, lat0);
if lon0==mean(sampleLon) && lat0==mean(sampleLat)
    fprintf('Map origin is set to sample centroid.\n')
else
    fprintf('Map origin is set to the center of the topographic grid.\n')
end
fprintf('Size of cosine window as fraction of grid size: %g (dimensionless)\n', rTukey)
fprintf('Coriolis frequency at map-origin latitude (mrad/s): %.3f\n', ...
    fC*1e3)
fprintf('Lon, lat for section origin (degrees): %.5f, %.5f\n', sectionLon0, sectionLat0);
fprintf('\n-------------------- Sample File --------------------\n')
fprintf('Sample file: %s\n', sampleFile)
fprintf('Number of all samples: %d\n', nSamples + nSamplesAlt)
fprintf('Number of altered samples: %d\n', nSamplesAlt)
fprintf('Number of primary samples: %d\n', nSamples)
fprintf('Number of local primary samples: %d\n', sum(sampleLC=='L'))
fprintf('Number of catchment primary samples: %d\n', sum(sampleLC=='C'))
fprintf('Centroid for primary samples, longitude, latitude: %.5f, %.5f degrees\n', ...
    mean(sampleLon), mean(sampleLat))
fprintf('Minimum and maximum for longitude: %.5f, %.5f\n', ...
    min(sampleLon), max(sampleLon))
fprintf('Minimum and maximum for latitude: %.5f, %.5f\n', ...
    min(sampleLat), max(sampleLat))
fprintf('\n---------------------- Constants --------------------\n')
fprintf('Characteristic distance for isotopic exchange (m): %.0f\n', hR)
fprintf('Standard-deviation ratio for data residuals: %.2f (dimensionless)\n', sdResRatio)
fprintf('\n----------------- Evaporation Option ----------------\n')
if lB(9)==1 && uB(9)==1 && lB(19)==1 && uB(19)==1
    fprintf('No Evaporative recycling active for both precipitation states.\n')
else
    fprintf('Evaporative recycling active for both precipitation states.\n')
end
fprintf('\n---------------------- Solution ---------------------\n')
fprintf('Precipitation State #1:\n')
fprintf('Wind speed: %.1f m/s\n', U_1)
fprintf('Azimuth: %.1f degrees\n', azimuth_1)
fprintf('Sea-level surface-air temperature: %.1f K (%.1f °C)\n', T0_1, T0_1 - TC2K)
fprintf('Mountain-height number: %.3f (dimensionless)\n', M_1)
fprintf('Horizontal eddy diffusivity: %.0f m^2/s\n', kappa_1)
fprintf('Average residence time for cloud water: %.0f s\n', tauC_1)
fprintf('d2H for base precipitation: %.1f per mil\n', d2H0_1*1e3)
fprintf('d2H latitude gradient for base precipitation: %.3f per mil/deg lat\n', dD2H0_dLat_1*1e3)
fprintf('Residual precipitation after evaporation: %.2f (dimensionless)\n', fP0_1)
fprintf('Fraction for precipitation state #1: %.2f\n', fraction);
fprintf('\nPrecipitation State #2:\n')
fprintf('Wind speed: %.1f m/s\n', U_2)
fprintf('Azimuth: %.1f degrees\n', azimuth_2)
fprintf('Sea-level surface-air temperature: %.1f K (%.1f °C)\n', T0_2, T0_2 - TC2K)
fprintf('Mountain-height number: %.3f (dimensionless)\n', M_2)
fprintf('Horizontal eddy diffusivity: %.0f m^2/s\n', kappa_2)
fprintf('Average residence time for cloud water: %.0f s\n', tauC_2)
fprintf('d2H for base precipitation: %.1f per mil\n', d2H0_2*1e3)
fprintf('d2H latitude gradient for base precipitation: %.3f per mil/deg lat\n', dD2H0_dLat_2*1e3)
fprintf('Residual precipitation after evaporation: %.2f (dimensionless)\n', fP0_2)
fprintf('\n---- Other Variables Related to Best-Fit Solution ---\n')
fprintf('\nPrecipitation State #1:\n')
fprintf('Moist buoyancy frequency: %.3f mrad/s\n', NM_1*1e3)
fprintf('d18O for base precipitation: %.1f per mil\n', d18O0_1*1e3)
fprintf('d18O latitude gradient for base precipitation: %.3f per mil/deg lat\n', dD18O0_dLat_1*1e3)
fprintf('Average residence time for falling precipitation: %.0f s\n', tauF_1)
fprintf('Water-vapor density at sea level: %.2f g/m^3\n', rhoS0_1*1e3)
fprintf('Scale height for water vapor: %.0f m\n', hS_1)
fprintf('Average velocity for falling precipitation: %.1f m/s\n', hS_1/tauF_1)
fprintf('Total density at sea level: %.2f g/m^3\n', rho0_1);
fprintf('Scale height for total density: %.0f m\n', hRho_1)
fprintf('Average lapse-rate ratio, gammaSat/gammaEnv: %.2f (dimensionless)\n', gammaRatio_1)
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
fprintf('Average lapse-rate ratio, gammaSat/gammaEnv: %.2f (dimensionless)\n', gammaRatio_2)
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
fprintf('\n------------ Estimates for Lifting Lines-------------\n')
fprintf('The precipitation isotopes are represented using their predicted\n')
fprintf('values from the best-fit OPI solution, either as point estimates if\n')
fprintf('the sample location is designated as "local" (type L), or as the\n')
fprintf('precipitated-weighted value for the upstream catchment if the sample is\n')
fprintf('designated as "catchment" (type C).\n')
fprintf('The lifting is represented either by elevation or by the maximum\n')
fprintf('lifting along the upwind path. The elevation and maximum lifting are\n')
fprintf('calculated as either "local" or "catchment" values depending on the\n')
fprintf('designation of the sample (type L or C). \n')
fprintf('The isotopes are not adjusted for latitudinal gradients\n')
fprintf('(dd2H0_dLat, dd18O0_dLat), but this source of error is small and \n')
fprintf('symmetric. The slope of the lines are estimated by least squares\n')
fprintf('but the intercept is held fixed so that it matches the estimated\n')
fprintf('base isotope values (d2H0, d18O).\n')

fprintf('\nPredicted Isotopes vs Maximum Lifting, State #1\n')
fprintf('Intercept (per mil) and slope (per mil/km) for d2H: %.1f, %.1f\n', ...
    bLiftMaxLine_d2HPred_1(2)*1e3, bLiftMaxLine_d2HPred_1(1)*1e6)
fprintf('Intercept (per mil) and slope (per mil/km) for d18O: %.1f, %.1f\n', ...
    bLiftMaxLine_d18OPred_1(2)*1e3, bLiftMaxLine_d18OPred_1(1)*1e6)
fprintf('[%g\t%g\t%g\t%g]\n', ...
    bLiftMaxLine_d2HPred_1(2)*1e3, bLiftMaxLine_d2HPred_1(1)*1e6, ...    
    bLiftMaxLine_d18OPred_1(2)*1e3, bLiftMaxLine_d18OPred_1(1)*1e6);

fprintf('\nPredicted Isotopes vs Maximum Lifting, State #2\n')
fprintf('Intercept (per mil) and slope (per mil/km) for d2H: %.1f, %.1f\n', ...
    bLiftMaxLine_d2HPred_2(2)*1e3, bLiftMaxLine_d2HPred_2(1)*1e6)
fprintf('Intercept (per mil) and slope (per mil/km) for d18O: %.1f, %.1f\n', ...
    bLiftMaxLine_d18OPred_2(2)*1e3, bLiftMaxLine_d18OPred_2(1)*1e6)
fprintf('[%g\t%g\t%g\t%g]\n', ...
    bLiftMaxLine_d2HPred_2(2)*1e3, bLiftMaxLine_d2HPred_2(1)*1e6, ...    
    bLiftMaxLine_d18OPred_2(2)*1e3, bLiftMaxLine_d18OPred_2(1)*1e6);

fprintf('\nPredicted Isotopes vs Elevation, State #1\n')
fprintf('Intercept (per mil) and slope (per mil/km) for d2H: %.1f, %.1f\n', ...
    bLineElev_d2HPred_1(2)*1e3, bLineElev_d2HPred_1(1)*1e6)
fprintf('Intercept (per mil) and slope (per mil/km) for d18O: %.1f, %.1f\n', ...
    bLineElev_d18OPred_1(2)*1e3, bLineElev_d18OPred_1(1)*1e6)

fprintf('\nPredicted Isotopes vs Elevation, State #2\n')
fprintf('Intercept (per mil) and slope (per mil/km) for d2H: %.1f, %.1f\n', ...
    bLineElev_d2HPred_2(2)*1e3, bLineElev_d2HPred_2(1)*1e6)
fprintf('Intercept (per mil) and slope (per mil/km) for d18O: %.1f, %.1f\n', ...
    bLineElev_d18OPred_2(2)*1e3, bLineElev_d18OPred_2(1)*1e6)

fprintf('\n------------------ Quality of Fit -------------------\n')
fprintf('Reduced chi-square: %.4f\n', chiR2)
fprintf('Degrees of freedom: %d\n', nu);
fprintf('Number of primary samples in wet locations: %d\n', nSamplesWet)
fprintf('Number of primary samples: %d\n', nSamples);

%... Close the diary
diary off;

%% Plot figures
Fig01 % Plot predicted vs observed isotope values
Fig02 % Plot observed and predicted d2H and d18O
Fig03 % Plot standarized residuals as a function of x and y
Fig04 % Plot predicted isotopes versus maximum lifting
Fig05 % Plot predicted isotopes versus elevation

%% Fig01, Plot predicted vs observed isotope values
    function Fig01
        figure(1)
        subplot(2,1,1) %... delta 2H results
        %... Plot isotope data, with color indicating the
        % indicating the location of the sample relative to the 
        % drainage divide. Samples that are on the state #1 side of the
        % topography are red, and those on the opposite side are blue. 
        hold on
        plot(sampleD2H(iWet&isSampleSide01)*1e3, ...
            d2HPred(iWet&isSampleSide01)*1e3, 'o', 'LineWidth', 2, 'Color', blue);
        plot(sampleD2H(iWet&~isSampleSide01)*1e3, ...
            d2HPred(iWet&~isSampleSide01)*1e3, 'o', 'LineWidth', 2, 'Color', red);
        %... Plot isotopic composition of base precipitation
        plot(d2H0_1*1e3, d2H0_1*1e3,'s', 'MarkerSize', 24, ...
            'LineWidth', 3, 'Color', blue)
        plot(d2H0_2*1e3, d2H0_2*1e3,'s', 'MarkerSize', 24, ...
            'LineWidth', 3, 'Color', red)
        axis square
        %... Plot 1:1 reference line (place after axis square)
        hLine = refline(1, 0);
        hLine.Color = [0.5 0.5 0.5];
        hLine.LineWidth = 3;
        uistack(hLine, 'down')
        %... Additional formatting
        grid on
        grid minor
        box on
        set(gca, 'FontSize', 16, 'LineWidth', 1);
        %... Write labels
        hT = title('Fig. 1. Observed versus predicted isotope values', ...
            'FontSize', 16);
        hT.Units = 'normalized'; 
        hT.Position(2) = hT.Position(2) + 0.02;
        xlabel(['Observed \delta^{2} (', char(8240), ')'], 'FontSize', 16)
        ylabel(['Predicted \delta^{2} (', char(8240), ')'], 'FontSize', 16)
                
        subplot(2,1,2) %... delta 18O results
        %... Plot isotope data, with color indicating the
        % indicating the location of the sample relative to the 
        % drainage divide. Samples that are on the state #1 side of the
        % topography are blue, and those on the opposite side are red. 
        hold on
        plot(sampleD18O(iWet&isSampleSide01)*1e3, ...
            d18OPred(iWet&isSampleSide01)*1e3, 'o', 'LineWidth', 2, 'Color', blue);        
        plot(sampleD18O(iWet&~isSampleSide01)*1e3, ...
            d18OPred(iWet&~isSampleSide01)*1e3, 'o', 'LineWidth', 2, 'Color', red);        
        %... Plot isotopic composition of base precipitation
        plot(d18O0_1*1e3, d18O0_1*1e3, 's', 'MarkerSize', 24, ...
            'LineWidth', 3, 'Color', blue)
        plot(d18O0_2*1e3, d18O0_2*1e3, 's', 'MarkerSize', 24, ...
            'LineWidth', 3, 'Color', red)
        axis square
        %... Plot 1:1 reference line (place after axis square)
        hLine = refline(1, 0);
        hLine.Color = [0.5 0.5 0.5];
        hLine.LineWidth = 3;
        uistack(hLine, 'down')
        %... Additional formatting
        grid on
        grid minor
        box on
        set(gca, 'FontSize', 16, 'LineWidth', 1);
        xlabel(['Observed \delta^{18}O (', char(8240), ')'], 'FontSize', 16)
        ylabel(['Predicted \delta^{18}O (', char(8240), ')'], 'FontSize', 16)
        %... Save figure in pdf format
        printFigure(matPathResults)
    end

%% Fig02, Craig plot, with primary samples only
    function Fig02
        figure(2)
        hold on
        % Plot observed using medium gray
        hP1 = plot(sampleD18O(iWet)*1e3, sampleD2H(iWet)*1e3, ...
            'o', 'Color', [0.7 0.7 0.7], 'MarkerSize', 20, 'LineWidth', 2);
        % Plot predicted using blue and red for sides 1 and 2, respectively
        hP2 = plot(d18OPred(iWet&isSampleSide01)*1e3, ...
            d2HPred(iWet&isSampleSide01)*1e3, 'o', ...
            'LineWidth', 2, 'MarkerSize', 20, 'Color', blue);
        hP3 = plot(d18OPred(iWet&~isSampleSide01)*1e3, ...
            d2HPred(iWet&~isSampleSide01)*1e3, 'o', ...
            'LineWidth', 2, 'MarkerSize', 20, 'Color', red);
        %... Plot isotopic composition of base precipitation
        plot(d18O0_1*1e3, d2H0_1*1e3, 's', 'MarkerSize', 24, ...
            'LineWidth', 3, 'Color', blue);
        plot(d18O0_2*1e3, d2H0_2*1e3, 's', 'MarkerSize', 24, ...
            'LineWidth', 3, 'Color', red);
        %... Plot one-sigma ellipse around base precipitation composition
        % Cholesky decomposition provides the square root of cov matrix
        % See "Drawing Confidence Ellipses and Ellipsoids", jellymatter.wordpress.com
        rHat = [cosd(0:0.5:360); sind(0:0.5:360)]';
        ellipse = rHat*chol(cov) + [d2H0_1, d18O0_1];
        plot(ellipse(:,2)*1e3, ellipse(:,1)*1e3, '-', ...
            'LineWidth', 3, 'Color', blue)
        ellipse = rHat*chol(cov) + [d2H0_2, d18O0_2];
        plot(ellipse(:,2)*1e3, ellipse(:,1)*1e3, '-', ...
            'LineWidth', 3, 'Color', red)
        axis square tight
        %... Plot meteoric water line
        hLine = refline(bMWLSample(2), bMWLSample(1)*1e3);
        hLine.Color = [0.5 0.5 0.5];
        hLine.LineWidth = 3;        
        %... Write legend
        legend([hP1, hP2, hP3, hLine], ...
            'Observed', 'Predicted, Side 1', 'Predicted, Side 2', ...
            'Sample MWL', 'Location', 'Northwest')
        %... Format plot
        grid on
        grid minor
        %... Write labels
        hT = title('Fig. 2. Craig plot of primary samples', ...
            'FontSize', 16);
        hT.Units = 'normalized'; 
        hT.Position(2) = hT.Position(2) + 0.02;
        xlabel(['\delta^{18}O (', char(8240), ')'], 'FontSize', 16)
        ylabel(['\delta^{2}H (', char(8240), ')'], 'FontSize', 16)
        set(gca, 'FontSize', 16, 'LineWidth', 1);
        %... Save figure in pdf format
        printFigure(matPathResults)
    end

%% Fig03, Plot standardized residuals as a function of easting and northing
    function Fig03
        figure(3)
        %... Plot standardized residuals as a function of x direction
        subplot(2,1,1)
        hold on
        plot(sampleX(iWet&isSampleSide01)*1e-3, ...
            stdResiduals(iWet&isSampleSide01), 'o', 'Color', blue);
        plot(sampleX(iWet&~isSampleSide01)*1e-3, ...
            stdResiduals(iWet&~isSampleSide01), 'o', 'Color', red);
        grid on
        grid minor
        %... Write labels for first plot
        hT = title('Fig. 3. Standardized residuals for best-fit solution', ...
            'FontSize', 16);
        hT.Units = 'normalized'; 
        hT.Position(2) = hT.Position(2) + 0.02;                
        xlabel('Easting (km)', 'FontSize', 16);
        ylabel('Standardized Residuals', 'FontSize', 16)
        set(gca, 'FontSize', 16, 'LineWidth', 1);

        %... Plot standardized residuals as a function of y direction
        subplot(2,1,2)
        hold on
        plot(sampleY(iWet&isSampleSide01)*1e-3, ...
            stdResiduals(iWet&isSampleSide01), 'o', 'Color', blue);
        plot(sampleY(iWet&~isSampleSide01)*1e-3, ...
            stdResiduals(iWet&~isSampleSide01), 'o', 'Color', red);
        grid on
        grid minor
        %... Write labels for second plot
        xlabel('Northing (km)', 'FontSize', 16);
        ylabel('Standardized Residuals', 'FontSize', 16)
        set(gca, 'FontSize', 16, 'LineWidth', 1);
        %... Save figure in pdf format
        printFigure(matPathResults)
    end

%% Fig04, Plot predicted isotopes versus maximum lifting
    function Fig04
        figure(4)
        %... Plot predicted d2H relative to maximum lifting
        % Blue and red here indicate precipitation associated
        % with state #1 and #2, respectively.
        subplot(2,1,1)
        hold on
        hP1 = plot(liftMaxPred_1(iWet_1), d2HPred_1(iWet_1)*1e3, ...
            'o', 'Color', blue);
        hP2 = plot(liftMaxPred_2(iWet_2), d2HPred_2(iWet_2)*1e3, ...
            'o', 'Color', red);
        xlim([0, inf])
        axis square
        %... Plot predicted lifting lines for each moisture source
        hL1 = refline(bLiftMaxLine_d2HPred_1*1e3);
        hL1.LineWidth = 3;
        uistack(hL1, 'down')
        hL1.Color = 'b';
        hL2 = refline(bLiftMaxLine_d2HPred_2*1e3);
        uistack(hL2, 'down')
        hL2.LineWidth = 3;
        hL2.Color = 'r';
        %... Move data to the top of layer
        uistack([hP1, hP2], 'top');
        %... Additional formatting
        grid on
        grid minor
        %... Write labels for first plot
        hT = title('Fig. 4. Predicted isotopes versus maximum lifting', ...
            'FontSize', 16);
        hT.Units = 'normalized';
        hT.Position(2) = hT.Position(2) + 0.02;
        xlabel('Maximum lifting (m)', 'FontSize', 16);
        ylabel(['Predicted \delta^{2} (', char(8240), ')'], 'FontSize', 16)
        hA = gca;
        hA.FontSize = 16;
        hA.LineWidth = 1;
        
        %... Plot predicted d18O relative to maximum lifting
        % Blue and red here indicate precipitation associated
        % with state #1 and #2, respectively.
        subplot(2,1,2)
        hold on
        hP1 = plot(liftMaxPred_1(iWet_1), d18OPred_1(iWet_1)*1e3, ...
            'o', 'Color', blue);
        hP2 = plot(liftMaxPred_2(iWet_2), d18OPred_2(iWet_2)*1e3, ...
            'o', 'Color', red);
        xlim([0, inf])
        axis square
        %... Plot predicted lifting lines for each moisture source
        hL1 = refline(bLiftMaxLine_d18OPred_1*1e3);
        hL1.LineWidth = 3;
        uistack(hL1, 'down')
        hL1.Color = 'b';        
        hL2 = refline(bLiftMaxLine_d18OPred_2*1e3);
        uistack(hL2, 'down')
        hL2.LineWidth = 3;
        hL2.Color = 'r';        
        %... Move data to the top of layer
        uistack([hP1, hP2], 'top');
        %... Additional formatting
        grid on
        grid minor
        %... Write labels for second plot
        xlabel('Maximum lifting (m)', 'FontSize', 16);
        ylabel(['Predicted \delta^{18} (', char(8240), ')'], 'FontSize', 16)
        hA = gca;
        hA.FontSize = 16;
        hA.LineWidth = 1;
        %... Save figure in pdf format
        printFigure(matPathResults)
    end

%% Fig05, Plot predicted isotopes versus elevation
    function Fig05
        figure(5)
        %... Plot predicted d2H relative to elevation
        % Blue and red here indicate precipitation associated
        % with state #1 and #2, respectively.
        subplot(2,1,1)
        hold on
        hP1 = plot(elevationPred_1(iWet_1), d2HPred_1(iWet_1)*1e3, ...
            'o', 'Color', blue);
        hP2 = plot(elevationPred_2(iWet_2), d2HPred_2(iWet_2)*1e3, ...
            'o', 'Color', red);
        xlim([0, inf])
        axis square
        %... Plot predicted lifting lines for each moisture source
        hL1 = refline(bLineElev_d2HPred_1*1e3);
        hL1.LineWidth = 3;
        uistack(hL1, 'down')
        hL1.Color = 'b';
        hL2 = refline(bLineElev_d2HPred_2*1e3);
        uistack(hL2, 'down')
        hL2.LineWidth = 3;
        hL2.Color = 'r';
        %... Move data to the top of layer
        uistack([hP1, hP2], 'top');
        %... Additional formatting
        grid on
        grid minor
        %... Write labels for first plot
        hT = title('Fig. 5. Predicted isotopes versus elevation', ...
            'FontSize', 16);
        hT.Units = 'normalized'; 
        hT.Position(2) = hT.Position(2) + 0.02;                
        xlabel('Elevation (m)', 'FontSize', 16);
        ylabel(['Predicted \delta^{2} (', char(8240), ')'], 'FontSize', 16)
        hA = gca;
        hA.FontSize = 16;
        hA.LineWidth = 1;
        
        %... Plot predicted d18O relative to elevation
        % Blue and red here indicate precipitation associated
        % with state #1 and #2, respectively.
        subplot(2,1,2)
        hold on
        hP1 = plot(elevationPred_1(iWet_1), d18OPred_1(iWet_1)*1e3, ...
            'o', 'Color', blue);
        hP2 = plot(elevationPred_2(iWet_2), d18OPred_2(iWet_2)*1e3, ...
            'o', 'Color', red);
        xlim([0, inf])
        axis square
        %... Plot predicted lines for each moisture source
        hL1 = refline(bLineElev_d18OPred_1*1e3);
        hL1.LineWidth = 3;
        uistack(hL1, 'down')
        hL1.Color = 'b';        
        hL2 = refline(bLineElev_d18OPred_2*1e3);
        uistack(hL2, 'down')
        hL2.LineWidth = 3;
        hL2.Color = 'r';        
        %... Move data to the top of layer
        uistack([hP1, hP2], 'top');
        %... Additional formatting
        grid on
        grid minor
        %... Write labels for second plot
        xlabel('Elevation (m)', 'FontSize', 16);
        ylabel(['Predicted \delta^{18} (', char(8240), ')'], 'FontSize', 16)
        hA = gca;
        hA.FontSize = 16;
        hA.LineWidth = 1;
        %... Save figure in pdf format
        printFigure(matPathResults)
    end
end
