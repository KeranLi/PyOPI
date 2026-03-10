function opiCalc_OneWind
% opiCalc_OneWind Calculates the full range of results for an OPI 
% solution, as represented by a run file with a "one-wind" solution. 
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
%... Convert from Celsius to kelvin
TC2K = 273.15;
%... Start time for calculation
startTimeOpiCalc = datetime; 
%... Mean radius of the Earth (m)
radiusEarth = 6371e3; 
%... Meters per arc degree for the Earth's surface
mPerDegree = pi*radiusEarth/180;
%... Average distance (m) for isotopic exchange (Friedman et al., 1962)
% Based on the average velocity of falling rain, 6 m/s, and the
% average time for isotopic exchange, 90 second, from Lee et al., 2007).
hR = 540;
%... Estimated standard-deviation ratio for isotopic variation at 
% a location due to seaonal variation, where the ratio is equal to
% the maximum sd/minimum sd.
% The estimate here is based on monthly measurements of 
% precipitation isotopes at the GNIP station at Coyhaique, Chile.
sdResRatio = 28.3;

%% Get run file information
% [runPath, runFile, runTitle, isParallel, dataPath, ...
%     topoFile, rTukey, sampleFile, contDivideFile, restartFile, ...
%     mapLimits, sectionLon0, sectionLat0, mu, epsilon0, ...
%     parameterLabels, exponents, lB, uB, isPenalty, beta] ...
%     = getRunFile(runFile)
[runPath, runFile, runTitle, ~, dataPath, ...
    topoFile, rTukey, sampleFile, contDivideFile, ~, ...
    mapLimits, sectionLon0, sectionLat0, ~, ~, ...
    ~, ~, lB, uB, beta] ...
    = getRunFile;
if isempty(beta)
    error('opiCalc requires that the run file include an OPI solution at the end of the run file.')
end
if length(beta)~=9
    error('Number of parameters is incorrect for this program')
end
nParametersFree = sum(lB~=uB);

%% Get input data
% [lon, lat, x, y, hGrid, lon0, lat0, ...
%     sampleLine, sampleLon, sampleLat, sampleX, sampleY, ...
%     sampleD2H, sampleD18O, sampleDExcess, sampleLC, ...
%     sampleLineAlt, sampleLonAlt, sampleLatAlt, sampleXAlt, sampleYAlt, ...
%     sampleD2HAlt, sampleD18OAlt, sampleDExcessAlt, sampleLCAlt, ...
%     bMWLSample, sdDataMin, sdDataMax, cov, fC] ...
%     = getInput(dataPath, topoFile, rTukey, sampleFile, sdResRatio)
[lon, lat, x, y, hGrid, lon0, lat0, ...
    sampleLine, sampleLon, sampleLat, sampleX, sampleY, ...
    sampleD2H, sampleD18O, sampleDExcess, sampleLC, ...
    sampleLineAlt, sampleLonAlt, sampleLatAlt, sampleXAlt, sampleYAlt, ...
    sampleD2HAlt, sampleD18OAlt, sampleDExcessAlt, sampleLCAlt, ...
    bMWLSample, sdDataMin, sdDataMax, cov, fC] ...
    = getInput(dataPath, topoFile, rTukey, sampleFile, sdResRatio);

%... Number of samples
nSamples = length(sampleLon);
nSamplesAlt = length(sampleLonAlt);

%... Unpack beta, the solution vector
U = beta(1);           % wind speed (m/s)
azimuth = beta(2);     % azimuth (degrees)
T0 = beta(3);          % sea-level temperature (K)
M =  beta(4);          % mountain-height number (dimensionless)
kappa = beta(5);       % eddy diffusion (m/s^2)
tauC = beta(6);        % condensation time (s)
d2H0 = beta(7);        % d2H of base precipitation (per unit)
dD2H0_dLat = beta(8);  % latitudinal gradient of base-prec d2H (1/deg lat)  
fP0 = beta(9);         % residual precipitation after evaporation (fraction) 
%... Convert mountain-height number to buoyancy frequency (rad/s)
hMax = max(hGrid, [], 'all');
NM = M*U/hMax;

%... If empty, set section origin to map origin 
if isempty([sectionLon0, sectionLat0])
    sectionLon0 = lon0;
    sectionLat0 = lat0;
end

%... Define synthetic sample points if sample file is not specified
if isempty(sampleFile)
    % Define sample points when sample file is not specified
    % Create a set of evenly spaced sample points along the wind 
    % direction and passing through the section origin.
    [sectionX0, sectionY0] = ...
        lonlat2xy(sectionLon0, sectionLat0, lon0, lat0);
    [sampleX, sampleY] = windPath(sectionX0, sectionY0, azimuth, x, y);
    [sampleLon, sampleLat] = xy2lonlat(sampleX, sampleY, lon0, lat0);
    nSamples = length(sampleX);
    sampleLine = (1:nSamples)';
    sampleLC = repmat('L', size(sampleX));
    sampleD2H = nan(nSamples, 1);
    sampleD18O = nan(nSamples, 1);
end

% Find grid nodes, for local and catchment samples ("L" and "C" in
% sampleLC).
[ijCatch, ptrCatch] = ...
    catchmentNodes(sampleX, sampleY, sampleLC, x, y, hGrid);
[ijCatchAlt, ptrCatchAlt] = ...
    catchmentNodes(sampleXAlt, sampleYAlt, sampleLCAlt, x, y, hGrid);

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
if isempty(sampleFile)
    fprintf('\n---------------- SYNTHETIC SAMPLES ------------------\n')
    fprintf([ ...
    'Data shown here is synthetic, and is intended for experimentation.\n', ...
    'There are no observed data, so the output has been reduced accordingly.\n'])
end    
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
fprintf('Size of cosine window as fraction of grid size: %g (dimensionless)\n', rTukey)
fprintf('Coriolis frequency at map-origin latitude: %.5f mrad/s\n', ...
    fC*1e3)
fprintf('Lon, lat for section origin: %.5f, %.5f degrees\n', sectionLon0, sectionLat0)
fprintf('\n-------------------- Sample File --------------------\n')
if isempty(sampleFile)
    fprintf('Sample file: No samples\n')
else    
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
end
fprintf('\n---------------------- Constants --------------------\n')
fprintf('Characteristic distance for isotopic exchange: %.0f m\n', hR)
fprintf('Standard-deviation ratio for data residuals: %.2f (dimensionless)\n', sdResRatio)
fprintf('\n--------- Constraints for Best-Fit Solution ---------\n')
fprintf('Wind speed: %g, %g m/s\n', lB(1), uB(1))
fprintf('Wind azimuth: %g, %g degrees\n', lB(2), uB(2))
fprintf('Sea-level temperature: %g, %g K  (%.1f, %.1f °C)\n', ...
    lB(3), uB(3), lB(3) - TC2K, uB(3) - TC2K)
fprintf('Mountain-height number: %g, %g (dimensionless)\n', lB(4), uB(4))
fprintf('Horizontal eddy diffusivity: %g, %g m^2/s\n', lB(5), uB(5))
fprintf('Average condensation time: %g, %g s\n', lB(6), uB(6))
fprintf('d2H for base precipitation: %g, %g per mil\n', [lB(7), uB(7)]*1e3)
fprintf('d2H latitude gradient for base precipitation: %g, %g per mil/deg lat\n', [lB(8), uB(8)]*1e3)
fprintf('Residual precipitation after evaporation: %g, %g (dimensionless)\n', lB(9), uB(9))
fprintf('\n----------------- Evaporation Option ----------------\n')
if lB(9)==1 && uB(9)==1
    fprintf('No Evaporative recycling for precipitation state.\n')
else
    fprintf('Evaporative recycling active for precipitation state.\n')
end
%... Force diary to write 
diary off; diary on

%% Calculate full result for best-fit solution
isFit = false;
[chiR2, nu, stdResiduals, ...
    zBar, T, gammaEnv, gammaSat, gammaRatio, ...
    rhoS0, hS, rho0, hRho, ...
    d18O0, dD18O0_dLat, tauF, pGrid, fMGrid, rHGrid, ...
    evapD2HGrid, uEvapD2HGrid, evapD18OGrid, uEvapD18OGrid, ...
    d2HGrid, d18OGrid, iWet, d2HPred, d18OPred] = ...
    calc_OneWind(beta, fC, hR, x, y, lat, lat0, ...
    hGrid, bMWLSample, ijCatch, ptrCatch, ...
    sampleD2H, sampleD18O, cov, nParametersFree, isFit);

%... Calculate the precipitation-weighted catchment average
% for maximum lifting and elevation for each sample point,
% while accounting for both type L and type C sample locations,
% and also for primary and alterated samples.
[liftMaxPred, elevationPred, subsidencePred] = ...
    lifting(x, y, hGrid, pGrid, azimuth, U, tauF, ijCatch, ptrCatch);
[liftMaxPredAlt, elevationPredAlt, subsidencePredAlt] = ...
    lifting(x, y, hGrid, pGrid, azimuth, U, tauF, ijCatchAlt, ptrCatchAlt);

%... Calculate catchment area for primary sample locations
if ~isempty(ijCatch) && ~isempty(ptrCatch)
    catchArea = nan(nSamples,1);
    dA = abs((x(2) - x(1))*(y(2) - y(1)));
    for k = 1:nSamples
        % Extract indices for sample catchment
        ij = catchmentIndices(k, ijCatch, ptrCatch);
        % Calculate catchment area (m^2)
        catchArea(k) = sum(pGrid(ij)>0)*dA;
    end
end

%... Precipitation isotopes d2HPred and d18OPred predicted for
% altered sample locations, and accounting for catchment area if selected.
d2HPredAlt = nan(nSamplesAlt,1);
d18OPredAlt = nan(nSamplesAlt,1);
iWetAlt = false(nSamplesAlt,1);
catchAreaAlt = nan(nSamplesAlt,1);
for k = 1:nSamplesAlt
    % Extract indices for sample catchment
    ij = catchmentIndices(k, ijCatchAlt, ptrCatchAlt);
    % Calculate catchment area (m^2)
    catchAreaAlt(k) = sum(pGrid(ij)>0)*dA;
    % Calculate weights using predicted precipitation
    pSum = sum(pGrid(ij));
    % Calculated precipitation-weighted isotope compositions for catchment
    if pSum>0
        % Normalize weights
        wtPrec = pGrid(ij)./pSum;
        % Calculate catchment-weighted composition of water isotopes
        iWetAlt(k) = true;
        d2HPredAlt(k) = sum(wtPrec.*d2HGrid(ij));
        d18OPredAlt(k) = sum(wtPrec.*d18OGrid(ij));
    else
        % Use simple mean for dry sites
        d2HPredAlt(k) = mean(d2HGrid(ij));
        d18OPredAlt(k) = mean(d18OGrid(ij));
    end
end

%... Save results in mat file
save([runPath, '/', mfilename, '_Results.mat'], ...
    'startTimeOpiCalc', 'runPath', 'runFile', 'runTitle', ...
    'dataPath', 'topoFile', 'rTukey', 'sampleFile', 'contDivideFile', ...
    'mapLimits', 'sectionLon0', 'sectionLat0', ...
    'lon', 'lat', 'x', 'y', 'lon0', 'lat0', ...
    'sampleLine', 'sampleLon', 'sampleLat', 'sampleX', 'sampleY', ...
    'sampleD2H', 'sampleD18O', 'sampleDExcess', 'sampleLC', ...
    'sampleLineAlt', 'sampleLonAlt', 'sampleLatAlt', 'sampleXAlt', 'sampleYAlt', ...
    'sampleD2HAlt', 'sampleD18OAlt', 'sampleDExcessAlt', 'sampleLCAlt', ...
    'bMWLSample', 'ijCatch', 'ptrCatch','ijCatchAlt', 'ptrCatchAlt', ...
    'hR', 'sdResRatio', 'sdDataMin', 'sdDataMax', 'cov', 'fC', ...
    'lB', 'uB', 'nParametersFree', ...
    'beta', 'chiR2', 'nu', 'stdResiduals', ...
    'zBar', 'T', 'gammaEnv', 'gammaSat', 'gammaRatio', ...
    'rhoS0', 'hS', 'rho0', 'hRho', ...
    'd18O0', 'dD18O0_dLat', 'tauF', 'pGrid', 'fMGrid', 'rHGrid', ...
    'evapD2HGrid', 'uEvapD2HGrid', 'evapD18OGrid', 'uEvapD18OGrid', ...
    'd2HGrid', 'd18OGrid', ...
    'iWet', 'd2HPred', 'd18OPred', 'catchArea', ...
    'iWetAlt', 'd2HPredAlt', 'd18OPredAlt', 'catchAreaAlt', ...
    'liftMaxPred', 'elevationPred', 'subsidencePred', ...
    'liftMaxPredAlt', 'elevationPredAlt', 'subsidencePredAlt', '-v7.3');

%... Sum number of precipitation-present samples
nSamplesWet = sum(iWet);
%... Calculate statistics for predicted water isotopes (uniform weighting)
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

%% Report results
fprintf('\n---------------------- Solution ---------------------\n')
fprintf('Wind speed: %.1f m/s\n', U)
fprintf('Azimuth: %.1f degrees\n', azimuth)
fprintf('Sea-level temperature: %.1f K (%.1f °C)\n', T0, T0 - TC2K)
fprintf('Mountain-height number: %.3f (dimensionless)\n', M)
fprintf('Horizontal eddy diffusivity: %.0f m^2/s\n', kappa)
fprintf('Average residence time for cloud water: %.0f s\n', tauC)
fprintf('d2H for base precipitation: %.1f per mil\n', d2H0*1e3)
fprintf('d2H latitude gradient for base precipitation: %.3f per mil/deg lat\n', dD2H0_dLat*1e3)
fprintf('Residual precipitation after evaporation: %.2f (fraction)\n', fP0)
fprintf('\n---- Other Variables Related to Best-Fit Solution ---\n')
fprintf('Moist buoyancy frequency: %.3f mrad/s\n', NM*1e3)
fprintf('d18O for base precipitation: %.1f per mil\n', d18O0*1e3)
fprintf('d18O latitude gradient for base precipitation: %.3f per mil/deg lat\n', dD18O0_dLat*1e3)
fprintf('Average residence time for falling precipitation: %.0f s\n', tauF)
fprintf('Water-vapor density at sea level: %.2f g/m^3\n', rhoS0*1e3)
fprintf('Scale height for water vapor: %.0f m\n', hS)
fprintf('Average velocity for falling precipitation: %.1f m/s\n', hS/tauF)
fprintf('Total density at sea level: %.2f g/m^3\n', rho0);
fprintf('Scale height for total density: %.0f m\n', hRho)
fprintf('Average lapse-rate ratio, gammaSat/gammaEnv: %.2f (dimensionless)\n', gammaRatio)
fprintf('\n------------ Observed Meteoric Water Line -----------\n')
if ~all(isnan([sampleD2H, sampleD18O]), 'all')
    fprintf('Principal standard deviations: %.2f, %.2f per mil\n', ...
        sdDataMin*1e3, sdDataMax*1e3);
    fprintf('Intercept and slope: %.1f per mil, %.2f\n', ...
        bMWLSample(1)*1e3, bMWLSample(2))
else
    fprintf('No samples.\n')
end
fprintf('\n----------- Predicted Meteoric Water Line -----------\n')
fprintf('Principal standard deviations: %.2f, %.2f per mil\n', ...
    sdPredMin*1e3, sdPredMax*1e3);
fprintf('Intercept and slope: %.1f per mil, %.2f\n', ...
    bMWLPred(1)*1e3, bMWLPred(2))
if ~all(isnan([sampleD2H, sampleD18O]), 'all')
    fprintf('\n------------------ Quality of Fit -------------------\n')
    fprintf('Reduced chi-square: %.4f\n', chiR2)
    fprintf('Degrees of freedom: %d\n', nu)
    fprintf('Number of primary samples in wet locations: %d\n', nSamplesWet)
    fprintf('Number of primary samples: %d\n', nSamples)    
    fprintf('Standard deviation of wet residuals for d2H: %.1f per mil\n', ...
        sdRes_d2H*1e3)
    fprintf('Standard deviation of wet residuals for d18O: %.1f per mil\n', ...
        sdRes_d18O*1e3)
    fprintf('Approximate standard error for predicted d2H: %.1f per mil\n', ...
        se_d2HPred*1e3)
    fprintf('Approximate standard error for predicted d18O: %.1f per mil\n', ...
        se_d18OPred*1e3)    
end
fprintf('\n----------------- Computation Time ------------------\n')
fprintf('Compute time: %.2f minutes\n', ...
    minutes(datetime - startTimeOpiCalc));
fprintf('\n--------------------- Mat File ----------------------\n')
fprintf('Results saved in the run directory as mat file:\n%s\n', ...
    [runPath, '/', mfilename, '_Results.mat']);
%... Close the diary
diary off;

end
