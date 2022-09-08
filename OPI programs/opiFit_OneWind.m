function opiFit_OneWind(runFile)
% opiFit Finds a best-fit OPI solution for isotope data using the 
% optimization function fminCRS for one moisture source (OneWind). 

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
if nargin==0
    close all
    clc
    dbstop if error
end

%% Constants
%... Convert from Celsius to kelvin
TC2K = 273.15;
%... Start time for calculation
startTime = datetime; 
%... Mean radius of the Earth (m)
radiusEarth = 6371e3; 
%... Meters per arc degree for the Earth's surface
mPerDegree = pi*radiusEarth/180;

%... Average distance (m) for isotopic exchange (Lee et al., 2007)
% Based on the average velocity of falling rain, 6 m/s, and the 
% average time for isotopic exchange, 90 second.
hR = 540;
%... Estimated standard-deviation ratio for isotopic variation at 
% a location due to seaonal variation, where the ratio is equal to
% the maximum sd/minimum sd.
% The estimate here is based on monthly measurements of 
% precipitation isotopes at the GNIP station at Coyhaique, Chile.
sdResRatio = 28.3;

%% Load input data and initialize variables
%... Load and parse run file
% [runPath, runFile, runTitle, isParallel, dataPath, ...
%     topoFile, rTukey, sampleFile, contDivideFile, restartFile, ...
%     mapLimits, sectionLon0, sectionLat0, mu, epsilon0, ...
%     parameterLabels, exponents, lB, uB, beta] ...
%     = getRunFile(runFile)
if nargin==0
    %... Interactive mode: run file selected using uigetfile
    [runPath, runFile, runTitle, isParallel, dataPath, ...
        topoFile, rTukey, sampleFile, ~, restartFile, ...
        ~, sectionLon0, sectionLat0, mu, epsilon0, ...
        parameterLabels, exponents, lB, uB, beta] ...
        = getRunFile;
else
    %... Batch mode: run file specified as argument for getRunFile
    fprintf('Open run file: %s\n\n\n', runFile);    
    [runPath, runFile, runTitle, isParallel, dataPath, ...
        topoFile, rTukey, sampleFile, ~, restartFile, ...
        ~, sectionLon0, sectionLat0, mu, epsilon0, ...
        parameterLabels, exponents, lB, uB, beta] ...
        = getRunFile(runFile);
end
if isempty(sampleFile)
    error('OpiFit requires a sample file.\n')
end
if ~isempty(beta)
    error('Run file includes a best-fit solution, which is not allowed for optFit.');
end
if length(lB)~=9
    error('Number of parameters is incorrect for this program')
end
%... Number of free parameters
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
    ~, sampleLon, sampleLat, sampleX, sampleY, ...
    sampleD2H, sampleD18O, ~, sampleLC, ...
    ~, sampleLonAlt, ~, ~, ~, ...
    ~, ~, ~, ~, ...
    bMWLSample, sdDataMin, sdDataMax, cov, fC] ...
    = getInput(dataPath, topoFile, rTukey, sampleFile, sdResRatio);

%... Number of samples
nSamples = length(sampleLon);
nSamplesAlt = length(sampleLonAlt);

%... Calculate maximum elevation
hMax = max(hGrid, [], 'all');

%... If empty, set section origin to map origin 
if isempty([sectionLon0, sectionLat0])
    sectionLon0 = lon0;
    sectionLat0 = lat0;
end

%... Find grid nodes for catchment upstream of each sample location
[ijCatch, ptrCatch] = ...
    catchmentNodes(sampleX, sampleY, sampleLC, x, y, hGrid);
%... Load solutions from restart file, if available
solutions = [];
nRestart = 0;
if ~isempty(restartFile)
    [solutionsTitle, ~, ~, ~, lbRestart, ubRestart, ...
        ~, ~, solutions, ~, ~, ~] ...
        = getSolutions(runPath, restartFile);
    if length(lB)~=(size(solutions,2) - 2) || ...
            any(lB~=lbRestart) || any(uB~=ubRestart) 
        error('Constraints in the restart file are different from those in the run file.')
    end
    nRestart = size(solutions,1);
end

%% Report starting conditions
%... Start diary file
logFilename=[runPath, '/', mfilename, '_Log.txt'];
if isfile(logFilename), delete (logFilename); end
diary(logFilename);
fprintf (['Program: ', mfilename, '\n'])
fprintf('Start time: %s\n', startTime)
fprintf('Run file path:\n%s\n', runPath)
fprintf('Run filename:\n%s\n', runFile)
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
fprintf('Grid spacing, dLon and dLat: %.5f, %.5f degrees\n', ...
    dLon, dLat)
fprintf('Grid spacing, dx and dy: %.2f, %.2f km\n', ...
    dLon*mPerDegree*1e-3*cosd(lat0), dLat*mPerDegree*1e-3)
fprintf('Lon, lat for map origin: %.5f, %.5f degrees\n', lon0, lat0);
if lon0==mean(sampleLon) && lat0==mean(sampleLat)
    fprintf('Map origin is set to sample centroid.\n')
else
    fprintf('Map origin is set to center of the topographic grid.\n')
end
fprintf('Size of cosine window as fraction of grid size: %g (dimensionless)\n', rTukey)
fprintf('Coriolis frequency at map-origin latitude: %.5f mrad/s\n', ...
    fC*1e3)
fprintf('Lon, lat for section origin: %.5f, %.5f degrees\n', sectionLon0, sectionLat0);
fprintf('\n-------------------- Sample File --------------------\n')
fprintf('Sample file: %s\n', sampleFile)
fprintf('Number of all samples: %d\n', nSamples + nSamplesAlt)
fprintf('Number of altered samples: %d\n', nSamplesAlt)
fprintf('Number of primary samples: %d\n', nSamples)
fprintf('Number of local primary samples: %d\n', sum(sampleLC=='L'))
fprintf('Number of catchment primary samples: %d\n', sum(sampleLC=='C'))
fprintf('Minimum and maximum for longitude: %.5f, %.5f degrees\n', ...
    min(sampleLon), max(sampleLon))
fprintf('Minimum and maximum for latitude: %.5f, %.5f degrees\n', ...
    min(sampleLat), max(sampleLat))
fprintf('\n---------------------- Constants --------------------\n')
fprintf('Average distance for isotopic exchange: %.0f m\n', hR)
fprintf('Standard-deviation ratio for data residuals: %.2f (dimensionless)\n', sdResRatio)
fprintf('\n--------- Constraints for Best-Fit Solution ---------\n')
fprintf('Wind speed: %g, %g m/s\n', lB(1), uB(1))
fprintf('Wind azimuth: %g, %g degrees\n', lB(2), uB(2))
fprintf('Sea-level surface-air temperature: %g, %g K  (%.1f, %.1f °C)\n', ...
    lB(3), uB(3), lB(3) - TC2K, uB(3) - TC2K)
fprintf('Mountain-height number: %g, %g (dimensionless)\n', lB(4), uB(4))
fprintf('Horizontal eddy diffusivity: %g, %g m^2/s\n', lB(5), uB(5))
fprintf('Average condensation time: %g, %g s\n', lB(6), uB(6))
fprintf('d2H for base precipitation: %g, %g per mil\n', [lB(7), uB(7)]*1e3)
fprintf('d2H latitude gradient for base precipitation: %g, %g per mil/deg lat\n', [lB(8), uB(8)]*1e3)
fprintf('Residual precipitation after evaporation: %g, %g (dimensionless)\n', lB(9), uB(9))
fprintf('\n----------------- Evaporation Option ----------------\n')
if lB(9)==1 && uB(9)==1
    fprintf('No Evaporative recycling active for precipitation state.\n')
else
    fprintf('Evaporative recycling active for precipitation state.\n')
end
fprintf('\n------------- Solutions From Restart File -----------\n')
if ~isempty(solutions)
    fprintf('Restart file: %s\n', restartFile);
    fprintf('Title for solutions in restart file:\n%s\n', solutionsTitle)
    fprintf('Number of restart solutions: %d\n', nRestart);
else
    fprintf('Restart file: none\n');
end

%% Estimate best-fit solution
%... Heading for start of best-fit search
fprintf('\n------------------ Best-Fit Search ------------------\n')
%... Force diary to write 
diary off; diary on

%... Initialize log file for fminCRS
writeSolutions('initialize', runPath, runTitle, nSamples, ...
    parameterLabels, exponents, lB, uB);

%... Search for best-fit solution
isFit = true;
beta = fminCRS3(@(beta) ...
    calc_OneWind(beta, fC, hR, x, y, lat, lat0, ...
    hGrid, bMWLSample, ijCatch, ptrCatch, ...
    sampleD2H, sampleD18O, cov, nParametersFree, isFit), ...
    lB, uB, mu, epsilon0, ...
    isParallel, @writeSolutions, solutions);

%% Calculate full result for best-fit solution
% [chiR2, nu, stdResiduals, ...
%     zBar, T, gammaEnv, gammaSat, gammaRatio, ...
%     rhoS0, hS, rho0, hRho, ...
%     d18O0, dD18O0_dLat, tauF, pGrid, fMGrid, rHGrid, ...
%     evapD2HGrid, uEvapD2HGrid, evapD18OGrid, uEvapD18OGrid, ...
%     d2HGrid, d18OGrid, iWet, d2HPred, d18OPred] = ...
    calc_OneWind(beta, fC, hR, x, y, lat, lat0, ...
    hGrid, bMWLSample, ijCatch, ptrCatch, ...
    sampleD2H, sampleD18O, cov, nParametersFree, isFit)
[chiR2, nu, ~, ...
    ~, ~, ~, ~, gammaRatio, ...
    rhoS0, hS, rho0, hRho, ...
    d18O0, dD18O0_dLat, tauF, ~, ~, ~, ...
    ~, ~, ~, ~, ...
    ~, ~, iWet, d2HPred, d18OPred] = ...
    calc_OneWind(beta, fC, hR, x, y, lat, lat0, ...
    hGrid, bMWLSample, ijCatch, ptrCatch, ...
    sampleD2H, sampleD18O, cov, nParametersFree, isFit);

%... Unpack the solution vector beta
U = beta(1);           % wind speed (m/s)
azimuth = beta(2);     % azimuth (degrees)
T0 = beta(3);          % sea-level temperature (K)
M = beta(4);           % mountain-height number (dimensionless)
kappa = beta(5);       % eddy diffusion (m/s^2)
tauC = beta(6);        % condensation time (s)
d2H0 = beta(7);        % d2H of base precipitation (per unit)
dD2H0_dLat = beta(8);  % latitudinal gradient of base-prec d2H (1/deg lat)  
fP0 = beta(9);         % residual precipitation after evaporation (fraction) 
%... Convert mountain-height number to buoyancy frequency (rad/s)
NM = M*U/hMax;
%... Sum number of precipitation-present samples
nSamplesWet = sum(iWet);
%... Calculate statistics for predicted water isotopes (uniform weighting)
[bMWLPred, sdPredMin, sdPredMax] = ...
    estimateMWL(d18OPred(iWet), d2HPred(iWet), sdResRatio);

%% Report best-fit results
fprintf('\n---------- Estimates for Best-Fit Solution ----------\n')
fprintf('Wind speed: %.1f m/s\n', U)
fprintf('Azimuth: %.1f degrees\n', azimuth)
fprintf('Sea-level surface-air temperature: %.1f K (%.1f °C)\n', T0, T0 - TC2K)
fprintf('Mountain-height number: %.3f (dimensionless)\n', M)
fprintf('Horizontal eddy diffusivity: %.0f m^2/s\n', kappa)
fprintf('Average residence time for cloud water: %.0f s\n', tauC)
fprintf('d2H for base precipitation: %.1f per mil\n', d2H0*1e3)
fprintf('d2H latitude gradient for base precipitation: %.3f per mil/deg lat\n', dD2H0_dLat*1e3)
fprintf('Residual precipitation after evaporation: %.2f (dimensionless)\n', fP0)
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
end
fprintf('\n----------- Predicted Meteoric Water Line -----------\n')
fprintf('Principal standard deviations: %.2f, %.2f per mil\n', ...
    sdPredMin*1e3, sdPredMax*1e3);
fprintf('Intercept and slope: %.1f per mil, %.2f\n', ...
    bMWLPred(1)*1e3, bMWLPred(2))
fprintf('\n------------------ Quality of Fit -------------------\n')
fprintf('Reduced chi-square: %.4f\n', chiR2)
fprintf('Degrees of freedom: %d\n', nu);
fprintf('Number of primary samples in wet locations: %d\n', nSamplesWet)
fprintf('Number of primary samples: %d\n', nSamples);
fprintf('Best-fit parameters:\n');
fprintf('%g\t', beta(1:end-1));
fprintf('%g\n', beta(end));

finishTime = datetime;
fprintf('\nFinish time: %s\n', finishTime)
fprintf('Time for current run: %.2f hours\n', ...
    hours(finishTime - startTime));

%... Close the diary
diary off

%... Append best-fit solution to end of run file
fid = fopen([runPath, '/', runFile], 'a+');
fprintf(fid, '\n\n%%... Best-Fit Solution\n');
fprintf(fid, '%% Start time for current run: %s\n', startTime);
fprintf(fid, '%% Finish time for current run: %s\n', finishTime);
fprintf(fid, '%% Time for current run: %.2f hours\n', ...
    hours(finishTime - startTime));    
fprintf(fid, '%% Reduced chi-square: %.4f\n', chiR2);
fprintf(fid, '%% Degrees of freedom: %d\n', nu);
fprintf(fid, '%% Number of primary samples in wet locations: %d\n', nSamplesWet);
fprintf(fid, '%% Number of primary samples: %d\n', nSamples);
fprintf(fid, '%% Best-fit parameters:\n');
fprintf(fid, '%g\t', beta(1:end-1));
fprintf(fid, '%g\n', beta(end));
fclose(fid);

end
