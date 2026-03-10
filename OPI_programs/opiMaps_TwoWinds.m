 function opiMaps_TwoWinds
% opiMaps_TwoWinds takes as input a run file and an associated mat file
% with a "two-winds" solution, and creates map figures of the results.
% DEM can have x,y in linear units (kilometers), or geographic units
% (decimal degrees). For the latter, x is longitude, and y is latitude.
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
%... Vertical exaggeration of topography in 3D figures
VE = 2e4;
% Starting height above ground for streamlines in map view
zL0Map = 2000;

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
% 'dataPath', 'topoFile', 'rTukey', 'sampleFile', 'contDivideFile', ...
% 'mapLimits', 'sectionLon0', 'sectionLat0', ...
% 'lon', 'lat', 'x', 'y', 'lon0', 'lat0', ...ß
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
load([matPathResults, '/', 'opiCalc_TwoWinds_Results.mat'], ...
    'runPath', 'runFile', 'runTitle', ...
    'dataPath', 'topoFile', 'rTukey', 'sampleFile', 'contDivideFile',  ...
    'mapLimits', 'sectionLon0', 'sectionLat0', ...
    'lon', 'lat', 'x', 'y', 'lon0', 'lat0', ...
    'sampleLon', 'sampleLat', ...
    'sampleLC', 'sampleLonAlt', 'ijCatch', 'ptrCatch', ...
    'hR', 'sdResRatio', 'fC', ...
    'lB', 'uB', ...    
    'stdResiduals', ...
    'zBar_1', 'T_1', 'gammaEnv_1', 'gammaSat_1', 'gammaRatio_1', ...
    'rhoS0_1', 'hS_1', 'rho0_1', 'hRho_1', ...
    'd18O0_1', 'dD18O0_dLat_1', 'tauF_1', 'pGrid_1', 'fMGrid_1', ...
    'd2HGrid_1', ...
    'zBar_2', 'T_2', 'gammaEnv_2', 'gammaSat_2', 'gammaRatio_2', ...
    'rhoS0_2', 'hS_2', 'rho0_2', 'hRho_2', ...
    'd18O0_2', 'dD18O0_dLat_2', 'tauF_2', 'pGrid_2', 'fMGrid_2', ...
    'd2HGrid_2', ...
    'pGrid', 'd2HGrid', 'd18OGrid', ...
    'fractionPGrid');

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

%... If empty, set section origin to map origin 
if isempty([sectionLon0, sectionLat0])
    sectionLon0 = lon0;
    sectionLat0 = lat0;
end
%... Get elevation for section orogin
sectionH0 = interp2(lon, lat, hGrid, sectionLon0, sectionLat0);

%... Check mapLimits
if ~all(mapLimits==0) && any([mapLimits(1:2)<lon(1), mapLimits(1:2)>lon(end), ...
        mapLimits(3:4)<lat(1), mapLimits(3:4)>lat(end)])
    error('Map limits must lie within the range of the topography grid');
end

%... Process mapLimits
if all(mapLimits==0) 
    % MapLimits not specified, set to grid limits
    mapLimits(1) = lon(1);
    mapLimits(2) = lon(end);
    mapLimits(3) = lat(1);
    mapLimits(4) = lat(end);
end

% Calculate x,y coordinates for mapLimits
[xyLimits(1), xyLimits(3)] = ...
    lonlat2xy(mapLimits(1), mapLimits(3), lon0, lat0);
[xyLimits(2), xyLimits(4)] = ...
    lonlat2xy(mapLimits(2), mapLimits(4), lon0, lat0);

%% Report results
%... Start diary file
logFilename=[runPath, '/', mfilename, '_Log.txt'];
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
fprintf('Topography file: %s\n', topoFile);
fprintf('Maximum elevation: %.0f m\n', hMax);
[nY, nX] = size(hGrid);
fprintf('Grid size, nx and ny: %d, %d\n', nX, nY);
fprintf('Longitude, minimum and maximum: %.5f, %.5f degrees\n', lon(1), lon(end));
fprintf('Latitude, minimum and maximum: %.5f, %.5f degrees\n', lat(1), lat(end));
dLon = lon(2) - lon(1);
dLat = lat(2) - lat(1);
fprintf('Grid spacing, dLon and dLat: %.5f, %.5f degrees\n', ...
    dLon, dLat)
fprintf('Grid spacing, dx and dy: %.2f, %.2f km\n', ...
    dLon*mPerDegree*1e-3*cosd(lat0), dLat*mPerDegree*1e-3)
fprintf('User-defined map limits, longitude: %.5f, %.5f\n', ...
    mapLimits(1), mapLimits(2));
fprintf('User-defined map limits, latitude: %.5f, %.5f\n', ...
    mapLimits(3), mapLimits(4));
if ~isempty(contDivideFile)
    fprintf('Continental-divide file: %s\n', contDivideFile);
else
    fprintf('Continental-divide file: none\n');
end
fprintf('Lon, lat for map origin (degrees): %.5f, %.5f\n', lon0, lat0);
if lon0==mean(sampleLon) && lat0==mean(sampleLat)
    fprintf('Map origin is set to sample centroid.\n')
else
    fprintf('Map origin is set to center of topographic grid.\n')
end
fprintf('Size of cosine window as fraction of grid size: %g\n', rTukey)
fprintf('Coriolis frequency at map-origin latitude: %.5f mrad/s\n', ...
    fC*1e3)
fprintf('Lon, lat for section origin (degrees): %.5f, %.5f\n', sectionLon0, sectionLat0);
fprintf('\n-------------------- Sample File --------------------\n')
if isempty(sampleFile)
    fprintf('Sample file: No samples\n')
else
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
end
fprintf('\n---------------------- Constants --------------------\n')
fprintf('Characteristic distance for isotopic exchange: %.0f m\n', hR)
fprintf('Standard-deviation ratio for data residuals: %.2f (dimensionless)n', sdResRatio)
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

%% Calculations for plots
%... Load continental-divide data, if present
contDivideLon = [];
contDivideLat = [];
if ~isempty(contDivideFile)
    load([dataPath, '/', contDivideFile], 'contDivideLon', 'contDivideLat');
end

%... Set relative scaling for data variables, including
% vertical exaggeration for topography
dataAspectVector = [mPerDegree*cosd(lat0) mPerDegree VE].^-1;

%... Calculate logical array for grid nodes inside map limits
% Used for cmapscale to focus on values inside map limits
[lonGrid, latGrid] = meshgrid(lon, lat);
isMap = (lonGrid>=mapLimits(1) & lonGrid<=mapLimits(2) ...
    & latGrid>=mapLimits(3) & latGrid<=mapLimits(4));
clear lonGrid latGrid

%... Calculate arrow for wind direction in map figures
% Note that mapLimits = [minLon, maxLon, minLat, maxLat]
arrowSize = 0.3;  % size relative to horizontal size of map
% Precipitation state #1
lonArrowOffset_1 = arrowSize*(mapLimits(2) - mapLimits(1))*sind(azimuth_1);
latArrowOffset_1 = arrowSize*(mapLimits(2) - mapLimits(1))*cosd(azimuth_1) ...
    *dataAspectVector(2)/dataAspectVector(1);
switch true    
    case azimuth_1 >=0 & azimuth_1 <= 90
        % Place in SW corner
        lonArrowStart_1 = mapLimits(1) + 0.04*((mapLimits(2) - mapLimits(1)));
        latArrowStart_1 = mapLimits(3) + 0.04*((mapLimits(4) - mapLimits(3)));
    case azimuth_1 > 90 & azimuth_1 <= 180
        % Place in NW corner
        lonArrowStart_1 = mapLimits(1) + 0.04*((mapLimits(2) - mapLimits(1)));
        latArrowStart_1 = mapLimits(4) - 0.04*((mapLimits(4) - mapLimits(3)));
    case azimuth_1 > 180 & azimuth_1 <= 270
        % Place in NE corner
        lonArrowStart_1 = mapLimits(2) - 0.04*((mapLimits(2) - mapLimits(1)));
        latArrowStart_1 = mapLimits(4) - 0.04*((mapLimits(4) - mapLimits(3)));
    case azimuth_1 > 270 & azimuth_1 <= 360
        % Place in SE corner
        lonArrowStart_1 = mapLimits(2) - 0.04*((mapLimits(2) - mapLimits(1)));
        latArrowStart_1 = mapLimits(3) + 0.04*((mapLimits(4) - mapLimits(3)));
end
% Precipitation state #2
lonArrowOffset_2 = arrowSize*(mapLimits(2) - mapLimits(1))*sind(azimuth_2);
latArrowOffset_2 = arrowSize*(mapLimits(2) - mapLimits(1))*cosd(azimuth_2) ...
    *dataAspectVector(2)/dataAspectVector(1);
switch true    
    case azimuth_2 >=0 & azimuth_2 <= 90
        % Place in SW corner
        lonArrowStart_2 = mapLimits(1) + 0.04*((mapLimits(2) - mapLimits(1)));
        latArrowStart_2 = mapLimits(3) + 0.04*((mapLimits(4) - mapLimits(3)));
    case azimuth_2 > 90 & azimuth_2 <= 180
        % Place in NW corner
        lonArrowStart_2 = mapLimits(1) + 0.04*((mapLimits(2) - mapLimits(1)));
        latArrowStart_2 = mapLimits(4) - 0.04*((mapLimits(4) - mapLimits(3)));
    case azimuth_2 > 180 & azimuth_2 <= 270
        % Place in NE corner
        lonArrowStart_2 = mapLimits(2) - 0.04*((mapLimits(2) - mapLimits(1)));
        latArrowStart_2 = mapLimits(4) - 0.04*((mapLimits(4) - mapLimits(3)));
    case azimuth_2 > 270 & azimuth_2 <= 360
        % Place in SE corner
        lonArrowStart_2 = mapLimits(2) - 0.04*((mapLimits(2) - mapLimits(1)));
        latArrowStart_2 = mapLimits(3) + 0.04*((mapLimits(4) - mapLimits(3)));
end

%... Create coastline for map figures
% Set minimum number of points in each contour line (to reduce clutter)
nContourPointsThreshold = 100;
% Set to 0 m for coastline contour
contourValues = [0 0];
% Row vector contourValues contains specified contours
contours = ...
    transpose(contourc(lon, lat, hGrid, contourValues));
%... Convert contour results for plotting
m = size(contours,1);
iCurrent = 1;
while iCurrent<m
    l = contours(iCurrent,2);
    if l>nContourPointsThreshold
        contours(iCurrent,:) = nan;
        iCurrent = iCurrent + l + 1;
    else
        contours = contours([1:iCurrent-1,iCurrent+l+1:end], :);
        m = m-l-1;
    end
end

%% Calculate streamlines for map plot
% Set number of streamlines (recommended: 40)
nStreamlines = 40;
%... Precipitation state #1
% Construct path orthogonal to wind path #1 and through the 
% section origin: sectionLon0, sectionLat0
[xSection0, ySection0] = lonlat2xy(sectionLon0, sectionLat0, lon0, lat0);
[xPath_1, yPath_1, sPath_1] = windPath(xSection0, ySection0, ...
    wrapTo360(azimuth_1 + 90), x, y);
% Find points on orthogonal path
sPt = interp1(sPath_1, linspace(1, length(sPath_1), nStreamlines+2));
sPt = sPt(2:end-1);
xPt = interp1(sPath_1, xPath_1, sPt);
yPt = interp1(sPath_1, yPath_1, sPt);
% Calculate streamlines
xLMap_1 = cell(nStreamlines,1);
yLMap_1 = cell(nStreamlines,1);
zLMap_1 = cell(nStreamlines,1);
for j = 1:nStreamlines
    % Construct path using current x,y reference point
    [xL0, yL0] = windPath(xPt(j), yPt(j), azimuth_1, x, y);
    % Break if empty (indicating a corner, with a single intersection)
    if isempty(xL0), break, end
    % Calculate streamline
    [xLMap_1{j}, yLMap_1{j}, zLMap_1{j}] = ...
        streamline(xL0(1), yL0(1), zL0Map, ...
        x, y, hGrid, U_1, azimuth_1, NM_1, fC, hRho_1, (j==1));
    % Add nans to indicate end of streamline (for plotting)
    xLMap_1{j}(end+1) = nan;
    yLMap_1{j}(end+1) = nan;
    zLMap_1{j}(end+1) = nan;
end
% Convert cell arrays to vectors
xLMap_1 = cell2mat(xLMap_1);
yLMap_1 = cell2mat(yLMap_1);
zLMap_1 = cell2mat(zLMap_1);
% Convert to geographic coordinates
[lonLMap_1, latLMap_1] = xy2lonlat(xLMap_1, yLMap_1, lon0, lat0);

%... Precipitation state #2
% Construct path orthogonal to wind path #2 and through the 
% section origin, sectionLon0, sectionLat0
[xPath_2, yPath_2, sPath_2] = windPath(xSection0, ySection0, ...
    wrapTo360(azimuth_2 + 90), x, y);
% Find points on orthogonal path
sPt = interp1(sPath_2, linspace(1, length(sPath_2), nStreamlines+2));
sPt = sPt(2:end-1);
xPt = interp1(sPath_2, xPath_2, sPt);
yPt = interp1(sPath_2, yPath_2, sPt);
% Calculate streamlines
xLMap_2 = cell(nStreamlines,1);
yLMap_2 = cell(nStreamlines,1);
zLMap_2 = cell(nStreamlines,1);
for j = 1:nStreamlines
    % Construct path using current x,y reference point
    [xL0, yL0] = windPath(xPt(j), yPt(j), azimuth_2, x, y);
    % Break if empty (indicating a corner, with a single intersection)
    if isempty(xL0), break, end
    % Calculate streamline
    [xLMap_2{j}, yLMap_2{j}, zLMap_2{j}] = ...
        streamline(xL0(1), yL0(1), zL0Map, ...
        x, y, hGrid, U_2, azimuth_2, NM_2, fC, hRho_2, (j==1));
    % Add nans to indicate end of streamline (for plotting)
    xLMap_2{j}(end+1) = nan;
    yLMap_2{j}(end+1) = nan;
    zLMap_2{j}(end+1) = nan;
end
% Convert cell arrays to vectors
xLMap_2 = cell2mat(xLMap_2);
yLMap_2 = cell2mat(yLMap_2);
zLMap_2 = cell2mat(zLMap_2);
% Convert to geographic coordinates
[lonLMap_2, latLMap_2] = xy2lonlat(xLMap_2, yLMap_2, lon0, lat0);

%% Calculate streamlines, cloud-water, and precipitation for section plot
%... Precipitation state #1
% Section path passing through section origin: xSection0, ySection0
[xPath_1, yPath_1, sPath_1, sLimits_1] = ...
    windPath(xSection0, ySection0, azimuth_1, x, y, xyLimits);

% Land surface elevation along wind path
hLPath_1 = interp2(x, y, hGrid, xPath_1, yPath_1, 'linear', 0);
% Set maximum height of section
hLMax_1 = max(hLPath_1) + 2*hS_1;

% Calculate cloud-water density,and 248 and 268 K isotherms
[zRhoC_1, rhoC_1, zCMean_1, z248Path_1, z268Path_1] = ...
    cloudWater(xPath_1, yPath_1, hLMax_1, ...
    x, y, hGrid, U_1, azimuth_1, NM_1, fC, kappa_1, tauC_1, tauF_1, hRho_1, ...
    zBar_1, T_1, gammaEnv_1, gammaSat_1, gammaRatio_1, rhoS0_1, hS_1);

% Calculate set of representative precipitation fall lines for section
nPrecLines = 20;
sPrecOffset_1 = hLMax_1*U_1*tauF_1/hS_1;
sPrecLines_1 = linspace(sPath_1(1) + sPrecOffset_1/2, ...
    sPath_1(end) - sPrecOffset_1/2, nPrecLines);
sPrecLines_1 = [sPrecLines_1 + sPrecOffset_1; sPrecLines_1];
hPrecLines_1 = [zeros(1,nPrecLines); hLMax_1*ones(1,nPrecLines)];

% Calculate streamlines for section (1 km spacing in elevation)
nStreamlines = floor(hLMax_1*1e-3);
zS0Section_1 = 1e3*(1:nStreamlines);
sLSection_1 = cell(nStreamlines,1);
zLSection_1 = cell(nStreamlines,1);
for j = 1:nStreamlines
    [~, ~, zLSection_1{j}, sLSection_1{j}] = ...
        streamline(xPath_1(1), yPath_1(1), zS0Section_1(j), ...
        x, y, hGrid, U_1, azimuth_1, NM_1, fC, hRho_1, (j==1));
    sLSection_1{j}(end+1) = nan;
    zLSection_1{j}(end+1) = nan;
end
sLSection_1 = cell2mat(sLSection_1);
zLSection_1 = cell2mat(zLSection_1);
% Align sLSection_1 with path origin
sLSection_1 = sLSection_1 - sLSection_1(1) ...
    - sqrt((xPath_1(1) - xSection0)^2 + (yPath_1(1) - ySection0)^2);

%... Precipitation rate along section
F = griddedInterpolant({y, x}, pGrid_1, 'linear', 'linear');    
pSection_1 = F(yPath_1, xPath_1);

%... Precipitation d2H and d2H0 along section
F = griddedInterpolant({y, x}, d2HGrid_1, 'linear', 'linear'); 
d2HSection_1 = F(yPath_1, xPath_1);
[~, latPath_1] = xy2lonlat(xPath_1, yPath_1, lon0, lat0);
d2H0Section_1 = d2H0_1 + dD2H0_dLat_1.*(abs(latPath_1) - abs(lat0));
clear F

%... Precipitation state #2
% Section path passing through section origin: xSection0, ySection0
[xPath_2, yPath_2, sPath_2, sLimits_2] = ...
    windPath(xSection0, ySection0, azimuth_2, x, y, xyLimits);
% Land surface elevation along wind path
hLPath_2 = interp2(x, y, hGrid, xPath_2, yPath_2, 'linear', 0);
% Set maximum height of section
hLMax_2 = max(hLPath_2) + 2*hS_2;

% Calculate cloud-water density,and 248 and 268 K isotherms
[zRhoC_2, rhoC_2, zCMean_2, z248Path_2, z268Path_2] = ...
    cloudWater(xPath_2, yPath_2, hLMax_2, ...
    x, y, hGrid, U_2, azimuth_2, NM_2, fC, kappa_2, tauC_2, tauF_2, hRho_2, ...
    zBar_2, T_2, gammaEnv_2, gammaSat_2, gammaRatio_2, rhoS0_2, hS_2);

% Calculate set of representative precipitation fall lines for section
nPrecLines = 20;
sPrecOffset_2 = hLMax_2*U_2*tauF_2/hS_2;
sPrecLines_2 = linspace(sPath_2(1) + sPrecOffset_2/2, ...
    sPath_2(end) - sPrecOffset_2/2, nPrecLines);
sPrecLines_2 = [sPrecLines_2 + sPrecOffset_2; sPrecLines_2];
hPrecLines_2 = [zeros(1,nPrecLines); hLMax_2*ones(1,nPrecLines)];
% Calculate streamlines for section (1 km spacing in elevation)
nStreamlines = floor(hLMax_2*1e-3);
zS0Section_2 = 1e3*(1:nStreamlines);
sLSection_2 = cell(nStreamlines,1);
zLSection_2 = cell(nStreamlines,1);
for j = 1:nStreamlines
    [~, ~, zLSection_2{j}, sLSection_2{j}] = ...
        streamline(xPath_2(1), yPath_2(1), zS0Section_2(j), ...
        x, y, hGrid, U_2, azimuth_2, NM_2, fC, hRho_2, (j==1));
    sLSection_2{j}(end+1) = nan;
    zLSection_2{j}(end+1) = nan;
end
sLSection_2 = cell2mat(sLSection_2);
zLSection_2 = cell2mat(zLSection_2);
% Align sLSection_2 with path origin
sLSection_2 = sLSection_2 - sLSection_2(1) ...
    - sqrt((xPath_2(1) - xSection0)^2 + (yPath_2(1) - ySection0)^2);

%... Precipitation rate along section
F = griddedInterpolant({y, x}, pGrid_2, 'linear', 'linear');    
pSection_2 = F(yPath_2, xPath_2);

%... Precipitation d2H and d2H0 along section
F = griddedInterpolant({y, x}, d2HGrid_2, 'linear', 'linear'); 
d2HSection_2 = F(yPath_2, xPath_2);
[~, latPath_2] = xy2lonlat(xPath_2, yPath_2, lon0, lat0);
d2H0Section_2 = d2H0_2 + dD2H0_dLat_2.*(abs(latPath_2) - abs(lat0));
clear F

%% Additional results
%... Streamline and cloudwater 
diary on
fprintf('\n------------ Streamlines and Cloud Water -------------\n')
fprintf('Vertical exaggeration for streamline figure: %g (dimensionless\n', VE);
fprintf('Starting elevation for streamlines: %.0f m\n', zL0Map);
fprintf('Precipitation State 1:\n')
fprintf('Mean height (above ground surface) for cloud water: %.0f m\n', zCMean_1);
fprintf('Precipitation State 2:\n')
fprintf('Mean height (above ground surface) for cloud water: %.0f m\n', zCMean_2);
diary off

%% Plot Figures
Fig01 % Plot topography
Fig02 % Plot precipitation rate
Fig03 % Plot streamlines, precipitation state #1
Fig04 % Plot streamlines, precipitation state #2
Fig05 % Plot cross section, precipitation state #1
Fig06 % Plot cross section, precipitation state #2
Fig07 % Plot predicted delta2H fractionation
Fig08 % Plot predicted delta18O fractionation
Fig09 % Plot percipitation source
Fig10 % Plot precipitation rate, precipitation state #1
Fig11 % Plot precipitation rate, precipitation state #2
Fig12 % Plot moisture ratio, precipitation state #1
Fig13 % Plot moisture ratio, precipitation state #2
Fig14 % Plot predicted delta2H fractionation, precipitation state #1
Fig15 % Plot predicted delta2H fractionation, precipitation state #2
Fig16 % Plot surface temperature, precipitation state #1
Fig17 % Plot surface temperature, precipitation state #2
Fig18 % Plot surface velocity ratio, u'/U, precipitation state #1
Fig19 % Plot surface velocity ratio, u'/U, precipitation state #2
Fig20 % Plot sample locations, with outliers labeled (stdResiduals > 3)

%% Report compute time
diary on
fprintf('Compute time: %.2f minutes\n', minutes(datetime - startTime));
% Close diary for the last time
diary off

%% Fig01, Plot topography and samples 
    function Fig01
        figure(1)
        pcolor(lon, lat, hGrid);
        shading interp        
        %... Set up and scale colormap
        % Color in first row is set to gray to highlight ocean regions.
        cMap = haxby(800);
        cMap = [[0.7 0.7 0.7]; cMap(201:end,:)];
        cMap = cmapscale(hGrid(isMap & hGrid>0), cMap, 0.5);
        cMap = [[0.7 0.7 0.7]; cMap];
        colormap(cMap);        
        hold on
        % Define cMin and cMax so that gray matches 0 m contour
        cMax = max(hGrid(isMap), [], 'all');
        clim([0, cMax]);
        %... Plot coast line 
        plot(contours(:,1), contours(:,2), '-k');
        %... Plot sample catchments
        if ~isempty(sampleFile)
            for k=1:nSamples
                isC = false(size(hGrid));
                if k~=nSamples
                    ijC = ijCatch(ptrCatch(k):ptrCatch(k+1)-1);
                else
                    ijC = ijCatch(ptrCatch(k):end);
                end
                isC(ijC) = true;
                [b, ~, nObjects] = bwboundaries(isC);
                b = cell2mat(b);
                if nObjects > 1, error('Catchment has more than one set of pixels'), end
                plot(lon(b(:,2)), lat(b(:,1)), '-r', 'LineWidth', 1)
            end
        end
        %... Plot continental divide data, if present
        if ~isempty(contDivideLon)
            plot(contDivideLon, contDivideLat, ...
                'Color', [0.5 0.5 0.5], 'LineWidth', 5);
        end        
        %... Plot sample locations
        if ~isempty(sampleFile)
            plot(sampleLon, sampleLat, ...
                'ok', 'MarkerSize', 12, 'LineWidth', 1);
        end
        %... Plot section origin
        plot(sectionLon0, sectionLat0, 'sw', 'LineWidth', 4, 'MarkerSize', 18)        
        plot(sectionLon0, sectionLat0, 'sk', 'LineWidth', 2, 'MarkerSize', 18)
        %... Draw arrows for wind directions
        quiver(lonArrowStart_1, latArrowStart_1, ...
            lonArrowOffset_1, latArrowOffset_1, ...
            'MaxHeadSize', 2, 'LineWidth', 3, 'Color', 'b');
        quiver(lonArrowStart_2, latArrowStart_2, ...
            lonArrowOffset_2, latArrowOffset_2, ...
            'MaxHeadSize', 2, 'LineWidth', 3, 'Color', 'r');        
        %... Format plot
        daspect(dataAspectVector);
        axis(mapLimits);
        box on
        hA = gca;
        hA.TickDir = 'Out';
        hA.Layer = 'top';
        hA.FontSize = 16;
        hA.LineWidth = 1;
        %... Create colorbar
        hCB = colorbar;
        hCB.Label.String = 'Elevation (m)';
        hCB.Label.FontSize = 16;
        hCB.Label.LineWidth = 1;
        %... Write labels
        hT = title('Fig. 1. Topography and sample locations', 'FontSize', 16);
        hT.Units = 'normalized'; 
        hT.Position(2) = hT.Position(2) + 0.02;
        xlabel('Longitude (deg)', 'FontSize', 16)
        ylabel('Latitude (deg)', 'FontSize', 16)
        %... Save figure in pdf format
        printFigure(runPath)
    end

%% Fig02, Plot precipitation rate (mm/h)
    function Fig02
        figure(2)
        %... Convert from kg/m^2/s to mm/h
        pcolor(lon, lat, pGrid*3.6e3);
        shading interp
        %... Set up and scale colormap
        % Colormap is reversed, and first row set to white to
        % highlight zero values. Number of rows is set to 1000, 
        % to ensure that white regions are approximately zero. 
        cMap = parula(1000);
        cMap = cMap(end:-1:1,:);
        cMap = cmapscale(pGrid(isMap)*3.6e3, cMap, 0.3);
        cMap = [[1 1 1]; cMap];
        colormap(cMap);
        % Define cMin so that color starts at 0.1 mm/h
        cMax = max(pGrid(isMap), [], 'all')*3.6e3;
        clim([0.1, cMax]);
        hold on
        %... Plot continental divide data, if present
        if ~isempty(contDivideLon)
            plot(contDivideLon, contDivideLat, ...
                'Color', [0.5 0.5 0.5], 'LineWidth', 5);
        end        
        %... Plot section origin
        plot(sectionLon0, sectionLat0, 'sk', 'LineWidth', 2, 'MarkerSize', 18)
        %... Draw arrow for wind direction
        quiver(lonArrowStart_1, latArrowStart_1, ...
            lonArrowOffset_1, latArrowOffset_1, ...
            'MaxHeadSize', 2, 'LineWidth', 3, 'Color', 'b');
        quiver(lonArrowStart_2, latArrowStart_2, ...
            lonArrowOffset_2, latArrowOffset_2, ...
            'MaxHeadSize', 2, 'LineWidth', 3, 'Color', 'r');
        %... Plot representative contours
        plot(contours(:,1), contours(:,2), '-k');
        %... Format plot
        daspect(dataAspectVector);
        axis(mapLimits);
        box on
        hA = gca;
        hA.TickDir = 'Out';
        hA.Layer = 'top';
        hA.FontSize = 16;
        hA.LineWidth = 1;
        %... Create colorbar
        hCB = colorbar;
        hCB.Label.String = 'mm/hr';
        hCB.Label.FontSize = 16;
        hCB.Label.LineWidth = 1;
        %... Write labels
        hT = title('Fig. 2. Precipitation rate', 'FontSize', 16);
        hT.Units = 'normalized'; 
        hT.Position(2) = hT.Position(2) + 0.02;
        xlabel('Longitude (deg)', 'FontSize', 16)
        ylabel('Latitude (deg)', 'FontSize', 16)
        %... Save figure in pdf format
        printFigure(runPath)
    end

%% Fig03, Plot streamline, precipitation state #1
    function Fig03
        figure(3)
        %... Plot topography
        surf(lon, lat, hGrid*1e-3);
        shading interp
        %... Set up and scale colormap
        % Color in first row is set to gray to highlight ocean regions.
        cMap = haxby(800);
        cMap = [[0.7 0.7 0.7]; cMap(201:end,:)];
        cMap = cmapscale(hGrid(isMap & hGrid>0), cMap, 0.5);
        cMap = [[0.7 0.7 0.7]; cMap];
        colormap(cMap);
        % Define cMin and cMax so that gray matches 0 m contour
        cMax = max(hGrid(isMap), [], 'all')*1e-3;
        cMin = 1e-3*cMax;
        clim([cMin, cMax]);        
        hold on
        %... Plot coast line
        plot3(contours(:,1), contours(:,2), zeros(size(contours(:,1))), '-k');
        %... Plot section origin 
        % Note that the square marker is centered on the origin, so only
        % the upper half of the marker is visible. 
        plot3(sectionLon0, sectionLat0, sectionH0*1e-3, ...
            'sw', 'LineWidth', 5, 'MarkerSize', 18)
        plot3(sectionLon0, sectionLat0, sectionH0*1e-3, ...
            'sk', 'LineWidth', 2, 'MarkerSize', 18)
        %... Plot streamlines
        plot3(lonLMap_1, latLMap_1, zLMap_1*1e-3, '-b');
        %... Draw arrow for wind direction
        quiver3(lonArrowStart_1, latArrowStart_1, ...
            max(zLMap_1,[],'all')*1e-3, ...
            lonArrowOffset_1, latArrowOffset_1, 0, ...
            'MaxHeadSize', 2, 'LineWidth', 3, 'Color', 'b');
        %... View at 60 degree from wind path from the south, and 
        % looking down at 45 degrees.
        viewAz = (60 - azimuth_1);
        viewAz = viewAz - 180*floor((viewAz - -90)/180);
        view(viewAz, 45)
        %... Format plot
        daspect(dataAspectVector);
        axis(mapLimits);
        axis manual
        hA = gca;
        hA.ZTick = [0 cMax];
        hA.ZTickLabels = [0 round(cMax,1)];
        hA.Box = 'on';
        hA.BoxStyle = 'full';
        hA.FontSize = 16;
        hA.LineWidth = 1;
        hA.ZLim(1) = min(hGrid(:));
        %... Write labels
        hT = title('Fig. 3. Streamlines for precipitation state #1', ...
            'FontSize', 16);
        hT.Units = 'normalized'; 
        hT.Position(2) = hT.Position(2) + 0.02;
        xlabel('Longitude (deg)', 'FontSize', 16)
        ylabel('Latitude (deg)', 'FontSize', 16)
        zlabel('z (km)')
        %... Save figure in pdf format
        printFigure(runPath)
    end

%% Fig04, Plot streamlines, precipitation state #2
    function Fig04
        figure(4)
        %... Plot topography
        surf(lon, lat, hGrid*1e-3);
        shading interp
        %... Set up and scale colormap
        % Color in first row is set to gray to highlight ocean regions.
        cMap = haxby(800);
        cMap = [[0.7 0.7 0.7]; cMap(201:end,:)];
        cMap = cmapscale(hGrid(isMap & hGrid>0), cMap, 0.5);
        cMap = [[0.7 0.7 0.7]; cMap];
        colormap(cMap);        
        % Define cMin and cMax so that gray matches 0 m contour
        cMax = max(hGrid(isMap), [], 'all')*1e-3;
        cMin = 1e-3*cMax;
        clim([cMin, cMax]);
        hold on
        %... Plot coast line
        plot3(contours(:,1), contours(:,2), zeros(size(contours(:,1))), '-k');
        %... Plot section origin
        % Note that the 's' marker is centered on the origin, so only
        % the upper half of the marker is visible. 
        plot3(sectionLon0, sectionLat0, sectionH0*1e-3, ...
            'sw', 'LineWidth', 5, 'MarkerSize', 18)
        plot3(sectionLon0, sectionLat0, sectionH0*1e-3, ...
            'sk', 'LineWidth', 2, 'MarkerSize', 18)
        %... Plot streamlines
        plot3(lonLMap_2, latLMap_2, zLMap_2*1e-3, '-r');
        %... Draw arrow for wind direction
        quiver3(lonArrowStart_2, latArrowStart_2, ...
            max(zLMap_2,[],'all')*1e-3, ...
            lonArrowOffset_2, latArrowOffset_2, 0, ...
            'MaxHeadSize', 2, 'LineWidth', 3, 'Color', 'r');        
        %... View at 60 degree from wind path from the south, and 
        % looking down at 45 degrees.
        viewAz = (60 - azimuth_2);
        viewAz = viewAz - 180*floor((viewAz - -90)/180);
        view([viewAz, 45])
        %... Format plot
        daspect(dataAspectVector);
        axis(mapLimits);
        axis manual
        hA = gca;
        hA.ZTick = [0 cMax];
        hA.ZTickLabels = [0 round(cMax,1)];
        hA.Box = 'on';
        hA.BoxStyle = 'full';
        hA.FontSize = 16;
        hA.LineWidth = 1;
        hA.ZLim(1) = min(hGrid(:));
        %... Write labels
        hT = title('Fig. 4. Streamlines for precipitation state #2', ...
            'FontSize', 16);
        hT.Units = 'normalized'; 
        hT.Position(2) = hT.Position(2) + 0.02;
        xlabel('Longitude (deg)', 'FontSize', 16)
        ylabel('Latitude (deg)', 'FontSize', 16)
        zlabel('z (km)')
        %... Save figure in pdf format
        printFigure(runPath)
    end

%% Fig05, Plot cross section, precipitation state #1
    function Fig05
        figure(5)
        %.... Generate color plot for cloud-water density
        hA1 = subplot(3,1,1);
        pcolor(sPath_1*1e-3, zRhoC_1*1e-3, rhoC_1);
        shading flat
        %... Calculate logical grid indicating nodes inside limits
        % for section and lying above the topography
        [S, Z] = meshgrid(sPath_1, zRhoC_1);
        isSection = sLimits_1(1)<=S & S<=sLimits_1(2) ...
            & 0<=zRhoC_1 & zRhoC_1<=hLMax_1 ...
            & ~inpolygon(S, Z, ...
            [sPath_1(1); sPath_1; sPath_1(end)], [0; hLPath_1;  0]);
        %... Colormap is generated with the first row set to light blue,
        % and the remaining rows set to a gray scale from 0.8 to 0.
        colormap([0.686, 1, 1; repmat(linspace(0.8, 0, 100)', 1, 3)]);
        %... Set climits to the range of positive values. The blue color
        % in the first row represents negative values for rhoC.
        cMax = max(rhoC_1(isSection));
        clim([0, cMax])
        hold on
        %... Plot freezing transition (WBF zone)
        plot(sPath_1*1e-3, z248Path_1*1e-3, '-b', 'LineWidth', 1);
        plot(sPath_1*1e-3, z268Path_1*1e-3, '-b', 'LineWidth', 1);
        %... Plot precipitation lines
        plot(sPrecLines_1*1e-3, hPrecLines_1*1e-3, ':k', 'LineWidth', 1);
        %... Plot topography
        fill([sPath_1(1); sPath_1; sPath_1(end)]*1e-3, [0; hLPath_1;  0]*1e-3, ...
            [0.82 0.70 0.55], 'EdgeColor', 'k', 'LineWidth', 0.5);
        %... Plot streamlines
        plot(sLSection_1*1e-3, zLSection_1*1e-3, '-k', 'LineWidth', 0.5)
        %... Adjust axis limits
        hA1.XLim = [sLimits_1(1), sLimits_1(2)]*1e-3;
        hA1.YLim = [0, hLMax_1]*1e-3;        
        %... Format plot
        box on
        grid off
        hA1.XTickLabel = {};
        hA1.FontSize = 12;
        hA1.LineWidth = 1;
        hA1.Layer = 'top';
        %... Write labels        
        hT = title('Fig. 5. Cross section for precipitation state #1', ...
            'FontSize', 12);
        hT.Units = 'normalized'; 
        hT.Position(2) = hT.Position(2) + 0.02;
        ylabel('Elevation (km)', 'FontSize', 10);  

        %... Precipitation rate along section
        hA2 = subplot(3,1,2);
        plot(sPath_1*1e-3, pSection_1*3.6e3, '-k', 'LineWidth', 2); 
        %... Adjust axis limits
        hA2.XLim = [sLimits_1(1), sLimits_1(2)]*1e-3;
        hA2.YLim = [0, max(pSection_1, [], 'all')*1.05]*3.6e3;
        %... Format plot        
        box on
        grid on
        hA2.XTickLabel = {};
        hA2.FontSize = 12;
        hA2.LineWidth = 1;
        %... Write labels        
        ylabel({'Precipitation', 'Rate (mm/hr)'}, 'FontSize', 12);  
        
        %... Precipitation d2H and d2H0 along section
        hA3 = subplot(3,1,3);
        hold on
        plot(sPath_1*1e-3, d2H0Section_1*1e3, '--k', 'LineWidth', 1); 
        plot(sPath_1*1e-3, d2HSection_1*1e3, '-k', 'LineWidth', 2); 
        %... Adjust axis limits
        hA3.XLim = [sLimits_1(1), sLimits_1(2)]*1e-3;
        hA3.YLim = [min(d2HSection_1(:)) - 10e-3, ...
            max([d2HSection_1(:); 10e-3])]*1e3;        
        %... Format plot        
        box on
        grid on
        hA3.FontSize = 12;
        hA3.LineWidth = 1;
        %... Write labels        
        xlabel('Section Distance (km)', 'FontSize', 12);
        ylabel(['\delta^{2}H (', char(8240), ')'], 'FontSize', 12);  
        
        % Adjust position relative to position and size of top plot
        hA2.Position(1) = hA1.Position(1);
        hA2.Position(2) = hA1.Position(2) - hA2.Position(4) - 0.1;
        hA2.Position(3) = hA1.Position(3);
        hA2.Position(4) = hA1.Position(4);   
        
        %... Plot wind direction in upper right corner of plot
        sArrowStart = hA2.Position(1) + 0.85*hA2.Position(3);
        hArrowStart = (hA1.Position(2) + hA2.Position(2) + hA2.Position(4))/2;
        sArrowEnd = hA2.Position(1) + 0.95*hA2.Position(3);
        hArrowEnd = hArrowStart;  
        hTA = annotation('textarrow',[sArrowStart, sArrowEnd], ...
            [hArrowStart, hArrowEnd], 'LineWidth', 1);
        hTA.String  = sprintf('%.0f° ', azimuth_1);
        hTA.FontSize = 12;

        % Adjust position relative to position and size of middle plot
        hA3.Position(1) = hA2.Position(1);
        hA3.Position(2) = hA2.Position(2) - hA3.Position(4) - 0.1;
        hA3.Position(3) = hA2.Position(3);
        hA3.Position(4) = hA2.Position(4);

        %... Save figure in pdf format
        printFigure(runPath)
    end

%% Fig06, Plot cross section, precipitation state #2
    function Fig06
        figure(6)
        %.... Generate color plot for cloudwater density
        hA1 = subplot(3,1,1);
        pcolor(sPath_2*1e-3, zRhoC_2*1e-3, rhoC_2);
        shading flat
        %... Calculate logical grid indicating nodes inside limits
        % for section and lying above the topography
        [S, Z] = meshgrid(sPath_2, zRhoC_2);
        isSection = sLimits_2(1)<=S & S<=sLimits_2(2) ...
            & 0<=zRhoC_2 & zRhoC_2<=hLMax_2 ...
            & ~inpolygon(S, Z, ...
            [sPath_2(1); sPath_2; sPath_2(end)], [0; hLPath_2;  0]);
        %... Colormap is generated with the first row set to light blue,
        % and the remaining rows set to a gray scale from 0.8 to 0.
        colormap([0.686, 1, 1; repmat(linspace(0.8, 0, 100)', 1, 3)]);
        %... Set climits to the range of positive values. The blue color
        % in the first row represents negative values for rhoC.
        cMax = max(rhoC_2(isSection));
        clim([0, cMax])
        hold on
        %... Plot freezing transition (WBF zone)
        plot(sPath_2*1e-3, z248Path_2*1e-3, '-b', 'LineWidth', 1);
        plot(sPath_2*1e-3, z268Path_2*1e-3, '-b', 'LineWidth', 1);
        %... Plot precipitation lines
        plot(sPrecLines_2*1e-3, hPrecLines_2*1e-3, ':k', 'LineWidth', 1);
        %... Plot topography
        fill([sPath_2(1); sPath_2; sPath_2(end)]*1e-3, [0; hLPath_2;  0]*1e-3, ...
            [0.82 0.70 0.55], 'EdgeColor', 'k', 'LineWidth', 0.5);
        %... Plot streamlines
        plot(sLSection_2*1e-3, zLSection_2*1e-3, '-k', 'LineWidth', 0.5)
        %... Adjust axis limits
        hA1.XLim = [sLimits_2(1), sLimits_2(2)]*1e-3;
        hA1.YLim = [0, hLMax_2]*1e-3;        
        %... Format plot
        box on
        grid off
        hA1.XTickLabel = {};        
        hA1.FontSize = 12;
        hA1.LineWidth = 1;
        hA1.Layer = 'top';
        %... Write labels        
        hT = title('Fig. 6. Cross section for precipitation state #2', ...
            'FontSize', 12);
        hT.Units = 'normalized'; 
        hT.Position(2) = hT.Position(2) + 0.02;
        ylabel('Elevation (km)', 'FontSize', 12);

        %... Precipitation rate along section
        hA2 = subplot(3,1,2);
        hold on
        plot(sPath_2*1e-3, pSection_2*3.6e3, '-k', 'LineWidth', 2); 
        %... Adjust axis limits
        hA2.XLim = [sLimits_2(1), sLimits_2(2)]*1e-3;
        hA2.YLim = [0, max(pSection_2, [], 'all')*1.05]*3.6e3;
        %... Format plot        
        box on
        grid on
        hA2.XTickLabel = {};
        hA2.FontSize = 12;
        hA2.LineWidth = 1;
        %... Write labels        
        ylabel({'Precipitation', 'Rate (mm/hr)'}, 'FontSize', 12);  

        %... Precipitation d2H and d2H0 along section
        hA3 = subplot(3,1,3);
        hold on
        plot(sPath_2*1e-3, d2H0Section_2*1e3, '--k', 'LineWidth', 1); 
        plot(sPath_2*1e-3, d2HSection_2*1e3, '-k', 'LineWidth', 2); 
        %... Adjust axis limits        
        hA3.XLim = [sLimits_2(1), sLimits_2(2)]*1e-3;
        hA3.YLim = [min(d2HSection_2(:)) - 10e-3, ...
            max([d2HSection_2(:); 10e-3])]*1e3;
        %... Format plot        
        box on
        grid on
        hA3.FontSize = 12;
        hA3.LineWidth = 1;
        %... Write labels        
        xlabel('Section Distance (km)', 'FontSize', 12);
        ylabel(['\delta^{2}H (', char(8240), ')'], 'FontSize', 12);  
        
        % Adjust position relative to position and size of top plot
        hA2.Position(1) = hA1.Position(1);
        hA2.Position(2) = hA1.Position(2) - hA2.Position(4) - 0.1;
        hA2.Position(3) = hA1.Position(3);
        hA2.Position(4) = hA1.Position(4);   

        %... Plot wind direction in upper right corner of plot
        sArrowStart = hA2.Position(1) + 0.85*hA2.Position(3);
        hArrowStart = (hA1.Position(2) + hA2.Position(2) + hA2.Position(4))/2;
        sArrowEnd = hA2.Position(1) + 0.95*hA2.Position(3);
        hArrowEnd = hArrowStart;  
        hTA = annotation('textarrow',[sArrowStart, sArrowEnd], ...
            [hArrowStart, hArrowEnd], 'LineWidth', 1);
        hTA.String  = sprintf('%.0f° ', azimuth_2);
        hTA.FontSize = 12;
        
        % Adjust position relative to position and size of middle plot
        hA3.Position(1) = hA2.Position(1);
        hA3.Position(2) = hA2.Position(2) - hA3.Position(4) - 0.1;
        hA3.Position(3) = hA2.Position(3);
        hA3.Position(4) = hA2.Position(4);   

        %... Save figure in pdf format
        printFigure(runPath)
    end

%% Fig07, Plot predicted precipitation d2H
    function Fig07
        figure(7)
        pcolor(lon, lat, d2HGrid*1e3);
        shading interp
        %... Set up and scale colormap
        cMap = parula(256);
        cMap = cMap(end:-1:1, :);
        cMap = cmapscale(d2HGrid(isMap)*1e3, cMap, 0.3);
        colormap(cMap);
        cMin = min(d2HGrid(isMap), [], 'all')*1e3;
        cMax = max(d2HGrid(isMap), [], 'all')*1e3;
        clim([cMin, cMax]);        
        hold on
        %... Plot continental divide data, if present
        if ~isempty(contDivideLon)
            plot(contDivideLon, contDivideLat, ...
                'Color', [0.5 0.5 0.5], 'LineWidth', 5);
        end        
        %... Plot section origin
        plot(sectionLon0, sectionLat0, 'sk', 'LineWidth', 2, 'MarkerSize', 18)
        %... Draw arrow for wind direction
        quiver(lonArrowStart_1, latArrowStart_1, ...
            lonArrowOffset_1, latArrowOffset_1, ...
            'MaxHeadSize', 2, 'LineWidth', 3, 'Color', 'b');
        quiver(lonArrowStart_2, latArrowStart_2, ...
            lonArrowOffset_2, latArrowOffset_2, ...
            'MaxHeadSize', 2, 'LineWidth', 3, 'Color', 'r');
        %... Plot representative contours
        plot(contours(:,1), contours(:,2), '-k');
        %... Format plot
        daspect(dataAspectVector);
        axis(mapLimits);
        box on
        hA = gca;
        hA.TickDir = 'Out';
        hA.Layer = 'top';
        hA.FontSize = 16;
        hA.LineWidth = 1;
        %... Create colorbar
        hCB = colorbar;
        hCB.Label.String = ['\delta^{2}H (', char(8240), ')'];
        hCB.Label.FontSize = 16;
        hCB.Label.LineWidth = 1;
        %... Write labels
        hT = title(['Fig. 7. Predicted precipitation ', ...
            '\delta^{2}H'], 'FontSize', 16);
        hT.Units = 'normalized';
        hT.Position(2) = hT.Position(2) + 0.02;
        xlabel('Longitude (deg)', 'FontSize', 16)
        ylabel('Latitude (deg)', 'FontSize', 16)
        %... Save figure in pdf format
        printFigure(runPath)
    end

%% Fig08, Plot predicted precipitation d18O
    function Fig08
        figure(8)
        pcolor(lon, lat, d18OGrid*1e3);
        hold on
        shading interp
        %... Set up and scale colormap
        cMap = parula(256);
        cMap = cMap(end:-1:1, :);
        cMap = cmapscale(d18OGrid(isMap)*1e3, cMap, 0.3);
        colormap(cMap);
        cMin = min(d18OGrid(isMap), [], 'all')*1e3;
        cMax = max(d18OGrid(isMap), [], 'all')*1e3;
        clim([cMin, cMax]);        
        hold on        
        %... Plot continental divide data, if present
        if ~isempty(contDivideLon)
            plot(contDivideLon, contDivideLat, ...
                'Color', [0.5 0.5 0.5], 'LineWidth', 5);
        end        
        %... Plot section origin
        plot(sectionLon0, sectionLat0, 'sk', 'LineWidth', 2, 'MarkerSize', 18)
        %... Draw arrow for wind direction
        quiver(lonArrowStart_1, latArrowStart_1, ...
            lonArrowOffset_1, latArrowOffset_1, ...
            'MaxHeadSize', 2, 'LineWidth', 3, 'Color', 'b');
        quiver(lonArrowStart_2, latArrowStart_2, ...
            lonArrowOffset_2, latArrowOffset_2, ...
            'MaxHeadSize', 2, 'LineWidth', 3, 'Color', 'r');
        %... Plot representative contours
        plot(contours(:,1), contours(:,2), '-k');
        %... Format plot
        daspect(dataAspectVector);
        axis(mapLimits);
        box on
        hA = gca;
        hA.TickDir = 'Out';
        hA.Layer = 'top';
        hA.FontSize = 16;
        hA.LineWidth = 1;
        %... Create colorbar
        hCB = colorbar;
        hCB.Label.String = ['\delta^{18}O (', char(8240), ')'];
        hCB.Label.FontSize = 16;
        hCB.Label.LineWidth = 1;
        %... Write labels
        hT = title(['Fig. 8. Predicted precipitation ', ...
            '\delta^{18}O'], 'FontSize', 16);
        hT.Units = 'normalized'; 
        hT.Position(2) = hT.Position(2) + 0.02;
        xlabel('Longitude (deg)', 'FontSize', 16)
        ylabel('Latitude (deg)', 'FontSize', 16)
        %... Save figure in pdf format
        printFigure(runPath)
    end

%% Fig09, Plot precipitation source
    function Fig09
        figure(9)
        pcolor(lon, lat, fractionPGrid)
        hold on
        shading interp
        %... Set up and scale colormap
        % Colormap is set from red to blue, to correspond to 
        % precipitation states #1 and #2 as blue and red, respectively. 
        cMap = coolwarm;
        cMap = cMap(end:-1:1,:);
        cMap = cmapscale(fractionPGrid(isMap), cMap, 0.3);
        colormap(cMap);
        %... Plot continental divide data, if present
        if ~isempty(contDivideLon)
            plot(contDivideLon, contDivideLat, ...
                'Color', [0.5 0.5 0.5], 'LineWidth', 5);
        end        
        %... Plot section origin
        plot(sectionLon0, sectionLat0, 'sk', 'LineWidth', 2, 'MarkerSize', 18)
        %... Draw arrow for wind direction
        quiver(lonArrowStart_1, latArrowStart_1, ...
            lonArrowOffset_1, latArrowOffset_1, ...
            'MaxHeadSize', 2, 'LineWidth', 3, 'Color', 'b');
        quiver(lonArrowStart_2, latArrowStart_2, ...
            lonArrowOffset_2, latArrowOffset_2, ...
            'MaxHeadSize', 2, 'LineWidth', 3, 'Color', 'r');
        %... Plot representative contours
        plot(contours(:,1), contours(:,2), '-k');
        %... Format plot
        daspect(dataAspectVector);
        axis(mapLimits);
        box on
        hA = gca;
        hA.TickDir = 'Out';
        hA.Layer = 'top';
        hA.FontSize = 16;
        hA.LineWidth = 1;
        %... Create colorbar
        hCB = colorbar;
        hCB.Label.String = 'Fraction, Source #1 (blue arrow)';
        hCB.Label.FontSize = 16;
        hCB.Label.LineWidth = 1;
        %... Write labels
        hT = title('Fig. 9. Precipitation source', 'FontSize', 16);
        hT.Units = 'normalized'; 
        hT.Position(2) = hT.Position(2) + 0.02;
        xlabel('Longitude (deg)', 'FontSize', 16)
        ylabel('Latitude (deg)', 'FontSize', 16)
        %... Save figure in pdf format
        printFigure(runPath)
    end

%% Fig10, Plot precipitation rate (mm/h) for precipitation state #1
    function Fig10
        figure(10)
        %... Convert from kg/m^2/s to mm/h
        pcolor(lon, lat, pGrid_1*3.6e3);
        shading interp
        %... Set up and scale colormap
        % Colormap is reversed, and first row set to white to
        % highlight zero values. Number of rows is set to 1000, 
        % to ensure that white regions are approximately zero. 
        cMap = parula;
        cMap = cMap(end:-1:1,:);
        cMap = cmapscale(pGrid_1(isMap)*3.6e3, cMap, 0.3);
        cMap = [[1 1 1]; cMap(2:end,:)];
        colormap(cMap);
        % Define cMin so that color starts at 0.1 mm/h
        cMax = max(pGrid_1(isMap), [], 'all')*3.6e3;
        cMin = 0.1;
        clim([cMin, cMax]);
        hold on        
        %... Plot continental divide data, if present
        if ~isempty(contDivideLon)
            plot(contDivideLon, contDivideLat, ...
                'Color', [0.5 0.5 0.5], 'LineWidth', 5);
        end        
        %... Plot section origin
        plot(sectionLon0, sectionLat0, 'sk', 'LineWidth', 2, 'MarkerSize', 18)
        %... Draw arrow for wind direction
        quiver(lonArrowStart_1, latArrowStart_1, ...
            lonArrowOffset_1, latArrowOffset_1, ...
            'MaxHeadSize', 2, 'LineWidth', 3, 'Color', 'k');
        %... Plot representative contours
        plot(contours(:,1), contours(:,2), '-k');
        %... Format plot
        daspect(dataAspectVector);
        axis(mapLimits);
        box on
        hA = gca;
        hA.TickDir = 'Out';
        hA.Layer = 'top';
        hA.FontSize = 16;
        hA.LineWidth = 1;
        %... Create colorbar
        hCB = colorbar;
        hCB.Label.String = 'mm/hr';
        hCB.Label.FontSize = 16;
        hCB.Label.LineWidth = 1;
        %... Write labels
        hT = title('Fig. 10. Precipitation rate, state #1', 'FontSize', 16);
        hT.Units = 'normalized'; 
        hT.Position(2) = hT.Position(2) + 0.02;
        xlabel('Longitude (deg)', 'FontSize', 16)
        ylabel('Latitude (deg)', 'FontSize', 16)
        %... Save figure in pdf format
        printFigure(runPath)
    end
%% Fig11, Plot precipitation rate (mm/h) for precipitation state #2
    function Fig11
        figure(11)
        %... Convert from kg/m^2/s to mm/h
        pcolor(lon, lat, pGrid_2*3.6e3);
        shading interp
        %... Set up and scale colormap
        % Colormap is reversed, and first row set to white to
        % highlight zero values. Number of rows is set to 1000, 
        % to ensure that white regions are approximately zero. 
        cMap = parula(1000);
        cMap = cMap(end:-1:1,:);
        cMap = cmapscale(pGrid_2(isMap)*3.6e3, cMap, 0.3);
        cMap = [[1 1 1]; cMap(2:end,:)];
        colormap(cMap);
        % Define cMin so that color starts at 0.1 mm/h
        cMax = max(pGrid_2(isMap), [], 'all')*3.6e3;
        cMin = 0.1;
        clim([cMin, cMax]);
        hold on        
        %... Plot continental divide data, if present
        if ~isempty(contDivideLon)
            plot(contDivideLon, contDivideLat, ...
                'Color', [0.5 0.5 0.5], 'LineWidth', 5);
        end        
        %... Plot section origin
        plot(sectionLon0, sectionLat0, 'sk', 'LineWidth', 2, 'MarkerSize', 18)
        %... Draw arrow for wind direction
        quiver(lonArrowStart_2, latArrowStart_2, ...
            lonArrowOffset_2, latArrowOffset_2, ...
            'MaxHeadSize', 2, 'LineWidth', 3, 'Color', 'k');
        %... Plot representative contours
        plot(contours(:,1), contours(:,2), '-k');
        %... Format plot
        daspect(dataAspectVector);
        axis(mapLimits);
        box on
        hA = gca;
        hA.TickDir = 'Out';
        hA.Layer = 'top';
        hA.FontSize = 16;
        hA.LineWidth = 1;
        %... Create colorbar
        hCB = colorbar;
        hCB.Label.String = 'mm/hr';
        hCB.Label.FontSize = 16;
        hCB.Label.LineWidth = 1;
        %... Write labels
        hT = title('Fig. 11. Precipitation rate, state #2', 'FontSize', 16);
        hT.Units = 'normalized'; 
        hT.Position(2) = hT.Position(2) + 0.02;
        xlabel('Longitude (deg)', 'FontSize', 16)
        ylabel('Latitude (deg)', 'FontSize', 16)
        %... Save figure in pdf format
        printFigure(runPath)
    end

%% Fig12, Plot moisture ratio (dimensionless) for precipitation state #1
    function Fig12
        figure(12)
        pcolor(lon, lat, fMGrid_1)
        hold on
        shading interp
        %... Set up and scale colormap
        cMap = parula(256);
        cMap = cMap(end:-1:1,:);
        cMap = cmapscale(fMGrid_1(isMap), cMap, 0.5);
        colormap(cMap);
        %... Plot continental divide data, if present
        if ~isempty(contDivideLon)
            plot(contDivideLon, contDivideLat, ...
                'Color', [0.5 0.5 0.5], 'LineWidth', 5);
        end        
        %... Plot section origin
        plot(sectionLon0, sectionLat0, 'sk', 'LineWidth', 2, 'MarkerSize', 18)
        %... Draw arrow for wind direction
        quiver(lonArrowStart_1, latArrowStart_1, ...
            lonArrowOffset_1, latArrowOffset_1, ...
            'MaxHeadSize', 2, 'LineWidth', 3, 'Color', 'k');
        %... Plot representative contours
        plot(contours(:,1), contours(:,2), '-k');
        %... Format plot
        daspect(dataAspectVector);
        axis(mapLimits);
        box on
        hA = gca;
        hA.TickDir = 'Out';
        hA.Layer = 'top';
        hA.FontSize = 16;
        hA.LineWidth = 1;
        %... Create colorbar
        hCB = colorbar;
        hCB.Label.String = 'Moisture Ratio';
        hCB.Label.FontSize = 16;
        hCB.Label.LineWidth = 1;
        %... Write labels
        hT = title('Fig. 12. Moisture ratio, state #1', 'FontSize', 16);
        hT.Units = 'normalized'; 
        hT.Position(2) = hT.Position(2) + 0.02;
        xlabel('Longitude (deg)', 'FontSize', 16)
        ylabel('Latitude (deg)', 'FontSize', 16)
        %... Save figure in pdf format
        printFigure(runPath)
    end

%% Fig13, Plot moisture ratio (dimensionless) for precipitation state #2
    function Fig13
        figure(13)
        pcolor(lon, lat, fMGrid_2)
        hold on
        shading interp
        %... Set up and scale colormap
        cMap = parula(256);
        cMap = cMap(end:-1:1,:);
        cMap = cmapscale(fMGrid_2(isMap), cMap, 0.5);
        colormap(cMap);
        %... Plot continental divide data, if present
        if ~isempty(contDivideLon)
            plot(contDivideLon, contDivideLat, ...
                'Color', [0.5 0.5 0.5], 'LineWidth', 5);
        end        
        %... Plot section origin
        plot(sectionLon0, sectionLat0, 'sk', 'LineWidth', 2, 'MarkerSize', 18)
        %... Draw arrow for wind direction
        quiver(lonArrowStart_2, latArrowStart_2, ...
            lonArrowOffset_2, latArrowOffset_2, ...
            'MaxHeadSize', 2, 'LineWidth', 3, 'Color', 'k');
        %... Plot representative contours
        plot(contours(:,1), contours(:,2), '-k');
        %... Format plot
        daspect(dataAspectVector);
        axis(mapLimits);
        box on
        hA = gca;
        hA.TickDir = 'Out';
        hA.Layer = 'top';
        hA.FontSize = 16;
        hA.LineWidth = 1;
        %... Create colorbar
        hCB = colorbar;
        hCB.Label.String = 'Moisture Ratio';
        hCB.Label.FontSize = 16;
        hCB.Label.LineWidth = 1;
        %... Write labels
        hT = title('Fig. 13. Moisture ratio, state #2', 'FontSize', 16);
        hT.Units = 'normalized'; 
        hT.Position(2) = hT.Position(2) + 0.02;
        xlabel('Longitude (deg)', 'FontSize', 16)
        ylabel('Latitude (deg)', 'FontSize', 16)
        %... Save figure in pdf format
        printFigure(runPath)
    end

%% Fig014, Plot predicted d2H for precipitation state #1
    function Fig14
        figure(14)
        pcolor(lon, lat, d2HGrid_1*1e3);
        hold on
        shading interp
        %... Set up and scale colormap
        cMap = parula(256);
        cMap = cMap(end:-1:1,:);
        cMap = cmapscale(d2HGrid_1(isMap)*1e3, cMap, 0.3);
        colormap(cMap);
        cMin = min(d2HGrid_1(isMap), [], 'all')*1e3;
        cMax = max(d2HGrid_1(isMap), [], 'all')*1e3;        
        clim([cMin, cMax]);
        %... Plot continental divide data, if present
        if ~isempty(contDivideLon)
            plot(contDivideLon, contDivideLat, ...
                'Color', [0.5 0.5 0.5], 'LineWidth', 5);
        end        
        %... Plot section origin
        plot(sectionLon0, sectionLat0, 'sk', 'LineWidth', 2, 'MarkerSize', 18)
        %... Draw arrow for wind direction
        quiver(lonArrowStart_1, latArrowStart_1, ...
            lonArrowOffset_1, latArrowOffset_1, ...
            'MaxHeadSize', 2, 'LineWidth', 3, 'Color', 'k');
        %... Plot representative contours
        plot(contours(:,1), contours(:,2), '-k');
        %... Format plot
        daspect(dataAspectVector);
        axis(mapLimits);
        box on
        hA = gca;
        hA.TickDir = 'out';
        hA.Layer = 'top';
        hA.FontSize = 16;
        hA.LineWidth = 1;
        %... Create colorbar
        hCB = colorbar;
        hCB.Label.String = ['\delta^{2}H (', char(8240), ')'];
        hCB.Label.FontSize = 16;
        hCB.Label.LineWidth = 1;
        %... Write labels
        hT = title(['Fig. 14. Predicted precipitation ', ...
            '\delta^{2}H, ', 'state #1'], 'FontSize', 16);
        hT.Units = 'normalized'; 
        hT.Position(2) = hT.Position(2) + 0.02;
        xlabel('Longitude (deg)', 'FontSize', 16)
        ylabel('Latitude (deg)', 'FontSize', 16)
        %... Save figure in pdf format
        printFigure(runPath)
    end

%% Fig15, Plot predicted d2H for precipitation state #2
    function Fig15
        figure(15)
        pcolor(lon, lat, d2HGrid_2*1e3);
        hold on
        shading interp
        %... Set up and scale colormap
        cMap = parula(256);
        cMap = cMap(end:-1:1,:);
        cMap = cmapscale(d2HGrid_2(isMap)*1e3, cMap, 0.3);
        colormap(cMap);
        cMin = min(d2HGrid_2(isMap), [], 'all')*1e3;
        cMax = max(d2HGrid_2(isMap), [], 'all')*1e3;
        clim([cMin, cMax]);        
        %... Plot continental divide data, if present
        if ~isempty(contDivideLon)
            plot(contDivideLon, contDivideLat, ...
                'Color', [0.5 0.5 0.5], 'LineWidth', 5);
        end        
        %... Plot section origin
        plot(sectionLon0, sectionLat0, 'sk', 'LineWidth', 2, 'MarkerSize', 18)
        %... Draw arrow for wind direction
        quiver(lonArrowStart_2, latArrowStart_2, ...
            lonArrowOffset_2, latArrowOffset_2, ...
            'MaxHeadSize', 2, 'LineWidth', 3, 'Color', 'k');
        %... Plot representative contours
        plot(contours(:,1), contours(:,2), '-k');
        %... Format plot
        daspect(dataAspectVector);
        axis(mapLimits);
        box on
        hA = gca;
        hA.TickDir = 'Out';
        hA.Layer = 'top';
        hA.FontSize = 16;
        hA.LineWidth = 1;
        %... Create colorbar
        hCB = colorbar;
        hCB.Label.String = ['\delta^{2}H (', char(8240), ')'];
        hCB.Label.FontSize = 16;
        hCB.Label.LineWidth = 1;
        %... Write labels
        hT = title(['Fig. 15. Predicted precipitation ', ...
            '\delta^{2}H, ', 'state #2'], 'FontSize', 16);
        hT.Units = 'normalized'; 
        hT.Position(2) = hT.Position(2) + 0.02;
        xlabel('Longitude (deg)', 'FontSize', 16)
        ylabel('Latitude (deg)', 'FontSize', 16)
        %... Save figure in pdf format
        printFigure(runPath)
    end

%% Fig16, Plot surface-air temperature for precipitation state #1
    function Fig16
        figure(16)
        %... Mean temperature at surface
        %... Construct grid for temperature at land surface
        TGrid = T_1(1) - gammaSat_1(1)*hGrid;
        pcolor(lon, lat, TGrid-TC2K)
        hold on
        shading interp
        %... Set up and scale colormap
        cMap = coolwarm;
        cMap = cmapscale(TGrid(isMap)-TC2K, cMap, 0.5, 0);
        colormap(cMap);
        cMin = min(TGrid(isMap), [], 'all')-TC2K;
        cMax = max(TGrid(isMap), [], 'all')-TC2K;
        clim([cMin, cMax]);        
        
        %... Plot representative contours
        plot(contours(:,1), contours(:,2), '-k');
        %... Plot continental divide data, if present
        if ~isempty(contDivideLon)
            plot(contDivideLon, contDivideLat, ...
                'Color', [0.5 0.5 0.5], 'LineWidth', 5);
        end        
        %... Plot section origin
        plot(sectionLon0, sectionLat0, 'sk', 'LineWidth', 2, 'MarkerSize', 18)
        %... Plot representative contours
        plot(contours(:,1), contours(:,2), '-k');
        %... Format plot
        daspect(dataAspectVector);
        axis(mapLimits);
        box on
        hA = gca;
        hA.TickDir = 'Out';
        hA.Layer = 'top';
        hA.FontSize = 18;
        hA.LineWidth = 1;
        %... Create colorbar
        hCB = colorbar;
        hCB.Label.String = '°C';
        hCB.Label.FontSize = 18;
        hCB.Label.LineWidth = 1;
        %... Write labels
        hT = title('Fig. 16. Surface-air temperature, state #1', 'FontSize', 16);
        hT.Units = 'normalized'; 
        hT.Position(2) = hT.Position(2) + 0.02;
        xlabel('Longitude (deg)', 'FontSize', 16)
        ylabel('Latitude (deg)', 'FontSize', 16)
        %... Save figure in pdf format
        printFigure(runPath)
    end

%% Fig17, Plot surface-air temperature for precipitation state #2
    function Fig17
        figure(17)
        %... Mean temperature at surface
        %... Construct grid for temperature at land surface
        TGrid = T_2(1) - gammaSat_2(1)*hGrid;
        pcolor(lon, lat, TGrid-TC2K)
        hold on
        shading interp
        %... Set up and scale colormap
        cMap = coolwarm;
        cMap = cmapscale(TGrid(isMap)-TC2K, cMap, 0.5, 0);
        colormap(cMap);
        cMin = min(TGrid(isMap), [], 'all')-TC2K;
        cMax = max(TGrid(isMap), [], 'all')-TC2K;
        clim([cMin, cMax]);        
        
        %... Plot representative contours
        plot(contours(:,1), contours(:,2), '-k');
        %... Plot continental divide data, if present
        if ~isempty(contDivideLon)
            plot(contDivideLon, contDivideLat, ...
                'Color', [0.5 0.5 0.5], 'LineWidth', 5);
        end        
        %... Plot section origin
        plot(sectionLon0, sectionLat0, 'sk', 'LineWidth', 2, 'MarkerSize', 18)
        %... Plot representative contours
        plot(contours(:,1), contours(:,2), '-k');
        %... Format plot
        daspect(dataAspectVector);
        axis(mapLimits);
        box on
        hA = gca;
        hA.TickDir = 'Out';
        hA.Layer = 'top';
        hA.FontSize = 18;
        hA.LineWidth = 1;
        %... Create colorbar
        hCB = colorbar;
        hCB.Label.String = '°C';
        hCB.Label.FontSize = 18;
        hCB.Label.LineWidth = 1;
        %... Write labels
        hT = title('Fig. 17. Surface-air temperature, state #2', 'FontSize', 16);
        hT.Units = 'normalized'; 
        hT.Position(2) = hT.Position(2) + 0.02;
        xlabel('Longitude (deg)', 'FontSize', 16)
        ylabel('Latitude (deg)', 'FontSize', 16)
        %... Save figure in pdf format
        printFigure(runPath)
    end

%% Fig18, Plot surface velocity ratio, u'/U, for precipitation state #1
    function Fig18
        figure(18)
        %... Velocity ratio, u'U, at zBar = 0
        uRatioGrid = uPrime( ... 
            0, x, y, hGrid, U_1, azimuth_1, NM_1, fC, hRho_1, true)/U_1;
        pcolor(lon, lat, uRatioGrid)
        hold on
        shading interp
        %... Set up and scale colormap
        cMap = coolwarm;
        cMap = cmapscale(uRatioGrid(isMap), cMap, 0.5, 0);
        colormap(cMap);
        cMin = min(uRatioGrid(isMap), [], 'all');
        cMax = max(uRatioGrid(isMap), [], 'all');
        clim([cMin, cMax]);        
        %... Plot representative contours
        plot(contours(:,1), contours(:,2), '-k');
        %... Plot continental divide data, if present
        if ~isempty(contDivideLon)
            plot(contDivideLon, contDivideLat, ...
                'Color', [0.5 0.5 0.5], 'LineWidth', 5);
        end        
        %... Plot section origin
        plot(sectionLon0, sectionLat0, 'sk', 'LineWidth', 2, 'MarkerSize', 18)
        %... Plot representative contours
        plot(contours(:,1), contours(:,2), '-k');
        %... Format plot
        daspect(dataAspectVector);
        axis(mapLimits);
        box on
        hA = gca;
        hA.TickDir = 'Out';
        hA.Layer = 'top';
        hA.FontSize = 18;
        hA.LineWidth = 1;
        %... Create colorbar
        hCB = colorbar;
        hCB.Label.String = 'u''/U';
        hCB.Label.FontSize = 18;
        hCB.Label.LineWidth = 1;
        %... Write labels
        hT = title('Fig. 18. Surface velocity ratio, state #1', 'FontSize', 16);
        hT.Units = 'normalized'; 
        hT.Position(2) = hT.Position(2) + 0.02;
        xlabel('Longitude (deg)', 'FontSize', 16)
        ylabel('Latitude (deg)', 'FontSize', 16)        
        %... Save figure in pdf format
        printFigure(runPath)
    end

%% Fig19, Plot surface velocity ratio, u'/U, for precipitation state #2
    function Fig19
        figure(19)
        %... Velocity ratio, u'U, at zBar = 0
        uRatioGrid = uPrime( ... 
            0, x, y, hGrid, U_2, azimuth_2, NM_2, fC, hRho_2, true)/U_2;
        pcolor(lon, lat, uRatioGrid)
        hold on
        shading interp
        %... Set up and scale colormap
        cMap = coolwarm;
        cMap = cmapscale(uRatioGrid(isMap), cMap, 0.5, 0);
        colormap(cMap);
        cMin = min(uRatioGrid(isMap), [], 'all');
        cMax = max(uRatioGrid(isMap), [], 'all');
        clim([cMin, cMax]);                
        %... Plot representative contours
        plot(contours(:,1), contours(:,2), '-k');
        %... Plot continental divide data, if present
        if ~isempty(contDivideLon)
            plot(contDivideLon, contDivideLat, ...
                'Color', [0.5 0.5 0.5], 'LineWidth', 5);
        end        
        %... Plot section origin
        plot(sectionLon0, sectionLat0, 'sk', 'LineWidth', 2, 'MarkerSize', 18)
        %... Plot representative contours
        plot(contours(:,1), contours(:,2), '-k');
        %... Format plot
        daspect(dataAspectVector);
        axis(mapLimits);
        box on
        hA = gca;
        hA.TickDir = 'Out';
        hA.Layer = 'top';
        hA.FontSize = 18;
        hA.LineWidth = 1;
        %... Create colorbar
        hCB = colorbar;
        hCB.Label.String = 'u''/U';
        hCB.Label.FontSize = 18;
        hCB.Label.LineWidth = 1;
        %... Write labels
        hT = title('Fig. 19. Surface velocity ratio, state #2', 'FontSize', 16);
        hT.Units = 'normalized'; 
        hT.Position(2) = hT.Position(2) + 0.02;
        xlabel('Longitude (deg)', 'FontSize', 16)
        ylabel('Latitude (deg)', 'FontSize', 16)        
        %... Save figure in pdf format
        printFigure(runPath)
    end

%% Fig20, Plot outliers, defined as stdResiduals>3
    function Fig20
        figure(20)
        %... Plot representative contours
        hold on
        plot(contours(:,1), contours(:,2), '-k');
        %... Plot continental divide data, if present
        if ~isempty(contDivideLon)
            plot(contDivideLon, contDivideLat, ...
                'Color', [0.5 0.5 0.5], 'LineWidth', 5);
        end        
        %... Plot sample locations
        plot(sampleLon, sampleLat, '.b');
        %... Show stdResiduals for locations with values > 3
        for k=1:nSamples
            if stdResiduals(k)>3
                text(sampleLon(k), sampleLat(k), ...
                    num2str(stdResiduals(k),'%4.1f'), 'FontSize', 16);
            end
        end
        %... Plot section origin
        plot(sectionLon0, sectionLat0, 'sk', 'LineWidth', 2, 'MarkerSize', 18)
        %... Format plot
        daspect(dataAspectVector);
        axis(mapLimits);
        box on
        hA = gca;
        hA.TickDir = 'Out';
        hA.Layer = 'top';
        hA.FontSize = 16;
        hA.LineWidth = 1;
        %... Write labels
        hT = title('Fig. 20. Location of outliers', 'FontSize', 16);
        hT.Units = 'normalized'; 
        hT.Position(2) = hT.Position(2) + 0.02;
        xlabel('Longitude (deg)', 'FontSize', 16)
        ylabel('Latitude (deg)', 'FontSize', 16)
        %... Save figure in pdf format
        printFigure(runPath)
    end

 end