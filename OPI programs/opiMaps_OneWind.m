function opiMaps_OneWind
% opiMaps_OneWind Creates map figures of OPI results as given by a
% run file and an associated mat file with a "one-wind" solution.

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

%% Load results from mat file
%... Mat file content for one-wind solution
% ([runPath, '/', progName, '_Results.mat'], ...
%     'startTimeOpiCalc', 'runPath', 'runFile', 'runTitle', ...
%     'dataPath', 'topoFile', 'rTukey', 'sampleFile', 'contDivideFile', ...
%     'lon', 'lat', 'x', 'y', 'lon0', 'lat0', ...
%     'sampleLine', 'sampleLon', 'sampleLat', 'sampleX', 'sampleY', ...
%     'sampleD2H', 'sampleD18O', 'sampleDExcess', 'sampleLC', ...
%     'sampleLineAlt', 'sampleLonAlt', 'sampleLatAlt', 'sampleXAlt', 'sampleYAlt', ...
%     'sampleD2HAlt', 'sampleD18OAlt', 'sampleDExcessAlt', 'sampleLCAlt', ...
%     'bMWLSample', 'ijCatch', 'ptrCatch','ijCatchAlt', 'ptrCatchAlt', ...
%     'hR', 'sdResRatio', 'sdDataMin', 'sdDataMax', 'cov', 'fC', ...
%     'lB', 'uB', 'nParametersFree', ...
%     'beta', 'chiR2', 'nu', 'stdResiduals', ...
%     'zBar', 'T', 'gammaEnv', 'gammaSat', 'gammaRatio', ...
%     'rhoS0', 'hS', 'rho0', 'hRho', ...
%     'd18O0', 'dD18O0_dLat', 'tauF', 'pGrid', 'fMGrid', 'rHGrid', ...
%     'evapD2HGrid', 'uEvapD2HGrid', 'evapD18OGrid', 'uEvapD18OGrid', ...
%     'd2HGrid', 'd18OGrid', ...
%     'iWet', 'd2HPred', 'd18OPred', 'catchArea', ...
%     'iWetAlt', 'd2HPredAlt', 'd18OPredAlt', 'catchAreaAlt', ...
%     'liftMaxPred', 'elevationPred', 'subsidencePred', ...
%     'liftMaxPredAlt', 'elevationPredAlt', 'subsidencePredAlt');
load([matPathResults, '/', matFile_Results], 'beta')
if length(beta)==19, error('Mat file is for a two-wind solution.'), end
load([matPathResults, '/', matFile_Results], ...
    'runPath', 'runFile', 'runTitle', ...
    'dataPath', 'topoFile', 'rTukey', 'sampleFile', 'contDivideFile', ...
    'mapLimits', 'sectionLon0', 'sectionLat0', ...    
    'lon', 'lat', 'x', 'y', 'lon0', 'lat0', ...
    'sampleLon', 'sampleLat', ...
    'sampleD2H', 'sampleD18O', 'sampleLC', ...
    'sampleLonAlt', 'sampleLatAlt', 'sampleDExcessAlt', ...
    'ijCatch', 'ptrCatch', 'ijCatchAlt', 'ptrCatchAlt', ...
    'hR', 'sdResRatio', 'fC', ...
    'lB', 'uB', ...
    'stdResiduals', ...
    'zBar', 'T', 'gammaEnv', 'gammaSat', 'gammaRatio', ...
    'rhoS0', 'hS', 'rho0', 'hRho', ...
    'd18O0', 'dD18O0_dLat', 'tauF', 'pGrid', 'fMGrid', 'rHGrid', ...
    'd2HGrid', 'd18OGrid');

%... Read topographic data
[lon, lat, hGrid] = gridRead([dataPath, '/', topoFile]);
if any(isnan(hGrid(:))), error('Elevation grid contains nans.'), end
... Set elevations below sea level to 0 meters
hGrid(hGrid<0) = 0;

% Create 2D cosine-taper window, with
% fractional width = rTukey/2 on each margin of the grid.
[nY, nX] = size(hGrid);
window = tukeywin(nY, rTukey)*tukeywin(nX, rTukey)';
hGrid = window.*hGrid;
clear window

%... Unpack the solution vector beta
U = beta(1);           % wind speed (m/s)
azimuth = beta(2);     % azimuth (degrees)
T0 = beta(3);          % sea-level temperature (K)
M = beta(4);           % mountain-height number (dimensionless)
kappa = beta(5);       % eddy diffusion (m/s^2)
tauC = beta(6);        % condensation time (s)
d2H0 = beta(7);        % d2H of base precipitation (per unit)
dD2H0_dLat = beta(8);   % latitudinal gradient of base-prec d2H (1/deg lat)
fP0 = beta(9);         % residual precipitation after evaporation (fraction) 
%... Convert mountain-height number to buoyancy frequency (rad/s)
hMax = max(hGrid, [], 'all');
NM = M*U/hMax;

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
if isempty(sampleFile)
    fprintf('\n---------------- SYNTHETIC SAMPLES ------------------\n')
    fprintf([ ...
    'Data shown here is synthetic, and is intended for experimentation.\n', ...
    'There are no observed data, so the output has a reduced set of\n', ...
    'figures: 1, 4, 5, and 7 to 12.\n'])
end
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
fprintf('User-defined map limits, longitude: %.5f, %.5f degrees\n', ...
    mapLimits(1), mapLimits(2));
fprintf('User-defined map limits, latitude: %.5f, %.5f degrees\n', ...
    mapLimits(3), mapLimits(4));
if ~isempty(contDivideFile)
    fprintf('Continental-divide file: %s\n', contDivideFile);
else
    fprintf('Continental-divide file: none\n');
end
fprintf('Lon, lat for map origin: %.5f, %.5f degrees\n', lon0, lat0);
if lon0==mean(sampleLon) && lat0==mean(sampleLat)
    fprintf('Map origin is set to sample centroid.\n')
else
    fprintf('Map origin is set to center of the topographic grid.\n')
end
fprintf('Size of cosine window as fraction of grid size: %g\n', rTukey)
fprintf('Coriolis frequency at map-origin latitude: %.5f mrad/s\n', ...
    fC*1e3)
fprintf('Lon, lat for section origin: %.5f, %.5f degrees\n', sectionLon0, sectionLat0);
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
    fprintf('Minimum and maximum for longitude: %.5f, %.5f degrees\n', ...
        min(sampleLon), max(sampleLon))
    fprintf('Minimum and maximum for latitude: %.5f, %.5f degrees\n', ...
        min(sampleLat), max(sampleLat))
end
fprintf('\n---------------------- Constants --------------------\n')
fprintf('Characteristic distance for isotopic exchange: %.0f m\n', hR)
fprintf('Standard-deviation ratio for data residuals: %.2f (dimensionless)\n', sdResRatio)
fprintf('\n----------------- Evaporation Option ----------------\n')
if lB(9)==1 && uB(9)==1
    fprintf('No Evaporative recycling active for precipitation state.\n')
else
    fprintf('Evaporative recycling active for precipitation state.\n')
end
fprintf('\n---------------------- Solution ---------------------\n')
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
diary off

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
lonArrowOffset = arrowSize*(mapLimits(2) - mapLimits(1))*sind(azimuth);
latArrowOffset = arrowSize*(mapLimits(2) - mapLimits(1))*cosd(azimuth) ...
    *dataAspectVector(2)/dataAspectVector(1);
switch true
    case azimuth >=0 & azimuth <= 90
        % Place in SW corner
        lonArrowStart = mapLimits(1) + 0.04*((mapLimits(2) - mapLimits(1)));
        latArrowStart = mapLimits(3) + 0.04*((mapLimits(4) - mapLimits(3)));
    case azimuth > 90 & azimuth <= 180
        % Place in NW corner
        lonArrowStart = mapLimits(1) + 0.04*((mapLimits(2) - mapLimits(1)));
        latArrowStart = mapLimits(4) - 0.04*((mapLimits(4) - mapLimits(3)));
    case azimuth > 180 & azimuth <= 270
        % Place in NE corner
        lonArrowStart = mapLimits(2) - 0.04*((mapLimits(2) - mapLimits(1)));
        latArrowStart = mapLimits(4) - 0.04*((mapLimits(4) - mapLimits(3)));
    case azimuth > 270 & azimuth <= 360
        % Place in SE corner
        lonArrowStart = mapLimits(2) - 0.04*((mapLimits(2) - mapLimits(1)));
        latArrowStart = mapLimits(3) + 0.04*((mapLimits(4) - mapLimits(3)));
end

%... Create coastline for map figures
% Set minimum number of points in each contour line (to reduce clutter)
nContourPointsThreshold = 100;
% Set to 0 m for coastline contour
contourValues = [0 0];
% Row vector contourValues contains specified contours
contours = transpose(contourc(lon, lat, hGrid, contourValues));
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
% Construct path orthogonal to wind path and through the 
% section origin: sectionLon0, sectionLat0
[xSection0, ySection0] = lonlat2xy(sectionLon0, sectionLat0, lon0, lat0);
[xPath, yPath, sPath] = windPath(xSection0, ySection0, ...
    wrapTo360(azimuth + 90), x, y);
% Find points on orthogonal path
sPt = interp1(sPath, linspace(1, length(sPath), nStreamlines+2));
sPt = sPt(2:end-1);
xPt = interp1(sPath, xPath, sPt);
yPt = interp1(sPath, yPath, sPt);
% Calculate streamlines
xLMap = cell(nStreamlines,1);
yLMap = cell(nStreamlines,1);
zLMap = cell(nStreamlines,1);
for j = 1:nStreamlines
    % Construct path using current x,y reference point
    [xL0, yL0] = windPath(xPt(j), yPt(j), azimuth, x, y);
    % Break if empty (indicating a corner, with a single intersection)
    if isempty(xL0), break, end
    % Calculate streamline
    [xLMap{j}, yLMap{j}, zLMap{j}] = ...
        streamline(xL0(1), yL0(1), zL0Map, ...
        x, y, hGrid, U, azimuth, NM, fC, hRho, (j==1));
    % Add nans to indicate end of streamline (for plotting)
    xLMap{j}(end+1) = nan;
    yLMap{j}(end+1) = nan;
    zLMap{j}(end+1) = nan;
end
% Convert cell arrays to vectors
xLMap = cell2mat(xLMap);
yLMap = cell2mat(yLMap);
zLMap = cell2mat(zLMap);
% Convert to geographic coordinates
[lonLMap, latLMap] = xy2lonlat(xLMap, yLMap, lon0, lat0);

%% Calculate features for section plots
% Section path passing through xSection0, ySection0, and
% converted into xPath, YPath, sPath, lonPath, and latPath
[xPath, yPath, sPath, sLimits] = ...
    windPath(xSection0, ySection0, azimuth, x, y, xyLimits);
[lonPath, latPath] = xy2lonlat(xPath, yPath, lon0, lat0);

% Land surface elevation along wind path
hLPath = interp2(x, y, hGrid, xPath, yPath, 'linear', 0);
% Set maximum height of section
hLMax = max(hLPath) + 2*hS;

% Calculate cloud-water density, and 248 and 268 K isotherms
[zRhoC, rhoC, zCMean, z248Path, z268Path] = ...
    cloudWater(xPath, yPath, hLMax, ...
    x, y, hGrid, U, azimuth, NM, fC, kappa, tauC, tauF, hRho, ...
    zBar, T, gammaEnv, gammaSat, gammaRatio, rhoS0, hS);

% Calculate set of representative precipitation fall lines for section
nPrecLines = 20;
dPrecOffset = hLMax*U*tauF/hS;
sPrecLines = linspace(sPath(1) + dPrecOffset/2, ...
    sPath(end) - dPrecOffset/2, nPrecLines);
sPrecLines = [sPrecLines + dPrecOffset; sPrecLines];
zPrecLines = [zeros(1,nPrecLines); hLMax*ones(1,nPrecLines)];

% Calculate streamlines for section (1 km spacing in elevation)
nStreamlines = floor(hLMax/1000);
zS0Section = 1e3*(1:nStreamlines);
sLSection = cell(nStreamlines,1);
zLSection = cell(nStreamlines,1);
for j = 1:nStreamlines
    [~, ~, zLSection{j}, sLSection{j}] = ...
        streamline(xPath(1), yPath(1), zS0Section(j), ...
        x, y, hGrid, U, azimuth, NM, fC, hRho, (j==1));
    sLSection{j}(end+1) = nan;
    zLSection{j}(end+1) = nan;
end
sLSection = cell2mat(sLSection);
zLSection = cell2mat(zLSection);

%... Precipitation rate along section
F = griddedInterpolant({y, x}, pGrid, 'linear', 'linear');
pSection = F(yPath, xPath);

%... Precipitation d2H and d2H0 along section
F = griddedInterpolant({y, x}, d2HGrid, 'linear', 'linear');
d2HSection = F(yPath, xPath);
d2H0Section = d2H0 + dD2H0_dLat.*(abs(latPath) - abs(lat0));
clear F

%% Additional results
diary on
fprintf('\n------------ Streamlines, and Cloud Water ------------\n')
fprintf('Vertical exaggeration for streamline figure: %g (dimensionless)\n', VE);
fprintf('Starting elevation for streamlines: %.0f m\n', zL0Map);
fprintf('Mean height of cloud water relative to land surface: %.0f m\n', zCMean);
diary off

%% Plot Figures
Fig01 % Plot topography and sample locations
Fig02 % Scatter plot showing d2H of primary samples
Fig03 % Scatter plot showing dExcess of altered samples
Fig04 % Plot precipitation rate
Fig05 % Plot moisture ratio
Fig06 % Plot surface relative humidity
Fig07 % Plot streamlines
Fig08 % Plot cross section
Fig09 % Plot predicted delta2H fractionation
Fig10 % Plot predicted delta18O fractionation
Fig11 % Plot surface temperature
Fig12 % Plot surface velocity ratio, u'/U
Fig13 % Plot sample locations, with outliers labeled (stdResiduals > 3)

%% Report compute time
diary on
fprintf('\nCompute time: %.2f minutes\n', ...
    minutes(datetime - startTime));
diary off

%% Fig01, Plot topography and sample locations
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
            % Catchments for primary samples
            for k=1:nSamples
                isC = false(size(hGrid));
                % Extract indices for sample catchment
                ijC = catchmentIndices(k, ijCatch, ptrCatch);
                isC(ijC) = true;
                [b, ~, nObjects] = bwboundaries(isC);
                b = cell2mat(b);
                if nObjects > 1, error('Catchment has more than one set of pixels'), end
                plot(lon(b(:,2)), lat(b(:,1)), '-r', 'LineWidth', 1)
            end
            % Catchments for altered samples
            for k=1:nSamplesAlt
                isC = false(size(hGrid));
                % Extract indices for sample catchment
                ijC = catchmentIndices(k, ijCatchAlt, ptrCatchAlt);
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
            % Locations for primary samples
            plot(sampleLon, sampleLat, ...
                'ok', 'MarkerSize', 12, 'LineWidth', 1);
            % Locations for altered samples
            plot(sampleLonAlt, sampleLatAlt, ...
                'ok', 'MarkerSize', 12, 'LineWidth', 1);
        end
        %... Plot path and origin for section
        plot(lonPath([1,end]), latPath([1,end]), '-k', 'LineWidth', 2);
        plot(sectionLon0, sectionLat0, 'sw', 'LineWidth', 5, 'MarkerSize', 18)
        plot(sectionLon0, sectionLat0, 'sk', 'LineWidth', 2, 'MarkerSize', 18)
        %... Draw arrow for wind direction
        quiver(lonArrowStart, latArrowStart, ...
            lonArrowOffset, latArrowOffset, ...
            'MaxHeadSize', 2, 'LineWidth', 3, 'Color', 'k');
        %... Format plot
        daspect(dataAspectVector);
        axis(mapLimits);
        box on
        h = gca;
        h.TickDir = 'Out';
        h.Layer = 'top';
        h.FontSize = 16;
        h.LineWidth = 1;
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

%% Fig02, Map showing d2H for primary samples
    function Fig02
        if isempty(sampleFile), return, end        
        figure(2)
        %... Define land areas using white patches
        iStart = 2;
        iEnd = find(isnan(contours(:,1))) - 1;
        iEnd = [iEnd(2:end); size(contours,1)];
        for k = 1:length(iEnd)
            patch(contours(iStart:iEnd(k),1), contours(iStart:iEnd(k),2), ...
                [1 1 1])
            iStart = iEnd(k) + 2;
        end        
        %... Plot coast line
        hold on
        plot(contours(:,1), contours(:,2), '-k');        
        %... Plot sample catchments
        if ~isempty(sampleFile)
            for k=1:nSamples
                isC = false(size(hGrid));
                % Extract indices for sample catchment
                ijC = catchmentIndices(k, ijCatch, ptrCatch);
                isC(ijC) = true;
                [b, ~, nObjects] = bwboundaries(isC);
                b = cell2mat(b);
                if nObjects > 1, error('Catchment has more than one set of pixels'), end
                plot(lon(b(:,2)), lat(b(:,1)), '-r', 'LineWidth', 1)
            end
        end
        %... Plot primary samples
        if ~isempty(sampleFile)
            iOrder = randperm(nSamples);
            scatter(sampleLon(iOrder), sampleLat(iOrder), ...
                250, sampleD2H(iOrder)*1e3, 'MarkerEdgeColor', 'k', ...
                'MarkerFaceColor', 'flat', 'LineWidth', 0.5)
            cMap = parula;
            cMap = cmapscale(sampleD2H*1e3, cMap, 0.7);
            colormap(cMap);
        end
        %... Plot continental divide data, if present
        if ~isempty(contDivideLon)
            plot(contDivideLon, contDivideLat, ...
                'Color', [0.5 0.5 0.5], 'LineWidth', 5);
        end        
        %... Plot path and origin for section
        plot(lonPath([1,end]), latPath([1,end]), '-k', 'LineWidth', 2);
        plot(sectionLon0, sectionLat0, 'sw', 'LineWidth', 5, 'MarkerSize', 18)
        plot(sectionLon0, sectionLat0, 'sk', 'LineWidth', 2, 'MarkerSize', 18)
        %... Draw arrow for wind direction
        quiver(lonArrowStart, latArrowStart, ...
            lonArrowOffset, latArrowOffset, ...
            'MaxHeadSize', 2, 'LineWidth', 3, 'Color', 'k');
        %... Format plot
        daspect(dataAspectVector);
        axis(mapLimits);
        box on
        h = gca;
        h.TickDir = 'Out';
        h.Layer = 'top';
        h.FontSize = 16;
        h.LineWidth = 1;
        h.Color = [0.7 0.7 0.7];
        %... Create colorbar
        hCB = colorbar;
        hCB.Label.String = ['\delta^{2}H (', char(8240), ')'];
        hCB.Label.FontSize = 16;
        hCB.Label.LineWidth = 1;
        %... Write labels
        hT = title(['Fig. 2. Observed ', '\delta^{2}H ', ... 
            'for primary samples'], 'FontSize', 16);
        hT.Units = 'normalized';
        hT.Position(2) = hT.Position(2) + 0.02;
        xlabel('Longitude (deg)', 'FontSize', 16)
        ylabel('Latitude (deg)', 'FontSize', 16)
        %... Save figure in pdf format
        printFigure(runPath)
    end

%% Fig03, Map showing dExcess for altered samples
    function Fig03
        if nSamplesAlt==0, return, end
        figure(3)
        %... Define land areas using white patches
        iStart = 2;
        iEnd = find(isnan(contours(:,1))) - 1;
        iEnd = [iEnd(2:end); size(contours,1)];
        for k = 1:length(iEnd)
            patch(contours(iStart:iEnd(k),1), contours(iStart:iEnd(k),2), ...
                [1 1 1])
            iStart = iEnd(k) + 2;
        end
        %... Plot coast line
        hold on
        plot(contours(:,1), contours(:,2), '-k');
        %... Plot sample catchments
        if ~isempty(sampleFile)
            for k=1:nSamplesAlt
                isC = false(size(hGrid));
                % Extract indices for sample catchment
                ijC = catchmentIndices(k, ijCatchAlt, ptrCatchAlt);
                isC(ijC) = true;
                [b, ~, nObjects] = bwboundaries(isC);
                b = cell2mat(b);
                if nObjects > 1, error('Catchment has more than one set of pixels'), end
                plot(lon(b(:,2)), lat(b(:,1)), '-r', 'LineWidth', 1)
            end
        end
        %... Plot altered samples
        if ~isempty(sampleFile)
        iOrder = randperm(nSamplesAlt);            
        scatter(sampleLonAlt(iOrder), sampleLatAlt(iOrder), ...
            250, sampleDExcessAlt(iOrder)*1e3, 'MarkerEdgeColor', 'k', ...
            'MarkerFaceColor', 'flat', 'LineWidth', 0.5)
             cMap = parula;
             cMap = cmapscale(sampleD18O*1e3, cMap, 0.7);
             colormap(cMap);
        end        
        %... Plot continental divide data, if present
        if ~isempty(contDivideLon)
            plot(contDivideLon, contDivideLat, ...
                'Color', [0.5 0.5 0.5], 'LineWidth', 5);
        end
        %... Plot path and origin for section
        plot(lonPath([1,end]), latPath([1,end]), '-k', 'LineWidth', 2);
        plot(sectionLon0, sectionLat0, 'sw', 'LineWidth', 5, 'MarkerSize', 18)
        plot(sectionLon0, sectionLat0, 'sk', 'LineWidth', 2, 'MarkerSize', 18)
        %... Draw arrow for wind direction
        quiver(lonArrowStart, latArrowStart, ...
            lonArrowOffset, latArrowOffset, ...
            'MaxHeadSize', 2, 'LineWidth', 3, 'Color', 'k');
        %... Format plot
        daspect(dataAspectVector);
        axis(mapLimits);
        box on
        h = gca;
        h.TickDir = 'Out';
        h.Layer = 'top';
        h.FontSize = 16;
        h.LineWidth = 1;
        h.CLim(2) = 5;
        h.Color = [0.7 0.7 0.7];
        %... Create colorbar
        hCB = colorbar;
        hCB.Label.String = ['\it{d} (', char(8240), ')'];
        hCB.Label.FontSize = 16;
        hCB.Label.LineWidth = 1;
        %... Write labels
        hT = title('Fig. 3. Deutrium-excess for altered samples', 'FontSize', 16);
        hT.Units = 'normalized';
        hT.Position(2) = hT.Position(2) + 0.02;
        xlabel('Longitude (deg)', 'FontSize', 16)
        % ylabel('Latitude (deg)', 'FontSize', 16)
        h.YTickLabel = {};
        %... Save figure in pdf format
        printFigure(runPath)
    end

%% Fig04, Plot precipitation rate (mm/h)
    function Fig04
        figure(4)
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
        %... Plot path and origin for section
        plot(lonPath([1,end]), latPath([1,end]), '-k', 'LineWidth', 2);
        plot(sectionLon0, sectionLat0, 'sk', 'LineWidth', 2, 'MarkerSize', 18)
        %... Draw arrow for wind direction
        quiver(lonArrowStart, latArrowStart, ...
            lonArrowOffset, latArrowOffset, ...
            'MaxHeadSize', 2, 'LineWidth', 3, 'Color', 'k');
        %... Plot representative contours
        plot(contours(:,1), contours(:,2), '-k');
        %... Format plot
        daspect(dataAspectVector);
        axis(mapLimits);
        box on
        h = gca;
        h.TickDir = 'Out';
        h.Layer = 'top';
        h.FontSize = 16;
        h.LineWidth = 1;
        %... Create colorbar
        hCB = colorbar;
        hCB.Label.String = 'Precipitation Rate (mm/hr)';
        hCB.Label.FontSize = 16;
        hCB.Label.LineWidth = 1;
        %... Write labels
        hT = title('Fig. 4. Precipitation rate', 'FontSize', 16);
        hT.Units = 'normalized';
        hT.Position(2) = hT.Position(2) + 0.02;
        xlabel('Longitude (deg)', 'FontSize', 16)
        ylabel('Latitude (deg)', 'FontSize', 16)
        %... Save figure in pdf format
        printFigure(runPath)
    end

%% Fig05, Plot moisture ratio (dimensionless)
    function Fig05
        figure(5)
        pcolor(lon, lat, fMGrid)
        hold on
        shading interp
        %... Set up and scale colormap
        cMap = parula(320);
        cMap = cMap(320:-1:1,:);
        cMap = cMap(1:256,:);
        cMap = cmapscale(fMGrid(isMap), cMap, 0.65);
        colormap(cMap);
        %... Plot continental divide data, if present
        if ~isempty(contDivideLon)
            plot(contDivideLon, contDivideLat, ...
                'Color', [0.5 0.5 0.5], 'LineWidth', 5);
        end
        %... Plot path and origin for section
        plot(lonPath([1,end]), latPath([1,end]), '-k', 'LineWidth', 2);
        plot(sectionLon0, sectionLat0, 'sk', 'LineWidth', 2, 'MarkerSize', 18)
        %... Draw arrow for wind direction
        quiver(lonArrowStart, latArrowStart, ...
            lonArrowOffset, latArrowOffset, ...
            'MaxHeadSize', 2, 'LineWidth', 3, 'Color', 'k');
        %... Plot representative contours
        plot(contours(:,1), contours(:,2), '-k');
        %... Format plot
        daspect(dataAspectVector);
        axis(mapLimits);
        box on
        h = gca;
        h.TickDir = 'Out';
        h.Layer = 'top';
        h.FontSize = 16;
        h.LineWidth = 1;
        %... Create colorbar
        hCB = colorbar;
        hCB.Label.String = 'Moisture Ratio';
        hCB.Label.FontSize = 16;
        hCB.Label.LineWidth = 1;
        %... Write labels
        hT = title('Fig. 5. Moisture ratio', 'FontSize', 16);
        hT.Units = 'normalized';
        hT.Position(2) = hT.Position(2) + 0.02;
        xlabel('Longitude (deg)', 'FontSize', 16)
        ylabel('Latitude (deg)', 'FontSize', 16)
        %... Save figure in pdf format
        printFigure(runPath)
    end

%% Fig06, Plot surface relative humidity (dimensionless)
    function Fig06
        if numel(rHGrid)==1, return, end        
        figure(6)
        pcolor(lon, lat, rHGrid)
        hold on
        shading interp
        %... Set up and scale colormap
        cMap = parula(320);
        cMap = cMap(320:-1:1,:);
        cMap = cMap(1:256,:);
        cMap = cmapscale(rHGrid(isMap), cMap, 0.8);
        colormap(cMap);
        %... Plot continental divide data, if present
        if ~isempty(contDivideLon)
            plot(contDivideLon, contDivideLat, ...
                'Color', [0.5 0.5 0.5], 'LineWidth', 5);
        end
        %... Plot path and origin for section
        plot(lonPath([1,end]), latPath([1,end]), '-k', 'LineWidth', 2);
        plot(sectionLon0, sectionLat0, 'sk', 'LineWidth', 2, 'MarkerSize', 18)
        %... Draw arrow for wind direction
        quiver(lonArrowStart, latArrowStart, ...
            lonArrowOffset, latArrowOffset, ...
            'MaxHeadSize', 2, 'LineWidth', 3, 'Color', 'k');
        %... Plot representative contours
        plot(contours(:,1), contours(:,2), '-k');
        %... Format plot
        daspect(dataAspectVector);
        axis(mapLimits);
        box on
        h = gca;
        h.TickDir = 'Out';
        h.Layer = 'top';
        h.FontSize = 16;
        h.LineWidth = 1;
        %... Create colorbar
        hCB = colorbar;
        hCB.Label.String = 'Surface Relative Humidity';
        hCB.Label.FontSize = 16;
        hCB.Label.LineWidth = 1;
        %... Write labels
        hT = title('Fig. 6. Surface relative humidity', 'FontSize', 16);
        hT.Units = 'normalized';
        hT.Position(2) = hT.Position(2) + 0.02;
        xlabel('Longitude (deg)', 'FontSize', 16)
        ylabel('Latitude (deg)', 'FontSize', 16)
        %... Save figure in pdf format
        printFigure(runPath)
    end

%% Fig07, Plot streamlines
    function Fig07
        figure(7)
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
            'sk', 'LineWidth', 2, 'MarkerSize', 18)
        %... Plot streamlines
        plot3(lonLMap, latLMap, zLMap*1e-3, '-r');
        %... Draw arrow for wind direction
        quiver3(lonArrowStart, latArrowStart, ...
            max(zLMap,[],'all')*1e-3, ...
            lonArrowOffset, latArrowOffset, 0, ...
            'MaxHeadSize', 2, 'LineWidth', 3, 'Color', 'k');
        %... View at 60 degree from wind path from the south, and
        % looking down at 45 degrees.
        viewAz = (60 - azimuth);
        viewAz = viewAz - 180*floor((viewAz - -90)/180);
        view([viewAz, 45])
        %... Format plot
        daspect(dataAspectVector);
        axis(mapLimits);
        axis manual
        hA = gca;
        hA.ZTick = [0 hA.ZLim(2)];
        hA.ZTickLabels = [0 round(hA.ZLim(2),1)];
        hA.Box = 'on';
        hA.BoxStyle = 'full';        
        hA.FontSize = 16;
        hA.LineWidth = 1;
        hA.ZLim(1) = min(hGrid(:));
        %... Write labels
        str = sprintf('Fig. 7. Streamlines starting at %.0f m elevation', zL0Map);
        hT = title(str, 'FontSize', 16);
        hT.Units = 'normalized';
        hT.Position(2) = hT.Position(2) + 0.02;
        xlabel('Longitude (deg)', 'FontSize', 16)
        ylabel('Latitude (deg)', 'FontSize', 16)
        zlabel('z (km)')
        %... Save figure in pdf format
        printFigure(runPath)
    end

%% Fig08, Plot cross section
    function Fig08
        figure(8)
        %.... Generate color plot for cloud-water density
        hA1 = subplot(3,1,1);
        pcolor(sPath*1e-3, zRhoC*1e-3, rhoC);
        shading flat
        %... Calculate logical grid indicating nodes inside limits
        % for section and lying above the topography
        [S, Z] = meshgrid(sPath, zRhoC);
        isSection = sLimits(1)<=S & S<=sLimits(2) ...
            & 0<=zRhoC & zRhoC<=hLMax ...
            & ~inpolygon(S, Z, ...
            [sPath(1); sPath; sPath(end)], [0; hLPath;  0]);
        %... Colormap is generated with the first row set to light blue,
        % and the remaining rows set to a gray scale from 0.8 to 0.
        colormap([0.686, 1, 1; repmat(linspace(0.8, 0, 100)', 1, 3)]);
        %... Set climits to the range of positive values. The blue color
        % in the first row represents negative values for rhoC.
        cMax = max(rhoC(isSection));
        clim([0, cMax])
        hold on
        %... Plot freezing transition (WBF zone)
        plot(sPath*1e-3, z248Path*1e-3, '-b', 'LineWidth', 1);
        plot(sPath*1e-3, z268Path*1e-3, '-b', 'LineWidth', 1);
        %... Plot precipitation lines
        plot(sPrecLines*1e-3, zPrecLines*1e-3, ':k', 'LineWidth', 1);
        %... Plot topography
        fill([sPath(1); sPath; sPath(end)]*1e-3, [0; hLPath;  0]*1e-3, ...
            [0.82 0.70 0.55], 'EdgeColor', 'k', 'LineWidth', 0.5);
        %... Plot streamlines
        plot(sLSection*1e-3, zLSection*1e-3, '-k', 'LineWidth', 0.5)
        %... Adjust axis limits
        hA1.XLim = [sLimits(1), sLimits(2)]*1e-3;
        hA1.YLim = [0, hLMax]*1e-3;
        %... Format plot
        box on
        grid off
        hA1.XTickLabel = {};
        hA1.FontSize = 12;
        hA1.LineWidth = 1;
        hA1.Layer = 'top';
        %... Write labels        
        hT = title(hA1, 'Fig. 8. Cross section', 'FontSize', 12);
        hT.Units = 'normalized'; 
        hT.Position(2) = hT.Position(2) + 0.02;
        ylabel('Elevation (km)', 'FontSize', 10);  
        
        %... Precipitation rate along section
        hA2 = subplot(3,1,2);
        plot(sPath*1e-3, pSection*3.6e3, '-k', 'LineWidth', 2);
        %... Adjust axis limits
        hA2.XLim = [sLimits(1), sLimits(2)]*1e-3;
        hA2.YLim = [0, max(pSection, [], 'all')*1.05]*3.6e3;
        %... Format plot
        box on
        grid on
        hA2.XTickLabel = {};
        hA2.FontSize = 12;
        hA2.LineWidth = 1;
        ylabel({'Precipitation'; 'Rate (mm/hr)'}, 'FontSize', 12);
        
        %... Precipitation d2H and d2H0 along section
        hA3 = subplot(3,1,3);
        hold on
        plot(sPath*1e-3, d2H0Section*1e3, '--k', 'LineWidth', 1);
        plot(sPath*1e-3, d2HSection*1e3, '-k', 'LineWidth', 2);
        %... Adjust axis limits
        hA3.XLim = [sLimits(1), sLimits(2)]*1e-3;
        hA3.YLim = [min(d2HSection(:)) - 10e-3, ...
            max([d2HSection(:); 10e-3])]*1e3;
        %... Format plot
        box on
        grid on
        hA3.FontSize = 12;
        hA3.LineWidth = 1;
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
        hTA.String  = sprintf('%.0f° ', azimuth);
        hTA.FontSize = 12;

        % Adjust position relative to position and size of middle plot
        hA3.Position(1) = hA2.Position(1);
        hA3.Position(2) = hA2.Position(2) - hA3.Position(4) - 0.1;
        hA3.Position(3) = hA2.Position(3);
        hA3.Position(4) = hA2.Position(4);
        
        %... Save figure in pdf format
        printFigure(runPath)
    end

%% Fig09, Plot predicted precipitation d2H
    function Fig09
        figure(9)
        % Next line used to "gray out" regions with no precipitation.
        % After pcolor, set background color to gray: h.Color = 0.5*ones(1,3)
        % d2HGrid(pGrid==0) = nan;
        % Plot color map
        pcolor(lon, lat, d2HGrid*1e3);
        hold on
        shading interp
        %... Set up and scale colormap
        cMap = parula(256);
        cMap = cMap(end:-1:1, :);
        cMap = cmapscale(d2HGrid(isMap)*1e3, cMap, 0.3);
        colormap(cMap);
        cMin = min(d2HGrid(isMap), [],'all')*1e3;
        cMax = max(d2HGrid(isMap), [],'all')*1e3;
        clim([cMin, cMax]);
        %... Plot continental divide data, if present
        if ~isempty(contDivideLon)
            plot(contDivideLon, contDivideLat, ...
                'Color', [0.5 0.5 0.5], 'LineWidth', 5);
        end
        %... Plot path and origin for section
        plot(lonPath([1,end]), latPath([1,end]), '-k', 'LineWidth', 2);
        plot(sectionLon0, sectionLat0, 'sk', 'LineWidth', 2, 'MarkerSize', 18)
        %... Draw arrow for wind direction
        quiver(lonArrowStart, latArrowStart, ...
            lonArrowOffset, latArrowOffset, ...
            'MaxHeadSize', 2, 'LineWidth', 3, 'Color', 'k');
        %... Plot representative contours
        plot(contours(:,1), contours(:,2), '-k');
        %... Format plot
        daspect(dataAspectVector);
        axis(mapLimits);
        box on
        h = gca;
        h.Color = 0.5*ones(1,3);
        h.TickDir = 'Out';
        h.Layer = 'top';
        h.FontSize = 16;
        h.LineWidth = 1;
        %... Create colorbar
        hCB = colorbar;
        hCB.Label.String = ['\delta^{2}H (', char(8240), ')'];
        hCB.Label.FontSize = 16;
        hCB.Label.LineWidth = 1;
        %... Write labels
        hT = title(['Fig. 9. Predicted precipitation ', ...
            '\delta^{2}H'], 'FontSize', 16);
        hT.Units = 'normalized';
        hT.Position(2) = hT.Position(2) + 0.02;
        xlabel('Longitude (deg)', 'FontSize', 16)
        ylabel('Latitude (deg)', 'FontSize', 16)
        %... Save figure in pdf format
        printFigure(runPath)
    end

%% Fig10, Plot predicted precipitation d18O
    function Fig10
        figure(10)
        % Next line used to "gray out" regions with no precipitation.
        % After pcolor, set background color to gray: h.Color = 0.5*ones(1,3)
        % d18OGrid(pGrid==0) = nan;
        % Plot color map
        pcolor(lon, lat, d18OGrid*1e3);
        hold on
        shading interp
        %... Set up and scale colormap
        cMap = parula(256);
        cMap = cMap(end:-1:1, :);
        cMap = cmapscale(d18OGrid(isMap)*1e3, cMap, 0.3);
        colormap(cMap);
        cMin = min(d18OGrid(isMap), [],'all')*1e3;
        cMax = max(d18OGrid(isMap), [],'all')*1e3;
        clim([cMin, cMax]);
        %... Plot continental divide data, if present
        if ~isempty(contDivideLon)
            plot(contDivideLon, contDivideLat, ...
                'Color', [0.5 0.5 0.5], 'LineWidth', 5);
        end
        %... Plot path and origin for section
        plot(lonPath([1,end]), latPath([1,end]), '-k', 'LineWidth', 2);
        plot(sectionLon0, sectionLat0, 'sk', 'LineWidth', 2, 'MarkerSize', 18)
        %... Draw arrow for wind direction
        quiver(lonArrowStart, latArrowStart, ...
            lonArrowOffset, latArrowOffset, ...
            'MaxHeadSize', 2, 'LineWidth', 3, 'Color', 'k');
        %... Plot representative contours
        plot(contours(:,1), contours(:,2), '-k');
        %... Format plot
        daspect(dataAspectVector);
        axis(mapLimits);
        box on
        h = gca;
        h.Color = 0.5*ones(1,3);
        h.TickDir = 'Out';
        h.Layer = 'top';
        h.FontSize = 16;
        h.LineWidth = 1;
        %... Create colorbar
        hCB = colorbar;
        hCB.Label.String = ['\delta^{18}O (', char(8240), ')'];
        hCB.Label.FontSize = 16;
        hCB.Label.LineWidth = 1;
        %... Write labels
        hT = title(['Fig. 10. Predicted precipitation ', ...
            '\delta^{18}O'], ...
            'FontSize', 16);
        hT.Units = 'normalized';
        hT.Position(2) = hT.Position(2) + 0.02;
        xlabel('Longitude (deg)', 'FontSize', 16)
        ylabel('Latitude (deg)', 'FontSize', 16)
        %... Save figure in pdf format
        printFigure(runPath)
    end

%% Fig11, Plot surface-air temperature
    function Fig11
        figure(11)
        %... Mean air temperature at surface
        %... Construct grid for air temperature at land surface
        TGrid = T(1) - gammaSat(1)*hGrid;
        pcolor(lon, lat, TGrid-TC2K)
        hold on
        shading interp
        %... Set up and scale colormap
        cMap = coolwarm;
        cMap = cmapscale(TGrid(isMap)-TC2K, cMap, 0.5, 0);
        colormap(cMap);
        %... Plot representative contours
        plot(contours(:,1), contours(:,2), '-k');
        %... Plot continental divide data, if present
        if ~isempty(contDivideLon)
            plot(contDivideLon, contDivideLat, ...
                'Color', [0.5 0.5 0.5], 'LineWidth', 5);
        end
        %... Plot path and origin for section
        plot(lonPath([1,end]), latPath([1,end]), '-k', 'LineWidth', 2);
        plot(sectionLon0, sectionLat0, 'sk', 'LineWidth', 2, 'MarkerSize', 18)
        %... Plot representative contours
        plot(contours(:,1), contours(:,2), '-k');
        %... Format plot
        daspect(dataAspectVector);
        axis(mapLimits);
        box on
        h = gca;
        h.TickDir = 'Out';
        h.Layer = 'top';
        h.FontSize = 18;
        h.LineWidth = 1;
        %... Create colorbar
        hCB = colorbar;
        hCB.Label.String = 'Surface-Air Temperture (°C)';
        hCB.Label.FontSize = 18;
        hCB.Label.LineWidth = 1;
        %... Write labels
        hT = title('Fig. 11. Surface-air temperature', 'FontSize', 16);
        hT.Units = 'normalized';
        hT.Position(2) = hT.Position(2) + 0.02;
        xlabel('Longitude (deg)', 'FontSize', 16)
        ylabel('Latitude (deg)', 'FontSize', 16)
        %... Save figure in pdf format
        printFigure(runPath)
    end

%% Fig12, Plot surface velocity ratio, u'/U
    function Fig12
        figure(12)
        %... Velocity ratio, u'U, at zBar = 0
        uRatioGrid = uPrime( ...
            0, x, y, hGrid, U, azimuth, NM, fC, hRho, true)/U;
        pcolor(lon, lat, uRatioGrid)
        hold on
        shading interp
        %... Set up and scale colormap
        cMap = coolwarm;
        cMap = cmapscale(uRatioGrid(isMap), cMap, 0.5, 0);
        colormap(cMap);
        %... Plot representative contours
        plot(contours(:,1), contours(:,2), '-k');
        %... Plot continental divide data, if present
        if ~isempty(contDivideLon)
            plot(contDivideLon, contDivideLat, ...
                'Color', [0.5 0.5 0.5], 'LineWidth', 5);
        end
        %... Plot path and origin for section
        plot(lonPath([1,end]), latPath([1,end]), '-k', 'LineWidth', 2);
        plot(sectionLon0, sectionLat0, 'sk', 'LineWidth', 2, 'MarkerSize', 18)
        %... Plot representative contours
        plot(contours(:,1), contours(:,2), '-k');
        %... Format plot
        daspect(dataAspectVector);
        axis(mapLimits);
        box on
        h = gca;
        h.TickDir = 'Out';
        h.Layer = 'top';
        h.FontSize = 18;
        h.LineWidth = 1;
        %... Create colorbar
        hCB = colorbar;
        hCB.Label.String = 'Surface Velocity Ratio, u''/U';
        hCB.Label.FontSize = 18;
        hCB.Label.LineWidth = 1;
        %... Write labels
        hT = title('Fig. 12. Surface velocity ratio, u''/U', 'FontSize', 16);
        hT.Units = 'normalized';
        hT.Position(2) = hT.Position(2) + 0.02;
        xlabel('Longitude (deg)', 'FontSize', 16)
        ylabel('Latitude (deg)', 'FontSize', 16)
        %... Save figure in pdf format
        printFigure(runPath)
    end

%% Fig13, Plot outliers, defined as stdResiduals>3
    function Fig13
        if isempty(sampleFile), return, end
        figure(13)
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
        %... Plot path and origin for section
        plot(lonPath([1,end]), latPath([1,end]), '-k', 'LineWidth', 2);
        plot(sectionLon0, sectionLat0, 'sw', 'LineWidth', 5, 'MarkerSize', 18)
        plot(sectionLon0, sectionLat0, 'sk', 'LineWidth', 2, 'MarkerSize', 18)
        %... Format plot
        daspect(dataAspectVector);
        axis(mapLimits);
        box on
        h = gca;
        h.TickDir = 'Out';
        h.Layer = 'top';
        h.FontSize = 16;
        h.LineWidth = 1;
        %... Write labels
        hT = title('Figure 13. Location of outliers', 'FontSize', 16);
        hT.Units = 'normalized';
        hT.Position(2) = hT.Position(2) + 0.02;
        xlabel('Longitude (deg)', 'FontSize', 16)
        ylabel('Latitude (deg)', 'FontSize', 16)
        %... Save figure in pdf format
        printFigure(runPath)
    end

end