function opiPlots_OneWind
% opiPlots_OneWind takes as input a run file and an associated mat file
% with a "one-wind" solution, and creates plot figures of the results.
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
startTime = datetime;
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

%... Mat file content for one-wind solution
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
    'dataPath', 'x', 'y', 'topoFile', 'rTukey', 'sampleFile', ...
    'sectionLon0', 'sectionLat0', ...
    'lon', 'lat', 'lon0', 'lat0', ...
    'sampleLon', 'sampleLat', 'sampleX', 'sampleY', ...
    'sampleLonAlt', ...
    'sampleD2H', 'sampleD18O', 'sampleLC', ...
    'sampleD2HAlt', 'sampleD18OAlt', ...
    'bMWLSample', 'ijCatch', 'ptrCatch','ijCatchAlt', 'ptrCatchAlt', ...
    'hR', 'sdResRatio', 'sdDataMin', 'sdDataMax','cov', 'fC', ...
    'lB', 'uB', ...    
    'chiR2', 'nu', 'stdResiduals', ...
    'zBar', 'T', 'gammaEnv', 'gammaSat', 'gammaRatio', ...
    'rhoS0', 'hS', 'rho0', 'hRho', ...
    'd18O0', 'dD18O0_dLat', 'tauF', 'pGrid', 'rHGrid', ...
    'iWet', 'd2HPred', 'd18OPred', ...
    'd2HPredAlt', 'd18OPredAlt', ...    
    'liftMaxPred', 'elevationPred');

%... Test for samples
if isempty([sampleLon, sampleLat])
    error('Samples must be present for opiPlots_OneWind.')
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
hMax = max(hGrid, [], 'all');
NM = M*U/hMax;

%... Number of samples
nSamples = length(sampleLon);
nSamplesAlt = length(sampleLonAlt);

%... Sum number of precipitation-present samples
nSamplesWet = sum(iWet);

%... Calculate statistics for predicted water isotopes (uniform weighting)
[bMWLPred, sdPredMin, sdPredMax] = ...
    estimateMWL(d18OPred(iWet), d2HPred(iWet), sdResRatio);

%% Calculate slope along wind direction for sample catchments
if ~isempty(ijCatch) && ~isempty(ptrCatch)
    %... Transform topography to wind coordinates (s,t,z is right handed)
    [Sxy, Txy, s, t, Xst, Yst] = windGrid(x, y, azimuth);

    %... Offset topography upwind to account for average downwind 
    % transport during fallout of precipitation, as represented by tauF. 
    % The offset is accomplished by subtracting the offset from the 
    % grid coordinates Xst and Yst. 
    Xst = Xst - U*sind(azimuth)*tauF;
    Yst = Yst - U*cosd(azimuth)*tauF;
    % Note: griddedInterpolant uses the ndgrid format, so that the order
    % of the grid vectors, x and y, are reversed to account for the
    % meshgrid format for hGrid. Also note that grid vectors must be
    % specified as cell variables. The setting 'none' causes extrapolated
    % values to be set to nans, which are then found and set to zero.
    F = griddedInterpolant({y, x}, hGrid, 'linear', 'none');
    hWind = F(Yst, Xst);
    clear F
    hWind(isnan(hWind)) = 0;
    %... Calculate topographic slope in windward direction, which
    % is oriented down each column.
    dS = abs(s(2) - s(1));
    dT = abs(t(2) - t(1));
    [~, slopeWind] = gradient(hWind, dT, dS);
    clear hWind
    %... Transform slopeWind back to geographic coordinates
    F = griddedInterpolant({s, t}, slopeWind, 'linear', 'none');
    clear slopeWind
    slopeGrid = F(Sxy, Txy);

    %... Calculate wind-path slope for primary sample locations, 
    % and accounting for catchment area if selected.
    slope = nan(nSamples,1);
    for i = 1:nSamples
        % Extract indices for sample catchment
        ij = catchmentIndices(i, ijCatch, ptrCatch);
        % Calculate weights using predicted precipitation
        pSum = sum(pGrid(ij));
        % Calculated precipitation-weighted of wind-path slope for catchment
        if pSum>0
            % Normalize weights
            wtPrec = pGrid(ij)./pSum;
            % Calculate catchment-weighted composition of wind-path slope
            slope(i) = sum(wtPrec.*slopeGrid(ij));
        else
            % Use simple mean for dry sites
            slope(i) = mean(slopeGrid(ij));
        end
    end
    %... Calculate wind-path slope for altered sample locations, 
    % and accounting for catchment area if selected.
    slopeAlt = nan(nSamplesAlt,1);
    for i = 1:nSamplesAlt
        % Extract indices for altered sample catchment
        ij = catchmentIndices(i, ijCatchAlt, ptrCatchAlt);
        % Calculate weights using predicted precipitation
        pSum = sum(pGrid(ij));
        % Calculated precipitation-weighted of wind-path slope for catchment
        if pSum>0
            % Normalize weights
            wtPrec = pGrid(ij)./pSum;
            % Calculate catchment-weighted composition of wind-path slope
            slopeAlt(i) = sum(wtPrec.*slopeGrid(ij));
        else
            % Use simple mean for dry sites
            slopeAlt(i) = mean(slopeGrid(ij));
        end
    end
end

%% Calculate surface relative humidity for sample catchments
if ~isempty(ijCatch) && ~isempty(ptrCatch) && numel(rHGrid)>1
    %... Calculate surface relative humidity for primary sample locations, 
    % and accounting for catchment area if selected.
    rH = nan(nSamples,1);
    for i = 1:nSamples
        % Extract indices for primary sample catchment
        ij = catchmentIndices(i, ijCatch, ptrCatch);
        % Calculate weights using predicted precipitation
        pSum = sum(pGrid(ij));
        % Calculated precipitation-weighted of surface relative humidity for catchment
        if pSum>0
            % Normalize weights
            wtPrec = pGrid(ij)./pSum;
            % Calculate catchment-weighted composition of surface relative humidity 
            rH(i) = sum(wtPrec.*rHGrid(ij));
        else
            % Use simple mean for dry sites
            rH(i) = mean(rHGrid(ij));
        end
    end
    %... Calculate surface relative humidity for altered sample locations, 
    % and accounting for catchment area if selected.
    rHAlt = nan(nSamplesAlt,1);
    for i = 1:nSamplesAlt
        % Extract indices for altered sample catchment
        ij = catchmentIndices(i, ijCatchAlt, ptrCatchAlt);
        % Calculate weights using predicted precipitation
        pSum = sum(pGrid(ij));
        % Calculated precipitation-weighted of  surface relative humidity for catchment
        if pSum>0
            % Normalize weights
            wtPrec = pGrid(ij)./pSum;
            % Calculate catchment-weighted composition of  surface relative humidity
            rHAlt(i) = sum(wtPrec.*rHGrid(ij));
        else
            % Use simple mean for dry sites
            rHAlt(i) = mean(rHGrid(ij));
        end
    end
end

%% Caculate best-fit lines for isotopes vs lifting. 
% The precipitation isotopes are represented using their predicted 
% values from the best-fit OPI solution, either as point estimates if
% the sample location is designated as "local" (type L), or as the 
% precipitated-weighted value for the upstream catchment if the sample is 
% designated as "catchment" (type C). 
% The lifting is represented either by elevation or by the maximum
% lifting along the upwind path. The elevation and maximum lifting are 
% calculated as either "local" or "catchment" values depending on the
% designation of the sample (type L or C).  
% The isotope values are not adjusted for latitudinal gradients 
% (dD2H0_dLat, dD18O0_dLat), but this effect is will only contribute small
% symmetric errors. The slope of the lines are estimated by least squares
% and the intercept is held fixed so that it matches the estimated 
% base isotope values (d2H0, d18O).
% Note: the terms sample, pred, and grid refer to 
% sample observations, predicted values for sample locations (which can be
% either "local" or "catchment", and local estimates for nodes in a grid.

%... Predicted isotopes vs maximum lifting
x1 = liftMaxPred(iWet);
y1 = d2HPred(iWet) - d2H0;
bLiftMaxLine_d2HPred(1) = sum(y1.*x1)./sum(x1.^2);
bLiftMaxLine_d2HPred(2) = d2H0;
y1 = d18OPred(iWet) - d18O0;
bLiftMaxLine_d18OPred(1) = sum(y1.*x1)./sum(x1.^2);
bLiftMaxLine_d18OPred(2) = d18O0;

%... Predicted isotopes vs elevation
x1 = elevationPred(iWet);
y1 = d2HPred(iWet) - d2H0;
bLineElev_d2HPred(1) = sum(y1.*x1)./sum(x1.^2);
bLineElev_d2HPred(2) = d2H0;
y1 = d18OPred(iWet) - d18O0;
bLineElev_d18OPred(1) = sum(y1.*x1)./sum(x1.^2);
bLineElev_d18OPred(2) = d18O0;

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
    'figures: 2, and 5 to 7.\n'])
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
if lon0==mean(sampleLon) && lat0==mean(sampleLat)
    fprintf('Map origin is set to sample centroid.\n')
else
    fprintf('Map origin is set center of the topographic grid.\n')
end
fprintf('Size of cosine window as fraction of grid size: %g (dimensionless)\n', rTukey)
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
fprintf('\n------------ Estimates for Lifting Lines-------------\n')
fprintf('The precipitation isotopes are represented using their predicted\n')
fprintf('values from the best-fit OPI solution, either as point estimates if\n')
fprintf('the sample location is designated as "local" (type L), or as the\n')
fprintf('precipitated-weighted value for the upstream catchment if the sample is\n')
fprintf('designated as "catchment" (type C).\n')
fprintf('The lifting is represented either by local elevation or by the maximum\n')
fprintf('lifting along the upwind path. The elevation and maximum lifting are\n')
fprintf('calculated as either "local" or "catchment" values depending on the\n')
fprintf('designation of the sample (type L or C). \n')
fprintf('The isotopes are not adjusted for latitudinal gradients\n')
fprintf('(dD2H0_dLat, dD18O0_dLat), but this source of error is small\n')
fprintf('and symmetric. The slope of the lines are estimated by least squares\n')
fprintf('and the intercept is held fixed so that it matches the estimated\n')
fprintf('base isotope values (d2H0, d18O).\n')
fprintf('\nPredicted Isotopes vs Maximum Lifting\n')
fprintf('Intercept and slope for d2H: %.1f per mil, %.1f per mil/km\n', ...
    bLiftMaxLine_d2HPred(2)*1e3, bLiftMaxLine_d2HPred(1)*1e6)
fprintf('Intercept and slope for d18O: %.1f per mil, %.1f per mil/km\n', ...
    bLiftMaxLine_d18OPred(2)*1e3, bLiftMaxLine_d18OPred(1)*1e6)
fprintf('[%g\t%g\t%g\t%g]\n', ...
    bLiftMaxLine_d2HPred(2)*1e3, bLiftMaxLine_d2HPred(1)*1e6, ...    
    bLiftMaxLine_d18OPred(2)*1e3, bLiftMaxLine_d18OPred(1)*1e6);
fprintf('\nPredicted Isotopes vs Elevation\n')
fprintf('Intercept and slope for d2H: %.1f per mil, %.1f per mil/km\n', ...
    bLineElev_d2HPred(2)*1e3, bLineElev_d2HPred(1)*1e6)
fprintf('Intercept and slope for d18O: %.1f per mil, %.1f per mil/km\n', ...
    bLineElev_d18OPred(2)*1e3, bLineElev_d18OPred(1)*1e6)

if ~isempty(sampleFile)
    fprintf('\n------------------ Quality of Fit -------------------\n')
    fprintf('Reduced chi-square: %.4f\n', chiR2)
    fprintf('Degrees of freedom: %d\n', nu)
    fprintf('Number of wet locations for primary samples: %d\n', nSamplesWet)
    fprintf('Number of primary samples: %d\n', nSamples);
end

%... Close the diary
diary off;

%% Plot figures
Fig01 % Predicted vs observed isotope values
Fig02 % Craig plot, with primary samples only
Fig03 % Craig plot, with both primary and altered samples
Fig04 % Plot standarized residuals as a function of x and y
Fig05 % Plot predicted isotopes versus maximum lifting
Fig06 % Plot predicted isotopes versus elevation
Fig07 % Temperature and lapse rates for base state

%% Report compute time
diary on
fprintf('\n----------------- Computation Time ------------------\n')
fprintf('Compute time: %.2f minutes\n', ...
    minutes(datetime - startTime));
diary off

%% Fig01, Plot predicted vs observed isotope values, for primary samples
    function Fig01
        if isempty(sampleFile), return, end
        figure(1)
        subplot(2,1,1) %... delta 2H results
        %... Plot isotope data
        plot(sampleD2H(iWet)*1e3, d2HPred(iWet)*1e3, 'o', ...
                'LineWidth', 2, 'MarkerSize', 10, 'Color', blue);
        hold on
        %... Plot initial values for isotopes
        plot(d2H0*1e3, d2H0*1e3,'s', ...
            'MarkerSize',24, 'LineWidth',3, 'Color', red)
        %... Format plot
        axis square manual
        grid on
        grid minor
        box on
        set(gca, 'FontSize', 16, 'LineWidth', 1);
        %... Plot 1:1 reference line
        hLine = refline(1, 0);
        hLine.Color = [0.5 0.5 0.5];
        hLine.LineWidth = 3;
        %... Write labels
        hT = title('Fig. 1. Observed versus predicted isotope values', ...
            'FontSize', 16);
        hT.Units = 'normalized'; 
        hT.Position(2) = hT.Position(2) + 0.02;
        xlabel(['Observed \delta^{2}H (', char(8240), ')'], 'FontSize', 16)
        ylabel(['Predicted \delta^{2}H (', char(8240), ')'], 'FontSize', 16)
                
        subplot(2,1,2) %... delta 18O results
        %... Plot isotope data
        plot(sampleD18O(iWet)*1e3, d18OPred(iWet)*1e3, 'o', ...
                'LineWidth', 2, 'MarkerSize', 10, 'Color', blue);
        hold on
        %... Plot initial values for isotopes
        plot(d18O0*1e3, d18O0*1e3, 's', ...
            'MarkerSize', 24, 'LineWidth', 3, 'Color', red)
        %... Format plot
        axis square manual
        grid on
        grid minor
        box on
        set(gca, 'FontSize', 16, 'LineWidth', 1);
        %... Plot 1:1 reference line
        hLine = refline(1, 0);
        hLine.Color = [0.5 0.5 0.5];
        hLine.LineWidth = 3;
        %... Write labels
        xlabel(['Observed \delta^{2}H (', char(8240), ')'], 'FontSize', 16)
        ylabel(['Predicted \delta^{2}H (', char(8240), ')'], 'FontSize', 16)
        %... Save figure in pdf format
        printFigure(runPath)
    end

%% Fig02, Craig plot, with primary samples only
    function Fig02
        figure(2)
        %... Plot isotopic data
        hold on
        if ~isempty(sampleFile)
            hP1 = plot(sampleD18O(iWet)*1e3, sampleD2H(iWet)*1e3, ...
                'o', 'Color', [0.7 0.7 0.7], 'LineWidth', 2);
            hP2 = plot(d18OPred(iWet)*1e3, d2HPred(iWet)*1e3, 'o', ...
                'LineWidth', 2, 'MarkerSize', 10, 'Color', blue);
        else
            hP2 = plot(d18OPred(iWet)*1e3, d2HPred(iWet)*1e3, 'o', ...
                'LineWidth', 2, 'MarkerSize', 10, 'Color', blue);
            plot(d18OPred*1e3, d2HPred*1e3, '--k');
        end
        %... Plot isotopic composition of base precipitation
        plot(d18O0*1e3, d2H0*1e3, 's', 'MarkerSize', 20, 'Color', red)
        if ~isempty(sampleFile)
            %... Plot one-sigma ellipse around base precipitation composition
            % Cholesky decomposition provides the square root of cov matrix
            % See "Drawing Confidence Ellipses and Ellipsoids", jellymatter.wordpress.com
            rHat = [cosd(0:0.5:360); sind(0:0.5:360)]';
            ellipse = rHat*chol(cov) + [d2H0, d18O0];
            plot(ellipse(:,2)*1e3, ellipse(:,1)*1e3, '-', ...
                'LineWidth', 2, 'Color', red)
        end
        axis square tight
        %... Plot meteoric water line
        hLine = refline(bMWLSample(2), bMWLSample(1)*1e3);
        hLine.Color = [0.5 0.5 0.5];
        hLine.LineWidth = 3;
        %... Legend
        if ~isempty(sampleFile)
            legend([hP1, hP2, hLine], {'Observed', 'Predicted', 'Sample MWL'}, ...
                'Location', 'southeast')
        else
            legend([hP2, hLine], {'Predicted Wet', 'Global MWL'}, ...
                'Location', 'Northwest')
        end        
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
        printFigure(runPath)
    end

%% Fig03, Craig plot, with both primary and altered samples
    function Fig03
        if isempty(sampleFile) || nSamplesAlt==0, return, end
        figure(3)
        %... Plot isotopic data
        hold on
        % Observed primary samples
        hP1 = plot(sampleD18O*1e3, sampleD2H*1e3, 'o', ...
                'LineWidth', 2, 'MarkerSize', 10, 'Color', blue);
        % Observd altered samples
        % Line connecting altered samples to predicted primary values
        for k = 1:nSamplesAlt
            plot([sampleD18OAlt(k), d18OPredAlt(k)]*1e3, ...
                [sampleD2HAlt(k), d2HPredAlt(k)]*1e3, '-', ...
                'Color', [0.50196 0.50196 0], ...
                'LineWidth', 1, 'MarkerSize', 10);
        end
        axis square tight
        %... Plot meteoric water line
        hLine = refline(bMWLSample(2), bMWLSample(1)*1e3);
        hLine.Color = [0.5 0.5 0.5];
        hLine.LineWidth = 3;
        hP2 = plot(sampleD18OAlt*1e3, sampleD2HAlt*1e3, ...
            'o', 'Color', red);
        %... Plot isotopic composition of base precipitation
        plot(d18O0*1e3, d2H0*1e3, 's', ...
            'LineWidth', 3, 'MarkerSize', 20, 'Color', red)
        if ~isempty(sampleFile)
            %... Plot one-sigma ellipse around base precipitation composition
            % Cholesky decomposition provides the square root of cov matrix
            % See "Drawing Confidence Ellipses and Ellipsoids", jellymatter.wordpress.com
            rHat = [cosd(0:0.5:360); sind(0:0.5:360)]';
            ellipse = rHat*chol(cov) + [d2H0, d18O0];
            plot(ellipse(:,2)*1e3, ellipse(:,1)*1e3, '-', ...
                'LineWidth', 3, 'Color', red)
        end
        %... Legend
        legend([hP1, hP2, hLine], {'Primary', 'Altered', 'Sample MWL'}, ...
            'Location', 'southeast')
        %... Format plot
        grid on
        grid minor
        %... Write labels
        hT = title('Fig. 3. Craig plot of primary and altered samples', ...
            'FontSize', 16);
        hT.Units = 'normalized'; 
        hT.Position(2) = hT.Position(2) + 0.02;        
        xlabel(['\delta^{18}O (', char(8240), ')'], 'FontSize', 16)
        ylabel(['\delta^{2}H (', char(8240), ')'], 'FontSize', 16)
        set(gca, 'FontSize', 16, 'LineWidth', 1);
        %... Save figure in pdf format
        printFigure(runPath)
    end

%% Fig04, Plot standardized residuals relative to easting and northing
    function Fig04
        if isempty(sampleFile), return, end
        figure(4)
        %... Plot standardized residuals relative to easting (x)
        subplot(2,1,1)
        hold on
        plot(sampleX(iWet)*1e-3, stdResiduals(iWet), 'o', ...
                'LineWidth', 2, 'MarkerSize', 10, 'Color', blue);
        grid on
        grid minor
        %... Write labels for first plot
        hT = title('Fig. 4. Standardized residuals for best-fit solution', ...
            'FontSize', 16);
        hT.Units = 'normalized'; 
        hT.Position(2) = hT.Position(2) + 0.02;        
        xlabel('Easting (km)', 'FontSize', 16);
        ylabel('Standardized Residuals', 'FontSize', 16)
        set(gca, 'FontSize', 16, 'LineWidth', 1);

        %... Plot standardized residuals relative to northing (y)
        subplot(2,1,2)
        plot(sampleY(iWet)*1e-3, stdResiduals(iWet), 'o', ...
                'LineWidth', 2, 'MarkerSize', 10, 'Color', blue);
        % Format plot
        grid on
        grid minor
        %... Write labels for second plot
        xlabel('Northing (km)', 'FontSize', 16);
        ylabel('Standardized Residuals', 'FontSize', 16)
        set(gca, 'FontSize', 16, 'LineWidth', 1);
        %... Save figure in pdf format
        printFigure(runPath)
    end

%% Fig05, Plot predicted isotopes versus maximum lifting
    function Fig05
        figure(5)
        %... Plot sample d2H relative to maximum lifting
        subplot(2,1,1)
        hold on
        plot(liftMaxPred(iWet), d2HPred(iWet)*1e3, 'o', ...
                'LineWidth', 2, 'MarkerSize', 10, 'Color', blue);
        if isempty(sampleFile)
            plot(liftMaxPred, d2HPred*1e3, '--k');
        end
        axis square manual
        %... Plot best-fit line
        hL = refline(bLiftMaxLine_d2HPred*1e3);
        hL.Color = [0.5 0.5 0.5];
        hL.LineWidth = 3;
        hA = gca;
        hA.XLim(1) = 0;
        str = sprintf('%s%.1f - %.1f%s', '  \delta^{2}H = ', ...
            bLiftMaxLine_d2HPred(2)*1e3, -1*bLiftMaxLine_d2HPred(1)*1e6,  '{\it L} (km)');
        text(hA.XLim(1), hA.YLim(1)+4, str, 'FontSize', 12, ...
            'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom');
        %... Additional formatting
        grid on
        grid minor
        %... Write labels for first plot
        hT = title('Fig. 5. Predicted isotopes versus maximum lifting', ...
            'FontSize', 16);
        hT.Units = 'normalized';
        hT.Position(2) = hT.Position(2) + 0.02;
        xlabel('Maximum lifting (m)', 'FontSize', 16);
        ylabel(['Predicted \delta^{2}H (', char(8240), ')'], 'FontSize', 16)
        hA.FontSize = 16;
        hA.LineWidth = 1;
        
        %... Plot predicted d18O relative to maximum lifting
        subplot(2,1,2)
        hold on
        plot(liftMaxPred(iWet), d18OPred(iWet)*1e3, 'o', ...
                'LineWidth', 2, 'MarkerSize', 10, 'Color', blue);
        if isempty(sampleFile)
            plot(liftMaxPred, d18OPred*1e3, '--k');
        end
        axis square manual
        %... Plot best-fit line
        hL = refline(bLiftMaxLine_d18OPred*1e3);
        hL.Color = [0.5 0.5 0.5];
        hL.LineWidth = 3;
        hA = gca;
        hA.XLim(1) = 0;
        str = sprintf('%s%.1f - %.1f%s', '  \delta^{18}O = ', ...
            bLiftMaxLine_d18OPred(2)*1e3, -1*bLiftMaxLine_d18OPred(1)*1e6,  '{\it L} (km)');
        text(hA.XLim(1), hA.YLim(1)+0.5, str, 'FontSize', 12, ...
            'HorizontalAlignment', 'left', ...
            'VerticalAlignment', 'bottom', 'Margin', 8);
        %... Additional formatting
        grid on
        grid minor
        %... Write labels for second plot
        xlabel('Maximum lifting (m)', 'FontSize', 16);
        ylabel(['Predicted \delta^{18}O (', char(8240), ')'], 'FontSize', 16)
        hA.FontSize = 16;
        hA.LineWidth = 1;
        %... Save figure in pdf format
        printFigure(runPath)
    end

%% Fig06, Plot predicted isotopes versus elevation
    function Fig06
        figure(6)
        %... Plot predicted d2H relative to elevation
        subplot(2,1,1)
        hold on
        plot(elevationPred(iWet), d2HPred(iWet)*1e3, 'o', ...
                'LineWidth', 2, 'MarkerSize', 10, 'Color', blue);
        if isempty(sampleFile)
            plot(elevationPred(iWet), d2HPred(iWet)*1e3, '--k');
        end
        axis square manual
        %... Plot best-fit line
        hL = refline(bLineElev_d2HPred*1e3);
        hL.Color = [0.5 0.5 0.5];
        hL.LineWidth = 3;
        hA = gca;
        hA.XLim(1) = 0;
        str = sprintf('%s%.1f - %.1f%s', '  \delta^{2}H = ', ...
            bLineElev_d2HPred(2)*1e3, -1*bLineElev_d2HPred(1)*1e6,  '{\it h} (km)');
        text(hA.XLim(1), hA.YLim(1)+4, str, 'FontSize', 12, ...
            'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom');
        %... Additional formatting
        grid on
        grid minor
        %... Write labels for first plot
        hT = title('Fig. 6. Predicted isotopes versus elevation', ...
            'FontSize', 16);
        hT.Units = 'normalized'; 
        hT.Position(2) = hT.Position(2) + 0.02;                
        xlabel('Elevation (m)', 'FontSize', 16);
        ylabel(['Predicted \delta^{2}H (', char(8240), ')'], 'FontSize', 16)
        hA.FontSize = 16;
        hA.LineWidth = 1;
        
        %... Plot predicted d18O relative to elevation
        subplot(2,1,2)
        hold on
        plot(elevationPred(iWet), d18OPred(iWet)*1e3, 'o', ...
                'LineWidth', 2, 'MarkerSize', 10, 'Color', blue);
        if isempty(sampleFile)
            plot(elevationPred(iWet), d18OPred(iWet)*1e3, '--k');
        end
        axis square manual
        %... Plot best-fit line
        hL = refline(bLineElev_d18OPred*1e3);
        hL.Color = [0.5 0.5 0.5];
        hL.LineWidth = 3;
        hA = gca;
        hA.XLim(1) = 0;
        str = sprintf('%s%.1f + %.1f%s', '  \delta^{18}O = ', ...
            bLineElev_d18OPred(2)*1e3, -1*bLineElev_d18OPred(1)*1e6,  '{\it h} (km)');
        text(hA.XLim(1), hA.YLim(1)+0.5, str, 'FontSize', 12, ...
            'HorizontalAlignment', 'left', ...
            'VerticalAlignment', 'bottom', 'Margin', 8);
        %... Additional formatting
        grid on
        grid minor
        %... Write labels for second plot
        xlabel('Elevation (m)', 'FontSize', 16);
        ylabel(['Predicted \delta^{18}O (', char(8240), ')'], 'FontSize', 16)
        hA.FontSize = 16;
        hA.LineWidth = 1;
        %... Save figure in pdf format
        printFigure(runPath)
    end

%% Fig07, Plot temperature and lapse rates for base state
    function Fig07
        figure(7)
        %... Plot environmental temperature versus base-state elevation
        subplot(1,2,1)
        hold on
        plot(T - TC2K, zBar*1e-3, '-', 'LineWidth', 3, 'Color', red);
        %... Additional formatting
        axis square manual
        grid on
        grid minor
        %... Write labels for first plot
        xlabel({'Environmental'; 'Temperature (°C)'}, 'FontSize', 16)
        ylabel('Elevation (km)', 'FontSize', 16);
        hA1 = gca;
        hA1.FontSize = 16;
        hA1.LineWidth = 1;
        hA1.XLim(1) = min(T - TC2K);
        hA1.XLim(2) = ceil(max(T - TC2K)/10)*10;
        hA1.YLim(1) = 0;
        hA1.YLim(2) = zBar(end)*1e-3;
        
        %... Plot lapse rates vs base-state elevation
        subplot(1,2,2)
        hold on
        plot(gammaEnv*1e3, zBar*1e-3, '-', 'LineWidth', 3, 'Color', red);        
        plot(gammaSat*1e3, zBar*1e-3, '-', 'LineWidth', 3, 'Color', blue);
        legend('Environmental', 'Moist', 'Location', 'northwest')
        %... Additional formatting
        axis square manual
        axis tight
        grid on
        grid minor
        %... Write labels for second plot
        xlabel('Lapse Rate (°C/km)', 'FontSize', 16)
        %ylabel('Elevation (km)', 'FontSize', 16);
        hA2 = gca;
        hA2.FontSize = 16;
        hA2.LineWidth = 1;
        hA2.YLim(1) = 0;
        hA2.YLim(2) = zBar(end)*1e-3;        
        hA2.YTickLabel = [];
        
        %... Add title for both plots
        str = 'Fig. 7. Temperature and lapse rates versus elevation';
        hT1 = title(hA1, str, 'FontSize', 16);
        hT2 = title(hA2, str, 'FontSize', 16);
        hT1.Units = 'normalized'; 
        hT2.Units = 'normalized'; 
        hT1.Position(2) = hT1.Position(2) + 0.02;
        hT1.Position(1) = 1.5*hT1.Position(1) + 0.5*hT2.Position(1);
        hT2.String = '';

        %... Save figure in pdf format
        printFigure(runPath)
    end

end
