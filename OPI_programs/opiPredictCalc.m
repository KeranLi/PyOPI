function opiPredictCalc
% Calculate predicted precipitation d2H at specified locations, and
% as a function of a specified OPI solution, and variants adjusted
% to account for different ages. The oneWindCalc function is used
% to calculate the predicted d2H values.
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
tic
% Suppress warning about preallocation
%#ok<*AGROW>

%% Constants
%... Age span (Ma) used for loess smoothing.
% This value is meant to account for the lower precision of the
% sample ages relative to the record ages.
spanAge = 5;
%... Start time for calculation
startTime = datetime;
% Kelvin to Celsius
TC2K = 273.15;

%% Initialize present2Past
%... opiPredict provides the selection of two different benthic foram
% climate datasets. Of the two, my preference is Miller et al., 2020.
% In comparison, the Miller et al., 2020 dataset has larger values
% for Tdw and d18Osw, by about 2 to 3 C and 0.5 per mil, relative to 
% those in the Cramer et al., 2011 dataset. The Miller et al. dataset
% has a shorter range, back to 65 Ma, whereas the Cramer et al. dataset
% extends to 110 Ma. 
% Specify path and names for climate matfiles
matfilesPath_Climate = 'private/';
while true
    optionClimate = ...
        input('Climate option: Cramer et al, 2011 or Miller et al, 2020 (C,M): ', 's');
    optionClimate = upper(optionClimate(1));
    if strcmp('C', optionClimate) || strcmp('M', optionClimate), break, end
    fprintf('Incorrect selection. Try again.\n')
end
switch optionClimate
    case 'C'
        matFileBenthicForamClimate = 'Cramer2011BenthicForamClimate.mat';
    case 'M'
        matFileBenthicForamClimate = 'Miller2020BenthicForamClimate.mat';
end
matFileMEBM = 'MEBM_vary_OLR.mat';
climateOption = matFileBenthicForamClimate(1:6);
climateOption(1) = upper(climateOption(1));

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

% One-wind solution
% 'startTimeOpiCalc', 'runPath', 'runFile', 'runTitle', ...
% 'dataPath', 'topoFile', 'rTukey', 'sampleFile', ...
% 'lon', 'lat', 'x', 'y', 'lon0', 'lat0', ...
% 'sampleLine', 'sampleLon', 'sampleLat', 'sampleX', 'sampleY', ...
% 'sampleD2H', 'sampleD18O', 'sampleDExcess', 'sampleLC', ...
% 'sampleLineAlt', 'sampleLonAlt', 'sampleLatAlt', 'sampleXAlt', 'sampleYAlt', ...
% 'sampleD2HAlt', 'sampleD18OAlt', 'sampleDExcessAlt', 'sampleLCAlt', ...
% 'bMWLSample', 'ijCatch', 'ptrCatch','ijCatchAlt', 'ptrCatchAlt', ...
% 'hR', 'sdResRatio', 'sdDataMin', 'sdDataMax', 'cov', 'fC', ...
% 'lB', 'uB', 'nParametersFree', ...
% 'beta', 'chiR2', 'nu', 'stdResiduals', ...
% 'zBar', 'T', 'gammaEnv', 'gammaSat', 'gammaRatio', ...
% 'rhoS0', 'hS', 'rho0', 'hRho', ...
% 'd18O0', 'dD18O0_dLat', 'tauF', 'pGrid', 'fMGrid', 'rHGrid', ...
% 'evapD2HGrid', 'uEvapD2HGrid', 'evapd18OGrid', 'uEvapD18OGrid', ...
% 'd2HGrid', 'd18OGrid', ...
% 'iWet', 'd2HPred', 'd18OPred', 'catchArea', ...
% 'iWetAlt', 'd2HPredAlt', 'd18OPredAlt', 'catchAreaAlt', ...
% 'liftMaxPred', 'elevationPred', 'subsidencePred', ...
% 'liftMaxPredAlt', 'elevationPredAlt', 'subsidencePredAlt'

% Two-winds solution
% 'startTimeOpiCalc', 'runPath', 'runFile', 'runTitle', ...
% 'dataPath', 'topoFile', 'rTukey', 'sampleFile', ...
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
% 'liftMaxPred', 'elevationPred', 'fractionPGrid'

%... Load variables common to both kinds of solutions
load([matPathResults, '/', matFile_Results], ...
    'runPath', 'runFile', 'runTitle', ...
    'dataPath', 'topoFile', 'rTukey', ...
    'lon', 'lat', 'x', 'y', 'lon0', 'lat0', ...
    'bMWLSample', 'hR', 'sdResRatio', 'fC', 'beta');

%... Select the preferred precipitation state for two-wind solution.
% 0 for a one-wind solution, or 1 or 2 for the two-wind solution.
switch true
    case length(beta)==9
        stateNum = 0;
    case length(beta)==19
        fprintf('\nSelected run file is for two-wind solution, with wind directions of\n');
        fprintf('%.1f and %.1f degrees for precipitation states 1 and 2, respectively.\n\n', ...
            beta(2), beta(13));
        while true
            stateNum = ...
                str2num(input('Select precipitation state to be analyzed here (1,2): ', 's')); %#ok<ST2NM>
            if ~isempty(stateNum) && (stateNum==1 || stateNum==2), break, end
        end
    otherwise
        error('Number of parameters in beta is incorrect for this program')
end
fprintf('\n\n')

%... Load variables specific to the solutions
switch true
    case stateNum==0
        load([matPathResults, '/', matFile_Results], ...
            'zBar', 'T', 'pGrid', 'd2HGrid'); 
    case stateNum==1
        beta = beta(1:9);
        load([matPathResults, '/', matFile_Results], ...
            'zBar_1', 'T_1', 'pGrid_1', 'd2HGrid_1');        
        zBar = zBar_1; T = T_1; pGrid = pGrid_1; d2HGrid = d2HGrid_1;
        clear zBar_1 T_1 pGrid_1 d2HGrid_1 liftMaxPred_1
    case stateNum==2
        beta = beta(11:19);
        load([matPathResults, '/', matFile_Results], ...
            'zBar_2', 'T_2', 'pGrid_2', 'd2HGrid_2');
        zBar = zBar_2; T = T_2; pGrid = pGrid_2; d2HGrid = d2HGrid_2;
        clear zBar_2 T_2 pGrid_2 d2HGrid_2
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
U = beta(1);            % wind speed (m/s)
azimuth = beta(2);      % azimuth (degrees)
T0Pres = beta(3);       % sea-level temperature (K)
M =  beta(4);           % mountain-height number (dimensionless)
kappa = beta(5);        % eddy diffusion (m/s^2)
tauC = beta(6);         % condensation time (s)
d2H0Centroid = beta(7); % d2H of base precipitation at centroid (per unit)
dD2H0_dLat = beta(8);   % latitudinal gradient of base-prec d2H (1/deg lat)
fP = beta(9);           % residual precipitation after evaporation (fraction)
%... Convert mountain-height number to buoyancy frequency (rad/s)
hMax = max(hGrid, [], 'all');
NM = M*U/max(hGrid, [], 'all');
%... Calculate latitudinal gradient of base-prec d18O (per unit/deg lat)
dD18O0_dLat = dD2H0_dLat/bMWLSample(2);
%... Slope of local meteoric water line
dD2H0_dD18O0 = bMWLSample(2);

%% Report starting conditions
%... Start diary file
stateDesc = char([]);
if stateNum~=0,  stateDesc = ['_State0', num2str(stateNum)]; end
logFilename=[matPathResults, '/', mfilename, climateOption, stateDesc, '_Log.txt'];
if isfile(logFilename)
    movefile(logFilename, ...
        [matPathResults, '/', mfilename, climateOption, stateDesc, '_Log.bk!']);
end
diary(logFilename);
fprintf (['Program: ', mfilename, '\n'])
fprintf('Start time: %s\n', startTime)
fprintf('\n------------------- Climate Data --------------------\n')
fprintf('Path for climate data:\n')
fprintf('%s\n', matfilesPath_Climate)
fprintf('Mat file for benthic foram climate data:\n')
fprintf('%s\n', matFileBenthicForamClimate)
fprintf('Mat file for moist energy balance data:\n')
fprintf('%s\n', matFileMEBM)
fprintf('Age range used for loess smoothing: %g Ma\n', spanAge)
fprintf('\n-------------------- OPI Solution -------------------\n')
fprintf('Results matfile path:\n%s\n', matPathResults)
fprintf('Results matfile name:\n%s\n',  matFile_Results')
fprintf('Run file path:\n%s\n', runPath)
fprintf('Run file name:\n%s\n', runFile)
fprintf('Run title:\n%s\n', runTitle);
fprintf('Path name for data directory:\n%s\n', dataPath)
fprintf('\n------------------ Topography File ------------------\n')
fprintf('Topography file: %s\n', topoFile)
fprintf('Maximum elevation (m): %g\n', hMax);
[nY, nX] = size(hGrid);
fprintf('Grid size, nx and ny: %d, %d\n', nX, nY)
fprintf('Minimum and maximum for longitude: %g, %g\n', lon(1), lon(nX))
fprintf('Minimum and maximum for latitude: %g, %g\n', lat(1), lat(nY))
dLon = lon(2) - lon(1);
dLat = lat(2) - lat(1);
dX = x(2) - x(1);
dY = y(2) - y(1);
fprintf('Grid spacing, dx and dy (degrees): %g, %g\n', dLon, dLat)
fprintf('Grid spacing, dx and dy (km): %g, %g\n', dX, dY)
fprintf('Lon, lat for map origin (degrees): %g, %g\n', lon0, lat0);
fprintf('Size of cosine window as fraction of grid size: %f\n', rTukey)
fprintf('Coriolis frequency at map-origin latitude (mrad/s): %g\n', ...
    fC*1e3)
fprintf('\n---------------------- Constants --------------------\n')
fprintf('Average distance for isotopic exchange (m): %g\n', hR)
fprintf('Standard-deviation ratio for data residuals: %g\n', sdResRatio)
fprintf('\n------------------ Best-Fit Solution ----------------\n')
if stateNum>0
    fprintf('Selected state for two-wind solution: %d\n', stateNum)
end
fprintf('Wind speed: %g m/s\n', U);
fprintf('Wind azimuth: %g degrees\n', azimuth)
fprintf('Sea-level temperature: %g K (%g C)\n', T0Pres, T0Pres -TC2K)
fprintf('Mountain-height number: %g (dimensionless)\n', M)
fprintf('Horizontal eddy diffusivity: %g m^2/s\n', kappa)
fprintf('Characteristic time for conversion: %g s\n', tauC)
fprintf('d2H for base precipitation: %g per mil\n', d2H0Centroid*1e3)
fprintf('d2H latitude gradient for base precipitation: %g per mil/deg lat\n', dD2H0_dLat*1e3)
fprintf('Residual precipitation after evaporation: %g (fraction)\n', fP)
fprintf('\n------------ Observed Meteoric Water Line -----------\n')
fprintf('Intercept and slope: %g per mil, %g\n', ...
    bMWLSample(1)*1e3, bMWLSample(2))
diary off

%% Get sample list
% Read sample list which is an xlsx file with data for specified samples.
% The first three entries are given line by line:
% 1) dd2H0_dT0: temperature derivative for the base-precipitation d2H.
% 2) seD2HPres: standard error for modern precipitaton d2H (from OPI).
% 3) seD2H0Pres: standard error for modern base-precitation d2H (from OPI).
% Each remaining line provides data for a sample: 
% name, longitude, latitude, d2H_obs, SE(d2H_obs), age, comment
listPath = fnPersistentPath;
[listFile, listPath] = uigetfile([listPath,'/*.xlsx']);
if listFile==0, error('List file not found'), end
%... Remove terminal slash, if present
if listPath(end)=='/' || listPath(end)=='\'
    listPath = listPath(1:end-1);
end
fnPersistentPath(listPath);

%... Read input xlsx file, and remove records where the first cell is 
% blank ('missing') or contains "%" as the first non-white space character.
X = readcell([listPath, '/', listFile]);
n = size(X,1);
iSelect = true(n,1); 
for i = 1:n
    if all(ismissing(X{i,1}))
        iSelect(i) = false;
    else
        c = strip(char(X{i,1}));
        iSelect(i) = (c(1)~='%');
    end
end
X = X(iSelect,:);

% Get dD2H0_dT0, which is the temperature derivative for 
% base-precipitation d2H. This value is determined using the slope for
% modern monthly data for d2H precipitation vs. monthly temperature.
dD2H0_dT0 = X{1,1};
% Get standard error for modern precipitation d2H.
seD2HPres = X{2,1};
% Get standard error for modern base-precipitation d2H.
seD2H0Pres = X{3,1};

% Trim X to remove variables already read, and isolate sample variables.
X = X(4:end,:);
nSample = size(X,1);
% Get locations and data for specfied samples
sampleName = string(X(:,1));
sampleLon = cell2mat(X(:,2));
sampleLat = cell2mat(X(:,3));
sampleD2H = str2double(string(X(:,4)))*1e-3;
seD2HSample = str2double(string(X(:,5)))*1e-3;
sampleAge = cell2mat(X(:,6));
sampleComment = X(:,7);

%... Convert from geographic to Cartesian coordinates
[sampleX, sampleY] = lonlat2xy(sampleLon, sampleLat, lon0, lat0);

% Find catchment nodes for each of the samples.
typeLC = char(ones(nSample,1)*'C');
[ijCatch, ptrCatch] = ...
    catchmentNodes(sampleX, sampleY, typeLC, x, y, hGrid);

%% Report list information
diary on
fprintf('\n--------------------- List File ---------------------\n')
fprintf(['Read sample list which is an xlsx file with data for specified samples.\n',...
    'The first line gives the temperature derivative for the base-precipitation d2H,\n', ...
    'the second line gives the standard error for the modern precipitation d2H,\n', ...
    'and the third line gives the standard error for the modern base-precipitation d2H.\n', ...
    'Each remaining line provides data for a sample:\n', ... 
    'name, longitude, latitude, d2H_obs, SE(d2H_obs), age, comment\n']);
fprintf('List file path:\n%s\n', listPath)
fprintf('List file name:\n%s\n', listFile)
fprintf('Temperature derivative for present d2H0 (K/per mil): %f\n', ...
    dD2H0_dT0*1e3)
fprintf(['The value above is estimated using the slope for monthly averages\n', ...
 'for precipitation d2H and temperature.\n'])
fprintf('Standard error for modern precipitation d2H (per mil): %f\n', ...
    seD2HPres*1e3)
fprintf('Standard error for modern base-precipitation d2H (per mil): %f\n', ...
    seD2H0Pres*1e3)
fprintf('Number of selected samples: %d\n', nSample)
diary off

%% Calculate T0Record and d2H0Record for centroid latitude
[T0Record, d2H0Record, ageRecord, T0RecordSmooth, d2H0RecordSmooth] = ... 
climateRecords(matFileMEBM, matFileBenthicForamClimate, lat0, ...
    T0Pres, d2H0Centroid, dD2H0_dD18O0, dD2H0_dT0, spanAge);

% Check sampleAge relative to the age span of the climate records
if any(sampleAge<min(ageRecord)) || any(sampleAge>max(ageRecord))
    error('Sample ages extend beyond the age range for the climate records.')
end

%% Calculate present temperature and precipitation at sample locations
% Initialize variables
pPresLocal = nan(nSample,1);
pPresCatch = nan(nSample,1);
d2H0Pres = nan(nSample,1);
d2HPresLocal = nan(nSample,1);
d2HPresCatch = nan(nSample,1);
iWetPresLocal = false(nSample,1);
iWetPresCatch = false(nSample,1);
TPres = nan(nSample,1);
sampleH = nan(nSample,1);
liftMaxPred = nan(nSample,1);
% Calculate predictions
for k = 1:nSample
    % Extract indices for sample catchment
    ij = catchmentIndices(k, ijCatch, ptrCatch);
    
    % Adjust d2H0 for sample latitude
    d2H0Pres(k) = d2H0Centroid + dD2H0_dLat*(abs(sampleLat(k)) - abs(lat0));

    % Calculate local precipitation
    pPresLocal(k) = pGrid(ij(1));
    d2HPresLocal(k) = d2HGrid(ij(1));
    iWetPresLocal(k) = (pPresLocal(k)>0);

    % Calculate catchment precipitation
    pPresCatch(k) = sum(pGrid(ij));
    if pPresCatch(k)>0
        % Normalize weights
        wtPrec = pGrid(ij)./pPresCatch(k);
        % Calculate catchment-weighted composition of d2H
        d2HPresCatch(k) = sum(wtPrec.*d2HGrid(ij));
    else
        % Use simple mean for dry samples
        d2HPresCatch(k) = mean(d2HGrid(ij));
    end
    iWetPresCatch(k) = (pPresCatch(k)>0);

    % Calculate land-surface temperatures at sample location
    sampleH(k) = interp2(lon, lat, hGrid, sampleLon(k), sampleLat(k));    
    TPres(k) = interp1(zBar, T, sampleH(k));
end

%% Calculate reference temperature and precipitation at sample locations 
% The "references" values calculated here are specific for the 
% location and age of each sample, but with the topography held 
% the same as present. 
pLocalRef = nan(nSample,1);
pCatchRef = nan(nSample,1);
d2H0Ref = nan(nSample,1);
d2HRefLocal = nan(nSample,1);
d2HRefCatch = nan(nSample,1);
iWetLocal = false(nSample,1);
iWetCatch = false(nSample,1);
T0Ref = nan(nSample,1);
TRef = nan(nSample,1);
%... Determine unique ages in the vector sampleAge.
age = unique(sampleAge);
nAge = length(age);
% Calculate predictions
for i = 1:nAge
    fprintf('Progress: %.1f %%\n', (i/nSample)*1e2);
    %... Interpolate T0, d2H0Centroid, and d18OCentroid for specified age
    T0Age = interp1(ageRecord, T0RecordSmooth, age(i));
    d2H0CentroidAge = interp1(ageRecord, d2H0RecordSmooth, age(i));    
    d18O0CentroidAge = (d2H0CentroidAge - bMWLSample(1))/bMWLSample(2);
    
    %... Calculate environmental profile for atmosphere upwind of topography.
    % The atmosphere is assumed to be stable (NM > 0), and baseState includes
    % a check to ensure that gammaEnv > 0.
    [zBar, T, gammaEnv, gammaSat, gammaRatio, ...
        rhoS0, hS, ~, hRho] = baseState(NM, T0Age);
    
    %... Calculate precipitation rate (kg/(m^2 s))
    [s, t, Sxy, Txy, pGrid, hWind, fMWind, rHWind, fPWind, ...
        z223Wind, z258Wind, tauF] = precipitationGrid ...
        (x, y,  hGrid, U, azimuth, NM, fC, kappa, tauC, hRho, zBar, ...
        T, gammaEnv, gammaSat, gammaRatio, rhoS0, hS, fP);
        
    %... Calculate grids for precipitation isotopes and moisture ratio
    isFit = false;
    d2HGrid = isotopeGrid( ...
        s, t, Sxy, Txy, lat, lat0, hWind, fMWind, rHWind, fPWind, ...
        z223Wind, z258Wind, tauF, ...
        U, T, gammaSat, hS, hR, d2H0CentroidAge, d18O0CentroidAge, ...
        dD2H0_dLat, dD18O0_dLat, isFit);
    clear hWind fPWind rHWind
    
    % Calculate predictions for reference case in the past
    for k = 1:nSample
        % Skip if sampleAge is not equal to age(i)
        if sampleAge(k)~=age(i), continue, end
        
        % Sea-level air temperature for sample age
        T0Ref(k) = T0Age;
        
        % Adjust d2H0 for sample age and latitude
        d2H0Ref(k) = d2H0CentroidAge + dD2H0_dLat*(abs(sampleLat(k)) - abs(lat0));
        
        % Extract indices for sample catchment
        ij = catchmentIndices(k, ijCatch, ptrCatch);
        
        % Calculate local precipitation at sample location
        pLocalRef(k) = pGrid(ij(1));
        d2HRefLocal(k) = d2HGrid(ij(1));
        iWetLocal(k) = (pLocalRef(k)>0);
        
        % Calculate catchment precipitation at sample location
        pCatchRef(k) = sum(pGrid(ij));
        if pCatchRef(k)>0
            % Normalize weights
            wtPrec = pGrid(ij)./pCatchRef(k);
            % Calculate catchment-weighted composition of d2H
            d2HRefCatch(k) = sum(wtPrec.*d2HGrid(ij));
        else
            % Use simple mean for dry samples
            d2HRefCatch(k) = mean(d2HGrid(ij));
        end
        iWetCatch(k) = (pCatchRef(k)>0);

        % Surface-air temperature for reference case at sample location and age.
        TRef(k) = interp1(zBar, T, sampleH(k));

        %... Calculate maximum lifting for each sample location.
        liftMaxPred(k) = ...
            lifting(x, y, hGrid, pGrid, azimuth, U, tauF, ij, 1);
    end
end

% Calculate lifting-fractionation ratios.
PhiLiftLocal = (log(1 + sampleD2H) - log(1 + d2H0Ref)) ...
    ./(log(1 + d2HRefLocal) - log(1 + d2H0Ref));
PhiLiftCatch = (log(1 + sampleD2H) - log(1 + d2H0Ref)) ...
    ./(log(1 + d2HRefCatch) - log(1 + d2H0Ref));

% Calculate climate-fractionation ratio for sample locations
% PhiClim is calculated at each sample location using local precipitation
% data. Testing indicates that there is less than about 0.01 difference
% for PhiClim calculated using local and climate data. This result is
% expected given that that the base temperature field in OPI only varies
% in the vertical. 
PhiClim = (log(1 + d2HRefLocal) - log(1 + d2H0Ref)) ...
        ./(log(1 + d2HPresLocal) - log(1 + d2H0Pres));

%% Save to mat file
stateDesc = char([]);
if stateNum~=0,  stateDesc = ['_State0', num2str(stateNum)]; end
matFile_Results = [matPathResults, '/', ...
    'opiPredictCalc', climateOption, stateDesc, '.mat'];
save(matFile_Results, ...
    'matfilesPath_Climate', 'matFileBenthicForamClimate', 'matFileMEBM', ...
    'runTitle', 'listPath', 'listFile', ...
    'dD2H0_dT0', 'seD2HPres', 'seD2H0Pres', ...
    'sampleName', 'sampleLon', 'sampleLat', 'sampleH', 'liftMaxPred', ...
    'sampleAge', 'sampleD2H', 'seD2HSample', ...
    'd2H0Pres', 'd2HPresLocal', 'd2HPresCatch', ...
    'iWetPresLocal', 'iWetPresCatch', ...
    'd2H0Ref', 'd2HRefLocal', 'd2HRefCatch', ...
    'PhiLiftLocal', 'PhiLiftCatch', ...
    'T0Pres', 'TPres', 'T0Ref', 'TRef', 'PhiClim', 'sampleComment', ...
    'ageRecord', 'T0Record', 'd2H0Record', ...
    'T0RecordSmooth', 'd2H0RecordSmooth', 'spanAge', '-v7.3');

%% Save to excel file
T = table(sampleName, sampleLon, sampleLat, ...
    round(sampleH,0), round(liftMaxPred,0), ...
    round(sampleAge,1), round(sampleD2H*1e3,1), round(seD2HSample*1e3,1), ...
    round(d2H0Pres*1e3,1), ...
    round(d2HPresLocal*1e3,1), round(d2HPresCatch*1e3,1), ...
    iWetPresLocal, iWetPresCatch, ...
    round(d2H0Ref*1e3,1), ...
    round(d2HRefLocal*1e3,1), round(d2HRefCatch*1e3,1), ...
    round(PhiLiftLocal,2), round(PhiLiftCatch,2), ...
    round(T0Pres-TC2K,1)*ones(nSample,1), round(TPres-TC2K,1), ...
    round(T0Ref-TC2K,1), round(TRef-TC2K,1), round(PhiClim,2), ...
    string(sampleComment), ...
    'VariableNames', {'Name', 'Lon', 'Lat', 'Elev (m)', 'Lift_Max (m)', ...
    'Age (Ma)', 'd2H_obs', 'SE(d2H)', ...
    'd2H0_Present', 'd2H_L_Present', 'd2H_C_Present', 'iWet_L', 'iWet_C', ...
    'ref d2H0', 'ref d2H_L', 'ref d2H_C', ...
    'PhiLiftLocal', 'PhiLiftCatch', ...
    'T0_Present', 'T_Present', 'T0_Ref', 'T_Ref', 'PhiClim', 'Comment'});
%... Read input xls file
excelFilename = [matPathResults, '/', ...
    'opiPredictCalc', climateOption, stateDesc, '.xlsx'];
writetable(T, excelFilename, 'WriteMode', 'overwritesheet')

%... Complete and close log file
diary on
fprintf('\nTime for computation: %.2f hours\n', toc/3600);
diary off

end
