function [lon, lat, x, y, hGrid, lon0, lat0, ...
    sampleLine, sampleLon, sampleLat, sampleX, sampleY, ...
    sampleD2H, sampleD18O, sampleDExcess, sampleLC, ...
    sampleLineAlt, sampleLonAlt, sampleLatAlt, sampleXAlt, sampleYAlt, ...
    sample_d2HAlt, sample_d18OAlt, sampleDExcessAlt, sampleLCAlt, ...
    bMWLSample, sdDataMin, sdDataMax, cov, fC] ...
    = getInput(dataPath, topoFile, rTukey, sampleFile, sdResRatio)
% getInput  Select and parse an OPI run file.

% Mark Brandon, Yale University, 2016 - 2020

%% Initialize system
%... Turn off warning for cases where arrays are changed in size
% with each loop iteration.
%#ok<*AGROW>

%% Initialize variables
% Global meteoric water line (MWL), as determined from GNIP sites
% https://en.wikipedia.org/wiki/Global_meteoric_water_line
% If a sample file is not provided, then the MWL is set using these data.
bMWLSample = [9.47e-3, 8.03];

%... Set following variables to empty, in case they are not assigned.
sampleLine = []; sampleLon = []; sampleLat = [];
sampleX = []; sampleY = [];
sampleD2H = []; sampleD18O = []; sampleDExcess = []; sampleLC = [];
sampleLineAlt = []; sampleLonAlt = []; sampleLatAlt = [];
sampleXAlt = []; sampleYAlt = [];
sample_d2HAlt = []; sample_d18OAlt = []; sampleDExcessAlt = []; sampleLCAlt = [];
sdDataMin = []; sdDataMax = []; cov = [];

%% Read topographic data
[lon, lat, hGrid] = gridRead([dataPath, '/', topoFile]);
if any(isnan(hGrid(:)))
    error('Error: Elevation grid contains nans\n')
end
%... Set elevations below sea level to 0 meters
hGrid(hGrid<0) = 0;

% Create 2D cosine-taper window, with
% fractional width = rTukey/2 on each margin of the grid.
[nY, nX] = size(hGrid);
window = tukeywin(nY, rTukey)*tukeywin(nX, rTukey)';
hGrid = window.*hGrid;
clear window

%... Read data from sample file, if specified
if ~isempty(sampleFile)
    % Check sample file and initialize variables
    if isfile(sampleFile), error('Sample file not found: %s\n', sampleFile), end
    %... Read sample data in xlsx file, and remove records where the 
    % first cell contains "%" as the first non-whitespace character.
    X = readcell([dataPath, '/', sampleFile], 'Range', 'A:F');
    % Create list that relates sample back to row numbers in the xlsx file.
    sampleLine = (1:size(X,1))';
    % Isolate first column and set cells with "missing" to blank.
    X1 = X(:,1);
    X1(cellfun(@(y) all(ismissing(y)), X1)) = {''};
    % Strip whitespace characters from start and end of strings.
    c = char(strip(X1));
    % Find rows with leading comment character "%".
    iData = (c(:,1)~='%');
    % Remove commented rows and first column from the cell array X.
    X = X(iData,2:end);
    % Remove commented rows in sampleLine.
    sampleLine = sampleLine(iData);
    % Check for missing values in cell array X.
    if any(cellfun(@(y) all(ismissing(y)), X), 'all')
        iProblem = any(cellfun(@(y) all(ismissing(y)), X), 2);
        fprintf('One or more samples have missing values.\n');
        fprintf('Row number: %d\n', sampleLine(iProblem));
        error('Fix the sample file for the errors noted above.')
    end
    % Parse to sample variables
    sampleLon = cell2mat(X(:,1));
    sampleLat = cell2mat(X(:,2));
    sampleD2H = cell2mat(X(:,3))*1e-3;
    sampleD18O = cell2mat(X(:,4))*1e-3;
    sampleLC = upper(char(X(:,5)));
    sampleLC = sampleLC(:,1);
    % Check data
    isError = false;
    %... Check that all samples lie within hGrid
    if any(lon(1)>sampleLon | lon(end)<sampleLon ...
            | lat(1)>sampleLat | lat(end)<sampleLat)
        isError = true;
        iProblem =(lon(1)>sampleLon | lon(end)<sampleLon ...
            | lat(1)>sampleLat | lat(end)<sampleLat);
        fprintf('One or more samples lie outside span of the topographic grid\n');
        fprintf('%d: %f, %f\n', [sampleLine(iProblem), sampleLon(iProblem), ...
            sampleLat(iProblem)]');
        warning('Fix the sample file for the errors noted above.')
    end
    %... Check sampleLC character
    if any(strcmp(sampleLC,'L') & strcmp(sampleLC,'C'))
        isError = true;
        warning('Type variable in data file can only have values of L or C');
    end
    %... Stop if errors
    if isError==true, error('Fix problems described above.'), end
    
    %... Estimate the sample-based meteoric water line
    [bMWLSample, sdDataMin, sdDataMax, cov, iFit] = ...
        estimateMWL(sampleD18O, sampleD2H, sdResRatio);

    %... Divide samples into primary and altered precipitation, where
    % dExcess is greater or less, respectively, than 5 per mil.
    sampleLineAlt = sampleLine(~iFit);
    sampleLonAlt = sampleLon(~iFit);
    sampleLatAlt = sampleLat(~iFit);
    sample_d2HAlt = sampleD2H(~iFit);
    sample_d18OAlt = sampleD18O(~iFit);
    sampleLCAlt = sampleLC(~iFit);    
    sampleLine = sampleLine(iFit);
    sampleLon = sampleLon(iFit);
    sampleLat = sampleLat(iFit);
    sampleD2H = sampleD2H(iFit);
    sampleD18O = sampleD18O(iFit);
    sampleLC = sampleLC(iFit);
    %... Calculate dExcess
    sampleDExcess = sampleD2H - 8*sampleD18O;
    sampleDExcessAlt = sample_d2HAlt - 8*sample_d18OAlt;
end

%... Define map origin
if ~isempty([sampleLon, sampleLat])
    lon0 = mean(sampleLon);
    lat0 = mean(sampleLat);
    %... Convert to projected coordinate for samples
    [sampleX, sampleY] = lonlat2xy(sampleLon, sampleLat, lon0, lat0);
    [sampleXAlt, sampleYAlt] = lonlat2xy(sampleLonAlt, sampleLatAlt, lon0, lat0);
else
    lon0 = mean(lon);
    lat0 = mean(lat);
end

%... Convert to projected coordinate for hGrid and samples
[x, y] = lonlat2xy(lon, lat, lon0, lat0);

%... Coriolis frequency (rad/s) calculated at centroid for hGrid or samples.
omega = 7.2921e-5; % rotation rate of the Earth (rad/s)
fC = 2*omega*sind(lat0);

end
