function [runPath, runFile, runTitle, isParallel, dataPath, ...
    topoFile, rTukey, sampleFile, contDivideFile, restartFile, ...
    mapLimits, sectionLon0, sectionLat0, mu, epsilon0, ...
    parameterLabels, exponents, lB, uB, beta] ...
    = getRunFile(runFile)
% getRunFile  Select and parse an OPI run file.

% Mark Brandon, Yale University, 2016-2021

%% Compute
%... Open run file
if nargin==0
    %... Get user to select run file
    runPath = fnPersistentPath;
    [runFile, runPath] = uigetfile([runPath, '/*.run']);
    if runFile==0, error('Run file not found.'), end
    %... Remove terminal slash, if present
    if runPath(end)=='/' || runPath(end)=='\'
        runPath = runPath(1:end-1);
    end
    fnPersistentPath(runPath);
else
    %... Use input argument for run file
    [runPath, runFile, runExt] = fileparts(runFile);
    runFile = [runFile, runExt];
end
fid = fopen([runPath, '/', runFile], 'r', 'native', 'UTF-8');
if fid==-1
    error('Run file not found.')
end

%... Get run title
while true
    str = strip(fgetl(fid));
    if ~isempty(str) && ~strcmp(str(1),'%'), break, end
end
runTitle = str;

%... Set parallel mode versus serial mode
% 0: serial mode
% 1: parallel mode
while true
    str = strip(fgetl(fid));
    if ~isempty(str) && ~strcmp(str(1),'%'), break, end
end
if str~='0' && str~='1'
    error('Serial vs. parallel option must be either 0 or 1.')
end
isParallel = logical(str2num(str));

%... Read path name for data directory
% Enter two path names, each on a separate line, to account for runs on
% two different computers (e.g., local computer, cluster computer).
% Set one path to "no" if only one path is needed.
dataPath = strings(2,1);
for i =1:2
    while true
        str = strip(fgetl(fid));
        if ~isempty(str) && ~strcmp(str(1),'%'), break, end
    end
    % Remove any final backward or forward slash
    if str(end)=='/' || str(end)=='\'
        str = str(1:end-1);
    end
    dataPath(i) = string(str);
end
for i = 1:2
    if ~strcmp(dataPath(i), 'no') && isfolder(dataPath(i))
        dataPath = char(dataPath(i));
        break
    else
        if i==2
            error('None of the paths for the data directories in the run file exist.')
        end
    end
end

%... Read filename for digital topography. The file should be placed
% in the directory set above for the dataPath variable.
while true
    str = strip(fgetl(fid));
    if ~isempty(str) && ~strcmp(str(1),'%'), break, end
end
topoFile = str;
if ~isfile([dataPath, '/', topoFile])
    error('File for digital topography does not exist.')
end

%... Size of cosine window, as fraction of grid size (0<= rTukey <=1)
% For example, rTukey = 0.25 (recommended value) will create a cosine
% taper along each margin of the grid, with the taper width equal to
% rTukey/2 (0.125) times the dimension of the grid in each direction.
% For no window, set rTukey = 0.
while true
    str = strip(fgetl(fid));
    if ~isempty(str) && ~strcmp(str(1),'%'), break, end
end
rTukey = str2num(str); %#ok<*ST2NM>
if ~isnumeric(rTukey) || length(rTukey)~=1 ||rTukey<0 || rTukey>1
    error('Incorrect entry for size of cosine window.')
end

%... Read filename for water-isotope data. The file should be placed
% in the directory set above for the dataPath variable.
while true
    str = strip(fgetl(fid));
    if ~isempty(str) && ~strcmp(str(1),'%'), break, end
end
sampleFile = str;
switch true
    case strcmpi(str, 'no')
        sampleFile = [];
    otherwise
        if ~isfile([dataPath, '/', sampleFile])
            error('File for water-isotope data does not exist.')
        end
end

%... Get filename for continental-divide coordinates
% Set to "no" if no continental-divide file is available, or
% if a plot of the divide is not wanted. Note that this file
% is required for the two-wind versions of the opi programs.
% The file should be placed in the directory set above for the
% dataPath variable.
while true
    str = strip(fgetl(fid));
    if ~isempty(str) && ~strcmp(str(1),'%'), break, end
end
switch true
    case strcmpi(str, 'no')
        contDivideFile = [];
    otherwise
        contDivideFile = str;
end

%... Get limits for figures showing maps [minLon, maxLon, minLat, maxLat]
while true
    str = strip(fgetl(fid));
    if ~isempty(str) && ~strcmp(str(1),'%'), break, end
end
mapLimits = str2num(str); %#ok<*ST2NM>
if ~isnumeric(mapLimits) || length(mapLimits)~=4
    error('Map limits requires four numeric values.')
end

%... Get section origin, sectionLon0, sectionLat0. Set to "map" to tell
% the program to set this point to the map origin.
while true
    str = strip(fgetl(fid));
    if ~isempty(str) && ~strcmp(str(1),'%'), break, end
end
ll = str2num(str);
switch true
    case isnumeric(ll) && length(ll)==2
        %... Set section origin to user values
        sectionLon0 = ll(1); sectionLat0 = ll(2);
    case strcmpi(str, 'map')
        %... Set section origin to empty to indicate to use map origin
        sectionLon0 = []; sectionLat0 = [];
    otherwise
        error('Specification of section origin in run file is incorrect.')
end

%... The CRS3 search is controlled by 2 parameters.
% The first, mu, is a factor that determines the size of the search set,
% mSet = ceil(mu*(nFree - 1)), where nFree is the number of free 
% parameters. 
% The second, epsilon0, defines the stopping criterion, 
% epsilon0 > stdev(f(1:mSet)) < epsilon0, where f(1:mSet) is the mSet 
% of candidate solutions with the smallest f values, 
% and stdev is the standard deviation function. 
% Recommended values: mu = 10, epsilon0 = 1e-4.
while true
    str = strip(fgetl(fid));
    if ~isempty(str) && ~strcmp(str(1),'%'), break, end
end
CRSParameters = str2num(str); %#ok<*ST2NM>
if ~isnumeric(CRSParameters) || length(CRSParameters)~=2
    error('CRS3 parameters requires two numeric values.')
end
mu = CRSParameters(1);
epsilon0 = CRSParameters(2);

%... Get filename for restart file, which allows one to restart
% a search using a previous solutions file. Set to "no" to run
% without a restart file. The file should be placed in the directory
% set above for the dataPath variable.
while true
    str = strip(fgetl(fid));
    if ~isempty(str) && ~strcmp(str(1),'%'), break, end
end
switch true
    case strcmpi(str, 'no')
        restartFile = [];
    otherwise
        restartFile = str;
end

%... Get axis labels for parameters, which are used in
% the opiPairsPlot program.
while true
    str = strip(fgetl(fid));
    if ~isempty(str) && ~strcmp(str(1),'%'), break, end
end
parameterLabels = string(str);
nParameterLabels = length(strfind(parameterLabels, "|")) + 1;
if nParameterLabels~=9 && nParameterLabels~=19
    error('Number of parameter labels must be either 9 or 19 for one-wind and two-wind solutions.');
end

%... Get exponents for "power of 10" scaling factors for each
% parameter, as used by opiPairsPlot program.
% For example, an exponent set to 3 means that the parameter value
% will be multiplied by 10^-3 before plotting.
while true
    str = strip(fgetl(fid));
    if ~isempty(str) && ~strcmp(str(1),'%'), break, end
end
exponents = str2num(str);
if length(exponents)~=nParameterLabels
    error('Mismatch between exponents for "power of 10" scaling and number of paramters.')
end

%... Get constraints for parameter search
while true
    str = strip(fgetl(fid));
    if ~isempty(str) && ~strcmp(str(1),'%'), break, end
end
lB = str2num(str);
while true
    str = strip(fgetl(fid));
    if ~isempty(str) && ~strcmp(str(1),'%'), break, end
end
uB = str2num(str);
if ~isnumeric(lB) || ~isnumeric(uB)
    error('Parameter constraints must be numeric.')
end
if length(lB)~=length(uB)
    error('Constraints for parameter search must have the same length.')
end
if length(lB)~=nParameterLabels
    error('Number of constraints must be equal to number of parameters.')
end
if any(lB>uB)
    error('Lower constraints must be less than or equal to upper constraints.')
end

% Get best-fit solution vector (if present)
beta = [];  % Set to empty if beta is not present
while ~feof(fid)
    str = strip(fgetl(fid));
    if ~isempty(str) && ~strcmp(str(1),'%')
        beta = str2num(str);
        if ~isnumeric(beta)
            error('Solution vector must be numeric')
        end
        break
    end
end

%... Check to ensure that file is correctly terminated.
% Run file may not have uncommented lines after the penalty option,
% or after the best-fit solution.
while ~feof(fid)
    if feof(fid), break, end
    str = strip(fgetl(fid));
    if ~isempty(str) && ~strcmp(str(1),'%')
        error('Run file has incorrect content at end of file.')
    end
end
%... Close run file
fclose(fid);
end