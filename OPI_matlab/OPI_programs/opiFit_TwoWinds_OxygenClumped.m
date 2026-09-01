function opiFit_TwoWinds_OxygenClumped(runFile)
% opiFit_TwoWinds_OxygenClumped fits d18O plus clumped-dolomite temperature.

startTime = datetime;
radiusEarth = 6371e3;
mPerDegree = pi*radiusEarth/180;
hR = 540;
sdResRatio = 28.3;
TC2K = 273.15;

if nargin==0
    [runPath, runFile, runTitle, isParallel, dataPath, ...
        topoFile, rTukey, sampleFile, ~, restartFile, ...
        ~, sectionLon0, sectionLat0, mu, epsilon0, ...
        parameterLabels, exponents, lB, uB, beta] = getRunFile;
else
    fprintf('Open run file: %s\n\n', runFile);
    [runPath, runFile, runTitle, isParallel, dataPath, ...
        topoFile, rTukey, sampleFile, ~, restartFile, ...
        ~, sectionLon0, sectionLat0, mu, epsilon0, ...
        parameterLabels, exponents, lB, uB, beta] = getRunFile(runFile);
end
if ~isempty(beta)
    error('Run file includes a best-fit solution, which is not allowed for optFit.');
end
if length(lB)~=19
    error('Number of parameters is incorrect for this program')
end
nParametersFree = sum(lB~=uB);

[lon, lat, x, y, hGrid, lon0, lat0, ...
    ~, sampleLon, sampleLat, sampleX, sampleY, ...
    sampleD2H, sampleD18O, ~, sampleLC, ...
    ~, sampleLonAlt, ~, ~, ~, ...
    ~, ~, ~, ~, ...
    bMWLSample, ~, ~, cov, fC] = ...
    getInput(dataPath, topoFile, rTukey, sampleFile, sdResRatio);

if isempty([sectionLon0, sectionLat0])
    sectionLon0 = lon0;
    sectionLat0 = lat0;
end
[ijCatch, ptrCatch] = catchmentNodes(sampleX, sampleY, sampleLC, x, y, hGrid);
nSamples = length(sampleLon);

clumpedFile = fullfile(dataPath, 'proxy_clumped', 'clumped_temperature.xlsx');
clumped = prepareClumpedDolomiteData(clumpedFile, sampleLon, sampleLat);
% Sedimentology supports direct interpretation as warmest lake-water T.
% Nonzero offsets remain available through clumped_fit_config.csv for
% sensitivity experiments.
dolomiteOffsetC = 0;
sigmaOffsetC = 0;
clumpedSeason = "warmest";
[dolomiteOffsetC, sigmaOffsetC, clumpedSeason, clumpedFitConfigFile] = ...
    readClumpedFitConfig(runPath, dataPath, dolomiteOffsetC, ...
    sigmaOffsetC, clumpedSeason);
if ismember(lower(string(clumpedSeason)), ["warmest", "warmest_month"])
    temperatureInterpretation = "warmest-month surface-air temperature";
else
    temperatureInterpretation = "legacy seasonal transfer interpretation";
end

solutions = [];
if ~isempty(restartFile)
    [~, ~, ~, ~, lbRestart, ubRestart, ~, ~, solutions, ~, ~, ~] = ...
        getSolutions(runPath, restartFile);
    if length(lB)~=(size(solutions,2) - 2) || ...
            any(lB~=lbRestart) || any(uB~=ubRestart)
        error('Constraints in the restart file are different from those in the run file.')
    end
end

logFilename=[runPath, '/', mfilename, '_Log.txt'];
if isfile(logFilename), delete(logFilename); end
diary(logFilename);
fprintf('Program: %s\n', mfilename)
fprintf('Objective: d18O plus clumped dolomite temperature.\n')
fprintf('Start time: %s\n', startTime)
fprintf('Run file path:\n%s\n', runPath)
fprintf('Run filename:\n%s\n', runFile)
fprintf('Run title:\n%s\n', runTitle)
fprintf('Clumped file: %s\n', clumpedFile)
fprintf('Clumped rows: %d\n', height(clumped))
fprintf('Clumped fit config: %s\n', clumpedFitConfigFile)
fprintf('Dolomite environment offset: %.2f +/- %.2f C\n', dolomiteOffsetC, sigmaOffsetC)
fprintf('Clumped comparison season: %s\n', clumpedSeason)
fprintf('OPI temperature interpretation for clumped objective: %s\n', ...
    temperatureInterpretation)
fprintf('Number of primary samples: %d\n', nSamples)
fprintf('Number of free parameters: %d\n', nParametersFree)
fprintf('Sea-level T bounds state 1: %.1f, %.1f C\n', lB(3)-TC2K, uB(3)-TC2K)
fprintf('Sea-level T bounds state 2: %.1f, %.1f C\n', lB(13)-TC2K, uB(13)-TC2K)
fprintf('Grid spacing dx/dy km: %.2f, %.2f\n', ...
    (lon(2)-lon(1))*mPerDegree*1e-3*cosd(lat0), (lat(2)-lat(1))*mPerDegree*1e-3)

writeSolutionsOxygenOnly('initialize', runPath, runTitle, nSamples, ...
    parameterLabels, exponents, lB, uB);

isFit = true;
beta = fminCRS3(@(beta) ...
    calc_TwoWinds_OxygenClumpedObjective(beta, fC, hR, ...
    x, y, lat, lat0, hGrid, bMWLSample, ijCatch, ptrCatch, ...
    sampleD2H, sampleD18O, cov, nParametersFree, isFit, ...
    clumped, sampleLon, sampleLat, dolomiteOffsetC, sigmaOffsetC, clumpedSeason), ...
    lB, uB, mu, epsilon0, isParallel, @writeSolutionsOxygenOnly, solutions);

[chiR2Total, nuTotal, detail] = calc_TwoWinds_OxygenClumpedObjective(beta, fC, hR, ...
    x, y, lat, lat0, hGrid, bMWLSample, ijCatch, ptrCatch, ...
    sampleD2H, sampleD18O, cov, nParametersFree, false, ...
    clumped, sampleLon, sampleLat, dolomiteOffsetC, sigmaOffsetC, clumpedSeason);

fprintf('\nBest-fit combined objective:\n')
fprintf('chiR2Total: %.4f\n', chiR2Total)
fprintf('nuTotal: %d\n', nuTotal)
fprintf('chiR2OxygenOnlyPart: %.4f\n', detail.chiR2O)
fprintf('nuOxygenOnlyPart: %d\n', detail.nuO)
fprintf('clumped chi2: %.4f\n', detail.chi2T)
fprintf('clumped mean residual: %.3f C\n', detail.meanResidualT_C)
fprintf('clumped mean z: %.3f\n', detail.meanZT)
fprintf('Best-fit beta:\n')
fprintf('%.8g\t', beta)
fprintf('\n')

save(fullfile(runPath, 'opiFit_TwoWinds_OxygenClumped_BestFit.mat'), ...
    'beta', 'chiR2Total', 'nuTotal', 'detail', 'dolomiteOffsetC', ...
    'sigmaOffsetC', 'clumpedSeason', 'clumpedFile', ...
    'clumpedFitConfigFile', 'temperatureInterpretation', 'runFile', 'runTitle');

finishTime = datetime;
fprintf('Finish time: %s\n', finishTime)
fprintf('Elapsed time: %.2f hours\n', hours(finishTime - startTime))
diary off

end

function [dolomiteOffsetC, sigmaOffsetC, clumpedSeason, configFile] = ...
    readClumpedFitConfig(runPath, dataPath, dolomiteOffsetC, ...
    sigmaOffsetC, clumpedSeason)
configFile = "default hard-coded values";
candidateFiles = [
    string(fullfile(runPath, 'proxy_clumped', 'clumped_fit_config.csv'));
    string(fullfile(dataPath, 'proxy_clumped', 'clumped_fit_config.csv'))];

for i = 1:numel(candidateFiles)
    if ~isfile(candidateFiles(i))
        continue
    end
    T = readtable(candidateFiles(i), 'TextType', 'string');
    names = string(T.Properties.VariableNames);
    required = ["dolomiteOffsetC", "sigmaOffsetC", "clumpedSeason"];
    missing = setdiff(required, names);
    if ~isempty(missing)
        error('Missing clumped fit config column(s): %s', strjoin(missing, ', '));
    end
    if height(T) < 1
        error('Clumped fit config is empty: %s', candidateFiles(i));
    end
    dolomiteOffsetC = T.dolomiteOffsetC(1);
    sigmaOffsetC = T.sigmaOffsetC(1);
    clumpedSeason = string(T.clumpedSeason(1));
    configFile = candidateFiles(i);
    return
end
end
