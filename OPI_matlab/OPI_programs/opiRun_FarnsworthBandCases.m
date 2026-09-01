function opiRun_FarnsworthBandCases(caseName, rootScenario, experimentName)
% Run fixed-parameter Farnsworth-inspired band cases and summarize d18O.

if nargin < 1 || strlength(string(caseName)) == 0
    caseName = "";
end
if nargin < 2 || strlength(string(rootScenario)) == 0
    rootScenario = fullfile(fileparts(mfilename('fullpath')), '..', ...
        'scenarios', ...
        'Qiangtang_30Ma_4000m_dH045_T290_fP030_Valid_platform_northsouth');
end
if nargin < 3 || strlength(string(experimentName)) == 0
    experimentName = 'topography_farnsworth_band';
end
rootScenario = char(string(rootScenario));
experimentName = char(string(experimentName));
calcRoot = fullfile(rootScenario, experimentName, 'calc_only');
analysisRoot = fullfile(rootScenario, experimentName, 'analysis');
if ~isfolder(calcRoot)
    error('Farnsworth band calc-only directory not found: %s', calcRoot);
end
if ~isfolder(analysisRoot)
    mkdir(analysisRoot);
end

raw = dir(calcRoot);
caseDirs = raw([raw.isdir] & ~startsWith({raw.name}, '.'));
[~, order] = sort(lower({caseDirs.name}));
caseDirs = caseDirs(order);
if strlength(string(caseName)) > 0
    caseDirs = caseDirs(string({caseDirs.name}) == string(caseName));
    if isempty(caseDirs)
        error('Case not found: %s', caseName);
    end
end

statusFile = fullfile(analysisRoot, 'case_status.csv');
statusId = fopen(statusFile, 'w');
if statusId == -1
    error('Could not create case status file.');
end
cleanup = onCleanup(@() fclose(statusId)); %#ok<NASGU>
fprintf(statusId, 'case_id,status,message\n');
rows = table();
for i = 1:numel(caseDirs)
    caseDir = fullfile(caseDirs(i).folder, caseDirs(i).name);
    caseId = string(caseDirs(i).name);
    fprintf('\n===== Farnsworth band case %d/%d: %s =====\n', ...
        i, numel(caseDirs), caseId);
    try
        bestRun = findBestRun(caseDir);
        opiCalc_TwoWinds_OxygenOnly(bestRun);
        resultFile = fullfile(caseDir, ...
            'opiCalc_TwoWinds_OxygenOnly_Results.mat');
        clumpedFile = fullfile(caseDir, 'proxy_clumped', ...
            'clumped_temperature.xlsx');
        if isfile(clumpedFile)
            opiCompare_ClumpedTemperature(resultFile, clumpedFile, ...
                fullfile(caseDir, 'proxy_clumped'));
        end
        rows = [rows; summarizeResult(resultFile, caseId)]; %#ok<AGROW>
        fprintf(statusId, '%s,complete,calc-only result written\n', caseId);
    catch ME
        fprintf(statusId, '%s,failed,"%s"\n', caseId, escapeCsv(ME.message));
        warning('Farnsworth case failed: %s\n%s', caseId, getReport(ME));
    end
end
if isempty(rows)
    error('No Farnsworth band cases completed.');
end
summaryFile = fullfile(analysisRoot, 'fixed_precipitation_d18O_summary.csv');
writetable(rows, summaryFile);
writeReport(rows, analysisRoot);
fprintf('\nWrote fixed-parameter precipitation summary:\n%s\n', summaryFile);
fprintf('Wrote case status:\n%s\n', statusFile);
end

function fileName = findBestRun(caseDir)
files = dir(fullfile(caseDir, '*_Best.run'));
if numel(files) ~= 1
    error('Expected one _Best.run in %s, found %d.', caseDir, numel(files));
end
fileName = fullfile(files(1).folder, files(1).name);
end

function row = summarizeResult(resultFile, caseId)
S = load(resultFile, 'lon', 'lat', 'd18OGrid', 'pGrid', ...
    'd18OPred', 'pSumPred', 'sampleD18O');
d = S.d18OGrid * 1e3;
p = S.pGrid;
valid = isfinite(d) & isfinite(p) & p > 0;
globalD = weightedMean(d(valid), p(valid));
north = S.lat(:) >= 32.5;
south = S.lat(:) <= 30.0;
northMask = north .* ones(1, numel(S.lon));
southMask = south .* ones(1, numel(S.lon));
northValid = valid & northMask;
southValid = valid & southMask;
northD = weightedMean(d(northValid), p(northValid));
southD = weightedMean(d(southValid), p(southValid));
sampleLon = 87.2;
sampleLat = 32.9;
[xx, yy] = meshgrid(S.lon, S.lat);
distanceKm = hypot((xx - sampleLon) .* 100.5, ...
    (yy - sampleLat) .* 111.1);
local = valid & distanceKm <= 50;
localD = weightedMean(d(local), p(local));
sampleValid = isfinite(S.d18OPred) & isfinite(S.pSumPred) & S.pSumPred > 0;
samplePred = mean(S.d18OPred(sampleValid), 'omitnan') * 1e3;
row = table(caseId, sum(valid(:)), sum(p(valid)), globalD, northD, ...
    localD, southD, northD - southD, std(d(valid), 1, 'omitnan'), ...
    northD + 13, samplePred, mean(S.sampleD18O, 'omitnan') * 1e3, ...
    'VariableNames', {'case_id', 'n_wet_nodes', 'total_precip', ...
    'weighted_d18O_global_permil', 'weighted_d18O_qiangtang_permil', ...
    'weighted_d18O_50km_permil', 'weighted_d18O_south_permil', ...
    'north_minus_south_permil', 'spatial_std_permil', ...
    'residual_vs_minus13_permil', 'sample_prediction_permil', ...
    'sample_observation_mean_permil'});
end

function value = weightedMean(values, weights)
if isempty(values) || sum(weights, 'omitnan') <= 0
    value = nan;
else
    value = sum(values .* weights, 'omitnan') / sum(weights, 'omitnan');
end
end

function writeReport(T, analysisRoot)
fid = fopen(fullfile(analysisRoot, 'fixed_precipitation_d18O_report.md'), 'w');
if fid == -1
    error('Could not create fixed-parameter report.');
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
[~, iTarget] = min(abs(T.residual_vs_minus13_permil));
[~, iLow] = min(T.weighted_d18O_qiangtang_permil);
[~, iHigh] = max(T.weighted_d18O_qiangtang_permil);
fprintf(fid, '# Farnsworth-Inspired Fixed-Parameter Precipitation d18O\n\n');
fprintf(fid, 'Completed cases: **%d**. All values below are from fixed `_Best.run` calculations.\n\n', height(T));
fprintf(fid, 'The primary target is the Qiangtang north-band precipitation-weighted d18O; the independent constraint is approximately -13 per mil.\n\n');
fprintf(fid, '- Closest Qiangtang value to -13 per mil: `%s` (%.3f per mil; residual %.3f).\n', ...
    T.case_id(iTarget), T.weighted_d18O_qiangtang_permil(iTarget), T.residual_vs_minus13_permil(iTarget));
fprintf(fid, '- Most negative Qiangtang value: `%s` (%.3f per mil).\n', T.case_id(iLow), T.weighted_d18O_qiangtang_permil(iLow));
fprintf(fid, '- Least negative Qiangtang value: `%s` (%.3f per mil).\n\n', T.case_id(iHigh), T.weighted_d18O_qiangtang_permil(iHigh));
fprintf(fid, 'The sample prediction column is diagnostic only: all 16 oxygen observations share one coordinate and therefore do not provide independent spatial discrimination.\n');
end

function text = escapeCsv(text)
text = replace(string(text), '"', '""');
end
