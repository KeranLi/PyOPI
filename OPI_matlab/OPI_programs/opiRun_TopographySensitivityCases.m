function opiRun_TopographySensitivityCases(caseName, rootScenario, experimentName)
% Run the calc-only topography cases using clumped-temperature best runs.
% This function does not fit parameters and does not touch refit_selected.

if nargin < 1
    caseName = "";
end
caseName = string(caseName);
if nargin < 2 || strlength(string(rootScenario)) == 0
    rootScenario = fullfile(fileparts(mfilename('fullpath')), '..', ...
        'scenarios', ...
        'Qiangtang_30Ma_4000m_dH045_T290_fP030_Valid_platform_northsouth');
end
if nargin < 3 || strlength(string(experimentName)) == 0
    experimentName = 'topography_sensitivity_clumped';
end
rootScenario = char(string(rootScenario));
experimentName = char(string(experimentName));
calcRoot = fullfile(rootScenario, experimentName, ...
    'calc_only');
analysisRoot = fullfile(rootScenario, experimentName, ...
    'analysis');
if ~isfolder(calcRoot)
    error('Topography calc-only directory not found: %s', calcRoot);
end
if ~isfolder(analysisRoot)
    mkdir(analysisRoot);
end

raw = dir(calcRoot);
caseDirs = raw([raw.isdir] & ~startsWith({raw.name}, '.'));
[~, order] = sort(lower({caseDirs.name}));
caseDirs = caseDirs(order);
if caseName ~= ""
    caseDirs = caseDirs(string({caseDirs.name}) == caseName);
    if isempty(caseDirs)
        error('Topography case not found: %s', caseName);
    end
end

statusFile = fullfile(analysisRoot, 'calc_only_case_status.csv');
fid = fopen(statusFile, 'w');
if fid == -1
    error('Could not create status file: %s', statusFile);
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, 'case_id,status,message\n');

rows = table();
for i = 1:numel(caseDirs)
    caseDir = fullfile(caseDirs(i).folder, caseDirs(i).name);
    thisCase = string(caseDirs(i).name);
    fprintf('\n===== Topography case %d/%d: %s =====\n', ...
        i, numel(caseDirs), thisCase);
    try
        bestRun = findSingleFile(caseDir, '*_Best.run', 'best run');
        clumpedFile = fullfile(caseDir, 'proxy_clumped', ...
            'clumped_temperature.xlsx');
        if ~isfile(clumpedFile)
            error('Missing case clumped-temperature file: %s', clumpedFile);
        end
        opiCalc_TwoWinds_OxygenOnly(bestRun);
        resultFile = fullfile(caseDir, ...
            'opiCalc_TwoWinds_OxygenOnly_Results.mat');
        opiCompare_ClumpedTemperature(resultFile, clumpedFile, ...
            fullfile(caseDir, 'proxy_clumped'));
        opiMaps_TwoWinds(resultFile);
        row = summarizeCase(resultFile, thisCase);
        rows = [rows; row]; %#ok<AGROW>
        fprintf(fid, '%s,complete,calc and diagnostics written\n', thisCase);
    catch ME
        fprintf(fid, '%s,failed,"%s"\n', thisCase, escapeCsv(ME.message));
        warning('Topography case failed: %s\n%s', thisCase, getReport(ME));
    end
end

summaryFile = fullfile(analysisRoot, 'calc_only_summary_all_cases.csv');
writetable(rows, summaryFile);
fprintf('\nWrote calc-only summary:\n%s\n', summaryFile);
fprintf('Wrote case status:\n%s\n', statusFile);
if isempty(rows)
    error('No topography cases completed successfully.');
end
end

function fileName = findSingleFile(caseDir, pattern, label)
files = dir(fullfile(caseDir, pattern));
if numel(files) ~= 1
    error('Expected one %s in %s, found %d.', label, caseDir, numel(files));
end
fileName = fullfile(files(1).folder, files(1).name);
end

function row = summarizeCase(resultFile, caseId)
S = load(resultFile, 'beta', 'chiR2', 'nu', 'sampleLon', 'sampleLat', ...
    'sampleD18O', 'd18OPred', 'pSumPred', 'stdResiduals', ...
    'pGrid', 'd18OGrid', 'elevationPred');
obs = S.sampleD18O(:) * 1e3;
pred = S.d18OPred(:) * 1e3;
resid = obs - pred;
wet = isfinite(obs) & isfinite(pred) & S.pSumPred(:) > 0;
[mode, targetHeight, pattern] = parseCaseId(caseId);
row = table(caseId, mode, targetHeight, pattern, ...
    S.chiR2, S.nu, sum(isfinite(obs)), sum(wet), ...
    sqrt(mean(resid(wet).^2, 'omitnan')), mean(abs(resid(wet)), 'omitnan'), ...
    mean(resid(wet), 'omitnan'), max(abs(resid(wet)), [], 'omitnan'), ...
    mean(S.stdResiduals(wet), 'omitnan'), max(S.stdResiduals(wet), [], 'omitnan'), ...
    mean(S.elevationPred(wet), 'omitnan'), max(S.elevationPred(wet), [], 'omitnan'), ...
    S.beta(1), S.beta(2), S.beta(3), S.beta(4), S.beta(9), S.beta(10), ...
    S.beta(11), S.beta(12), S.beta(13), S.beta(14), S.beta(19), ...
    'VariableNames', {'case_id', 'normalization_mode', 'height_target_m', ...
    'pattern_id', 'chiR2_oxygen', 'nu_oxygen', 'n_samples_total', ...
    'n_samples_wet', 'rmse_d18O_permille', 'mae_d18O_permille', ...
    'bias_d18O_permille', 'maxabs_d18O_permille', 'mean_std_residual', ...
    'max_std_residual', 'mean_sample_elevation_m', 'max_sample_elevation_m', ...
    'U1', 'Az1_deg', 'T0_1_K', 'M1', 'fP1', 'fraction', 'U2', ...
    'Az2_deg', 'T0_2_K', 'M2', 'fP2'});
end

function [mode, targetHeight, pattern] = parseCaseId(caseId)
parts = split(caseId, '_');
mode = parts(1);
heightToken = parts(2);
targetHeight = str2double(extractAfter(heightToken, 1));
pattern = join(parts(3:end), '_');
end

function text = escapeCsv(text)
text = replace(string(text), '"', '""');
end
