function opiRun_TopographyRefitCases(caseName, rootScenario, experimentName, force)
% Run selected joint oxygen plus clumped-temperature topography refits.

if nargin < 1
    caseName = "";
end
if nargin < 2 || strlength(string(rootScenario)) == 0
    rootScenario = fullfile(fileparts(mfilename('fullpath')), '..', ...
        'scenarios', ...
        'Qiangtang_30Ma_4000m_dH045_T290_fP030_Valid_platform_northsouth');
end
if nargin < 3 || strlength(string(experimentName)) == 0
    experimentName = 'topography_sensitivity_clumped_band';
end
if nargin < 4
    force = false;
end
caseName = string(caseName);
rootScenario = char(string(rootScenario));
experimentName = char(string(experimentName));
refitRoot = fullfile(rootScenario, experimentName, 'refit_selected');
if ~isfolder(refitRoot)
    error('Refit directory not found: %s', refitRoot);
end

raw = dir(refitRoot);
caseDirs = raw([raw.isdir] & ~startsWith({raw.name}, '.'));
[~, order] = sort(lower({caseDirs.name}));
caseDirs = caseDirs(order);
if caseName ~= ""
    caseDirs = caseDirs(string({caseDirs.name}) == caseName);
    if isempty(caseDirs)
        error('Refit case not found: %s', caseName);
    end
end

analysisRoot = fullfile(rootScenario, experimentName, 'analysis');
if ~isfolder(analysisRoot)
    mkdir(analysisRoot);
end
statusFile = fullfile(analysisRoot, 'refit_case_status.csv');
fid = fopen(statusFile, 'w');
if fid == -1
    error('Could not create status file: %s', statusFile);
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, 'case_id,status,message\n');

rows = table();
for i = 1:numel(caseDirs)
    thisCase = string(caseDirs(i).name);
    caseDir = fullfile(caseDirs(i).folder, caseDirs(i).name);
    fprintf('\n===== Topography refit %d/%d: %s =====\n', ...
        i, numel(caseDirs), thisCase);
    try
        runFile = findPrimaryRun(caseDir);
        fitFile = fullfile(caseDir, ...
            'opiFit_TwoWinds_OxygenClumped_BestFit.mat');
        if isfile(fitFile) && ~force
            S = load(fitFile, 'chiR2Total', 'nuTotal', 'detail', 'beta');
            bestRun = ensureBestRunFile(runFile, fitFile);
            ensureCalcResult(bestRun, caseDir);
            row = makeSummaryRow(thisCase, S);
            rows = [rows; row]; %#ok<AGROW>
            fprintf(fid, '%s,complete,existing BestFit reused\n', thisCase);
            continue
        end
        opiFit_TwoWinds_OxygenClumped(runFile);
        if ~isfile(fitFile)
            error('Fit completed without BestFit output: %s', fitFile);
        end
        S = load(fitFile, 'chiR2Total', 'nuTotal', 'detail', 'beta');
        bestRun = ensureBestRunFile(runFile, fitFile);
        ensureCalcResult(bestRun, caseDir);
        row = makeSummaryRow(thisCase, S);
        rows = [rows; row]; %#ok<AGROW>
        fprintf(fid, '%s,complete,fit and BestFit output written\n', thisCase);
    catch ME
        fprintf(fid, '%s,failed,"%s"\n', thisCase, escapeCsv(ME.message));
        warning('Topography refit failed: %s\n%s', thisCase, getReport(ME));
    end
end

summaryFile = fullfile(analysisRoot, 'refit_summary.csv');
writetable(rows, summaryFile);
fprintf('\nWrote refit summary:\n%s\n', summaryFile);
fprintf('Wrote refit status:\n%s\n', statusFile);
if isempty(rows)
    error('No refit cases completed successfully.');
end
end

function row = makeSummaryRow(caseId, S)
[mode, targetHeight, pattern] = parseCaseId(caseId);
row = table(caseId, mode, targetHeight, pattern, S.chiR2Total, ...
    S.nuTotal, S.detail.chiR2O, S.detail.nuO, S.detail.chi2T, ...
    S.detail.meanResidualT_C, S.detail.meanZT, S.beta(4), ...
    S.beta(14), 'VariableNames', {'case_id', ...
    'normalization_mode', 'height_target_m', 'pattern_id', ...
    'chiR2_total', 'nu_total', 'chiR2_oxygen', 'nu_oxygen', ...
    'chi2_clumped', 'mean_clumped_residual_C', 'mean_clumped_z', ...
    'M1', 'M2'});
end

function bestRunFile = ensureBestRunFile(runFile, fitFile)
[runPath, runName, runExt] = fileparts(runFile);
bestRunFile = fullfile(runPath, [runName, '_Best', runExt]);
if isfile(bestRunFile)
    return
end
S = load(fitFile, 'beta', 'chiR2Total', 'nuTotal', 'detail', ...
    'dolomiteOffsetC', 'sigmaOffsetC', 'clumpedSeason');
text = fileread(runFile);
fid = fopen(bestRunFile, 'w', 'native', 'UTF-8');
if fid == -1
    error('Could not create best run file: %s', bestRunFile);
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '%s\n', strip(text));
fprintf(fid, '\n%%... Best-Fit Solution\n');
fprintf(fid, '%% Source: opiFit_TwoWinds_OxygenClumped_BestFit.mat\n');
fprintf(fid, '%% Combined reduced chi-square: %.6f\n', S.chiR2Total);
fprintf(fid, '%% Combined degrees of freedom: %d\n', S.nuTotal);
fprintf(fid, '%% Clumped residual mean: %.3f C\n', S.detail.meanResidualT_C);
fprintf(fid, '%% Dolomite environment offset: %.1f +/- %.1f C\n', ...
    S.dolomiteOffsetC, S.sigmaOffsetC);
fprintf(fid, '%% Clumped comparison season: %s\n', S.clumpedSeason);
fprintf(fid, '%% Best-fit parameters:\n');
fprintf(fid, '%.8g\t', S.beta);
fprintf(fid, '\n');
end

function ensureCalcResult(bestRunFile, caseDir)
resultFile = fullfile(caseDir, 'opiCalc_TwoWinds_OxygenOnly_Results.mat');
if ~isfile(resultFile)
    opiCalc_TwoWinds_OxygenOnly(bestRunFile);
end
end

function runFile = findPrimaryRun(caseDir)
files = dir(fullfile(caseDir, '*.run'));
files = files(~endsWith(string({files.name}), '_Best.run'));
if numel(files) ~= 1
    error('Expected one primary refit run in %s, found %d.', caseDir, numel(files));
end
runFile = fullfile(files(1).folder, files(1).name);
end

function [mode, targetHeight, pattern] = parseCaseId(caseId)
parts = split(string(caseId), '_');
mode = parts(1);
targetHeight = str2double(extractAfter(parts(2), 1));
pattern = join(parts(3:end-1), '_');
end

function text = escapeCsv(text)
text = replace(string(text), '"', '""');
end
