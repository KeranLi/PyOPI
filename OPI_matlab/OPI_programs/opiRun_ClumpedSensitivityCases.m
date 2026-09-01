function opiRun_ClumpedSensitivityCases(groupName, caseName)
% opiRun_ClumpedSensitivityCases runs self-contained clumped sensitivity cases.
%
% Usage
%   opiRun_ClumpedSensitivityCases
%       Run all supported groups.
%
%   opiRun_ClumpedSensitivityCases("parameter")
%       Run all local-parameter cases.
%
%   opiRun_ClumpedSensitivityCases("mechanism")
%       Run all mechanism-oriented local cases.
%
%   opiRun_ClumpedSensitivityCases("azimuth_fine")
%       Run all fine wind-azimuth calc-only cases.
%
%   opiRun_ClumpedSensitivityCases("az2_transition")
%       Run State 2 azimuth transition calc-only cases.
%
%   opiRun_ClumpedSensitivityCases("divide_calc_only")
%       Run all divide-shift calc-only cases.
%
%   opiRun_ClumpedSensitivityCases("divide", "south_020deg")
%       Run one divide-shift case only.

if nargin < 1
    groupName = "";
end
if nargin < 2
    caseName = "";
end

groupName = lower(string(groupName));
caseName = string(caseName);
rootScenario = getRootScenario();
groupDefs = buildGroupDefinitions(rootScenario);

if groupName ~= ""
    groupDefs = groupDefs([groupDefs.groupName] == groupName);
    if isempty(groupDefs)
        error('Unknown sensitivity group: %s', groupName);
    end
end

for iGroup = 1:numel(groupDefs)
    runGroup(groupDefs(iGroup), caseName);
end
end

function rootScenario = getRootScenario()
rootScenario = fullfile(fileparts(mfilename('fullpath')), '..', 'scenarios', ...
    'Qiangtang_30Ma_4000m_dH045_T290_fP030_Valid_platform_northsouth');
rootScenario = char(string(rootScenario));
end

function groupDefs = buildGroupDefinitions(rootScenario)
groupDefs = [
    struct('groupName', "parameter", ...
    'folderName', "sensitivity_parameter_local_clumped", ...
    'mode', "calc_only", ...
    'summaryPrefix', "parameter_local_clumped")
    struct('groupName', "mechanism", ...
    'folderName', "sensitivity_mechanism_local_clumped", ...
    'mode', "calc_only", ...
    'summaryPrefix', "mechanism_local_clumped")
    struct('groupName', "azimuth_fine", ...
    'folderName', "sensitivity_azimuth_fine_clumped", ...
    'mode', "calc_only", ...
    'summaryPrefix', "azimuth_fine_clumped")
    struct('groupName', "az2_transition", ...
    'folderName', "sensitivity_az2_transition_clumped", ...
    'mode', "calc_only", ...
    'summaryPrefix', "az2_transition_clumped")
    struct('groupName', "divide_calc_only", ...
    'folderName', "sensitivity_divide_calc_only_clumped", ...
    'mode', "calc_only", ...
    'summaryPrefix', "divide_calc_only_clumped")
    struct('groupName', "divide", ...
    'folderName', "sensitivity_divide_shift_clumped", ...
    'mode', "refit", ...
    'summaryPrefix', "divide_shift_clumped")
    struct('groupName', "proxy", ...
    'folderName', "sensitivity_proxy_clumped", ...
    'mode', "refit", ...
    'summaryPrefix', "proxy_clumped")
    ];

for i = 1:numel(groupDefs)
    groupDefs(i).groupDir = fullfile(rootScenario, groupDefs(i).folderName);
end
end

function runGroup(groupDef, caseName)
if ~isfolder(groupDef.groupDir)
    error('Sensitivity group folder not found: %s', groupDef.groupDir);
end

caseDirs = listCaseDirectories(groupDef.groupDir);
if caseName ~= ""
    names = string({caseDirs.name});
    caseDirs = caseDirs(names == caseName);
    if isempty(caseDirs)
        error('Case not found in group %s: %s', groupDef.groupName, caseName);
    end
end

fprintf('\n===== Running sensitivity group: %s =====\n', groupDef.groupName);
fprintf('Group folder: %s\n', groupDef.groupDir);
fprintf('Number of cases: %d\n', numel(caseDirs));

for iCase = 1:numel(caseDirs)
    caseDir = fullfile(caseDirs(iCase).folder, caseDirs(iCase).name);
    caseLabel = string(caseDirs(iCase).name);
    fprintf('\n----- Case %d/%d: %s -----\n', iCase, numel(caseDirs), caseLabel);
    runCase(caseDir, groupDef.mode);
end

writeGroupSummary(groupDef);
end

function caseDirs = listCaseDirectories(groupDir)
raw = dir(groupDir);
isKeep = [raw.isdir] & ~startsWith({raw.name}, '.');
caseDirs = raw(isKeep);
[~, order] = sort(lower({caseDirs.name}));
caseDirs = caseDirs(order);
end

function runCase(caseDir, mode)
runFile = findPrimaryRunFile(caseDir);
proxyClumpedDir = fullfile(caseDir, 'proxy_clumped');
clumpedFile = fullfile(proxyClumpedDir, 'clumped_temperature.xlsx');
if ~isfile(clumpedFile)
    error('Clumped temperature file not found: %s', clumpedFile);
end

switch mode
    case "calc_only"
        bestRunFile = findBestRunFile(caseDir);
        opiCalc_TwoWinds_OxygenOnly(bestRunFile);
    case "refit"
        opiFit_TwoWinds_OxygenClumped(runFile);
        fitFile = fullfile(caseDir, 'opiFit_TwoWinds_OxygenClumped_BestFit.mat');
        bestRunFile = writeBestRunFile(runFile, fitFile);
        opiCalc_TwoWinds_OxygenOnly(bestRunFile);
    otherwise
        error('Unsupported sensitivity mode: %s', mode);
end

matFile = fullfile(caseDir, 'opiCalc_TwoWinds_OxygenOnly_Results.mat');
opiCompare_ClumpedTemperature(matFile, clumpedFile, proxyClumpedDir);
opiMaps_TwoWinds(matFile);
end

function runFile = findPrimaryRunFile(caseDir)
files = dir(fullfile(caseDir, '*.run'));
isBest = endsWith(string({files.name}), "_Best.run");
files = files(~isBest);
if isempty(files)
    error('No primary run file found in case folder: %s', caseDir);
end
if numel(files) > 1
    error('Expected one primary run file in %s, found %d.', caseDir, numel(files));
end
runFile = fullfile(caseDir, files(1).name);
end

function bestRunFile = findBestRunFile(caseDir)
files = dir(fullfile(caseDir, '*_Best.run'));
if isempty(files)
    error('Best-run file not found in case folder: %s', caseDir);
end
if numel(files) > 1
    error('Expected one best-run file in %s, found %d.', caseDir, numel(files));
end
bestRunFile = fullfile(caseDir, files(1).name);
end

function bestRunFile = writeBestRunFile(runFile, fitFile)
S = load(fitFile, 'beta', 'chiR2Total', 'nuTotal', 'detail', ...
    'dolomiteOffsetC', 'sigmaOffsetC', 'clumpedSeason');
[runPath, runName, runExt] = fileparts(runFile);
bestRunFile = fullfile(runPath, [runName, '_Best', runExt]);

text = fileread(runFile);
fid = fopen(bestRunFile, 'w', 'native', 'UTF-8');
if fid == -1
    error('Could not create best-run file: %s', bestRunFile);
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

function writeGroupSummary(groupDef)
caseDirs = listCaseDirectories(groupDef.groupDir);
summaryRows = table();
sampleRows = table();

for iCase = 1:numel(caseDirs)
    caseDir = fullfile(caseDirs(iCase).folder, caseDirs(iCase).name);
    caseName = string(caseDirs(iCase).name);
    [summaryRow, sampleTable] = summarizeCase(groupDef, caseDir, caseName);
    summaryRows = [summaryRows; summaryRow]; %#ok<AGROW>
    sampleRows = [sampleRows; sampleTable]; %#ok<AGROW>
end

summaryFile = fullfile(groupDef.groupDir, 'summary_all_cases.csv');
sampleFile = fullfile(groupDef.groupDir, 'sample_metrics_all_cases.csv');
summaryNamedFile = fullfile(groupDef.groupDir, ...
    groupDef.summaryPrefix + "_summary_all_cases.csv");
sampleNamedFile = fullfile(groupDef.groupDir, ...
    groupDef.summaryPrefix + "_sample_metrics_all_cases.csv");

writetable(summaryRows, summaryFile);
writetable(sampleRows, sampleFile);
writetable(summaryRows, summaryNamedFile);
writetable(sampleRows, sampleNamedFile);

fprintf('\nWrote group summary:\n%s\n', summaryFile);
fprintf('Wrote sample summary:\n%s\n', sampleFile);
end

function [summaryRow, sampleTable] = summarizeCase(groupDef, caseDir, caseName)
runFile = findOptionalPrimaryRunFile(caseDir);
bestRunFile = findOptionalBestRunFile(caseDir);
calcFile = fullfile(caseDir, 'opiCalc_TwoWinds_OxygenOnly_Results.mat');
fitFile = fullfile(caseDir, 'opiFit_TwoWinds_OxygenClumped_BestFit.mat');
hrenFile = fullfile(caseDir, 'proxy_clumped', ...
    'clumped_temperature_HrenSheldon2012_comparison.csv');

summaryRow = table( ...
    string(caseName), string(groupDef.groupName), string(groupDef.mode), ...
    string(runFile), string(bestRunFile), false, false, ...
    'VariableNames', {'case_name', 'group_name', 'run_mode', ...
    'run_file', 'best_run_file', 'has_fit', 'has_calc'});
summaryRow = [summaryRow, emptyCalcSummaryTable(), ...
    emptyFitSummaryTable(), emptyHrenSummaryTable()];

sampleTable = table();

if isfile(calcFile)
    summaryRow.has_calc = true;
    [calcSummary, sampleTable] = summarizeCalcCase(calcFile, caseName);
    summaryRow{1, calcSummary.Properties.VariableNames} = calcSummary{1, :};
end

if isfile(fitFile)
    summaryRow.has_fit = true;
    fitSummary = summarizeFitCase(fitFile);
    summaryRow{1, fitSummary.Properties.VariableNames} = fitSummary{1, :};
end

if isfile(hrenFile)
    hrenSummary = summarizeHrenCase(hrenFile);
    summaryRow{1, hrenSummary.Properties.VariableNames} = hrenSummary{1, :};
end
end

function [summaryTable, sampleTable] = summarizeCalcCase(calcFile, caseName)
S = load(calcFile, 'beta', 'chiR2', 'nu', 'sampleLon', 'sampleLat', ...
    'sampleD18O', 'd18OPred', 'pSumPred', 'stdResiduals');

obs = S.sampleD18O(:) * 1e3;
pred = S.d18OPred(:) * 1e3;
resid = obs - pred;
wet = isfinite(obs) & isfinite(pred) & S.pSumPred(:) > 0;
stdResiduals = S.stdResiduals(:);

summaryTable = table( ...
    S.chiR2, S.nu, numel(obs), sum(wet), ...
    calcRmse(resid(wet)), mean(abs(resid(wet)), 'omitnan'), ...
    mean(resid(wet), 'omitnan'), max(abs(resid(wet)), [], 'omitnan'), ...
    mean(stdResiduals(wet), 'omitnan'), max(stdResiduals(wet), [], 'omitnan'), ...
    S.beta(1), S.beta(2), S.beta(3), S.beta(4), S.beta(9), S.beta(10), ...
    S.beta(11), S.beta(12), S.beta(13), S.beta(14), S.beta(19), ...
    'VariableNames', {'chiR2_oxygen', 'nu_oxygen', 'n_samples_total', ...
    'n_samples_wet', 'rmse_d18O_permille', 'mae_d18O_permille', ...
    'bias_d18O_permille', 'maxabs_d18O_permille', 'mean_std_residual', ...
    'max_std_residual', 'U1', 'Az1_deg', 'T0_1_K', 'M1', 'fP1', ...
    'fraction', 'U2', 'Az2_deg', 'T0_2_K', 'M2', 'fP2'});

sampleTable = table( ...
    repmat(string(caseName), numel(obs), 1), (1:numel(obs))', ...
    S.sampleLon(:), S.sampleLat(:), obs, pred, resid, wet, stdResiduals, ...
    'VariableNames', {'case_name', 'sample_index', 'sample_lon', ...
    'sample_lat', 'observed_d18O_permille', 'predicted_d18O_permille', ...
    'residual_d18O_permille', 'is_wet_sample', 'std_residual'});
end

function fitSummary = summarizeFitCase(fitFile)
S = load(fitFile, 'chiR2Total', 'nuTotal', 'detail', ...
    'dolomiteOffsetC', 'sigmaOffsetC', 'clumpedSeason');

fitSummary = table( ...
    S.chiR2Total, S.nuTotal, S.detail.chiR2O, S.detail.chi2T, ...
    S.detail.meanResidualT_C, S.detail.meanZT, ...
    S.dolomiteOffsetC, S.sigmaOffsetC, string(S.clumpedSeason), ...
    'VariableNames', {'chiR2_total_fit', 'nu_total_fit', ...
    'chiR2_oxygen_fit_component', 'chi2_temperature_fit_component', ...
    'clumped_mean_residual_C', 'clumped_mean_z', ...
    'dolomite_offset_C', 'sigma_offset_C', 'clumped_season'});
end

function calcSummary = emptyCalcSummaryTable()
calcSummary = table( ...
    nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, ...
    nan, nan, nan, nan, nan, nan, nan, nan, ...
    'VariableNames', {'chiR2_oxygen', 'nu_oxygen', 'n_samples_total', ...
    'n_samples_wet', 'rmse_d18O_permille', 'mae_d18O_permille', ...
    'bias_d18O_permille', 'maxabs_d18O_permille', 'mean_std_residual', ...
    'max_std_residual', 'U1', 'Az1_deg', 'T0_1_K', 'M1', 'fP1', ...
    'fraction', 'U2', 'Az2_deg', 'T0_2_K', 'M2', 'fP2'});
end

function fitSummary = emptyFitSummaryTable()
fitSummary = table( ...
    nan, nan, nan, nan, nan, nan, nan, nan, "", ...
    'VariableNames', {'chiR2_total_fit', 'nu_total_fit', ...
    'chiR2_oxygen_fit_component', 'chi2_temperature_fit_component', ...
    'clumped_mean_residual_C', 'clumped_mean_z', ...
    'dolomite_offset_C', 'sigma_offset_C', 'clumped_season'});
end

function hrenSummary = summarizeHrenCase(hrenFile)
T = readtable(hrenFile);
residName = 'residual_Tclumped_minus_OPI_Hren2012_Tw_warmest_C';
zName = 'z_Tclumped_minus_OPI_Hren2012_Tw_warmest';

if ~ismember(residName, T.Properties.VariableNames) || ...
        ~ismember(zName, T.Properties.VariableNames)
    hrenSummary = emptyHrenSummaryTable();
    return
end

resid = T.(residName);
z = T.(zName);
hrenSummary = table( ...
    mean(resid, 'omitnan'), max(abs(resid), [], 'omitnan'), ...
    mean(z, 'omitnan'), max(abs(z), [], 'omitnan'), ...
    'VariableNames', {'hren_warmest_mean_residual_C', ...
    'hren_warmest_maxabs_residual_C', 'hren_warmest_mean_z', ...
    'hren_warmest_maxabs_z'});
end

function hrenSummary = emptyHrenSummaryTable()
hrenSummary = table( ...
    nan, nan, nan, nan, ...
    'VariableNames', {'hren_warmest_mean_residual_C', ...
    'hren_warmest_maxabs_residual_C', 'hren_warmest_mean_z', ...
    'hren_warmest_maxabs_z'});
end

function value = calcRmse(x)
if isempty(x) || all(~isfinite(x))
    value = nan;
else
    value = sqrt(mean(x.^2, 'omitnan'));
end
end

function runFile = findOptionalPrimaryRunFile(caseDir)
files = dir(fullfile(caseDir, '*.run'));
if isempty(files)
    runFile = "";
    return
end
isBest = endsWith(string({files.name}), "_Best.run");
files = files(~isBest);
if isempty(files)
    runFile = "";
    return
end
runFile = string(fullfile(caseDir, files(1).name));
end

function bestRunFile = findOptionalBestRunFile(caseDir)
files = dir(fullfile(caseDir, '*_Best.run'));
if isempty(files)
    bestRunFile = "";
    return
end
bestRunFile = string(fullfile(caseDir, files(1).name));
end
