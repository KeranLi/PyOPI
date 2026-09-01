function opiAnalyze_TopographyPrecipD18O(rootScenario, experimentName)
% Compare reconstructed precipitation d18O fields across topography cases.

if nargin < 1 || strlength(string(rootScenario)) == 0
    rootScenario = fullfile(fileparts(mfilename('fullpath')), '..', ...
        'scenarios', ...
        'Qiangtang_30Ma_4000m_dH045_T290_fP030_Valid_platform_northsouth');
end
if nargin < 2 || strlength(string(experimentName)) == 0
    experimentName = 'topography_sensitivity_clumped_band';
end
rootScenario = char(string(rootScenario));
experimentName = char(string(experimentName));
experimentRoot = fullfile(rootScenario, experimentName);
analysisRoot = fullfile(experimentRoot, 'analysis');
if ~isfolder(analysisRoot)
    mkdir(analysisRoot);
end

calcRoot = fullfile(experimentRoot, 'calc_only');
refitRoot = fullfile(experimentRoot, 'refit_selected');
calcReference = fullfile(calcRoot, ...
    'Mfixed_H4150_P01_double_platform', ...
    'opiCalc_TwoWinds_OxygenOnly_Results.mat');
refitReference = fullfile(refitRoot, ...
    'Mfixed_H3000_P01_double_platform_R01', ...
    'opiCalc_TwoWinds_OxygenOnly_Results.mat');
if ~isfile(calcReference)
    error('Calc-only reference result not found: %s', calcReference);
end

calcBase = load(calcReference, 'd18OGrid');
calcBaseGrid = calcBase.d18OGrid;
if isfile(refitReference)
    refitBase = load(refitReference, 'd18OGrid');
    refitBaseGrid = refitBase.d18OGrid;
else
    refitBaseGrid = calcBaseGrid;
end

rows = table();
rows = appendCaseRows(rows, calcRoot, 'calc_only', calcBaseGrid);
if isfolder(refitRoot)
    rows = appendCaseRows(rows, refitRoot, 'refit_selected', refitBaseGrid);
end
if isempty(rows)
    error('No precipitation d18O result files were found.');
end

summaryFile = fullfile(analysisRoot, 'precipitation_d18O_case_summary.csv');
writetable(rows, summaryFile);

[G, stage, mode, pattern] = findgroups(rows.stage, ...
    rows.normalization_mode, rows.pattern_id);
patternSummary = table(stage, mode, pattern, splitapply(@numel, rows.case_id, G), ...
    splitapply(@mean, rows.weighted_d18O_permil, G), ...
    splitapply(@mean, rows.weighted_anomaly_vs_reference_permil, G), ...
    splitapply(@mean, rows.spatial_rms_anomaly_permil, G), ...
    splitapply(@mean, rows.sample_mean_d18O_permil, G), ...
    'VariableNames', {'stage', 'normalization_mode', 'pattern_id', ...
    'n_cases', 'mean_weighted_d18O_permil', ...
    'mean_weighted_anomaly_permil', 'mean_spatial_rms_anomaly_permil', ...
    'mean_sample_d18O_permil'});
writetable(patternSummary, fullfile(analysisRoot, ...
    'precipitation_d18O_pattern_summary.csv'));

[~, lowIdx] = min(rows.weighted_anomaly_vs_reference_permil);
[~, highIdx] = max(rows.weighted_anomaly_vs_reference_permil);
reportFile = fullfile(analysisRoot, 'precipitation_d18O_report.md');
fid = fopen(reportFile, 'w');
if fid == -1
    error('Could not create precipitation d18O report: %s', reportFile);
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '# Reconstructed Precipitation d18O Comparison\n\n');
fprintf(fid, 'The primary comparison uses precipitation-weighted `d18OGrid`, not reduced chi-square.\n\n');
fprintf(fid, 'Reference fields: calc-only `%s`; refit `%s`.\n\n', ...
    'Mfixed_H4150_P01_double_platform', ...
    'Mfixed_H3000_P01_double_platform_R01');
fprintf(fid, '## Case extremes\n\n');
fprintf(fid, '- Most negative weighted anomaly: `%s` (%.4f per mil).\n', ...
    rows.case_id(lowIdx), rows.weighted_anomaly_vs_reference_permil(lowIdx));
fprintf(fid, '- Most positive weighted anomaly: `%s` (%.4f per mil).\n\n', ...
    rows.case_id(highIdx), rows.weighted_anomaly_vs_reference_permil(highIdx));
fprintf(fid, 'Each row also contains the weighted field mean, unweighted field mean,\n');
fprintf(fid, 'spatial RMS anomaly, and the reconstructed sample d18O statistics.\n');
fprintf(fid, 'For refit cases, the d18O field is reconstructed from the joint-fit beta\n');
fprintf(fid, 'through the case-local `_Best.run` file.\n');

fprintf('Wrote reconstructed precipitation d18O analysis under:\n%s\n', analysisRoot);
end

function rows = appendCaseRows(rows, rootDir, stage, referenceGrid)
raw = dir(rootDir);
caseDirs = raw([raw.isdir] & ~startsWith({raw.name}, '.'));
[~, order] = sort(lower({caseDirs.name}));
caseDirs = caseDirs(order);
for i = 1:numel(caseDirs)
    caseDir = fullfile(caseDirs(i).folder, caseDirs(i).name);
    resultFile = fullfile(caseDir, 'opiCalc_TwoWinds_OxygenOnly_Results.mat');
    if ~isfile(resultFile)
        continue
    end
    S = load(resultFile, 'd18OGrid', 'pGrid', 'd18OPred', 'pSumPred');
    if ~isequal(size(S.d18OGrid), size(referenceGrid))
        error('Grid size mismatch in %s.', resultFile);
    end
    valid = isfinite(S.d18OGrid) & isfinite(S.pGrid) & S.pGrid > 0;
    p = S.pGrid(valid);
    d = S.d18OGrid(valid);
    weighted = sum(d .* p) / sum(p) * 1e3;
    unweighted = mean(d) * 1e3;
    delta = S.d18OGrid - referenceGrid;
    validDelta = valid & isfinite(referenceGrid);
    deltaValid = delta(validDelta);
    pDelta = S.pGrid(validDelta);
    weightedDelta = sum(deltaValid .* pDelta) / sum(pDelta) * 1e3;
    spatialRms = sqrt(mean(deltaValid.^2)) * 1e3;
    sampleValid = isfinite(S.d18OPred) & isfinite(S.pSumPred) & S.pSumPred > 0;
    sampleD = S.d18OPred(sampleValid) * 1e3;
    [mode, heightTarget, pattern] = parseCaseId(string(caseDirs(i).name), stage);
    row = table(string(caseDirs(i).name), string(stage), mode, heightTarget, ...
        pattern, sum(valid(:)), sum(p), weighted, unweighted, min(d)*1e3, ...
        max(d)*1e3, std(d, 1)*1e3, weightedDelta, spatialRms, ...
        mean(sampleD), sqrt(mean((sampleD - mean(sampleD)).^2)), ...
        'VariableNames', {'case_id', 'stage', 'normalization_mode', ...
        'height_target_m', 'pattern_id', 'n_wet_grid_nodes', ...
        'total_precip', 'weighted_d18O_permil', 'unweighted_d18O_permil', ...
        'min_d18O_permil', 'max_d18O_permil', 'std_d18O_permil', ...
        'weighted_anomaly_vs_reference_permil', 'spatial_rms_anomaly_permil', ...
        'sample_mean_d18O_permil', 'sample_spread_d18O_permil'});
    rows = [rows; row]; %#ok<AGROW>
end
end

function [mode, heightTarget, pattern] = parseCaseId(caseId, stage)
parts = split(caseId, '_');
mode = parts(1);
heightTarget = str2double(extractAfter(parts(2), 1));
if stage == "refit_selected"
    pattern = join(parts(3:end-1), '_');
else
    pattern = join(parts(3:end), '_');
end
end
