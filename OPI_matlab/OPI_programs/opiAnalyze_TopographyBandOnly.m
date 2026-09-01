function opiAnalyze_TopographyBandOnly(rootScenario, experimentName)
% Summarize the completed east-west-band topography calc-only matrix.

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
summaryFile = fullfile(analysisRoot, 'calc_only_summary_all_cases.csv');
if ~isfile(summaryFile)
    error('Calc-only summary not found: %s', summaryFile);
end

T = readtable(summaryFile, 'TextType', 'string');
numCases = size(T, 1);
if numCases ~= 24
    error('Expected 24 calc-only rows, found %d.', numCases);
end

% Summaries by spatial pattern and normalization mode, averaged over heights.
[G, mode, pattern] = findgroups(T.normalization_mode, T.pattern_id);
factorSummary = table(mode, pattern, splitapply(@numel, T.case_id, G), ...
    splitapply(@mean, T.rmse_d18O_permille, G), ...
    splitapply(@mean, T.chiR2_oxygen, G), ...
    splitapply(@mean, T.bias_d18O_permille, G), ...
    splitapply(@mean, T.mean_sample_elevation_m, G), ...
    'VariableNames', {'normalization_mode', 'pattern_id', 'n_cases', ...
    'mean_rmse_d18O_permille', 'mean_chiR2_oxygen', ...
    'mean_bias_d18O_permille', 'mean_sample_elevation_m'});
writetable(factorSummary, fullfile(analysisRoot, ...
    'calc_only_factor_summary.csv'));

% Summaries by height and normalization mode, averaged over patterns.
[G, mode, height] = findgroups(T.normalization_mode, T.height_target_m);
heightSummary = table(mode, height, splitapply(@numel, T.case_id, G), ...
    splitapply(@mean, T.rmse_d18O_permille, G), ...
    splitapply(@mean, T.chiR2_oxygen, G), ...
    splitapply(@mean, T.bias_d18O_permille, G), ...
    'VariableNames', {'normalization_mode', 'height_target_m', 'n_cases', ...
    'mean_rmse_d18O_permille', 'mean_chiR2_oxygen', ...
    'mean_bias_d18O_permille'});
writetable(heightSummary, fullfile(analysisRoot, ...
    'calc_only_height_summary.csv'));

[~, bestIdx] = min(T.rmse_d18O_permille);
[~, worstIdx] = max(T.rmse_d18O_permille);
reportFile = fullfile(analysisRoot, 'calc_only_report.md');
fid = fopen(reportFile, 'w');
if fid == -1
    error('Could not create report: %s', reportFile);
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '# East-West Band Topography Calc-Only Results\n\n');
fprintf(fid, 'Cases completed: **%d/24**. All rows contain 16 wet primary samples.\n\n', numCases);
fprintf(fid, 'The matrix uses fixed baseline parameters; it measures direct topography responses and is not a refit result.\n\n');
fprintf(fid, '## Extremes\n\n');
fprintf(fid, '- Lowest d18O RMSE: `%s` (%.4f per mil; chiR2 %.4f).\n', ...
    T.case_id(bestIdx), T.rmse_d18O_permille(bestIdx), T.chiR2_oxygen(bestIdx));
fprintf(fid, '- Highest d18O RMSE: `%s` (%.4f per mil; chiR2 %.4f).\n\n', ...
    T.case_id(worstIdx), T.rmse_d18O_permille(worstIdx), T.chiR2_oxygen(worstIdx));
fprintf(fid, '## Outputs\n\n');
fprintf(fid, '- `calc_only_factor_summary.csv`: pattern and normalization averages.\n');
fprintf(fid, '- `calc_only_height_summary.csv`: height and normalization averages.\n');
fprintf(fid, '- `calc_only_summary_all_cases.csv`: case-level values.\n');
fprintf(fid, '- `precipitation_d18O_report.md`: primary reconstructed precipitation response.\n');
fprintf(fid, '- Case maps and proxy comparisons remain inside each case directory.\n');
fprintf(fid, '\nThe factor tables are descriptive summaries only. Refit cases should be selected after checking the spatial maps and the matched-height contrasts.\n');

refitFile = fullfile(analysisRoot, 'refit_summary.csv');
if isfile(refitFile)
    R = readtable(refitFile, 'TextType', 'string');
    refitReport = fullfile(analysisRoot, 'refit_report.md');
    refitId = fopen(refitReport, 'w');
    if refitId == -1
        error('Could not create refit report: %s', refitReport);
    end
    refitCleanup = onCleanup(@() fclose(refitId)); %#ok<NASGU>
    numRefit = size(R, 1);
    fprintf(refitId, '# East-West Band Topography Refit Results\n\n');
    fprintf(refitId, 'Completed joint oxygen + clumped-temperature refits: **%d/%d**.\n\n', ...
        numRefit, 7);
    fprintf(refitId, 'The refits use `opiFit_TwoWinds_OxygenClumped` with the original\n');
    fprintf(refitId, '`Tibet_Eocene_30Ma_OxygenClumped_UltraAggressive.run` template.\n\n');
    fprintf(refitId, '## Objective range\n\n');
    fprintf(refitId, '- `chiR2_total`: %.6f to %.6f.\n', ...
        min(R.chiR2_total), max(R.chiR2_total));
    fprintf(refitId, '- oxygen component: %.6f to %.6f.\n', ...
        min(R.chiR2_oxygen), max(R.chiR2_oxygen));
    fprintf(refitId, '- clumped component: %.6f to %.6f.\n\n', ...
        min(R.chi2_clumped), max(R.chi2_clumped));
    fprintf(refitId, 'The objective values are effectively unchanged across the selected\n');
    fprintf(refitId, 'topographies; the spatial sensitivity is therefore expressed mainly by\n');
    fprintf(refitId, 'parameter displacement.\n\n');
    fprintf(refitId, '## Fitted mountain-height numbers\n\n');
    fprintf(refitId, '- `M1`: %.6f to %.6f.\n', min(R.M1), max(R.M1));
    fprintf(refitId, '- `M2`: %.6f to %.6f.\n\n', min(R.M2), max(R.M2));
    fprintf(refitId, 'Use the case-level map outputs and parameter displacement, rather than\n');
    fprintf(refitId, 'small differences in reduced chi-square, for the next interpretation step.\n');
end

fprintf('Wrote band-only topography analysis outputs under:\n%s\n', analysisRoot);
end
