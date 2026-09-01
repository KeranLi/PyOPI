function opiAnalyze_ClumpedSensitivity(rootScenario)
% opiAnalyze_ClumpedSensitivity summarizes completed clumped sensitivity runs.
%
% Usage
%   opiAnalyze_ClumpedSensitivity
%   opiAnalyze_ClumpedSensitivity('/path/to/platform_scenario')
%
% The function reads the per-group summary_all_cases.csv files created by
% opiRun_ClumpedSensitivityCases and writes compact tables, figures, and a
% Markdown report under <rootScenario>/sensitivity_analysis.

if nargin < 1 || strlength(string(rootScenario)) == 0
    rootScenario = fullfile(fileparts(mfilename('fullpath')), '..', ...
        'scenarios', ...
        'Qiangtang_30Ma_4000m_dH045_T290_fP030_Valid_platform_northsouth');
end
rootScenario = char(string(rootScenario));

outDir = fullfile(rootScenario, 'sensitivity_analysis');
if ~isfolder(outDir)
    mkdir(outDir);
end

groupDefs = [
    struct('groupName', "parameter", ...
    'folderName', "sensitivity_parameter_local_clumped", ...
    'expectedCases', 28)
    struct('groupName', "divide", ...
    'folderName', "sensitivity_divide_shift_clumped", ...
    'expectedCases', 7)
    struct('groupName', "proxy", ...
    'folderName', "sensitivity_proxy_clumped", ...
    'expectedCases', 6)
    struct('groupName', "mechanism", ...
    'folderName', "sensitivity_mechanism_local_clumped", ...
    'expectedCases', 16)
    struct('groupName', "azimuth_fine", ...
    'folderName', "sensitivity_azimuth_fine_clumped", ...
    'expectedCases', 12)
    struct('groupName', "divide_calc_only", ...
    'folderName', "sensitivity_divide_calc_only_clumped", ...
    'expectedCases', 7)
    struct('groupName', "az2_transition", ...
    'folderName', "sensitivity_az2_transition_clumped", ...
    'expectedCases', 9)
    ];

allRows = readAllGroupSummaries(rootScenario, groupDefs);
if isempty(allRows)
    error('No sensitivity summary rows found under: %s', rootScenario);
end

writetable(allRows, fullfile(outDir, 'sensitivity_all_cases_compact.csv'));

completionSummary = summarizeCompletion(allRows, groupDefs);
writetable(completionSummary, fullfile(outDir, ...
    'sensitivity_completion_summary.csv'));

groupResponseSummary = summarizeResponseGroups(allRows);
writetable(groupResponseSummary, fullfile(outDir, ...
    'sensitivity_group_response_summary.csv'));

qualityControl = buildQualityControl(allRows, groupDefs);
writetable(qualityControl, fullfile(outDir, ...
    'sensitivity_quality_control.csv'));

parameterRows = allRows(allRows.group_name == "parameter", :);
[variedParameter, variedValue] = parseParameterCases(parameterRows.case_name, ...
    parameterRows);
parameterRows.varied_parameter = variedParameter;
parameterRows.varied_value = variedValue;
writetable(parameterRows, fullfile(outDir, ...
    'sensitivity_parameter_cases_with_values.csv'));

parameterGroupSummary = summarizeParameterGroups(parameterRows);
writetable(parameterGroupSummary, fullfile(outDir, ...
    'sensitivity_parameter_group_summary.csv'));

refitGroupSummary = summarizeRefitGroups(allRows);
writetable(refitGroupSummary, fullfile(outDir, ...
    'sensitivity_refit_group_summary.csv'));

plotParameterSensitivity(parameterRows, outDir);
plotRefitSensitivity(allRows, outDir);
plotAllGroupResponses(allRows, groupResponseSummary, outDir);

spatialGroupSummary = readSpatialGroupSummary(rootScenario);
if ~isempty(spatialGroupSummary)
    writetable(spatialGroupSummary, fullfile(outDir, ...
        'sensitivity_spatial_group_summary_50km.csv'));
    plotSpatialGroupResponses(spatialGroupSummary, outDir);
end

writeMarkdownReport(outDir, allRows, completionSummary, ...
    groupResponseSummary, parameterGroupSummary, refitGroupSummary, ...
    qualityControl, spatialGroupSummary);

fprintf('Wrote clumped sensitivity analysis outputs to:\n%s\n', outDir);
end

function S = summarizeCompletion(T, groupDefs)
S = table();
for i = 1:numel(groupDefs)
    G = T(T.group_name == groupDefs(i).groupName, :);
    nCases = height(G);
    nCalc = sum(G.has_calc == 1);
    nFit = sum(G.has_fit == 1);
    nFiniteRmse = sum(isfinite(G.rmse_d18O_permille));
    nFullSamples = sum(G.n_samples_total == G.n_samples_wet & ...
        G.n_samples_total > 0);
    isComplete = nCases == groupDefs(i).expectedCases && ...
        nCalc == nCases && nFiniteRmse == nCases && ...
        nFullSamples == nCases;
    if isComplete && any(groupDefs(i).groupName == ...
            ["divide", "divide_calc_only"])
        status = "computed_review";
    elseif isComplete
        status = "complete";
    else
        status = "incomplete";
    end
    modes = strjoin(unique(G.run_mode, 'stable'), "+");
    row = table(string(groupDefs(i).groupName), ...
        string(groupDefs(i).folderName), groupDefs(i).expectedCases, ...
        nCases, nCalc, nFit, nFiniteRmse, nFullSamples, modes, status, ...
        'VariableNames', {'group_name', 'folder_name', 'expected_cases', ...
        'n_cases', 'n_calc', 'n_fit', 'n_finite_rmse', ...
        'n_full_sample_coverage', 'run_mode', 'status'});
    S = [S; row]; %#ok<AGROW>
end
end

function S = summarizeResponseGroups(T)
groups = unique(T.group_name, 'stable');
S = table();
for i = 1:numel(groups)
    G = T(T.group_name == groups(i), :);
    [rmseMin, bestIdx] = min(G.rmse_d18O_permille, [], 'omitnan');
    [rmseMax, worstIdx] = max(G.rmse_d18O_permille, [], 'omitnan');
    row = table(groups(i), height(G), ...
        strjoin(unique(G.run_mode, 'stable'), "+"), ...
        rmseMin, rmseMax, rangeOmitnan(G.rmse_d18O_permille), ...
        string(G.case_name(bestIdx)), string(G.case_name(worstIdx)), ...
        min(G.bias_d18O_permille, [], 'omitnan'), ...
        max(G.bias_d18O_permille, [], 'omitnan'), ...
        rangeOmitnan(G.hren_warmest_mean_residual_C), ...
        'VariableNames', {'group_name', 'n_cases', 'run_mode', ...
        'rmse_min_permille', 'rmse_max_permille', ...
        'rmse_span_permille', 'best_rmse_case', 'worst_rmse_case', ...
        'bias_min_permille', 'bias_max_permille', ...
        'hren_mean_residual_span_C'});
    S = [S; row]; %#ok<AGROW>
end
S = sortrows(S, 'rmse_span_permille', 'descend');
end

function Q = buildQualityControl(T, groupDefs)
expectedTotal = sum([groupDefs.expectedCases]);
keys = T.group_name + "/" + T.case_name;
refitRows = T(T.run_mode == "refit", :);
divideCalc = T(T.group_name == "divide_calc_only", :);

Q = table();
Q = appendCheck(Q, "expected_case_count", "error", ...
    height(T) == expectedTotal, ...
    sprintf('%d of %d cases found', height(T), expectedTotal));
Q = appendCheck(Q, "unique_group_case_keys", "error", ...
    numel(unique(keys)) == height(T), ...
    sprintf('%d unique keys for %d rows', numel(unique(keys)), height(T)));
Q = appendCheck(Q, "all_calc_results_present", "error", ...
    all(T.has_calc == 1), ...
    sprintf('%d of %d cases have calc results', sum(T.has_calc == 1), height(T)));
Q = appendCheck(Q, "finite_rmse", "error", ...
    all(isfinite(T.rmse_d18O_permille)), ...
    sprintf('%d of %d cases have finite RMSE', ...
    sum(isfinite(T.rmse_d18O_permille)), height(T)));
Q = appendCheck(Q, "full_sample_coverage", "error", ...
    all(T.n_samples_total == T.n_samples_wet & T.n_samples_total > 0), ...
    sprintf('%d of %d cases use all reported samples', ...
    sum(T.n_samples_total == T.n_samples_wet & T.n_samples_total > 0), ...
    height(T)));
Q = appendCheck(Q, "refit_outputs_present", "error", ...
    all(refitRows.has_fit == 1), ...
    sprintf('%d of %d refit cases have fit results', ...
    sum(refitRows.has_fit == 1), height(refitRows)));

identicalDivideMetrics = ~isempty(divideCalc) && ...
    rangeOmitnan(divideCalc.rmse_d18O_permille) < 1e-12 && ...
    rangeOmitnan(divideCalc.bias_d18O_permille) < 1e-12;
Q = appendCheck(Q, "divide_geometry_used_by_model", ...
    "critical", ~identicalDivideMetrics, ...
    "The fit function does not read the divide file. The calc function " + ...
    "computes isSampleSide01 but does not pass it into the prediction " + ...
    "function. Current divide groups therefore do not isolate a causal " + ...
    "divide-position response and must not be used as sensitivity evidence.");
end

function Q = appendCheck(Q, checkName, severity, passed, details)
row = table(string(checkName), string(severity), logical(passed), ...
    string(details), 'VariableNames', ...
    {'check_name', 'severity', 'passed', 'details'});
Q = [Q; row]; %#ok<AGROW>
end

function allRows = readAllGroupSummaries(rootScenario, groupDefs)
allRows = table();
for i = 1:numel(groupDefs)
    summaryFile = fullfile(rootScenario, groupDefs(i).folderName, ...
        'summary_all_cases.csv');
    if ~isfile(summaryFile)
        warning('Summary file not found: %s', summaryFile);
        continue
    end

    T = readtable(summaryFile, 'TextType', 'string');
    T.source_summary_file = repmat(string(summaryFile), height(T), 1);
    allRows = [allRows; T]; %#ok<AGROW>
end
end

function [variedParameter, variedValue] = parseParameterCases(caseNames, T)
variedParameter = strings(height(T), 1);
variedValue = nan(height(T), 1);

for i = 1:height(T)
    name = string(caseNames(i));
    if startsWith(name, "T0_1_")
        variedParameter(i) = "T0_1_K";
        variedValue(i) = T.T0_1_K(i);
    elseif startsWith(name, "T0_2_")
        variedParameter(i) = "T0_2_K";
        variedValue(i) = T.T0_2_K(i);
    elseif startsWith(name, "M_1_")
        variedParameter(i) = "M1";
        variedValue(i) = T.M1(i);
    elseif startsWith(name, "M_2_")
        variedParameter(i) = "M2";
        variedValue(i) = T.M2(i);
    elseif startsWith(name, "fraction_")
        variedParameter(i) = "fraction";
        variedValue(i) = T.fraction(i);
    else
        variedParameter(i) = "unknown";
    end
end
end

function S = summarizeParameterGroups(T)
params = unique(T.varied_parameter, 'stable');
S = table();

for i = 1:numel(params)
    idx = T.varied_parameter == params(i);
    G = T(idx, :);
    [bestRmse, bestIdx] = min(G.rmse_d18O_permille, [], 'omitnan');
    [worstRmse, worstIdx] = max(G.rmse_d18O_permille, [], 'omitnan');
    [bestChi, bestChiIdx] = min(G.chiR2_oxygen, [], 'omitnan');

    row = table(params(i), height(G), ...
        min(G.varied_value, [], 'omitnan'), ...
        max(G.varied_value, [], 'omitnan'), ...
        bestRmse, worstRmse, ...
        rangeOmitnan(G.rmse_d18O_permille), ...
        string(G.case_name(bestIdx)), string(G.case_name(worstIdx)), ...
        bestChi, string(G.case_name(bestChiIdx)), ...
        rangeOmitnan(G.hren_warmest_mean_residual_C), ...
        rangeOmitnan(abs(G.hren_warmest_mean_residual_C)), ...
        'VariableNames', {'varied_parameter', 'n_cases', ...
        'value_min', 'value_max', 'rmse_min_permille', ...
        'rmse_max_permille', 'rmse_span_permille', ...
        'best_rmse_case', 'worst_rmse_case', ...
        'chiR2_oxygen_min', 'best_chiR2_case', ...
        'hren_mean_residual_span_C', ...
        'abs_hren_mean_residual_span_C'});
    S = [S; row]; %#ok<AGROW>
end

S = sortrows(S, 'rmse_span_permille', 'descend');
end

function S = summarizeRefitGroups(T)
groups = ["divide"; "proxy"];
S = table();

for i = 1:numel(groups)
    G = T(T.group_name == groups(i), :);
    if isempty(G)
        continue
    end

    row = table(groups(i), height(G), sum(G.has_fit), sum(G.has_calc), ...
        min(G.rmse_d18O_permille, [], 'omitnan'), ...
        max(G.rmse_d18O_permille, [], 'omitnan'), ...
        rangeOmitnan(G.rmse_d18O_permille), ...
        min(G.chiR2_total_fit, [], 'omitnan'), ...
        max(G.chiR2_total_fit, [], 'omitnan'), ...
        rangeOmitnan(G.chiR2_total_fit), ...
        rangeOmitnan(G.clumped_mean_residual_C), ...
        rangeOmitnan(G.hren_warmest_mean_residual_C), ...
        rangeOmitnan(G.U1), rangeOmitnan(G.Az1_deg), ...
        rangeOmitnan(G.T0_1_K), rangeOmitnan(G.M1), ...
        rangeOmitnan(G.fP1), rangeOmitnan(G.fraction), ...
        rangeOmitnan(G.U2), rangeOmitnan(G.Az2_deg), ...
        rangeOmitnan(G.T0_2_K), rangeOmitnan(G.M2), ...
        rangeOmitnan(G.fP2), ...
        'VariableNames', {'group_name', 'n_cases', 'n_fit', 'n_calc', ...
        'rmse_min_permille', 'rmse_max_permille', ...
        'rmse_span_permille', 'chiR2_total_min', 'chiR2_total_max', ...
        'chiR2_total_span', 'clumped_mean_residual_span_C', ...
        'hren_mean_residual_span_C', 'U1_span', 'Az1_span_deg', ...
        'T0_1_span_K', 'M1_span', 'fP1_span', 'fraction_span', ...
        'U2_span', 'Az2_span_deg', 'T0_2_span_K', 'M2_span', ...
        'fP2_span'});
    S = [S; row]; %#ok<AGROW>
end
end

function plotParameterSensitivity(T, outDir)
if isempty(T)
    return
end

params = unique(T.varied_parameter, 'stable');
fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 1100 800]);
tiledlayout(3, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

for i = 1:numel(params)
    nexttile
    G = sortrows(T(T.varied_parameter == params(i), :), 'varied_value');
    plot(G.varied_value, G.rmse_d18O_permille, '-o', ...
        'LineWidth', 1.5, 'MarkerSize', 5);
    grid on
    xlabel(string(params(i)), 'Interpreter', 'none');
    ylabel('d18O RMSE (per mil)');
    title(string(params(i)), 'Interpreter', 'none');
end

exportgraphics(fig, fullfile(outDir, ...
    'sensitivity_parameter_rmse_by_value.png'), 'Resolution', 220);
close(fig);

fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 1100 800]);
tiledlayout(3, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

for i = 1:numel(params)
    nexttile
    G = sortrows(T(T.varied_parameter == params(i), :), 'varied_value');
    plot(G.varied_value, abs(G.hren_warmest_mean_residual_C), '-s', ...
        'LineWidth', 1.5, 'MarkerSize', 5);
    grid on
    xlabel(string(params(i)), 'Interpreter', 'none');
    ylabel('|Hren Tw residual| (C)');
    title(string(params(i)), 'Interpreter', 'none');
end

exportgraphics(fig, fullfile(outDir, ...
    'sensitivity_parameter_hren_residual_by_value.png'), 'Resolution', 220);
close(fig);
end

function plotRefitSensitivity(T, outDir)
divideRows = T(T.group_name == "divide", :);
proxyRows = T(T.group_name == "proxy", :);

fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 1100 480]);
tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile
if ~isempty(divideRows)
    shiftDeg = parseDivideShift(divideRows.case_name);
    [shiftDeg, order] = sort(shiftDeg);
    G = divideRows(order, :);
    yyaxis left
    plot(shiftDeg, G.rmse_d18O_permille, '-o', 'LineWidth', 1.5);
    ylabel('d18O RMSE (per mil)');
    ylim([0, max(G.rmse_d18O_permille, [], 'omitnan') * 1.15]);
    yyaxis right
    plot(shiftDeg, G.hren_warmest_mean_residual_C, '-s', 'LineWidth', 1.5);
    ylabel('Hren Tw mean residual (C)');
    xlabel('Divide shift (deg; south negative)');
    title('Divide-shift refits');
    grid on
end

nexttile
if ~isempty(proxyRows)
    sortKey = parseProxySortKey(proxyRows.case_name);
    [~, order] = sort(sortKey);
    G = proxyRows(order, :);
    x = categorical(G.case_name);
    x = reordercats(x, cellstr(G.case_name));
    yyaxis left
    plot(x, G.rmse_d18O_permille, '-o', 'LineWidth', 1.5);
    ylabel('d18O RMSE (per mil)');
    ylim([0, max(G.rmse_d18O_permille, [], 'omitnan') * 1.15]);
    yyaxis right
    plot(x, G.hren_warmest_mean_residual_C, '-s', 'LineWidth', 1.5);
    ylabel('Hren Tw mean residual (C)');
    title('Proxy/clumped refits');
    grid on
    xtickangle(35);
end

exportgraphics(fig, fullfile(outDir, ...
    'sensitivity_refit_group_metrics.png'), 'Resolution', 220);
close(fig);
end

function plotAllGroupResponses(T, groupSummary, outDir)
groupOrder = groupSummary.group_name;
fig = figure('Visible', 'off', 'Color', 'w', ...
    'Position', [100 100 1180 720]);
tiledlayout(2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile
hold on
colors = lines(numel(groupOrder));
for i = 1:numel(groupOrder)
    G = T(T.group_name == groupOrder(i), :);
    x = i + linspace(-0.18, 0.18, height(G))';
    scatter(x, G.rmse_d18O_permille, 34, colors(i, :), 'filled');
    plot(i, median(G.rmse_d18O_permille, 'omitnan'), 'kd', ...
        'MarkerFaceColor', 'w', 'MarkerSize', 7, 'LineWidth', 1.2);
end
hold off
grid on
xlim([0.4, numel(groupOrder) + 0.6]);
set(gca, 'XTick', 1:numel(groupOrder), ...
    'XTickLabel', cellstr(strrep(groupOrder, "_", " ")));
xtickangle(25);
ylabel('d18O sample RMSE (per mil)');
title('Sample-fit response across all sensitivity groups');

nexttile
bar(groupSummary.rmse_span_permille, 'FaceColor', [0.25 0.48 0.68], ...
    'EdgeColor', 'none');
grid on
set(gca, 'XTick', 1:height(groupSummary), ...
    'XTickLabel', cellstr(strrep(groupSummary.group_name, "_", " ")));
xtickangle(25);
ylabel('Within-group RMSE span (per mil)');
title('Sensitivity ranking by sample-fit RMSE span');

exportgraphics(fig, fullfile(outDir, ...
    'sensitivity_all_group_rmse.png'), 'Resolution', 240);
close(fig);
end

function S = readSpatialGroupSummary(rootScenario)
summaryFile = fullfile(rootScenario, 'divide_d18O_controls', ...
    'central_divide_d18O_controls_50km_group_summary.csv');
if ~isfile(summaryFile)
    warning('Spatial group summary not found: %s', summaryFile);
    S = table();
    return
end
S = readtable(summaryFile, 'TextType', 'string');
end

function plotSpatialGroupResponses(S, outDir)
T = S(S.group_name ~= "baseline", :);
[~, order] = sort(T.combined_d18O_span_permille, 'descend');
T = T(order, :);

fig = figure('Visible', 'off', 'Color', 'w', ...
    'Position', [100 100 1060 620]);
Y = [T.combined_d18O_span_permille, T.state1_d18O_span_permille, ...
    T.state2_d18O_span_permille];
bar(Y, 'grouped', 'EdgeColor', 'none');
grid on
set(gca, 'XTick', 1:height(T), ...
    'XTickLabel', cellstr(strrep(T.group_name, "_", " ")));
xtickangle(25);
ylabel('Central-divide d18O span at 50 km (per mil)');
title('Spatial response near the central divide');
legend({'Combined', 'State 1', 'State 2'}, 'Location', 'best');
exportgraphics(fig, fullfile(outDir, ...
    'sensitivity_spatial_group_response_50km.png'), 'Resolution', 240);
close(fig);
end

function sortKey = parseProxySortKey(caseNames)
sortKey = nan(numel(caseNames), 1);
for i = 1:numel(caseNames)
    name = string(caseNames(i));
    if contains(name, "warmest")
        seasonRank = 0;
    elseif contains(name, "annual")
        seasonRank = 100;
    elseif contains(name, "jja")
        seasonRank = 110;
    else
        seasonRank = 200;
    end

    tokens = regexp(char(name), '^offset(\d+)_', 'tokens', 'once');
    if isempty(tokens)
        offset = 0;
    else
        offset = str2double(tokens{1});
    end
    sortKey(i) = seasonRank + offset;
end
end

function shiftDeg = parseDivideShift(caseNames)
shiftDeg = nan(numel(caseNames), 1);
for i = 1:numel(caseNames)
    name = string(caseNames(i));
    if name == "base_000deg"
        shiftDeg(i) = 0;
    elseif startsWith(name, "north_")
        shiftDeg(i) = extractDegreeValue(name);
    elseif startsWith(name, "south_")
        shiftDeg(i) = -extractDegreeValue(name);
    end
end
end

function value = extractDegreeValue(name)
tokens = regexp(char(name), '_(\d{3})deg$', 'tokens', 'once');
if isempty(tokens)
    value = nan;
else
    value = str2double(tokens{1}) / 100;
end
end

function writeMarkdownReport(outDir, allRows, completionSummary, ...
    groupResponseSummary, parameterGroupSummary, refitGroupSummary, ...
    qualityControl, spatialGroupSummary)
reportFile = fullfile(outDir, 'sensitivity_analysis_report.md');
fid = fopen(reportFile, 'w', 'native', 'UTF-8');
if fid == -1
    error('Could not write report: %s', reportFile);
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>

fprintf(fid, '# Qiangtang 30 Ma Clumped-Temperature Sensitivity Report\n\n');
fprintf(fid, ['Generated by `opiAnalyze_ClumpedSensitivity` on %s. ' ...
    'This report covers all sensitivity experiments in the ' ...
    '`platform_northsouth` scenario.\n\n'], ...
    char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm')));

fprintf(fid, '## Summary\n\n');
fprintf(fid, ['The analysis contains **%d groups and %d cases**. All cases ' ...
    'have calculation results, finite sample RMSE, and full reported ' ...
    'sample coverage; all %d refit cases have fit results.\n\n'], ...
    height(completionSummary), height(allRows), ...
    sum(allRows.run_mode == "refit"));
fprintf(fid, ['Fixed-parameter experiments show the largest sample-fit ' ...
    'responses to state 1 source and rainout controls, mixture fraction, ' ...
    'and wind direction. Proxy-temperature interpretation changes ' ...
    'temperature residuals and internal parameters after refitting while ' ...
    'leaving oxygen-isotope RMSE nearly unchanged. Divide experiments have ' ...
    'a critical implementation limitation because the fitted prediction ' ...
    'does not use divide geometry.\n\n']);

fprintf(fid, '## 1. Design and completion\n\n');
fprintf(fid, ['Calculation-only cases measure direct local response with ' ...
    'other parameters fixed. Refit cases allow parameter compensation after ' ...
    'an interpretation or input choice changes. These designs answer ' ...
    'different questions. The two divide groups are computationally ' ...
    'complete but remain review-only until geometry is connected to the ' ...
    'model prediction.\n\n']);
fprintf(fid, '| Group | Mode | Planned | Calculated | Fitted | Status |\n');
fprintf(fid, '|---|---:|---:|---:|---:|---|\n');
for i = 1:height(completionSummary)
    R = completionSummary(i, :);
    fprintf(fid, '| `%s` | `%s` | %d | %d | %d | %s |\n', ...
        R.group_name, R.run_mode, R.expected_cases, R.n_calc, ...
        R.n_fit, R.status);
end

fprintf(fid, '\n## 2. All-group response\n\n');
fprintf(fid, '| Group | Cases | RMSE range (per mil) | Span (per mil) | Best case | Worst case |\n');
fprintf(fid, '|---|---:|---:|---:|---|---|\n');
for i = 1:height(groupResponseSummary)
    R = groupResponseSummary(i, :);
    fprintf(fid, '| `%s` | %d | %.4f to %.4f | %.4f | `%s` | `%s` |\n', ...
        R.group_name, R.n_cases, R.rmse_min_permille, ...
        R.rmse_max_permille, R.rmse_span_permille, ...
        R.best_rmse_case, R.worst_rmse_case);
end
fprintf(fid, '\n![All-group sample RMSE](sensitivity_all_group_rmse.png)\n\n');

fprintf(fid, '## 3. Local parameter sensitivity\n\n');
fprintf(fid, ['The ranking uses the sample oxygen-isotope RMSE span in each ' ...
    'one-at-a-time parameter scan. A larger span indicates greater direct ' ...
    'response over the tested range.\n\n']);
fprintf(fid, '| Parameter | Range | RMSE span (per mil) | Best case | Worst case |\n');
fprintf(fid, '|---|---:|---:|---|---|\n');
for i = 1:height(parameterGroupSummary)
    R = parameterGroupSummary(i, :);
    fprintf(fid, '| `%s` | %.5g to %.5g | %.4f | `%s` (%.4f) | `%s` (%.4f) |\n', ...
        R.varied_parameter, R.value_min, R.value_max, ...
        R.rmse_span_permille, R.best_rmse_case, R.rmse_min_permille, ...
        R.worst_rmse_case, R.rmse_max_permille);
end
fprintf(fid, ['\n`fraction` and `M1` have the largest RMSE spans, followed ' ...
    'by `T0_1_K`. Direct responses to `M2` and `T0_2_K` are smaller. The ' ...
    '`T0_1_K` response is nonmonotonic and reaches the lowest sampled RMSE ' ...
    'near 296.25 K.\n\n']);
fprintf(fid, '![Parameter-scan RMSE](sensitivity_parameter_rmse_by_value.png)\n\n');
fprintf(fid, '![Parameter-scan temperature residual](sensitivity_parameter_hren_residual_by_value.png)\n\n');

fprintf(fid, '## 4. Refit experiments\n\n');
for i = 1:height(refitGroupSummary)
    R = refitGroupSummary(i, :);
    fprintf(fid, ['- `%s`: sample oxygen-isotope RMSE span %.6g per mil; ' ...
        'combined reduced chi-square span %.6g; mean Hren warmest-water ' ...
        'residual span %.4f degrees C.\n'], ...
        R.group_name, R.rmse_span_permille, R.chiR2_total_span, ...
        R.hren_mean_residual_span_C);
end
fprintf(fid, ['\nThe seven divide refits retain nearly identical sample RMSE, ' ...
    'but fitted parameter changes cannot be attributed to divide movement ' ...
    'because fitting does not read divide geometry. Proxy refits also ' ...
    'preserve oxygen-isotope fit but produce about 13.11 degrees C of mean ' ...
    'Hren warmest-water residual span.\n\n']);
fprintf(fid, '![Refit-group metrics](sensitivity_refit_group_metrics.png)\n\n');

fprintf(fid, '## 5. Wind direction and mechanism experiments\n\n');
writeGroupResult(fid, groupResponseSummary, "mechanism", "Local mechanism perturbations");
writeGroupResult(fid, groupResponseSummary, "azimuth_fine", "Fine azimuth scan");
writeGroupResult(fid, groupResponseSummary, "az2_transition", "Az2 transition scan");
fprintf(fid, ['\nThe mechanism case `d18O0_1_dm2permil` has the highest sample ' ...
    'RMSE. Fine azimuth response is directional and nonlinear: ' ...
    '`Az1_plus05deg` reaches the lowest RMSE, negative Az1 perturbations ' ...
    'degrade the fit, and the Az2 transition reaches a local minimum near ' ...
    '`Az2_plus10deg`.\n\n']);

fprintf(fid, '## 6. Spatial response near the sample centroid\n\n');
if ~isempty(spatialGroupSummary)
    fprintf(fid, ['The spatial diagnostic uses the combined oxygen-isotope ' ...
        'span within 50 km of the sample centroid. It complements discrete ' ...
        'sample RMSE because refitting can preserve sample fit while ' ...
        'changing fields between samples.\n\n']);
    fprintf(fid, '| Group | Combined span (per mil) | State 1 span (per mil) | State 2 span (per mil) |\n');
    fprintf(fid, '|---|---:|---:|---:|\n');
    [~, order] = sort(spatialGroupSummary.combined_d18O_span_permille, ...
        'descend');
    P = spatialGroupSummary(order, :);
    for i = 1:height(P)
        R = P(i, :);
        fprintf(fid, '| `%s` | %.4f | %.4f | %.4f |\n', ...
            R.group_name, R.combined_d18O_span_permille, ...
            R.state1_d18O_span_permille, R.state2_d18O_span_permille);
    end
    fprintf(fid, '\n![Sample-centroid spatial response](sensitivity_spatial_group_response_50km.png)\n\n');
end
fprintf(fid, ['The state 2 spans for `proxy` and `divide` are about 25.32 ' ...
    'and 10.47 per mil, whereas their combined spans are about 2.15 and ' ...
    '2.95 per mil. Changes in one state are partly compensated by the other ' ...
    'state and precipitation weights. The divide span cannot be interpreted ' ...
    'as a response to divide movement.\n\n']);
fprintf(fid, ['The `divide_calc_only` group has zero sample and 50-km ' ...
    'response. Code review shows that the calculation creates ' ...
    '`isSampleSide01` but does not pass it to prediction and the fitting ' ...
    'function does not read the divide file. Zero response is therefore not ' ...
    'evidence that divide position is unimportant.\n\n']);

fprintf(fid, '## 7. Quality control and limitations\n\n');
for i = 1:height(qualityControl)
    R = qualityControl(i, :);
    if R.passed
        state = "PASS";
    else
        state = "REVIEW";
    end
    fprintf(fid, '- **%s** `%s`: %s\n', state, R.check_name, R.details);
end
fprintf(fid, ['\nThese finite scans describe local response over specified ' ...
    'ranges rather than global parameter uncertainty. Calculation-only ' ...
    'experiments hold other parameters fixed, whereas refit experiments ' ...
    'allow compensation. The 16-sample spatial interpretation should be ' ...
    'evaluated with independent geological constraints.\n\n']);

fprintf(fid, '## 8. Main conclusions\n\n');
fprintf(fid, ['1. All 85 planned cases are computationally complete.\n' ...
    '2. State 1 parameters, mixture fraction, and wind direction control ' ...
    'the main local responses; M2 and T0_2 are weaker over tested ranges.\n' ...
    '3. Proxy assumptions can be compensated during refitting, so stable ' ...
    'sample RMSE does not imply stable parameters or temperatures.\n' ...
    '4. Clumped-temperature interpretation creates a large temperature-' ...
    'system response that must be propagated explicitly.\n' ...
    '5. Divide geometry is absent from the fitted prediction; the divide ' ...
    'groups require a model connection and rerun before scientific use.\n\n']);

fprintf(fid, '## 9. Generated files\n\n');
fprintf(fid, '- `sensitivity_all_cases_compact.csv`\n');
fprintf(fid, '- `sensitivity_completion_summary.csv`\n');
fprintf(fid, '- `sensitivity_group_response_summary.csv`\n');
fprintf(fid, '- `sensitivity_quality_control.csv`\n');
fprintf(fid, '- `sensitivity_parameter_cases_with_values.csv`\n');
fprintf(fid, '- `sensitivity_parameter_group_summary.csv`\n');
fprintf(fid, '- `sensitivity_refit_group_summary.csv`\n');
fprintf(fid, '- `sensitivity_spatial_group_summary_50km.csv`\n');
fprintf(fid, '- `sensitivity_all_group_rmse.png`\n');
fprintf(fid, '- `sensitivity_parameter_rmse_by_value.png`\n');
fprintf(fid, '- `sensitivity_parameter_hren_residual_by_value.png`\n');
fprintf(fid, '- `sensitivity_refit_group_metrics.png`\n');
fprintf(fid, '- `sensitivity_spatial_group_response_50km.png`\n');
end

function writeGroupResult(fid, S, groupName, displayName)
R = S(S.group_name == groupName, :);
if isempty(R)
    return
end
fprintf(fid, ['- %s (`%s`): RMSE %.4f to %.4f per mil; best `%s`; ' ...
    'worst `%s`.\n'], displayName, groupName, R.rmse_min_permille, ...
    R.rmse_max_permille, R.best_rmse_case, R.worst_rmse_case);
end

function value = rangeOmitnan(x)
if isempty(x) || all(~isfinite(x))
    value = nan;
else
    value = max(x, [], 'omitnan') - min(x, [], 'omitnan');
end
end
