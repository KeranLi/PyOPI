function package = opiPrepare_SensitivityManuscriptPackage( ...
    packageRoot, rootScenario)
% Build an English Science Advances package for the OPI sensitivity runs.


opiRoot = fileparts(fileparts(mfilename('fullpath')));
if nargin < 1 || strlength(string(packageRoot)) == 0
    packageRoot = fullfile(opiRoot, ...
        'manuscript_package_OPI_sensitivity');
end
if nargin < 2 || strlength(string(rootScenario)) == 0
    rootScenario = fullfile(opiRoot, 'scenarios', ...
        ['Qiangtang_30Ma_4000m_dH045_T290_fP030_Valid_' ...
        'platform_northsouth']);
end

sourceRoot = fullfile(rootScenario, 'sensitivity_analysis');
required = string({ ...
    fullfile(sourceRoot, 'sensitivity_all_cases_compact.csv'), ...
    fullfile(sourceRoot, 'sensitivity_completion_summary.csv'), ...
    fullfile(sourceRoot, 'sensitivity_group_response_summary.csv'), ...
    fullfile(sourceRoot, 'sensitivity_parameter_cases_with_values.csv'), ...
    fullfile(sourceRoot, 'sensitivity_parameter_group_summary.csv'), ...
    fullfile(sourceRoot, 'sensitivity_refit_group_summary.csv'), ...
    fullfile(sourceRoot, 'sensitivity_spatial_group_summary_50km.csv'), ...
    fullfile(sourceRoot, 'sensitivity_quality_control.csv')});
missing = required(~isfile(required));
if ~isempty(missing)
    error('Missing sensitivity summary input(s):\n%s', ...
        strjoin(missing, newline));
end

figureRoot = fullfile(packageRoot, 'figures');
overviewRoot = fullfile(figureRoot, 'Figure_S3_Overview');
parameterRoot = fullfile(figureRoot, 'Figure_S4_Parameter_Sensitivity');
windRoot = fullfile(figureRoot, 'Figure_S5_Wind_Spatial_Sensitivity');
proxyRoot = fullfile(figureRoot, 'Figure_S6_Proxy_Interpretation');
mlFigureRoot = fullfile(figureRoot, ...
    'Figure_S7_Machine_Learning_Comparison');
rawRoot = fullfile(packageRoot, 'raw_tables');
archiveRoot = fullfile(packageRoot, 'archive', 'legacy_figures');
configurationRoot = fullfile(packageRoot, 'configuration');
scriptsRoot = fullfile(packageRoot, 'scripts');
mlArchiveRoot = fullfile(packageRoot, 'machine_learning');
roots = string({packageRoot, figureRoot, overviewRoot, parameterRoot, ...
    windRoot, proxyRoot, mlFigureRoot, rawRoot, archiveRoot, ...
    configurationRoot, scriptsRoot, mlArchiveRoot, ...
    fullfile(overviewRoot, 'data'), ...
    fullfile(parameterRoot, 'data'), fullfile(windRoot, 'data'), ...
    fullfile(proxyRoot, 'data'), fullfile(mlFigureRoot, 'data')});
for i = 1:numel(roots)
    if ~isfolder(roots(i))
        mkdir(roots(i));
    end
end

allCases = readtable(required(1), 'TextType', 'string');
completion = readtable(required(2), 'TextType', 'string');
groupSummary = readtable(required(3), 'TextType', 'string');
parameterCases = readtable(required(4), 'TextType', 'string');
parameterSummary = readtable(required(5), 'TextType', 'string');
refitSummary = readtable(required(6), 'TextType', 'string');
spatialSummary = readtable(required(7), 'TextType', 'string');
qualityControl = readtable(required(8), 'TextType', 'string');

validGroups = ["mechanism", "parameter", "azimuth_fine", ...
    "az2_transition", "proxy"];
validCases = allCases(ismember(allCases.group_name, validGroups), :);
validGroupSummary = groupSummary( ...
    ismember(groupSummary.group_name, validGroups), :);
writetable(validCases, fullfile(overviewRoot, 'data', ...
    'Figure_S3_valid_case_metrics.csv'));
writetable(validGroupSummary, fullfile(overviewRoot, 'data', ...
    'Figure_S3_valid_group_summary.csv'));
renderOverview(overviewRoot, validCases, validGroupSummary);

writetable(parameterCases, fullfile(parameterRoot, 'data', ...
    'Figure_S4_parameter_case_metrics.csv'));
writetable(parameterSummary, fullfile(parameterRoot, 'data', ...
    'Figure_S4_parameter_group_summary.csv'));
renderParameterFigure(parameterRoot, parameterCases, parameterSummary);

windCases = buildWindPlotData(allCases);
spatialValid = spatialSummary( ...
    ismember(spatialSummary.group_name, validGroups), :);
writetable(windCases, fullfile(windRoot, 'data', ...
    'Figure_S5_wind_case_metrics.csv'));
writetable(spatialValid, fullfile(windRoot, 'data', ...
    'Figure_S5_spatial_group_metrics.csv'));
renderWindFigure(windRoot, windCases, spatialValid);

proxyCases = allCases(allCases.group_name == "proxy", :);
proxyCases = orderProxyCases(proxyCases);
proxyRefit = refitSummary(refitSummary.group_name == "proxy", :);
writetable(proxyCases, fullfile(proxyRoot, 'data', ...
    'Figure_S6_proxy_case_metrics.csv'));
writetable(proxyRefit, fullfile(proxyRoot, 'data', ...
    'Figure_S6_proxy_refit_summary.csv'));
renderProxyFigure(proxyRoot, proxyCases);

prepareMLComparison(mlFigureRoot, mlArchiveRoot, opiRoot, rootScenario);

copyRawTables(sourceRoot, rawRoot);
copyLegacyFigures(sourceRoot, archiveRoot);
copyConfiguration(opiRoot, rootScenario, configurationRoot);
copyScripts(opiRoot, scriptsRoot);
writeFileInventory(packageRoot);

package = struct;
package.root = string(packageRoot);
package.nCases = height(allCases);
package.nValidCases = height(validCases);
package.nGroups = height(completion);
package.nQualityChecksPassed = sum(qualityControl.passed == 1);
fprintf('Wrote OPI sensitivity manuscript package to:\n%s\n', packageRoot);
end

function renderOverview(root, T, S)
groupOrder = ["mechanism", "parameter", "azimuth_fine", ...
    "az2_transition", "proxy"];
labels = ["Mechanism", "Parameter", "Azimuth", ...
    "Az2 transition", "Proxy"];
colors = [0.00, 0.45, 0.70; 0.84, 0.37, 0.00; ...
    0.93, 0.69, 0.13; 0.49, 0.18, 0.56; 0.00, 0.62, 0.45];

fig = figure('Color', 'w', 'Position', [100, 100, 1180, 500]);
layout = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', ...
    'Padding', 'compact');
ax1 = nexttile(layout, 1);
hold(ax1, 'on');
for i = 1:numel(groupOrder)
    G = sortrows(T(T.group_name == groupOrder(i), :), ...
        'rmse_d18O_permille');
    jitter = linspace(-0.16, 0.16, max(1, height(G)))';
    scatter(ax1, i + jitter, G.rmse_d18O_permille, 24, 'o', ...
        'MarkerFaceColor', colors(i, :), 'MarkerEdgeColor', 'w', ...
        'LineWidth', 0.5);
    scatter(ax1, i, median(G.rmse_d18O_permille), 48, 'd', ...
        'MarkerFaceColor', 'w', 'MarkerEdgeColor', [0.08, 0.09, 0.10], ...
        'LineWidth', 1.1);
end
formatAxis(ax1);
xlim(ax1, [0.55, numel(groupOrder) + 0.45]);
xticks(ax1, 1:numel(groupOrder));
xticklabels(ax1, labels);
ax1.XTickLabelRotation = 20;
ylabel(ax1, '\delta^{18}O sample RMSE (per mil)', 'Interpreter', 'tex');
title(ax1, 'Case-level response', 'FontWeight', 'normal', ...
    'FontSize', 10.5, 'Color', [0.08, 0.09, 0.10]);
addPanelLetter(ax1, 'a');

ax2 = nexttile(layout, 2);
span = nan(numel(groupOrder), 1);
for i = 1:numel(groupOrder)
    row = S.group_name == groupOrder(i);
    span(i) = S.rmse_span_permille(row);
end
b = barh(ax2, 1:numel(groupOrder), span, 0.62, ...
    'FaceColor', 'flat', 'EdgeColor', 'none');
b.CData = colors;
formatAxis(ax2);
yticks(ax2, 1:numel(groupOrder));
yticklabels(ax2, labels);
ax2.YDir = 'reverse';
xlabel(ax2, 'Within-group \delta^{18}O RMSE span (per mil)', ...
    'Interpreter', 'tex');
title(ax2, 'Sensitivity ranking', 'FontWeight', 'normal', ...
    'FontSize', 10.5, 'Color', [0.08, 0.09, 0.10]);
addPanelLetter(ax2, 'b');
exportFigureSet(fig, root, 'Figure_S3_OPI_Sensitivity_Overview_EN');
end

function renderParameterFigure(root, T, S)
parameters = ["fraction", "M1", "M2", "T0_1_K", "T0_2_K"];
titles = ["Mixture fraction", "M_1", "M_2", "T_{0,1}", "T_{0,2}"];
xLabels = ["Fraction", "M_1", "M_2", ...
    "Temperature (K)", "Temperature (K)"];
colors = [0.00, 0.45, 0.70; 0.84, 0.37, 0.00; ...
    0.00, 0.62, 0.45; 0.49, 0.18, 0.56; 0.93, 0.69, 0.13];
fig = figure('Color', 'w', 'Position', [100, 80, 1180, 720]);
layout = tiledlayout(fig, 2, 3, 'TileSpacing', 'compact', ...
    'Padding', 'compact');
for i = 1:numel(parameters)
    ax = nexttile(layout, i);
    G = sortrows(T(T.varied_parameter == parameters(i), :), ...
        'varied_value');
    plot(ax, G.varied_value, G.rmse_d18O_permille, '-o', ...
        'Color', colors(i, :), 'MarkerFaceColor', 'w', ...
        'LineWidth', 1.35, 'MarkerSize', 4.5);
    formatAxis(ax);
    xlabel(ax, xLabels(i));
    ylabel(ax, '\delta^{18}O RMSE (per mil)', 'Interpreter', 'tex');
    title(ax, titles(i), 'FontWeight', 'normal', 'FontSize', 10.5, ...
        'Interpreter', 'tex', 'Color', [0.08, 0.09, 0.10]);
    addPanelLetter(ax, char('a' + i - 1));
end
ax = nexttile(layout, 6);
span = nan(numel(parameters), 1);
for i = 1:numel(parameters)
    span(i) = S.rmse_span_permille(S.varied_parameter == parameters(i));
end
[span, order] = sort(span, 'descend');
b = barh(ax, 1:numel(parameters), span, 0.62, ...
    'FaceColor', 'flat', 'EdgeColor', 'none');
b.CData = colors(order, :);
formatAxis(ax);
yticks(ax, 1:numel(parameters));
yticklabels(ax, titles(order));
ax.YDir = 'reverse';
xlabel(ax, '\delta^{18}O RMSE span (per mil)', 'Interpreter', 'tex');
title(ax, 'Local sensitivity ranking', 'FontWeight', 'normal', ...
    'FontSize', 10.5, 'Color', [0.08, 0.09, 0.10]);
addPanelLetter(ax, 'f');
exportFigureSet(fig, root, 'Figure_S4_OPI_Parameter_Sensitivity_EN');
end

function wind = buildWindPlotData(T)
keep = ismember(T.group_name, ["azimuth_fine", "az2_transition"]);
T = T(keep, :);
family = strings(height(T), 1);
azimuthOffsetDeg = nan(height(T), 1);
for i = 1:height(T)
    token = regexp(char(T.case_name(i)), ...
        '(Az[12])_(minus|plus)([0-9]+)deg', 'tokens', 'once');
    if isempty(token)
        error('Could not parse azimuth case: %s', T.case_name(i));
    end
    family(i) = string(token{1});
    azimuthOffsetDeg(i) = str2double(token{3});
    if strcmp(token{2}, 'minus')
        azimuthOffsetDeg(i) = -azimuthOffsetDeg(i);
    end
end
wind = addvars(T, family, azimuthOffsetDeg, 'After', 'group_name', ...
    'NewVariableNames', {'azimuth_family', 'azimuth_offset_deg'});
wind = sortrows(wind, {'group_name', 'azimuth_family', ...
    'azimuth_offset_deg'});
end

function renderWindFigure(root, T, spatial)
fig = figure('Color', 'w', 'Position', [100, 100, 1320, 470]);
layout = tiledlayout(fig, 1, 3, 'TileSpacing', 'compact', ...
    'Padding', 'compact');
colors = [0.00, 0.45, 0.70; 0.84, 0.37, 0.00];

ax1 = nexttile(layout, 1);
hold(ax1, 'on');
families = ["Az1", "Az2"];
for i = 1:2
    G = T(T.group_name == "azimuth_fine" & ...
        T.azimuth_family == families(i), :);
    G = sortrows(G, 'azimuth_offset_deg');
    plot(ax1, G.azimuth_offset_deg, G.rmse_d18O_permille, '-o', ...
        'Color', colors(i, :), 'MarkerFaceColor', 'w', ...
        'LineWidth', 1.35, 'MarkerSize', 4.5, ...
        'DisplayName', families(i));
end
formatAxis(ax1);
xlabel(ax1, 'Azimuth perturbation (degrees)');
ylabel(ax1, '\delta^{18}O RMSE (per mil)', 'Interpreter', 'tex');
title(ax1, 'Fine azimuth scan', 'FontWeight', 'normal', ...
    'FontSize', 10.5, 'Color', [0.08, 0.09, 0.10]);
lgd = legend(ax1, 'Location', 'best', 'Box', 'off', 'FontSize', 8);
lgd.TextColor = [0.08, 0.09, 0.10];
addPanelLetter(ax1, 'a');

ax2 = nexttile(layout, 2);
G = T(T.group_name == "az2_transition", :);
G = sortrows(G, 'azimuth_offset_deg');
plot(ax2, G.azimuth_offset_deg, G.rmse_d18O_permille, '-o', ...
    'Color', [0.49, 0.18, 0.56], 'MarkerFaceColor', 'w', ...
    'LineWidth', 1.35, 'MarkerSize', 4.5);
formatAxis(ax2);
xlabel(ax2, 'Az2 perturbation (degrees)');
ylabel(ax2, '\delta^{18}O RMSE (per mil)', 'Interpreter', 'tex');
title(ax2, 'Az2 transition scan', 'FontWeight', 'normal', ...
    'FontSize', 10.5, 'Color', [0.08, 0.09, 0.10]);
addPanelLetter(ax2, 'b');

ax3 = nexttile(layout, 3);
groupOrder = ["azimuth_fine", "mechanism", "az2_transition", ...
    "parameter", "proxy"];
labels = ["Azimuth", "Mechanism", "Az2 transition", ...
    "Parameter", "Proxy"];
values = nan(numel(groupOrder), 3);
for i = 1:numel(groupOrder)
    row = spatial.group_name == groupOrder(i);
    values(i, :) = [spatial.combined_d18O_span_permille(row), ...
        spatial.state1_d18O_span_permille(row), ...
        spatial.state2_d18O_span_permille(row)];
end
b = bar(ax3, values, 'grouped', 'EdgeColor', 'none');
b(1).FaceColor = [0.00, 0.45, 0.70];
b(2).FaceColor = [0.84, 0.37, 0.00];
b(3).FaceColor = [0.93, 0.69, 0.13];
formatAxis(ax3);
xticks(ax3, 1:numel(groupOrder));
xticklabels(ax3, labels);
ax3.XTickLabelRotation = 25;
ylabel(ax3, '50-km \delta^{18}O span (per mil)', 'Interpreter', 'tex');
title(ax3, 'Central-domain spatial response', ...
    'FontWeight', 'normal', 'FontSize', 10.5, ...
    'Color', [0.08, 0.09, 0.10]);
lgd = legend(ax3, {'Combined', 'State 1', 'State 2'}, ...
    'Location', 'northwest', 'Box', 'off', 'FontSize', 8);
lgd.TextColor = [0.08, 0.09, 0.10];
addPanelLetter(ax3, 'c');
exportFigureSet(fig, root, 'Figure_S5_OPI_Wind_Spatial_Sensitivity_EN');
end

function T = orderProxyCases(T)
orderNames = ["offset0_warmest", "offset5_warmest", ...
    "offset7_warmest", "offset10_warmest", ...
    "offset7_annual", "offset7_jja"];
[found, order] = ismember(orderNames, T.case_name);
if ~all(found)
    error('Proxy sensitivity cases are incomplete.');
end
T = T(order, :);
end

function renderProxyFigure(root, T)
labels = ["0 / warmest", "5 / warmest", "7 / warmest", ...
    "10 / warmest", "7 / annual", "7 / JJA"];
fig = figure('Color', 'w', 'Position', [100, 100, 1120, 460]);
layout = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', ...
    'Padding', 'compact');
ax1 = nexttile(layout, 1);
plot(ax1, 1:height(T), T.rmse_d18O_permille, '-o', ...
    'Color', [0.00, 0.45, 0.70], 'MarkerFaceColor', 'w', ...
    'LineWidth', 1.35, 'MarkerSize', 4.5);
formatAxis(ax1);
xticks(ax1, 1:height(T));
xticklabels(ax1, labels);
ax1.XTickLabelRotation = 25;
ylim(ax1, [0, 0.42]);
ylabel(ax1, '\delta^{18}O RMSE (per mil)', 'Interpreter', 'tex');
xlabel(ax1, 'Offset (degrees C) / season');
title(ax1, 'Refitted sample fit', 'FontWeight', 'normal', ...
    'FontSize', 10.5, 'Color', [0.08, 0.09, 0.10]);
addPanelLetter(ax1, 'a');

ax2 = nexttile(layout, 2);
bar(ax2, 1:height(T), T.hren_warmest_mean_residual_C, 0.62, ...
    'FaceColor', [0.84, 0.37, 0.00], 'EdgeColor', 'none');
formatAxis(ax2);
xticks(ax2, 1:height(T));
xticklabels(ax2, labels);
ax2.XTickLabelRotation = 25;
yline(ax2, 0, '-', 'Color', [0.25, 0.25, 0.25], ...
    'HandleVisibility', 'off');
ylabel(ax2, 'Hren warmest-water residual (degrees C)');
xlabel(ax2, 'Offset (degrees C) / season');
title(ax2, 'Temperature-system response', ...
    'FontWeight', 'normal', 'FontSize', 10.5, ...
    'Color', [0.08, 0.09, 0.10]);
addPanelLetter(ax2, 'b');
exportFigureSet(fig, root, 'Figure_S6_OPI_Proxy_Interpretation_EN');
end

function prepareMLComparison(figureRoot, archiveRoot, opiRoot, rootScenario)
modelRoot = fullfile(opiRoot, 'data', 'derived', ...
    'LakeTransferFunction', 'TerrazasWarmestML');
validationFile = fullfile(modelRoot, ...
    'TerrazasWarmestML_validation_metrics.csv');
if ~isfile(validationFile)
    error('Missing Terrazas ML validation metrics: %s', validationFile);
end
validation = readtable(validationFile, 'TextType', 'string');

caseNames = ["offset0_warmest", "offset5_warmest", ...
    "offset7_warmest", "offset10_warmest", ...
    "offset7_annual", "offset7_jja"];
application = table();
for i = 1:numel(caseNames)
    caseRoot = fullfile(rootScenario, 'sensitivity_proxy_clumped', ...
        caseNames(i), 'proxy_clumped');
    comparisonFile = fullfile(caseRoot, ...
        'clumped_temperature_TerrazasWarmestML_comparison.csv');
    if ~isfile(comparisonFile)
        error('Missing Terrazas ML comparison output: %s', comparisonFile);
    end
    T = readtable(comparisonFile, 'TextType', 'string');
    hrenFile = fullfile(caseRoot, ...
        'clumped_temperature_HrenSheldon2012_comparison.csv');
    if ~isfile(hrenFile)
        error('Missing Hren-Sheldon comparison output: %s', hrenFile);
    end
    H = readtable(hrenFile, 'TextType', 'string');
    if height(H) ~= height(T) || any(H.sample_index ~= T.sample_index)
        error('ML and Hren comparison rows do not align for %s.', caseNames(i));
    end
    hrenPred = H.OPI_Hren2012_Tw_warmest_C;
    hrenSigma = H.sigma_OPI_Hren2012_Tw_warmest_C;
    hrenResidual = T.ML_lake_temperature_corrected_C - hrenPred;
    hrenCombinedSigma = sqrt( ...
        T.sigma_ML_lake_temperature_corrected_C.^2 + hrenSigma.^2);
    T = addvars(T, hrenPred, hrenSigma, hrenResidual, ...
        hrenCombinedSigma, hrenResidual ./ hrenCombinedSigma, ...
        'NewVariableNames', { ...
        'OPI_Hren2012_Tw_warmest_C', ...
        'sigma_OPI_Hren2012_Tw_warmest_C', ...
        'residual_corrected_lake_minus_OPI_Hren2012_Tw_warmest_C', ...
        'sigma_combined_corrected_lake_minus_OPI_Hren2012_Tw_warmest_C', ...
        'z_corrected_lake_minus_OPI_Hren2012_Tw_warmest'});
    T = addvars(T, repmat(caseNames(i), height(T), 1), ...
        'Before', 1, 'NewVariableNames', 'case_name');
    application = [application; T]; %#ok<AGROW>
end
summary = summarizeMLApplication(application, caseNames);

writetable(validation, fullfile(figureRoot, 'data', ...
    'Figure_S7_ML_validation_metrics.csv'));
writetable(application, fullfile(figureRoot, 'data', ...
    'Figure_S7_ML_proxy_sample_metrics.csv'));
writetable(summary, fullfile(figureRoot, 'data', ...
    'Figure_S7_ML_proxy_case_summary.csv'));
renderMLFigure(figureRoot, validation, application, summary, caseNames);
copyMLArtifacts(archiveRoot, modelRoot, rootScenario, caseNames);
end

function S = summarizeMLApplication(T, caseNames)
S = table();
for i = 1:numel(caseNames)
    G = T(T.case_name == caseNames(i), :);
    forward = G.residual_corrected_lake_minus_OPI_ML_Tw_warmest_C;
    hrenForward = ...
        G.residual_corrected_lake_minus_OPI_Hren2012_Tw_warmest_C;
    inverse = G.residual_OPI_minus_ML_Tair_warmest_C;
    row = table(caseNames(i), height(G), ...
        mean(forward, 'omitnan'), rmsFinite(forward), ...
        mean(G.z_corrected_lake_minus_OPI_ML_Tw_warmest, 'omitnan'), ...
        mean(hrenForward, 'omitnan'), rmsFinite(hrenForward), ...
        mean(G.z_corrected_lake_minus_OPI_Hren2012_Tw_warmest, 'omitnan'), ...
        mean(G.OPI_ML_Tw_warmest_C - ...
        G.OPI_Hren2012_Tw_warmest_C, 'omitnan'), ...
        mean(inverse, 'omitnan'), rmsFinite(inverse), ...
        mean(G.z_OPI_minus_ML_Tair_warmest, 'omitnan'), ...
        sum(G.ML_forward_outside_global_training_range), ...
        sum(G.ML_forward_outside_high_elevation_range), ...
        sum(G.ML_inverse_outside_global_training_range), ...
        sum(G.ML_inverse_outside_high_elevation_range), ...
        sum(G.ML_used_default_lake_area), ...
        sum(G.ML_used_default_lake_depth), ...
        'VariableNames', {'case_name', 'n_samples', ...
        'forward_mean_residual_C', 'forward_rmse_C', 'forward_mean_z', ...
        'hren_forward_mean_residual_C', 'hren_forward_rmse_C', ...
        'hren_forward_mean_z', 'ml_minus_hren_predicted_lake_mean_C', ...
        'inverse_mean_residual_C', 'inverse_rmse_C', 'inverse_mean_z', ...
        'n_forward_outside_global_training_range', ...
        'n_forward_outside_high_elevation_range', ...
        'n_inverse_outside_global_training_range', ...
        'n_inverse_outside_high_elevation_range', ...
        'n_default_lake_area', 'n_default_lake_depth'});
    S = [S; row]; %#ok<AGROW>
end
end

function renderMLFigure(root, V, A, S, caseNames)
subsetOrder = ["all", "latitude_25_to_40", ...
    "elevation_above_3km", "elevation_above_4km"];
subsetLabels = ["All", "25-40 deg", ">3 km", ">4 km"];
caseLabels = ["0 / warmest", "5 / warmest", "7 / warmest", ...
    "10 / warmest", "7 / annual", "7 / JJA"];
fig = figure('Color', 'w', 'Position', [100, 60, 1240, 760]);
layout = tiledlayout(fig, 2, 2, 'TileSpacing', 'compact', ...
    'Padding', 'compact');

ax1 = nexttile(layout, 1);
plotMLValidationBars(ax1, V, "warmest_air_to_lake", ...
    ["ML_quadratic_ridge_spatial_CV", ...
    "linear_air_temperature_spatial_CV"], subsetOrder, subsetLabels);
title(ax1, 'Forward spatial cross-validation', 'FontWeight', 'normal', ...
    'FontSize', 10.5, 'Color', [0.08, 0.09, 0.10]);
addPanelLetter(ax1, 'a');

ax2 = nexttile(layout, 2);
plotMLValidationBars(ax2, V, "warmest_lake_to_air", ...
    ["ML_quadratic_ridge_spatial_CV", ...
    "linear_lake_temperature_spatial_CV"], subsetOrder, subsetLabels);
title(ax2, 'Inverse spatial cross-validation', 'FontWeight', 'normal', ...
    'FontSize', 10.5, 'Color', [0.08, 0.09, 0.10]);
addPanelLetter(ax2, 'b');

ax3 = nexttile(layout, 3);
plotMLForwardComparison(ax3, A, S, caseNames, caseLabels);
ylabel(ax3, 'Corrected lake minus predicted lake (degrees C)');
title(ax3, 'Forward operator comparison', 'FontWeight', 'normal', ...
    'FontSize', 10.5, 'Color', [0.08, 0.09, 0.10]);
addPanelLetter(ax3, 'c');

ax4 = nexttile(layout, 4);
plotMLApplicationResiduals(ax4, A, S, caseNames, caseLabels, ...
    'residual_OPI_minus_ML_Tair_warmest_C', ...
    'ML_inverse_outside_high_elevation_range');
ylabel(ax4, 'OPI air minus inferred air (degrees C)');
title(ax4, 'Inverse paleolake diagnostic', 'FontWeight', 'normal', ...
    'FontSize', 10.5, 'Color', [0.08, 0.09, 0.10]);
addPanelLetter(ax4, 'd');

exportFigureSet(fig, root, ...
    'Figure_S7_OPI_Machine_Learning_Comparison_EN');
end

function plotMLForwardComparison(ax, T, S, caseNames, labels)
hold(ax, 'on');
for i = 1:numel(caseNames)
    G = T(T.case_name == caseNames(i), :);
    mlResidual = G.residual_corrected_lake_minus_OPI_ML_Tw_warmest_C;
    hrenResidual = ...
        G.residual_corrected_lake_minus_OPI_Hren2012_Tw_warmest_C;
    flagged = logical(G.ML_forward_outside_high_elevation_range);
    jitter = linspace(-0.06, 0.06, max(1, height(G)))';
    scatter(ax, i - 0.12 + jitter(~flagged), mlResidual(~flagged), ...
        18, 'o', 'MarkerFaceColor', [0.00, 0.45, 0.70], ...
        'MarkerEdgeColor', 'w', 'LineWidth', 0.4);
    if any(flagged)
        scatter(ax, i - 0.12 + jitter(flagged), mlResidual(flagged), ...
            26, 'o', 'MarkerFaceColor', 'w', ...
            'MarkerEdgeColor', [0.80, 0.20, 0.12], 'LineWidth', 1.0);
    end
    scatter(ax, i + 0.12 + jitter, hrenResidual, 18, 'o', ...
        'MarkerFaceColor', [0.60, 0.60, 0.60], ...
        'MarkerEdgeColor', 'w', 'LineWidth', 0.4);
    scatter(ax, i - 0.12, ...
        S.forward_mean_residual_C(S.case_name == caseNames(i)), ...
        42, 'd', 'MarkerFaceColor', 'w', ...
        'MarkerEdgeColor', [0.00, 0.35, 0.56], 'LineWidth', 1.1);
    scatter(ax, i + 0.12, ...
        S.hren_forward_mean_residual_C(S.case_name == caseNames(i)), ...
        42, 'd', 'MarkerFaceColor', 'w', ...
        'MarkerEdgeColor', [0.35, 0.35, 0.35], 'LineWidth', 1.1);
end
yline(ax, 0, '-', 'Color', [0.25, 0.25, 0.25], ...
    'HandleVisibility', 'off');
formatAxis(ax);
xlim(ax, [0.55, numel(caseNames) + 0.45]);
xticks(ax, 1:numel(caseNames));
xticklabels(ax, labels);
ax.XTickLabelRotation = 25;
xlabel(ax, 'Offset (degrees C) / fitted season');
h1 = scatter(ax, nan, nan, 22, 'o', 'MarkerFaceColor', ...
    [0.00, 0.45, 0.70], 'MarkerEdgeColor', 'w');
h2 = scatter(ax, nan, nan, 22, 'o', 'MarkerFaceColor', ...
    [0.60, 0.60, 0.60], 'MarkerEdgeColor', 'w');
lgd = legend(ax, [h1, h2], {'ML', 'Hren-Sheldon'}, ...
    'Location', 'southeast', 'Box', 'off', 'FontSize', 8);
lgd.TextColor = [0.08, 0.09, 0.10];
text(ax, 0.99, 0.98, 'Red outline: ML outside >3-km range', ...
    'Units', 'normalized', 'HorizontalAlignment', 'right', ...
    'VerticalAlignment', 'top', 'FontName', 'Arial', 'FontSize', 7.5, ...
    'Color', [0.45, 0.12, 0.08]);
end

function plotMLValidationBars(ax, T, task, models, subsetOrder, labels)
values = nan(numel(subsetOrder), numel(models));
for i = 1:numel(subsetOrder)
    for j = 1:numel(models)
        row = T.task == task & T.model == models(j) & ...
            T.subset == subsetOrder(i);
        if sum(row) ~= 1
            error('Incomplete ML validation metric for %s / %s / %s.', ...
                task, models(j), subsetOrder(i));
        end
        values(i, j) = T.rmse_C(row);
    end
end
b = bar(ax, values, 'grouped', 'EdgeColor', 'none');
b(1).FaceColor = [0.00, 0.45, 0.70];
b(2).FaceColor = [0.60, 0.60, 0.60];
formatAxis(ax);
xticks(ax, 1:numel(subsetOrder));
xticklabels(ax, labels);
ylabel(ax, 'Cross-validated RMSE (degrees C)');
lgd = legend(ax, {'ML', 'Linear'}, 'Location', 'northwest', ...
    'Box', 'off', 'FontSize', 8);
lgd.TextColor = [0.08, 0.09, 0.10];
end

function plotMLApplicationResiduals(ax, T, S, caseNames, labels, ...
    residualName, flagName)
hold(ax, 'on');
hasFlagged = false;
for i = 1:numel(caseNames)
    G = T(T.case_name == caseNames(i), :);
    y = G.(residualName);
    flagged = logical(G.(flagName));
    jitter = linspace(-0.14, 0.14, max(1, height(G)))';
    scatter(ax, i + jitter(~flagged), y(~flagged), 20, 'o', ...
        'MarkerFaceColor', [0.00, 0.45, 0.70], ...
        'MarkerEdgeColor', 'w', 'LineWidth', 0.4);
    if any(flagged)
        hasFlagged = true;
        scatter(ax, i + jitter(flagged), y(flagged), 28, 'o', ...
            'MarkerFaceColor', 'w', 'MarkerEdgeColor', [0.80, 0.20, 0.12], ...
            'LineWidth', 1.0);
    end
    if residualName == "residual_corrected_lake_minus_OPI_ML_Tw_warmest_C"
        meanValue = S.forward_mean_residual_C(S.case_name == caseNames(i));
    else
        meanValue = S.inverse_mean_residual_C(S.case_name == caseNames(i));
    end
    scatter(ax, i, meanValue, 46, 'd', 'MarkerFaceColor', 'w', ...
        'MarkerEdgeColor', [0.08, 0.09, 0.10], 'LineWidth', 1.1);
end
yline(ax, 0, '-', 'Color', [0.25, 0.25, 0.25], ...
    'HandleVisibility', 'off');
formatAxis(ax);
xlim(ax, [0.55, numel(caseNames) + 0.45]);
xticks(ax, 1:numel(caseNames));
xticklabels(ax, labels);
ax.XTickLabelRotation = 25;
xlabel(ax, 'Offset (degrees C) / fitted season');
if hasFlagged
    text(ax, 0.99, 0.98, 'Red outline: outside >3-km training range', ...
        'Units', 'normalized', 'HorizontalAlignment', 'right', ...
        'VerticalAlignment', 'top', 'FontName', 'Arial', 'FontSize', 7.5, ...
        'Color', [0.45, 0.12, 0.08]);
end
end

function copyMLArtifacts(targetRoot, modelRoot, rootScenario, caseNames)
modelTarget = fullfile(targetRoot, 'model');
if ~isfolder(modelTarget), mkdir(modelTarget); end
modelFiles = dir(fullfile(modelRoot, '*'));
modelFiles = modelFiles(~[modelFiles.isdir]);
for i = 1:numel(modelFiles)
    copyfile(fullfile(modelFiles(i).folder, modelFiles(i).name), ...
        fullfile(modelTarget, modelFiles(i).name));
end

for i = 1:numel(caseNames)
    source = fullfile(rootScenario, 'sensitivity_proxy_clumped', ...
        caseNames(i), 'proxy_clumped');
    target = fullfile(targetRoot, 'cases', caseNames(i));
    if ~isfolder(target), mkdir(target); end
    names = ["clumped_temperature_TerrazasWarmestML_comparison.csv", ...
        "clumped_temperature_HrenSheldon2012_comparison.csv", ...
        "TerrazasWarmestML_application.mat", ...
        "Fig_TerrazasWarmestML_Forward_vs_Clumped.png", ...
        "Fig_TerrazasWarmestML_Forward_vs_Clumped.fig", ...
        "Fig_TerrazasWarmestML_InferredAir_vs_OPI.png", ...
        "Fig_TerrazasWarmestML_InferredAir_vs_OPI.fig"];
    for j = 1:numel(names)
        sourceFile = fullfile(source, names(j));
        if ~isfile(sourceFile)
            error('Missing ML application artifact: %s', sourceFile);
        end
        copyfile(sourceFile, fullfile(target, names(j)));
    end
end
end

function value = rmsFinite(x)
keep = isfinite(x);
if ~any(keep)
    value = nan;
else
    value = sqrt(mean(x(keep).^2));
end
end

function formatAxis(ax)
ax.Color = 'w';
ax.XColor = [0.08, 0.09, 0.10];
ax.YColor = [0.08, 0.09, 0.10];
ax.FontName = 'Arial';
ax.FontSize = 8.5;
ax.LineWidth = 0.8;
ax.TickDir = 'out';
ax.Box = 'off';
ax.YGrid = 'on';
ax.XGrid = 'off';
ax.GridColor = [0.84, 0.84, 0.84];
ax.GridAlpha = 0.35;
ax.Title.Color = [0.08, 0.09, 0.10];
ax.XLabel.Color = [0.08, 0.09, 0.10];
ax.YLabel.Color = [0.08, 0.09, 0.10];
if isprop(ax, 'Toolbar') && ~isempty(ax.Toolbar)
    ax.Toolbar.Visible = 'off';
end
disableDefaultInteractivity(ax);
end

function addPanelLetter(ax, letter)
text(ax, -0.10, 1.03, letter, 'Units', 'normalized', ...
    'FontName', 'Arial', 'FontSize', 11, 'FontWeight', 'bold', ...
    'Color', [0.08, 0.09, 0.10], 'Clipping', 'off');
end

function exportFigureSet(fig, root, stem)
base = fullfile(root, stem);
exportgraphics(fig, base + ".png", 'Resolution', 300, ...
    'BackgroundColor', 'white');
exportgraphics(fig, base + ".pdf", 'ContentType', 'vector', ...
    'BackgroundColor', 'white');
savefig(fig, base + ".fig");
close(fig);
end

function copyRawTables(source, target)
files = dir(fullfile(source, '*.csv'));
for i = 1:numel(files)
    copyfile(fullfile(files(i).folder, files(i).name), ...
        fullfile(target, files(i).name));
end
end

function copyLegacyFigures(source, target)
files = dir(fullfile(source, '*.png'));
for i = 1:numel(files)
    copyfile(fullfile(files(i).folder, files(i).name), ...
        fullfile(target, files(i).name));
end
end

function copyConfiguration(opiRoot, rootScenario, target)
baseline = fullfile(rootScenario, 'oxygen_clumped_ultra_aggressive');
files = string({ ...
    fullfile(baseline, ...
        'Tibet_Eocene_30Ma_OxygenClumped_UltraAggressive.run'), ...
    fullfile(baseline, ...
        'Tibet_Eocene_30Ma_OxygenClumped_UltraAggressive_Best.run'), ...
    fullfile(opiRoot, 'scenarios', ...
        'Qiangtang_30Ma_4000m_dH045_T290_fP030_Valid', ...
        'SENSITIVITY_ANALYSIS_PLAN.md')});
for i = 1:numel(files)
    [~, name, extension] = fileparts(files(i));
    copyfile(files(i), fullfile(target, name + extension));
end
end

function copyScripts(opiRoot, target)
names = [ ...
    "opiPrepare_SensitivityManuscriptPackage.m", ...
    "opiAnalyze_ClumpedSensitivity.m", ...
    "opiPlot_QiangtangSensitivityMechanisms.m", ...
    "opiRun_ClumpedSensitivityCases.m", ...
    "opiSetup_ClumpedSensitivityCases.m", ...
    "opiDiagnose_QiangtangDivideD18OControls.m", ...
    "opiDiagnose_QiangtangDivideCorridorControls.m", ...
    "opiCompare_ClumpedTemperature.m", ...
    "lakeTransferML_TerrazasWarmest.m", ...
    "opiTrain_TerrazasWarmestML.m", ...
    "opiInferElevation_TerrazasWarmestML.m", ...
    "opiCalc_TwoWinds_OxygenOnly.m", ...
    "opiFit_TwoWinds_OxygenClumped.m"];
for i = 1:numel(names)
    copyfile(fullfile(opiRoot, 'OPI_programs', names(i)), ...
        fullfile(target, names(i)));
end
end

function writeFileInventory(packageRoot)
listing = dir(fullfile(packageRoot, '**', '*'));
listing = listing(~[listing.isdir]);
isInventory = strcmp({listing.name}, 'file_inventory.csv') & ...
    strcmp({listing.folder}, char(packageRoot));
listing = listing(~isInventory);
relativePath = strings(numel(listing), 1);
sizeBytes = nan(numel(listing), 1);
modified = strings(numel(listing), 1);
for i = 1:numel(listing)
    absolute = fullfile(listing(i).folder, listing(i).name);
    relativePath(i) = erase(string(absolute), string(packageRoot) + filesep);
    sizeBytes(i) = listing(i).bytes;
    modified(i) = string(datetime(listing(i).datenum, ...
        'ConvertFrom', 'datenum', 'Format', "yyyy-MM-dd'T'HH:mm:ss"));
end
inventory = table(relativePath, sizeBytes, modified, ...
    'VariableNames', {'relative_path', 'size_bytes', 'modified'});
writetable(sortrows(inventory, 'relative_path'), ...
    fullfile(packageRoot, 'file_inventory.csv'));
end
