function package = prepareManuscriptMaterials( ...
    packageRoot, outputRoot, experimentRoot)
% Build a self-contained manuscript figure and data package.


assimilationRoot = fileparts(fileparts(mfilename('fullpath')));
opiRoot = fullfile(assimilationRoot, '..', 'OPI_matlab');
if nargin < 1 || strlength(string(packageRoot)) == 0
    packageRoot = fullfile(assimilationRoot, 'manuscript_package_30Ma');
end
if nargin < 2 || strlength(string(outputRoot)) == 0
    outputRoot = fullfile(assimilationRoot, 'results', ...
        'topography_north_south_grid_coarse', ...
        'westerhold_age_marginalized_sensitivity');
end
if nargin < 3 || strlength(string(experimentRoot)) == 0
    experimentRoot = fullfile(opiRoot, 'scenarios', ...
        'Qiangtang_30Ma_4000m_dH045_T290_fP030_Valid_platform_northsouth', ...
        'topography_north_south_grid_coarse');
end

figure1Root = fullfile(packageRoot, 'main_figures', ...
    'Figure_1_Paleotopography');
figure2Root = fullfile(packageRoot, 'main_figures', ...
    'Figure_2_Xiong_Sensitivity');
s1Root = fullfile(packageRoot, 'sensitivity_analysis', ...
    'Figure_S1_Temporal_Scale');
s2Root = fullfile(packageRoot, 'sensitivity_analysis', ...
    'Figure_S2_Literature_Weight');
configurationRoot = fullfile(packageRoot, 'configuration');
referenceDataRoot = fullfile(packageRoot, 'reference_data');
scriptsRoot = fullfile(packageRoot, 'scripts');
roots = string({packageRoot, figure1Root, figure2Root, s1Root, s2Root, ...
    configurationRoot, referenceDataRoot, scriptsRoot, ...
    fullfile(figure1Root, 'data'), ...
    fullfile(figure2Root, 'data'), fullfile(s1Root, 'data'), ...
    fullfile(s2Root, 'data')});
for i = 1:numel(roots)
    if ~isfolder(roots(i))
        mkdir(roots(i));
    end
end

fullCase = readtable(fullfile(outputRoot, ...
    'north_south_case_posterior.csv'), 'TextType', 'string');
withoutCase = readtable(fullfile(outputRoot, ...
    'north_south_without_xiong_case_posterior.csv'), 'TextType', 'string');
[fullMarginal, fullSummary] = aggregateUnits(fullCase);
[withoutMarginal, withoutSummary] = aggregateUnits(withoutCase);

writeMainFigureData(fullfile(figure1Root, 'data'), ...
    fullMarginal, fullSummary, "with_Xiong");
renderMainFigure1(figure1Root, "en", fullMarginal, fullSummary);

writeMainFigureData(fullfile(figure2Root, 'data'), ...
    fullMarginal, fullSummary, "with_Xiong");
writeMainFigureData(fullfile(figure2Root, 'data'), ...
    withoutMarginal, withoutSummary, "without_Xiong");
copyFigureSet(outputRoot, figure2Root, ...
    'Fig_NorthSouth_Xiong_Comparison_EN', 'Figure_2_Xiong_Sensitivity_EN');

manifest = readtable(fullfile(experimentRoot, 'design', ...
    'case_manifest.csv'), 'TextType', 'string');
[temporalPosterior, temporalSummary] = ...
    buildTemporalSensitivity(outputRoot, manifest);
writetable(temporalPosterior, fullfile(s1Root, 'data', ...
    'Figure_S1_temporal_scale_unit_posterior.csv'));
writetable(temporalSummary, fullfile(s1Root, 'data', ...
    'Figure_S1_temporal_scale_unit_summary.csv'));
copyfile(fullfile(outputRoot, 'temporal_rate_scale_posterior.csv'), ...
    fullfile(s1Root, 'data', 'temporal_scale_model_weights.csv'));
renderSensitivityFigure(s1Root, "en", temporalPosterior, ...
    temporalSummary, "temporal");

[weightPosterior, weightSummary] = ...
    buildWeightSensitivity(outputRoot, manifest);
writetable(weightPosterior, fullfile(s2Root, 'data', ...
    'Figure_S2_literature_weight_unit_posterior.csv'));
writetable(weightSummary, fullfile(s2Root, 'data', ...
    'Figure_S2_literature_weight_unit_summary.csv'));
renderSensitivityFigure(s2Root, "en", weightPosterior, ...
    weightSummary, "weight");

copyConfiguration(assimilationRoot, opiRoot, experimentRoot, ...
    configurationRoot);
copyReferenceData(opiRoot, referenceDataRoot);
copyScripts(assimilationRoot, opiRoot, scriptsRoot);
writeFileInventory(packageRoot);

package = struct;
package.root = string(packageRoot);
package.figure1 = string(figure1Root);
package.figure2 = string(figure2Root);
package.figureS1 = string(s1Root);
package.figureS2 = string(s2Root);
fprintf('Wrote manuscript materials to:\n%s\n', packageRoot);
end

function writeMainFigureData(dataRoot, marginal, summary, scenario)
prefix = char(scenario);
writetable(marginal, fullfile(dataRoot, ...
    [prefix, '_unit_elevation_posterior.csv']));
writetable(summary, fullfile(dataRoot, ...
    [prefix, '_paleotopography_summary.csv']));
[bandGrid, profile] = makeBandPlotData(marginal, summary);
writetable(bandGrid, fullfile(dataRoot, ...
    [prefix, '_posterior_band_plot_grid.csv']));
writetable(profile, fullfile(dataRoot, ...
    [prefix, '_posterior_profile_plot_lines.csv']));
end

function [bandGrid, profile] = makeBandPlotData(marginal, summary)
unitOrder = ["Qiangtang", "Central_valley_zone", "Gangdese"];
x = (1:3)';
xFine = linspace(1, 3, 401);
yKm = linspace(0.75, 5.75, 501)';
bandwidthKm = 0.20;
densityAtUnit = zeros(numel(yKm), 3);
for i = 1:3
    rows = marginal.unit == unitOrder(i);
    zKm = marginal.elevationM(rows) ./ 1000;
    probability = marginal.posterior_probability(rows);
    probability = probability ./ sum(probability);
    offset = (yKm - zKm') ./ bandwidthKm;
    kernel = exp(-0.5 .* offset.^2) ./ ...
        (sqrt(2*pi) .* bandwidthKm);
    densityAtUnit(:, i) = kernel * probability;
end
probabilityPerHalfKm = 0.5 .* ...
    interp1(x, densityAtUnit', xFine, 'pchip')';
probabilityPerHalfKm = max(probabilityPerHalfKm, 0);
[xGrid, yGrid] = meshgrid(xFine, yKm);
bandGrid = table(xGrid(:), yGrid(:), probabilityPerHalfKm(:), ...
    'VariableNames', {'north_south_position', 'elevation_km', ...
    'posterior_probability_per_0p5km'});

meanKm = summary.posterior_mean_elevation_m ./ 1000;
p16Km = summary.p16_elevation_m ./ 1000;
p84Km = summary.p84_elevation_m ./ 1000;
profile = table(xFine', ...
    interp1(x, meanKm, xFine, 'linear')', ...
    interp1(x, p16Km, xFine, 'linear')', ...
    interp1(x, p84Km, xFine, 'linear')', ...
    'VariableNames', {'north_south_position', 'posterior_mean_km', ...
    'p16_km', 'p84_km'});
end

function renderMainFigure1(root, ~, marginal, summary)
fontName = 'Arial';
labels = ["Qiangtang", "Central valley", "Gangdese"];
xLabel = 'North  →  South';
yLabel = 'Elevation (km)';
meanLabel = 'Posterior mean';
boundLabel = '16–84% posterior bounds';
colorLabel = 'Posterior probability per 0.5 km elevation band';
suffix = 'EN';
[grid, profile] = makeBandPlotData(marginal, summary);
xFine = unique(grid.north_south_position, 'stable')';
yKm = unique(grid.elevation_km, 'stable');
probability = reshape(grid.posterior_probability_per_0p5km, ...
    numel(yKm), numel(xFine));
fig = figure('Color', 'w', 'Position', [100, 100, 900, 610]);
ax = axes(fig);
hold(ax, 'on');
if isprop(ax, 'Toolbar') && ~isempty(ax.Toolbar)
    ax.Toolbar.Visible = 'off';
end
disableDefaultInteractivity(ax);
imageData = min(probability, 0.5);
hImage = imagesc(ax, xFine, yKm, imageData);
hImage.AlphaData = (imageData ./ 0.5).^0.68;
hImage.AlphaData(imageData < 0.005) = 0;
hImage.AlphaDataMapping = 'none';
hImage.HandleVisibility = 'off';
ax.YDir = 'normal';
palette = manuscriptPalette();
colormap(ax, palette);
clim(ax, [0, 0.5]);
hMean = plot(ax, profile.north_south_position, ...
    profile.posterior_mean_km, '-', 'Color', [0.08, 0.09, 0.10], ...
    'LineWidth', 1.9, 'DisplayName', meanLabel);
hBounds = plot(ax, profile.north_south_position, profile.p16_km, '--', ...
    'Color', [0.20, 0.25, 0.27], 'LineWidth', 1.05, ...
    'DisplayName', boundLabel);
plot(ax, profile.north_south_position, profile.p84_km, '--', ...
    'Color', [0.20, 0.25, 0.27], 'LineWidth', 1.05, ...
    'HandleVisibility', 'off');
meanKm = summary.posterior_mean_elevation_m ./ 1000;
scatter(ax, 1:3, meanKm, 34, 'o', 'MarkerFaceColor', 'w', ...
    'MarkerEdgeColor', [0.08, 0.09, 0.10], 'LineWidth', 1.1, ...
    'HandleVisibility', 'off');
for i = 1:3
    text(ax, i, meanKm(i) + 0.09, sprintf('%.2f', meanKm(i)), ...
        'HorizontalAlignment', 'center', 'FontName', fontName, ...
        'FontSize', 8.5, 'Color', [0.08, 0.09, 0.10]);
end
formatManuscriptAxis(ax, fontName, labels, xLabel, yLabel, true);
legend(ax, [hMean, hBounds], {meanLabel, boundLabel}, ...
    'Location', 'northeast', 'FontName', fontName, ...
    'FontSize', 8.5, 'Box', 'off', 'TextColor', [0.08, 0.09, 0.10]);
cb = colorbar(ax, 'eastoutside');
cb.Ticks = 0:0.1:0.5;
cb.Label.String = colorLabel;
cb.Label.FontName = fontName;
cb.Label.FontSize = 9;
cb.FontName = fontName;
cb.FontSize = 8.5;
cb.TickDirection = 'out';
cb.LineWidth = 0.7;
cb.Color = [0.08, 0.09, 0.10];
cb.Label.Color = [0.08, 0.09, 0.10];
base = fullfile(root, "Figure_1_Paleotopography_" + suffix);
exportgraphics(fig, base + ".png", 'Resolution', 300, ...
    'BackgroundColor', 'white');
exportgraphics(fig, base + ".pdf", 'ContentType', 'image', ...
    'Resolution', 300, 'BackgroundColor', 'white');
savefig(fig, base + ".fig");
close(fig);
end

function [posterior, summary] = buildTemporalSensitivity(outputRoot, manifest)
S = readtable(fullfile(outputRoot, ...
    'temporal_uncertainty_sensitivity_summary.csv'), 'TextType', 'string');
posterior = table();
summary = table();
for i = 1:height(S)
    label = "scale_" + replace(compose('%.1f', ...
        S.temporal_rate_scale(i)), '.', 'p');
    cases = readtable(fullfile(outputRoot, label, 'combined_blocks', ...
        'combined_block_case_posterior.csv'), 'TextType', 'string');
    casePosterior = attachElevations(cases.case_id, ...
        cases.combined_primary_probability, manifest);
    [marginal, unitSummary] = aggregateUnits(casePosterior);
    marginal = addvars(marginal, ...
        repmat(S.temporal_rate_scale(i), height(marginal), 1), ...
        'Before', 1, 'NewVariableNames', 'temporal_rate_scale');
    unitSummary = addvars(unitSummary, ...
        repmat(S.temporal_rate_scale(i), height(unitSummary), 1), ...
        'Before', 1, 'NewVariableNames', 'temporal_rate_scale');
    posterior = [posterior; marginal]; %#ok<AGROW>
    summary = [summary; unitSummary]; %#ok<AGROW>
end
end

function [posterior, summary] = buildWeightSensitivity(outputRoot, manifest)
cases = readtable(fullfile(outputRoot, 'scale_1p0', 'combined_blocks', ...
    'combined_block_case_posterior.csv'), 'TextType', 'string');
values = [0.25, 0.50, 1.00];
variables = ["combined_external_w0p25", ...
    "combined_external_w0p50", "combined_external_w1p00"];
posterior = table();
summary = table();
for i = 1:numel(values)
    casePosterior = attachElevations(cases.case_id, ...
        cases.(variables(i)), manifest);
    [marginal, unitSummary] = aggregateUnits(casePosterior);
    marginal = addvars(marginal, repmat(values(i), height(marginal), 1), ...
        'Before', 1, 'NewVariableNames', 'literature_block_weight');
    unitSummary = addvars(unitSummary, ...
        repmat(values(i), height(unitSummary), 1), 'Before', 1, ...
        'NewVariableNames', 'literature_block_weight');
    posterior = [posterior; marginal]; %#ok<AGROW>
    summary = [summary; unitSummary]; %#ok<AGROW>
end
end

function casePosterior = attachElevations(caseId, probability, manifest)
[found, order] = ismember(caseId, manifest.case_id);
if ~all(found)
    error('Sensitivity cases do not align with the case manifest.');
end
M = manifest(order, :);
centralM = M.valley_target_m;
ramp = ~isfinite(centralM);
centralM(ramp) = 0.5 .* (M.qiangtang_target_m(ramp) + ...
    M.gangdese_target_m(ramp));
casePosterior = table(caseId, M.qiangtang_target_m, centralM, ...
    M.gangdese_target_m, M.valley_mode, probability, ...
    'VariableNames', {'case_id', 'qiangtang_elevation_m', ...
    'central_valley_zone_elevation_m', 'gangdese_elevation_m', ...
    'central_morphology', 'posterior_probability'});
end

function renderSensitivityFigure(root, ~, posterior, summary, mode)
if mode == "temporal"
    parameterName = 'temporal_rate_scale';
    parameterValues = unique(summary.temporal_rate_scale);
    xLabel = 'Residual temporal-error scale';
    panelA = 'Posterior elevation by tectonic unit';
    panelB = 'Central-valley elevation probability';
    baseStem = 'Figure_S1_Temporal_Scale';
else
    parameterName = 'literature_block_weight';
    parameterValues = unique(summary.literature_block_weight);
    xLabel = 'Literature-data block weight';
    panelA = 'Posterior elevation by tectonic unit';
    panelB = 'Central-valley elevation probability';
    baseStem = 'Figure_S2_Literature_Weight';
end
fontName = 'Arial';
unitLabels = ["Qiangtang", "Central valley", "Gangdese"];
yLabel = 'Elevation (km)';
probabilityLabel = 'Posterior probability';
suffix = 'EN';
unitOrder = ["Qiangtang", "Central_valley_zone", "Gangdese"];
unitColors = [0.00, 0.45, 0.70; 0.84, 0.37, 0.00; 0.00, 0.62, 0.45];
markers = {'o', 's', 'd'};
fig = figure('Color', 'w', 'Position', [100, 100, 1180, 500]);
layout = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', ...
    'Padding', 'compact');

ax1 = nexttile(layout, 1);
hold(ax1, 'on');
disableAxesInteractivity(ax1);
for u = 1:3
    rows = summary.unit == unitOrder(u);
    T = sortrows(summary(rows, :), parameterName);
    x = T.(parameterName);
    y = T.posterior_mean_elevation_m ./ 1000;
    low = y - T.p16_elevation_m ./ 1000;
    high = T.p84_elevation_m ./ 1000 - y;
    errorbar(ax1, x, y, low, high, ['-', markers{u}], ...
        'Color', unitColors(u, :), 'MarkerFaceColor', 'w', ...
        'LineWidth', 1.25, 'MarkerSize', 5, ...
        'DisplayName', unitLabels(u));
end
formatSensitivityAxis(ax1, fontName, xLabel, yLabel, parameterValues);
title(ax1, panelA, 'FontName', fontName, 'FontWeight', 'normal', ...
    'FontSize', 10.5, 'Color', [0.08, 0.09, 0.10]);
legend(ax1, 'Location', 'best', 'Box', 'off', 'FontName', fontName, ...
    'FontSize', 8, 'TextColor', [0.08, 0.09, 0.10]);
addPanelLetter(ax1, 'a');

ax2 = nexttile(layout, 2);
hold(ax2, 'on');
disableAxesInteractivity(ax2);
settingColors = [0.20, 0.47, 0.67; 0.35, 0.70, 0.62; 0.84, 0.37, 0.16];
for i = 1:numel(parameterValues)
    rows = posterior.unit == "Central_valley_zone" & ...
        posterior.(parameterName) == parameterValues(i);
    T = sortrows(posterior(rows, :), 'elevationM');
    plot(ax2, T.elevationM ./ 1000, T.posterior_probability, '-o', ...
        'Color', settingColors(i, :), 'MarkerFaceColor', 'w', ...
        'LineWidth', 1.25, 'MarkerSize', 4, ...
        'DisplayName', char(compose('%.2g', parameterValues(i))));
end
formatSensitivityAxis(ax2, fontName, yLabel, probabilityLabel, ...
    1.5:0.5:5.0);
xlim(ax2, [1.35, 5.15]);
title(ax2, panelB, 'FontName', fontName, 'FontWeight', 'normal', ...
    'FontSize', 10.5, 'Color', [0.08, 0.09, 0.10]);
lgd = legend(ax2, 'Location', 'best', 'Box', 'off', ...
    'FontName', fontName, 'FontSize', 8, ...
    'TextColor', [0.08, 0.09, 0.10]);
lgd.Title.String = xLabel;
lgd.Title.FontName = fontName;
addPanelLetter(ax2, 'b');

base = fullfile(root, baseStem + "_" + suffix);
exportgraphics(fig, base + ".png", 'Resolution', 300, ...
    'BackgroundColor', 'white');
exportgraphics(fig, base + ".pdf", 'ContentType', 'vector', ...
    'BackgroundColor', 'white');
savefig(fig, base + ".fig");
close(fig);
end

function formatSensitivityAxis(ax, fontName, xLabel, yLabel, xTicks)
grid(ax, 'off');
ax.Color = 'w';
ax.XColor = [0.08, 0.09, 0.10];
ax.YColor = [0.08, 0.09, 0.10];
ax.YGrid = 'on';
ax.GridColor = [0.84, 0.84, 0.84];
ax.GridAlpha = 0.35;
ax.Box = 'off';
ax.TickDir = 'out';
ax.LineWidth = 0.8;
ax.FontName = fontName;
ax.FontSize = 8.5;
xticks(ax, xTicks);
xlabel(ax, xLabel, 'FontName', fontName, 'FontSize', 9);
ylabel(ax, yLabel, 'FontName', fontName, 'FontSize', 9);
end

function formatManuscriptAxis(ax, fontName, labels, xLabel, yLabel, showY)
grid(ax, 'off');
ax.Color = 'w';
ax.XColor = [0.08, 0.09, 0.10];
ax.YColor = [0.08, 0.09, 0.10];
ax.YGrid = 'on';
ax.GridColor = [0.84, 0.84, 0.84];
ax.GridAlpha = 0.35;
ax.Box = 'off';
ax.TickDir = 'out';
ax.LineWidth = 0.8;
ax.FontName = fontName;
ax.FontSize = 9.5;
ax.Layer = 'top';
xlim(ax, [0.55, 3.45]);
ylim(ax, [0.75, 5.75]);
xticks(ax, 1:3);
xticklabels(ax, labels);
yticks(ax, 1:0.5:5.5);
xlabel(ax, xLabel, 'FontName', fontName, 'FontSize', 10);
if showY
    ylabel(ax, yLabel, 'FontName', fontName, 'FontSize', 10);
end
end

function addPanelLetter(ax, letter)
text(ax, -0.08, 1.03, letter, 'Units', 'normalized', ...
    'FontName', 'Arial', 'FontSize', 11, 'FontWeight', 'bold', ...
    'Color', [0.08, 0.09, 0.10], 'Clipping', 'off');
end

function disableAxesInteractivity(ax)
if isprop(ax, 'Toolbar') && ~isempty(ax.Toolbar)
    ax.Toolbar.Visible = 'off';
end
disableDefaultInteractivity(ax);
end

function palette = manuscriptPalette()
anchors = [0.98, 0.98, 0.97; 0.76, 0.88, 0.84; ...
    0.30, 0.65, 0.63; 0.06, 0.34, 0.46; 0.82, 0.35, 0.16];
palette = interp1(linspace(0, 1, size(anchors, 1)), anchors, ...
    linspace(0, 1, 256), 'pchip');
end

function [marginal, summary] = aggregateUnits(casePosterior)
unitOrder = ["Qiangtang"; "Central_valley_zone"; "Gangdese"];
unit = [repmat(unitOrder(1), height(casePosterior), 1); ...
    repmat(unitOrder(2), height(casePosterior), 1); ...
    repmat(unitOrder(3), height(casePosterior), 1)];
elevationM = [casePosterior.qiangtang_elevation_m; ...
    casePosterior.central_valley_zone_elevation_m; ...
    casePosterior.gangdese_elevation_m];
probability = repmat(casePosterior.posterior_probability, 3, 1);
marginal = table(unit, elevationM, probability);
marginal = groupsummary(marginal, {'unit', 'elevationM'}, ...
    'sum', 'probability');
marginal.GroupCount = [];
marginal.Properties.VariableNames{'sum_probability'} = ...
    'posterior_probability';
marginal = sortrows(marginal, {'unit', 'elevationM'});
modeM = nan(3, 1);
meanM = nan(3, 1);
p16M = nan(3, 1);
p84M = nan(3, 1);
for i = 1:3
    rows = marginal.unit == unitOrder(i);
    z = marginal.elevationM(rows);
    p = marginal.posterior_probability(rows);
    [z, order] = sort(z);
    p = p(order) ./ sum(p);
    [~, modeIndex] = max(p);
    modeM(i) = z(modeIndex);
    meanM(i) = sum(z .* p);
    cdf = cumsum(p);
    p16M(i) = z(find(cdf >= 0.16, 1));
    p84M(i) = z(find(cdf >= 0.84, 1));
end
summary = table((1:3)', unitOrder, modeM, meanM, p16M, p84M, ...
    'VariableNames', {'north_to_south_order', 'unit', ...
    'highest_probability_elevation_m', 'posterior_mean_elevation_m', ...
    'p16_elevation_m', 'p84_elevation_m'});
end

function copyFigureSet(sourceRoot, targetRoot, sourceStem, targetStem)
extensions = [".png", ".pdf", ".fig"];
for i = 1:numel(extensions)
    copyfile(fullfile(sourceRoot, sourceStem + extensions(i)), ...
        fullfile(targetRoot, targetStem + extensions(i)));
end
end

function copyConfiguration(assimilationRoot, opiRoot, experimentRoot, target)
files = string({ ...
    fullfile(assimilationRoot, 'config', ...
        'collected_carbonate_assimilation_config.csv'), ...
    fullfile(assimilationRoot, 'config', 'assimilation_block_weights.csv'), ...
    fullfile(assimilationRoot, 'config', 'observation_chronology.csv'), ...
    fullfile(assimilationRoot, 'config', ...
        'observation_operator_assignments.csv'), ...
    fullfile(opiRoot, 'data', 'assimilation', ...
        'qiangtang_assimilation_config_3500_5500.csv'), ...
    fullfile(experimentRoot, 'design', 'case_manifest.csv'), ...
    fullfile(experimentRoot, 'design', 'topography_quality_control.csv'), ...
    fullfile(experimentRoot, 'assimilation', ...
        'assimilation_observation_summary.csv')});
for i = 1:numel(files)
    copyfile(files(i), fullfile(target, ...
        string(extractAfter(files(i), max(strfind(files(i), filesep))))));
end
end

function copyReferenceData(opiRoot, target)
files = string({ ...
    fullfile(opiRoot, 'data', 'reference', 'LakeTransferFunction', ...
        'ERA5_LakeTemp.csv'), ...
    fullfile(opiRoot, 'data', 'reference', 'LakeTransferFunction', ...
        'README.md'), ...
    fullfile(opiRoot, 'data', 'derived', 'LakeTransferFunction', ...
        'TerrazasWarmestML', 'TerrazasWarmestML_model.mat')});
for i = 1:numel(files)
    [~, name, extension] = fileparts(files(i));
    copyfile(files(i), fullfile(target, name + extension));
end
end

function copyScripts(assimilationRoot, opiRoot, target)
files = string({ ...
    fullfile(assimilationRoot, 'src', ...
        'runCollectedCarbonateAssimilation.m'), ...
    fullfile(assimilationRoot, 'src', 'temporalAgeQuadrature.m'), ...
    fullfile(assimilationRoot, 'src', 'kimONeil1997CalciteWater.m'), ...
    fullfile(assimilationRoot, 'src', ...
        'lakeTransferAirToLake_Terrazas2025.m'), ...
    fullfile(assimilationRoot, 'src', ...
        'paleosolTemperature_Xiong2022.m'), ...
    fullfile(assimilationRoot, 'src', ...
        'evaluatePaleosolTemperatureOperator.m'), ...
    fullfile(assimilationRoot, 'src', ...
        'evaluatePaleosolCarbonateOperator.m'), ...
    fullfile(assimilationRoot, 'src', 'weightedBlockPosterior.m'), ...
    fullfile(assimilationRoot, 'src', 'combineAssimilationBlocks.m'), ...
    fullfile(assimilationRoot, 'src', ...
        'plotNorthSouthXiongComparison.m'), ...
    fullfile(assimilationRoot, 'src', ...
        'reportNorthSouthPaleotopography.m'), ...
    fullfile(assimilationRoot, 'src', 'prepareManuscriptMaterials.m'), ...
    fullfile(assimilationRoot, 'run_temporal_uncertainty_sensitivity.m'), ...
    fullfile(opiRoot, 'OPI_programs', ...
        'opiAssimilate_TopographyScenarios.m'), ...
    fullfile(opiRoot, 'OPI_programs', 'opiAssimilationLikelihood.m'), ...
    fullfile(opiRoot, 'OPI_programs', ...
        'lakeTransferML_TerrazasWarmest.m'), ...
    fullfile(opiRoot, 'OPI_programs', 'opiSetup_NorthSouthElevationGrid.m')});
for i = 1:numel(files)
    [~, name, extension] = fileparts(files(i));
    copyfile(files(i), fullfile(target, name + extension));
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
