function files = plotNorthSouthXiongComparison( ...
    outputRoot, ownCaseFile, weightFile, manifestFile)
% Plot English north-south posterior bands with and without Xiong data.


assimilationRoot = fileparts(fileparts(mfilename('fullpath')));
opiRoot = fullfile(assimilationRoot, '..', 'OPI_matlab');
experimentRoot = fullfile(opiRoot, 'scenarios', ...
    'Qiangtang_30Ma_4000m_dH045_T290_fP030_Valid_platform_northsouth', ...
    'topography_north_south_grid_coarse');
if nargin < 1 || strlength(string(outputRoot)) == 0
    outputRoot = fullfile(assimilationRoot, 'results', ...
        'topography_north_south_grid_coarse', ...
        'westerhold_age_marginalized_sensitivity');
end
if nargin < 2 || strlength(string(ownCaseFile)) == 0
    ownCaseFile = fullfile(experimentRoot, 'assimilation', ...
        'assimilation_case_posterior.csv');
end
if nargin < 3 || strlength(string(weightFile)) == 0
    weightFile = fullfile(assimilationRoot, 'config', ...
        'assimilation_block_weights.csv');
end
if nargin < 4 || strlength(string(manifestFile)) == 0
    manifestFile = fullfile(experimentRoot, 'design', 'case_manifest.csv');
end

sensitivityFile = fullfile(outputRoot, ...
    'temporal_uncertainty_sensitivity_summary.csv');
fullCaseFile = fullfile(outputRoot, 'north_south_case_posterior.csv');
required = string({ownCaseFile, weightFile, manifestFile, ...
    sensitivityFile, fullCaseFile});
missing = required(~isfile(required));
if ~isempty(missing)
    error('Missing Xiong-comparison input(s):\n%s', strjoin(missing, newline));
end

own = readtable(ownCaseFile, 'TextType', 'string');
weights = readtable(weightFile, 'TextType', 'string');
manifest = readtable(manifestFile, 'TextType', 'string');
sensitivity = readtable(sensitivityFile, 'TextType', 'string');
fullCase = readtable(fullCaseFile, 'TextType', 'string');
primaryWeight = weights.external_weight(lower(weights.status) == "primary");
if ~isscalar(primaryWeight)
    error('Exactly one primary literature-data weight is required.');
end
[found, order] = ismember(own.case_id, manifest.case_id);
if ~all(found)
    error('Case manifest does not contain every local-likelihood case.');
end
manifest = manifest(order, :);

scale = sensitivity.temporal_rate_scale;
nScale = numel(scale);
withoutXiongLogWeight = nan(height(own), nScale);
for i = 1:nScale
    label = "scale_" + replace(compose('%.1f', scale(i)), '.', 'p');
    caseFile = fullfile(outputRoot, label, ...
        'collected_proxy_case_posterior.csv');
    predictionFile = fullfile(outputRoot, label, ...
        'collected_proxy_predictions.csv');
    if ~isfile(caseFile) || ~isfile(predictionFile)
        error('Missing temporal-scale output for %s.', label);
    end
    external = readtable(caseFile, 'TextType', 'string');
    prediction = readtable(predictionFile, 'TextType', 'string');
    xiong = prediction(prediction.source == "Xiong_et_al_2022_SA", :);
    [externalFound, externalOrder] = ismember(own.case_id, external.case_id);
    [xiongFound, xiongOrder] = ismember(own.case_id, xiong.case_id);
    if ~all(externalFound) || ~all(xiongFound) || height(xiong) ~= height(own)
        error('Xiong leave-one-out case alignment failed for %s.', label);
    end
    literatureWithoutXiong = external.log_likelihood_joint(externalOrder) - ...
        xiong.log_likelihood_joint(xiongOrder);
    withoutXiongLogWeight(:, i) = log(own.prior_weight) + ...
        own.log_evidence_joint + primaryWeight .* literatureWithoutXiong - ...
        log(nScale);
end
maximum = max(withoutXiongLogWeight, [], 'all');
withoutXiongJoint = exp(withoutXiongLogWeight - maximum);
withoutXiongJoint = withoutXiongJoint ./ sum(withoutXiongJoint, 'all');
withoutXiongProbability = sum(withoutXiongJoint, 2);

centralTargetM = manifest.valley_target_m;
ramp = ~isfinite(centralTargetM);
centralTargetM(ramp) = 0.5 .* (manifest.qiangtang_target_m(ramp) + ...
    manifest.gangdese_target_m(ramp));
withoutXiongCase = table(own.case_id, manifest.qiangtang_target_m, ...
    centralTargetM, manifest.gangdese_target_m, manifest.valley_mode, ...
    withoutXiongProbability, 'VariableNames', {'case_id', ...
    'qiangtang_elevation_m', 'central_valley_zone_elevation_m', ...
    'gangdese_elevation_m', 'central_morphology', ...
    'posterior_probability'});

[fullMarginal, fullSummary] = aggregateUnits(fullCase);
[withoutMarginal, withoutSummary] = aggregateUnits(withoutXiongCase);
writetable(withoutXiongCase, fullfile(outputRoot, ...
    'north_south_without_xiong_case_posterior.csv'));
writetable(withoutMarginal, fullfile(outputRoot, ...
    'north_south_without_xiong_unit_elevation_posterior.csv'));
writetable(withoutSummary, fullfile(outputRoot, ...
    'north_south_without_xiong_paleotopography_summary.csv'));

scenario = [repmat("with_Xiong", 3, 1); ...
    repmat("without_Xiong", 3, 1)];
comparison = [fullSummary; withoutSummary];
comparison = addvars(comparison, scenario, 'Before', 1, ...
    'NewVariableNames', 'scenario');
writetable(comparison, fullfile(outputRoot, ...
    'north_south_xiong_sensitivity_summary.csv'));

bandWith = buildBand(fullMarginal);
bandWithout = buildBand(withoutMarginal);
edges = 0:0.1:0.5;

en = renderComparison(outputRoot, "en", bandWith, bandWithout, ...
    fullSummary, withoutSummary, edges);
files = struct('en', en, ...
    'withoutXiongCaseCsv', string(fullfile(outputRoot, ...
    'north_south_without_xiong_case_posterior.csv')), ...
    'comparisonSummaryCsv', string(fullfile(outputRoot, ...
    'north_south_xiong_sensitivity_summary.csv')));
fprintf('Wrote English Xiong comparison figure to:\n%s\n', outputRoot);
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

function band = buildBand(marginal)
unitOrder = ["Qiangtang", "Central_valley_zone", "Gangdese"];
x = (1:3)';
yKm = linspace(0.75, 5.75, 501)';
bandwidthKm = 0.20;
densityAtUnit = zeros(numel(yKm), 3);
for i = 1:3
    rows = marginal.unit == unitOrder(i);
    zKm = marginal.elevationM(rows) ./ 1000;
    probability = marginal.posterior_probability(rows);
    probability = probability ./ sum(probability);
    offset = (yKm - zKm') ./ bandwidthKm;
    kernels = exp(-0.5 .* offset.^2) ./ ...
        (sqrt(2*pi) .* bandwidthKm);
    densityAtUnit(:, i) = kernels * probability;
end
xFine = linspace(1, 3, 401);
band = struct;
band.x = x;
band.xFine = xFine;
band.yKm = yKm;
band.probability = 0.5 .* ...
    interp1(x, densityAtUnit', xFine, 'pchip')';
band.probability = max(band.probability, 0);
end

function files = renderComparison(outputRoot, ~, withBand, ...
    withoutBand, withSummary, withoutSummary, edges)
fontName = 'Arial';
unitLabels = ["Qiangtang", "Central valley", "Gangdese"];
panelTitles = ["With Xiong et al. (2022)", ...
    "Without Xiong et al. (2022)"];
yLabel = 'Elevation (km)';
xLabel = 'North  →  South';
meanLegend = 'Posterior mean';
intervalLegend = '16–84% posterior bounds';
colorLabel = 'Posterior probability per 0.5 km elevation band';
suffix = 'EN';

fig = figure('Color', 'w', 'Position', [80, 100, 1650, 620]);
layout = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', ...
    'Padding', 'compact');
bands = {withBand, withoutBand};
summaries = {withSummary, withoutSummary};
axesList = gobjects(2, 1);
legendHandles = gobjects(2, 1);
paletteAnchors = [0.98, 0.98, 0.97; ...
    0.76, 0.88, 0.84; ...
    0.30, 0.65, 0.63; ...
    0.06, 0.34, 0.46; ...
    0.82, 0.35, 0.16];
palette = interp1(linspace(0, 1, size(paletteAnchors, 1)), ...
    paletteAnchors, linspace(0, 1, 256), 'pchip');
for panel = 1:2
    ax = nexttile(layout, panel);
    axesList(panel) = ax;
    if isprop(ax, 'Toolbar') && ~isempty(ax.Toolbar)
        ax.Toolbar.Visible = 'off';
    end
    disableDefaultInteractivity(ax);
    hold(ax, 'on');
    band = bands{panel};
    summary = summaries{panel};
    imageData = min(band.probability, edges(end));
    alphaData = (imageData ./ edges(end)).^0.68;
    alphaData(imageData < 0.005) = 0;
    densityImage = imagesc(ax, band.xFine, band.yKm, imageData);
    densityImage.AlphaData = alphaData;
    densityImage.AlphaDataMapping = 'none';
    densityImage.HandleVisibility = 'off';
    ax.YDir = 'normal';

    meanKm = summary.posterior_mean_elevation_m ./ 1000;
    p16Km = summary.p16_elevation_m ./ 1000;
    p84Km = summary.p84_elevation_m ./ 1000;
    meanFine = interp1(band.x, meanKm, band.xFine, 'linear');
    p16Fine = interp1(band.x, p16Km, band.xFine, 'linear');
    p84Fine = interp1(band.x, p84Km, band.xFine, 'linear');
    boundaryColor = [0.20, 0.25, 0.27];
    meanColor = [0.08, 0.09, 0.10];
    hBoundary = plot(ax, band.xFine, p16Fine, '--', ...
        'Color', boundaryColor, 'LineWidth', 1.05, ...
        'DisplayName', intervalLegend);
    plot(ax, band.xFine, p84Fine, '--', 'Color', boundaryColor, ...
        'LineWidth', 1.05, 'HandleVisibility', 'off');
    hMean = plot(ax, band.xFine, meanFine, '-', 'Color', meanColor, ...
        'LineWidth', 1.9, 'DisplayName', meanLegend);
    scatter(ax, band.x, meanKm, 34, 'o', 'MarkerFaceColor', 'w', ...
        'MarkerEdgeColor', meanColor, 'LineWidth', 1.1, ...
        'HandleVisibility', 'off');
    text(ax, band.x(2), meanKm(2) + 0.10, ...
        sprintf('%.2f km', meanKm(2)), ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
        'FontName', fontName, 'FontSize', 9, ...
        'Color', meanColor, 'HandleVisibility', 'off');
    if panel == 1
        legendHandles = [hMean; hBoundary];
    end
    grid(ax, 'off');
    ax.YGrid = 'on';
    ax.XGrid = 'off';
    ax.GridColor = [0.82, 0.82, 0.82];
    ax.GridAlpha = 0.35;
    ax.Layer = 'top';
    ax.FontName = fontName;
    ax.FontSize = 9.5;
    ax.LineWidth = 0.8;
    ax.TickDir = 'out';
    ax.TickLength = [0.008, 0.008];
    ax.Box = 'off';
    xlim(ax, [0.55, 3.45]);
    ylim(ax, [0.75, 5.75]);
    yticks(ax, 1:0.5:5.5);
    xticks(ax, 1:3);
    xticklabels(ax, unitLabels);
    if panel == 1
        ylabel(ax, yLabel, 'FontName', fontName, 'FontSize', 10);
    else
        ax.YTickLabel = [];
    end
    title(ax, panelTitles(panel), 'FontName', fontName, ...
        'FontWeight', 'normal', 'FontSize', 11);
    text(ax, -0.045, 1.025, char('a' + panel - 1), ...
        'Units', 'normalized', 'HorizontalAlignment', 'left', ...
        'VerticalAlignment', 'bottom', 'FontName', 'Arial', ...
        'FontSize', 12, 'FontWeight', 'bold', 'Color', [0, 0, 0], ...
        'Clipping', 'off', 'HandleVisibility', 'off');
end
xlabel(layout, xLabel, 'FontName', fontName, 'FontSize', 10);
colormap(fig, palette);
clim(axesList(1), [edges(1), edges(end)]);
clim(axesList(2), [edges(1), edges(end)]);
cb = colorbar(axesList(2), 'eastoutside');
cb.Ticks = edges;
cb.TickLabels = compose('%.1f', edges);
cb.Label.String = colorLabel;
cb.Label.FontName = fontName;
cb.Label.FontSize = 9;
cb.FontName = fontName;
cb.FontSize = 8.5;
cb.TickDirection = 'out';
cb.LineWidth = 0.7;
legend(axesList(1), legendHandles, {meanLegend, intervalLegend}, ...
    'Location', 'northeast', 'FontName', fontName, ...
    'FontSize', 8.5, 'Box', 'off');

baseName = fullfile(outputRoot, ...
    "Fig_NorthSouth_Xiong_Comparison_" + suffix);
pngFile = baseName + ".png";
pdfFile = baseName + ".pdf";
figFile = baseName + ".fig";
exportgraphics(fig, pngFile, 'Resolution', 300);
exportgraphics(fig, pdfFile, 'ContentType', 'image', 'Resolution', 300);
savefig(fig, figFile);
close(fig);
files = struct('png', string(pngFile), 'pdf', string(pdfFile), ...
    'fig', string(figFile));
end
