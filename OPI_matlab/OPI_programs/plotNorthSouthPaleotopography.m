function files = plotNorthSouthPaleotopography( ...
    summaryFile, outputDir, marginalFile)
% Plot a probability-density band from Qiangtang to Gangdese.


assimilationRoot = fileparts(fileparts(mfilename('fullpath')));
if nargin < 1 || strlength(string(summaryFile)) == 0
    summaryFile = fullfile(assimilationRoot, 'results', ...
        'topography_north_south_grid_coarse', ...
        'westerhold_age_marginalized_sensitivity', ...
        'north_south_paleotopography_summary.csv');
end
if nargin < 2 || strlength(string(outputDir)) == 0
    outputDir = fileparts(summaryFile);
end
if nargin < 3 || strlength(string(marginalFile)) == 0
    marginalFile = fullfile(fileparts(summaryFile), ...
        'north_south_unit_elevation_posterior.csv');
end
if ~isfile(summaryFile)
    error('North-south paleotopography summary not found: %s', summaryFile);
end
if ~isfile(marginalFile)
    error('North-south unit posterior not found: %s', marginalFile);
end
if ~isfolder(outputDir)
    mkdir(outputDir);
end

T = readtable(summaryFile, 'TextType', 'string');
required = ["north_to_south_order", "unit", ...
    "posterior_mean_elevation_m", "p16_elevation_m", "p84_elevation_m"];
missing = setdiff(required, string(T.Properties.VariableNames));
if ~isempty(missing)
    error('Summary is missing column(s): %s', strjoin(missing, ', '));
end
T = sortrows(T, 'north_to_south_order');
if height(T) ~= 3
    error('North-south profile requires exactly three ordered units.');
end
M = readtable(marginalFile, 'TextType', 'string');
requiredMarginal = ["unit", "elevationM", "posterior_probability"];
missing = setdiff(requiredMarginal, string(M.Properties.VariableNames));
if ~isempty(missing)
    error('Unit posterior is missing column(s): %s', strjoin(missing, ', '));
end

x = (1:3)';
meanKm = T.posterior_mean_elevation_m ./ 1000;
p16Km = T.p16_elevation_m ./ 1000;
p84Km = T.p84_elevation_m ./ 1000;
unitOrder = ["Qiangtang", "Central_valley_zone", "Gangdese"];
yKm = linspace(0.75, 5.75, 501)';
bandwidthKm = 0.22;
densityAtUnit = zeros(numel(yKm), 3);
for i = 1:3
    rows = M.unit == unitOrder(i);
    zKm = M.elevationM(rows) ./ 1000;
    probability = M.posterior_probability(rows);
    probability = probability ./ sum(probability);
    offset = (yKm - zKm') ./ bandwidthKm;
    kernels = exp(-0.5 .* offset.^2) ./ ...
        (sqrt(2*pi) .* bandwidthKm);
    densityAtUnit(:, i) = kernels * probability;
end
xFine = linspace(1, 3, 401);
density = interp1(x, densityAtUnit', xFine, 'linear')';
densityMax = max(density, [], 'all');
alphaData = (density ./ densityMax).^0.62;
alphaData(density < 0.012 .* densityMax) = 0;
meanFine = interp1(x, meanKm, xFine, 'linear');
p16Fine = interp1(x, p16Km, xFine, 'linear');
p84Fine = interp1(x, p84Km, xFine, 'linear');

fig = figure('Color', 'w', 'Name', 'North-south paleotopography', ...
    'Position', [100, 100, 1180, 690]);
ax = axes(fig);
if isprop(ax, 'Toolbar') && ~isempty(ax.Toolbar)
    ax.Toolbar.Visible = 'off';
end
disableDefaultInteractivity(ax);
hold(ax, 'on');
grid(ax, 'on');
ax.GridAlpha = 0.16;
ax.YMinorGrid = 'on';
ax.MinorGridAlpha = 0.08;
ax.Layer = 'top';
ax.FontName = 'PingFang SC';
ax.FontSize = 13;
ax.LineWidth = 1;

densityImage = imagesc(ax, xFine, yKm, density);
densityImage.AlphaData = alphaData;
densityImage.AlphaDataMapping = 'none';
densityImage.HandleVisibility = 'off';
ax.YDir = 'normal';

meanColor = [0.05, 0.13, 0.17];
boundaryColor = [0.12, 0.25, 0.30];
plot(ax, xFine, p16Fine, '--', 'Color', boundaryColor, ...
    'LineWidth', 1.7, 'DisplayName', '16-84% 后验边界');
plot(ax, xFine, p84Fine, '--', 'Color', boundaryColor, ...
    'LineWidth', 1.7, 'HandleVisibility', 'off');
plot(ax, xFine, meanFine, '-', 'Color', meanColor, 'LineWidth', 3.0, ...
    'DisplayName', '后验平均');
scatter(ax, x, meanKm, 70, 'o', 'MarkerFaceColor', 'w', ...
    'MarkerEdgeColor', meanColor, 'LineWidth', 2, ...
    'HandleVisibility', 'off');

labels = ["羌塘", "中央谷地", "冈底斯"];
for i = 1:3
    text(ax, x(i), meanKm(i) + 0.11, sprintf('%.2f km', meanKm(i)), ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
        'FontName', 'PingFang SC', 'FontSize', 11, ...
        'FontWeight', 'bold', 'Color', meanColor, ...
        'HandleVisibility', 'off');
end

anchors = [1.00, 1.00, 1.00; ...
    0.79, 0.91, 0.88; ...
    0.25, 0.64, 0.60; ...
    0.08, 0.34, 0.48; ...
    0.95, 0.56, 0.16];
anchorX = linspace(0, 1, size(anchors, 1));
colorMap = interp1(anchorX, anchors, linspace(0, 1, 256), 'pchip');
colormap(ax, colorMap);
clim(ax, [0, densityMax]);
cb = colorbar(ax, 'eastoutside');
cb.Label.String = '后验概率密度 (km^{-1})';
cb.Label.FontName = 'PingFang SC';
cb.FontName = 'PingFang SC';

xlim(ax, [0.55, 3.45]);
ylim(ax, [0.75, 5.75]);
xticks(ax, x);
xticklabels(ax, labels);
ylabel(ax, '海拔 (km)', 'FontName', 'PingFang SC');
xlabel(ax, '北  →  南', 'FontName', 'PingFang SC');
title(ax, '30 Ma 南北向古地形后验概率带', ...
    'FontName', 'PingFang SC', 'FontWeight', 'bold');
legend(ax, 'Location', 'southoutside', 'Orientation', 'horizontal', ...
    'FontName', 'PingFang SC', 'Box', 'off');

baseName = fullfile(outputDir, 'Fig_NorthSouth_Paleotopography_Posterior');
pngFile = baseName + ".png";
pdfFile = baseName + ".pdf";
figFile = baseName + ".fig";
exportgraphics(fig, pngFile, 'Resolution', 300);
exportgraphics(fig, pdfFile, 'ContentType', 'image', 'Resolution', 300);
savefig(fig, figFile);
close(fig);

files = struct('png', string(pngFile), 'pdf', string(pdfFile), ...
    'fig', string(figFile));
fprintf('Wrote north-south paleotopography figure:\n%s\n', pngFile);
end
