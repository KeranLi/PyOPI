function [summary, results] = opiTest_DolomiteOffset(matFileResults, clumpedFile, outputDir, offsetsC)
% opiTest_DolomiteOffset tests dolomite-environment temperature offsets.
%
% The tested proxy model is:
%   T_dolomite = T_OPI_predicted_LSWT + delta_dolomite_environment

if nargin < 4 || isempty(offsetsC)
    offsetsC = [0, 5, 10];
end
if nargin < 3 || isempty(outputDir)
    outputDir = fileparts(clumpedFile);
end
if ~isfolder(outputDir), mkdir(outputDir); end

results = opiCompare_ClumpedTemperature(matFileResults, clumpedFile, outputDir);
seasons = ["annual", "ao", "amj", "jja", "warmest"];

rows = {};
for i = 1:numel(seasons)
    season = seasons(i);
    twCol = "OPI_air_to_lake_Tw_" + season + "_C";
    sigmaCol = "sigma_combined_Tclumped_minus_OPI_Tw_" + season + "_C";
    tw = results.(twCol);
    sigma = results.(sigmaCol);
    obs = results.T_clumped_C;

    for offset = offsetsC(:).'
        rows(end+1, :) = makeSummaryRow(season, "fixed", offset, obs, tw, sigma, 0); %#ok<AGROW>
    end

    w = 1 ./ sigma.^2;
    offsetFit = sum(w .* (obs - tw), 'omitnan') ./ sum(w, 'omitnan');
    rows(end+1, :) = makeSummaryRow(season, "fitted", offsetFit, obs, tw, sigma, 1); %#ok<AGROW>
end

summary = cell2table(rows, 'VariableNames', ...
    {'season', 'offset_type', 'offset_C', 'n', 'n_parameters', ...
    'mean_residual_C', 'rms_residual_C', 'mean_z', 'chi2', 'nu', 'chiR2'});
summary.season = string(summary.season);
summary.offset_type = string(summary.offset_type);

outCsv = fullfile(outputDir, 'dolomite_offset_sensitivity.csv');
writetable(summary, outCsv);
makeOffsetResidualFigure(summary, outputDir);
makeOffsetChiFigure(summary, outputDir);

fprintf('Wrote dolomite offset sensitivity table:\n%s\n', outCsv);

end

function row = makeSummaryRow(season, offsetType, offsetC, obs, tw, sigma, nParameters)
res = obs - (tw + offsetC);
z = res ./ sigma;
iGood = isfinite(res) & isfinite(z);
n = sum(iGood);
chi2 = sum(z(iGood).^2);
nu = max(1, n - nParameters);
row = {season, offsetType, offsetC, n, nParameters, ...
    mean(res(iGood), 'omitnan'), ...
    sqrt(mean(res(iGood).^2, 'omitnan')), ...
    mean(z(iGood), 'omitnan'), ...
    chi2, nu, chi2/nu};
end

function makeOffsetResidualFigure(summary, outputDir)
fig = figure('Color', 'w', 'Name', 'Dolomite offset residual sensitivity');
ax = axes(fig);
set(ax, 'Color', 'w', 'XColor', 'k', 'YColor', 'k');
if isprop(ax, 'Toolbar') && ~isempty(ax.Toolbar)
    ax.Toolbar.Visible = 'off';
end
hold(ax, 'on');
seasons = unique(summary.season, 'stable');
colors = lines(numel(seasons));
for i = 1:numel(seasons)
    isSeason = summary.season == seasons(i) & summary.offset_type == "fixed";
    plot(ax, summary.offset_C(isSeason), summary.mean_residual_C(isSeason), ...
        '-o', 'Color', colors(i, :), 'MarkerFaceColor', colors(i, :), ...
        'DisplayName', upper(seasons(i)));
    isFit = summary.season == seasons(i) & summary.offset_type == "fitted";
    scatter(ax, summary.offset_C(isFit), summary.mean_residual_C(isFit), ...
        80, colors(i, :), 'd', 'filled', 'HandleVisibility', 'off');
end
yline(ax, 0, 'k--', 'HandleVisibility', 'off');
grid(ax, 'on');
xlabel(ax, 'Dolomite-environment offset delta (deg C)');
ylabel(ax, 'Mean residual: observed T - predicted T (deg C)');
title(ax, 'Dolomite offset sensitivity', 'Color', 'k');
lgd = legend(ax, 'Location', 'best');
set(lgd, 'Color', 'w', 'TextColor', 'k', 'EdgeColor', [0.3, 0.3, 0.3]);
exportgraphics(fig, fullfile(outputDir, 'Fig_DolomiteOffset_MeanResidual.png'), 'Resolution', 200);
savefig(fig, fullfile(outputDir, 'Fig_DolomiteOffset_MeanResidual.fig'));
end

function makeOffsetChiFigure(summary, outputDir)
fig = figure('Color', 'w', 'Name', 'Dolomite offset chi-square sensitivity');
ax = axes(fig);
set(ax, 'Color', 'w', 'XColor', 'k', 'YColor', 'k');
if isprop(ax, 'Toolbar') && ~isempty(ax.Toolbar)
    ax.Toolbar.Visible = 'off';
end
hold(ax, 'on');
seasons = unique(summary.season, 'stable');
colors = lines(numel(seasons));
for i = 1:numel(seasons)
    isSeason = summary.season == seasons(i) & summary.offset_type == "fixed";
    plot(ax, summary.offset_C(isSeason), summary.chiR2(isSeason), ...
        '-o', 'Color', colors(i, :), 'MarkerFaceColor', colors(i, :), ...
        'DisplayName', upper(seasons(i)));
    isFit = summary.season == seasons(i) & summary.offset_type == "fitted";
    scatter(ax, summary.offset_C(isFit), summary.chiR2(isFit), ...
        80, colors(i, :), 'd', 'filled', 'HandleVisibility', 'off');
end
yline(ax, 1, 'k--', 'HandleVisibility', 'off');
grid(ax, 'on');
xlabel(ax, 'Dolomite-environment offset delta (deg C)');
ylabel(ax, 'Reduced chi-square');
title(ax, 'Dolomite offset chi-square sensitivity', 'Color', 'k');
lgd = legend(ax, 'Location', 'best');
set(lgd, 'Color', 'w', 'TextColor', 'k', 'EdgeColor', [0.3, 0.3, 0.3]);
exportgraphics(fig, fullfile(outputDir, 'Fig_DolomiteOffset_ChiR2.png'), 'Resolution', 200);
savefig(fig, fullfile(outputDir, 'Fig_DolomiteOffset_ChiR2.fig'));
end
