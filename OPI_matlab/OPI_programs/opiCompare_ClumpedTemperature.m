function results = opiCompare_ClumpedTemperature(matFileResults, clumpedCsv, outputDir)
% opiCompare_ClumpedTemperature compares mock/observed clumped temperatures
% with OPI surface-temperature predictions through Terrazas et al. transfer
% functions.

if nargin < 1 || isempty(matFileResults)
    [f, p] = uigetfile('opiCalc*.mat', 'Select OPI result mat file');
    if isequal(f, 0), error('No OPI result file selected.'); end
    matFileResults = fullfile(p, f);
end
if nargin < 2 || isempty(clumpedCsv)
    [f, p] = uigetfile('*.csv', 'Select clumped-temperature csv file');
    if isequal(f, 0), error('No clumped-temperature csv file selected.'); end
    clumpedCsv = fullfile(p, f);
end
if nargin < 3 || isempty(outputDir)
    outputDir = fileparts(clumpedCsv);
end
if ~isfolder(outputDir), mkdir(outputDir); end

TC2K = 273.15;
S = load(matFileResults, ...
    'runPath', 'dataPath', 'topoFile', 'rTukey', ...
    'lon', 'lat', 'sampleLon', 'sampleLat', 'sampleLC', ...
    'ijCatch', 'ptrCatch', 'pGrid', 'pGrid_1', 'pGrid_2', ...
    'fractionPGrid', 'T_1', 'T_2', 'gammaSat_1', 'gammaSat_2');

[matPath, ~, ~] = fileparts(matFileResults);
topoPath = fullfile(S.dataPath, S.topoFile);
if ~isfile(topoPath)
    topoPath = fullfile(matPath, S.topoFile);
end
[~, ~, hGrid] = gridRead(topoPath);
hGrid(hGrid < 0) = 0;
window = tukeywin(size(hGrid, 1), S.rTukey) * tukeywin(size(hGrid, 2), S.rTukey)';
hGrid = window .* hGrid;

TGrid1_C = S.T_1(1) - S.gammaSat_1(1).*hGrid - TC2K;
TGrid2_C = S.T_2(1) - S.gammaSat_2(1).*hGrid - TC2K;
TGridCombined_C = (S.pGrid_1.*TGrid1_C + S.pGrid_2.*TGrid2_C)./S.pGrid;
TGridCombined_C(~isfinite(TGridCombined_C)) = nan;

clumped = readtable(clumpedCsv, 'TextType', 'string');
required = ["sample_index", "T_clumped_C", "sigma_T_C"];
missing = setdiff(required, string(clumped.Properties.VariableNames));
if ~isempty(missing)
    error('Missing required clumped CSV column(s): %s', strjoin(missing, ', '));
end

n = height(clumped);
sampleId = string(clumped.sample_index);
sampleIndex = resolveSampleIndex(clumped, S);

opiT_C = nan(n, 1);
opiElevM = nan(n, 1);
for i = 1:n
    k = sampleIndex(i);
    ij = catchmentIndices(k, S.ijCatch, S.ptrCatch);
    wt = S.pGrid(ij);
    if sum(wt, 'omitnan') > 0
        wt = wt ./ sum(wt, 'omitnan');
        opiT_C(i) = sum(wt .* TGridCombined_C(ij), 'omitnan');
        opiElevM(i) = sum(wt .* hGrid(ij), 'omitnan');
    else
        opiT_C(i) = interp2(S.lon, S.lat, TGridCombined_C, S.sampleLon(k), S.sampleLat(k));
        opiElevM(i) = interp2(S.lon, S.lat, hGrid, S.sampleLon(k), S.sampleLat(k));
    end
end

seasons = ["annual", "ao", "amj", "jja", "warmest"];
for s = seasons
    col = "MAAT_TF4_" + s + "_C";
    clumped.(col) = lakeTransfer_Terrazas2025( ...
        clumped.T_clumped_C, abs(clumped.lat), opiElevM./1e3, s, "TF4");
    resCol = "residual_OPI_minus_TF4_" + s + "_C";
    clumped.(resCol) = opiT_C - clumped.(col);
end

airToLakeModels = struct;
for s = seasons
    [twPred, sigmaTw, model] = lakeTransferAirToLake_Terrazas2025( ...
        opiT_C, abs(clumped.lat), opiElevM./1e3, s);
    col = "OPI_air_to_lake_Tw_" + s + "_C";
    sigmaCol = "sigma_OPI_air_to_lake_Tw_" + s + "_C";
    resCol = "residual_Tclumped_minus_OPI_Tw_" + s + "_C";
    combinedSigmaCol = "sigma_combined_Tclumped_minus_OPI_Tw_" + s + "_C";
    zCol = "z_Tclumped_minus_OPI_Tw_" + s;
    clumped.(col) = twPred;
    clumped.(sigmaCol) = sigmaTw;
    clumped.(resCol) = clumped.T_clumped_C - twPred;
    clumped.(combinedSigmaCol) = sqrt(clumped.sigma_T_C.^2 + sigmaTw.^2);
    clumped.(zCol) = clumped.(resCol) ./ clumped.(combinedSigmaCol);
    airToLakeModels.(s) = model;
end

hrenSeasons = ["annual", "amj", "jja", "amjjaso", "warmest"];
hrenModels = struct;
for s = hrenSeasons
    [maatHren, sigmaMaat, modelWaterToAir] = lakeTransfer_HrenSheldon2012( ...
        clumped.T_clumped_C, s, "water_to_air");
    maatCol = "MAAT_Hren2012_" + s + "_C";
    sigmaMaatCol = "sigma_MAAT_Hren2012_" + s + "_C";
    resMaatCol = "residual_OPI_minus_Hren2012_" + s + "_C";
    zMaatCol = "z_OPI_minus_Hren2012_" + s;
    clumped.(maatCol) = maatHren;
    clumped.(sigmaMaatCol) = sigmaMaat;
    clumped.(resMaatCol) = opiT_C - maatHren;
    clumped.(zMaatCol) = clumped.(resMaatCol) ./ sigmaMaat;

    [twHren, sigmaTw, modelAirToWater] = lakeTransfer_HrenSheldon2012( ...
        opiT_C, s, "air_to_water");
    twCol = "OPI_Hren2012_Tw_" + s + "_C";
    sigmaTwCol = "sigma_OPI_Hren2012_Tw_" + s + "_C";
    resTwCol = "residual_Tclumped_minus_OPI_Hren2012_Tw_" + s + "_C";
    zTwCol = "z_Tclumped_minus_OPI_Hren2012_Tw_" + s;
    combinedSigmaCol = "sigma_combined_Tclumped_minus_OPI_Hren2012_Tw_" + s + "_C";
    clumped.(twCol) = twHren;
    clumped.(sigmaTwCol) = sigmaTw;
    clumped.(resTwCol) = clumped.T_clumped_C - twHren;
    clumped.(combinedSigmaCol) = sqrt(clumped.sigma_T_C.^2 + sigmaTw.^2);
    clumped.(zTwCol) = clumped.(resTwCol) ./ clumped.(combinedSigmaCol);

    hrenModels.(s).waterToAir = modelWaterToAir;
    hrenModels.(s).airToWater = modelAirToWater;
end

[lakeAreaKm2, lakeDepthM] = optionalLakeProperties(clumped);
[dolomiteOffsetC, sigmaOffsetC, mlConfigSource] = ...
    readClumpedComparisonConfig(clumpedCsv);
mlLakeObservedC = clumped.T_clumped_C - dolomiteOffsetC;
sigmaMlLakeObservedC = sqrt(clumped.sigma_T_C.^2 + sigmaOffsetC.^2);
[mlTw, sigmaMlTw, mlForwardInfo] = lakeTransferML_TerrazasWarmest( ...
    opiT_C, abs(clumped.lat), "air_to_lake", ...
    'LakeAreaKm2', lakeAreaKm2, 'LakeDepthM', lakeDepthM);
[mlAir, sigmaMlAirModel, mlInverseInfo] = lakeTransferML_TerrazasWarmest( ...
    mlLakeObservedC, abs(clumped.lat), "lake_to_air", ...
    'LakeAreaKm2', lakeAreaKm2, 'LakeDepthM', lakeDepthM);
mlAirPlus = lakeTransferML_TerrazasWarmest( ...
    mlLakeObservedC + sigmaMlLakeObservedC, abs(clumped.lat), ...
    "lake_to_air", 'LakeAreaKm2', lakeAreaKm2, ...
    'LakeDepthM', lakeDepthM, 'IncludeResidual', false);
mlAirMinus = lakeTransferML_TerrazasWarmest( ...
    mlLakeObservedC - sigmaMlLakeObservedC, abs(clumped.lat), ...
    "lake_to_air", 'LakeAreaKm2', lakeAreaKm2, ...
    'LakeDepthM', lakeDepthM, 'IncludeResidual', false);
sigmaMlAirInput = abs(mlAirPlus - mlAirMinus) ./ 2;
sigmaMlAirTotal = sqrt(sigmaMlAirModel.^2 + sigmaMlAirInput.^2);

clumped.OPI_ML_Tw_warmest_C = mlTw;
clumped.sigma_OPI_ML_Tw_warmest_C = sigmaMlTw;
clumped.ML_dolomite_offset_C = repmat(dolomiteOffsetC, n, 1);
clumped.ML_sigma_dolomite_offset_C = repmat(sigmaOffsetC, n, 1);
clumped.ML_lake_temperature_corrected_C = mlLakeObservedC;
clumped.sigma_ML_lake_temperature_corrected_C = sigmaMlLakeObservedC;
clumped.residual_corrected_lake_minus_OPI_ML_Tw_warmest_C = ...
    mlLakeObservedC - mlTw;
clumped.sigma_combined_corrected_lake_minus_OPI_ML_Tw_warmest_C = ...
    sqrt(sigmaMlLakeObservedC.^2 + sigmaMlTw.^2);
clumped.z_corrected_lake_minus_OPI_ML_Tw_warmest = ...
    clumped.residual_corrected_lake_minus_OPI_ML_Tw_warmest_C ./ ...
    clumped.sigma_combined_corrected_lake_minus_OPI_ML_Tw_warmest_C;
clumped.ML_inferred_Tair_warmest_C = mlAir;
clumped.sigma_ML_inferred_Tair_model_C = sigmaMlAirModel;
clumped.sigma_ML_inferred_Tair_from_clumped_C = sigmaMlAirInput;
clumped.sigma_ML_inferred_Tair_total_C = sigmaMlAirTotal;
clumped.residual_OPI_minus_ML_Tair_warmest_C = opiT_C - mlAir;
clumped.z_OPI_minus_ML_Tair_warmest = ...
    clumped.residual_OPI_minus_ML_Tair_warmest_C ./ sigmaMlAirTotal;
clumped.ML_lake_area_km2 = mlForwardInfo.lakeAreaKm2;
clumped.ML_lake_depth_m = mlForwardInfo.lakeDepthM;
clumped.ML_used_default_lake_area = mlForwardInfo.usedDefaultArea;
clumped.ML_used_default_lake_depth = mlForwardInfo.usedDefaultDepth;
clumped.ML_forward_outside_global_training_range = ...
    mlForwardInfo.outsideGlobalTrainingRange;
clumped.ML_forward_outside_high_elevation_range = ...
    mlForwardInfo.outsideHighElevationTrainingRange;
clumped.ML_inverse_outside_global_training_range = ...
    mlInverseInfo.outsideGlobalTrainingRange;
clumped.ML_inverse_outside_high_elevation_range = ...
    mlInverseInfo.outsideHighElevationTrainingRange;

results = clumped;
results.sample_id = sampleId;
results.OPI_sample_index = sampleIndex;
results.OPI_surface_T_C = opiT_C;
results.OPI_elevation_m = opiElevM;
results.OPI_sample_lon = S.sampleLon(sampleIndex);
results.OPI_sample_lat = S.sampleLat(sampleIndex);

outCsv = fullfile(outputDir, 'clumped_temperature_Terrazas2025_comparison.csv');
writetable(results, outCsv);
baseCols = ["sample_index", "sample_id", "lon", "lat", "T_clumped_C", ...
    "sigma_T_C", "OPI_surface_T_C", "OPI_elevation_m", ...
    "OPI_sample_index", "OPI_sample_lon", "OPI_sample_lat"];
varNames = string(results.Properties.VariableNames);
baseCols = baseCols(ismember(baseCols, varNames));
hrenCols = varNames(contains(varNames, "Hren2012"));
hrenCsv = fullfile(outputDir, 'clumped_temperature_HrenSheldon2012_comparison.csv');
writetable(results(:, [baseCols, hrenCols]), hrenCsv);
mlCols = varNames(contains(varNames, "_ML_") | startsWith(varNames, "ML_"));
mlCsv = fullfile(outputDir, ...
    'clumped_temperature_TerrazasWarmestML_comparison.csv');
writetable(results(:, [baseCols, mlCols]), mlCsv);

makeComparisonFigure(results, seasons, outputDir);
makeResidualFigure(results, seasons, outputDir);
makeAirToLakeFigure(results, seasons, outputDir);
makeAirToLakeResidualFigure(results, seasons, outputDir);
makeHrenMaatFigure(results, hrenSeasons, outputDir);
makeHrenMaatResidualFigure(results, hrenSeasons, outputDir);
makeHrenAirToWaterFigure(results, hrenSeasons, outputDir);
makeHrenAirToWaterResidualFigure(results, hrenSeasons, outputDir);
makeMLForwardFigure(results, outputDir);
makeMLInverseFigure(results, outputDir);
save(fullfile(outputDir, 'Terrazas2025_air_to_lake_models.mat'), ...
    'airToLakeModels');
save(fullfile(outputDir, 'HrenSheldon2012_transfer_models.mat'), ...
    'hrenModels');
save(fullfile(outputDir, 'TerrazasWarmestML_application.mat'), ...
    'mlForwardInfo', 'mlInverseInfo', 'dolomiteOffsetC', ...
    'sigmaOffsetC', 'mlConfigSource');

fprintf('Wrote comparison table:\n%s\n', outCsv);
fprintf('Wrote Hren-Sheldon 2012 comparison table:\n%s\n', hrenCsv);
fprintf('Wrote Terrazas warmest ML comparison table:\n%s\n', mlCsv);
fprintf('Wrote figures to:\n%s\n', outputDir);

end

function [lakeAreaKm2, lakeDepthM] = optionalLakeProperties(clumped)
names = string(clumped.Properties.VariableNames);
if ismember("lake_area_km2", names)
    lakeAreaKm2 = clumped.lake_area_km2;
else
    lakeAreaKm2 = [];
end
if ismember("lake_depth_m", names)
    lakeDepthM = clumped.lake_depth_m;
else
    lakeDepthM = [];
end
end

function [offsetC, sigmaOffsetC, source] = readClumpedComparisonConfig(clumpedFile)
offsetC = 0;
sigmaOffsetC = 0;
source = "sedimentology default: formation T equals warmest lake-water T";
configFile = fullfile(fileparts(clumpedFile), 'clumped_fit_config.csv');
if ~isfile(configFile)
    return
end
T = readtable(configFile, 'TextType', 'string');
required = ["dolomiteOffsetC", "sigmaOffsetC"];
missing = setdiff(required, string(T.Properties.VariableNames));
if ~isempty(missing) || height(T) < 1
    error('Invalid clumped fit config: %s', configFile);
end
offsetC = T.dolomiteOffsetC(1);
sigmaOffsetC = T.sigmaOffsetC(1);
source = string(configFile);
end

function makeMLForwardFigure(results, outputDir)
fig = figure('Color', 'w', 'Name', 'Terrazas warmest ML forward comparison');
ax = axes(fig);
formatComparisonAxes(ax);
hold(ax, 'on');
x = results.OPI_ML_Tw_warmest_C;
y = results.ML_lake_temperature_corrected_C;
sx = results.sigma_OPI_ML_Tw_warmest_C;
sy = results.sigma_ML_lake_temperature_corrected_C;
errorbar(ax, x, y, sy, sy, sx, sx, 'o', ...
    'Color', [0.1, 0.45, 0.75], 'MarkerFaceColor', [0.1, 0.45, 0.75]);
setOneToOneLimits(ax, [x - sx; x + sx], [y - sy; y + sy]);
xlabel(ax, 'ML-predicted warmest lake temperature (deg C)');
ylabel(ax, 'Offset-corrected warmest lake temperature (deg C)');
title(ax, 'Terrazas warmest-season forward model', 'Color', 'k');
exportgraphics(fig, fullfile(outputDir, ...
    'Fig_TerrazasWarmestML_Forward_vs_Clumped.png'), 'Resolution', 200);
savefig(fig, fullfile(outputDir, ...
    'Fig_TerrazasWarmestML_Forward_vs_Clumped.fig'));
end

function makeMLInverseFigure(results, outputDir)
fig = figure('Color', 'w', 'Name', 'Terrazas warmest ML inverse comparison');
ax = axes(fig);
formatComparisonAxes(ax);
hold(ax, 'on');
x = results.ML_inferred_Tair_warmest_C;
y = results.OPI_surface_T_C;
sx = results.sigma_ML_inferred_Tair_total_C;
errorbar(ax, x, y, zeros(size(y)), zeros(size(y)), sx, sx, 'o', ...
    'Color', [0.75, 0.25, 0.15], 'MarkerFaceColor', [0.75, 0.25, 0.15]);
setOneToOneLimits(ax, [x - sx; x + sx], y);
xlabel(ax, 'Clumped-inferred warmest air temperature (deg C)');
ylabel(ax, 'OPI surface-air temperature (deg C)');
title(ax, 'Warmest-air reconstruction vs OPI temperature field', 'Color', 'k');
exportgraphics(fig, fullfile(outputDir, ...
    'Fig_TerrazasWarmestML_InferredAir_vs_OPI.png'), 'Resolution', 200);
savefig(fig, fullfile(outputDir, ...
    'Fig_TerrazasWarmestML_InferredAir_vs_OPI.fig'));
end

function makeAirToLakeFigure(results, seasons, outputDir)
fig = figure('Color', 'w', 'Name', 'OPI air-to-lake Tw comparison');
tiledlayout(1, 1);
ax = nexttile;
set(ax, 'Color', 'w', 'XColor', 'k', 'YColor', 'k');
if isprop(ax, 'Toolbar') && ~isempty(ax.Toolbar)
    ax.Toolbar.Visible = 'off';
end
hold(ax, 'on');
colors = lines(numel(seasons));
for i = 1:numel(seasons)
    col = "OPI_air_to_lake_Tw_" + seasons(i) + "_C";
    sigmaCol = "sigma_OPI_air_to_lake_Tw_" + seasons(i) + "_C";
    errorbar(ax, results.(col), results.T_clumped_C, ...
        results.sigma_T_C, results.sigma_T_C, ...
        results.(sigmaCol), results.(sigmaCol), ...
        'o', 'Color', colors(i, :), 'MarkerFaceColor', colors(i, :), ...
        'DisplayName', upper(seasons(i)));
end
allX = [];
for i = 1:numel(seasons)
    xCenter = results.("OPI_air_to_lake_Tw_" + seasons(i) + "_C");
    xSigma = results.("sigma_OPI_air_to_lake_Tw_" + seasons(i) + "_C");
    allX = [allX; xCenter - xSigma; xCenter + xSigma]; %#ok<AGROW>
end
allY = [results.T_clumped_C - results.sigma_T_C; ...
    results.T_clumped_C + results.sigma_T_C];
lims = [min([allX; allY], [], 'omitnan'), ...
    max([allX; allY], [], 'omitnan')];
pad = max(1, (lims(2) - lims(1))*0.1);
lims = lims + [-pad, pad];
plot(ax, lims, lims, 'k--', 'DisplayName', '1:1');
xlim(ax, lims);
ylim(ax, lims);
grid(ax, 'on');
xlabel(ax, 'OPI air-to-lake predicted LSWT (deg C)');
ylabel(ax, 'Observed clumped temperature (deg C)');
lgd = legend(ax, 'Location', 'best');
set(lgd, 'Color', 'w', 'TextColor', 'k', 'EdgeColor', [0.3, 0.3, 0.3]);
title(ax, 'Refit air-to-lake model vs clumped temperature', 'Color', 'k');
exportgraphics(fig, fullfile(outputDir, 'Fig_OPI_AirToLake_Tw_vs_Clumped.png'), 'Resolution', 200);
savefig(fig, fullfile(outputDir, 'Fig_OPI_AirToLake_Tw_vs_Clumped.fig'));
end

function makeAirToLakeResidualFigure(results, seasons, outputDir)
fig = figure('Color', 'w', 'Name', 'OPI air-to-lake Tw residuals');
ax = axes(fig);
set(ax, 'Color', 'w', 'XColor', 'k', 'YColor', 'k');
if isprop(ax, 'Toolbar') && ~isempty(ax.Toolbar)
    ax.Toolbar.Visible = 'off';
end
hold(ax, 'on');
means = nan(numel(seasons), 1);
stds = nan(numel(seasons), 1);
for i = 1:numel(seasons)
    col = "residual_Tclumped_minus_OPI_Tw_" + seasons(i) + "_C";
    means(i) = mean(results.(col), 'omitnan');
    stds(i) = std(results.(col), 'omitnan');
end
bar(ax, 1:numel(seasons), means);
errorbar(ax, 1:numel(seasons), means, stds, 'k.', 'LineWidth', 1.5);
yline(ax, 0, 'k--');
grid(ax, 'on');
xticks(ax, 1:numel(seasons));
xticklabels(ax, upper(seasons));
ylabel(ax, 'Observed clumped T minus OPI-predicted LSWT (deg C)');
title(ax, 'Air-to-lake seasonal residuals', 'Color', 'k');
exportgraphics(fig, fullfile(outputDir, 'Fig_OPI_AirToLake_Tw_Residuals.png'), 'Resolution', 200);
savefig(fig, fullfile(outputDir, 'Fig_OPI_AirToLake_Tw_Residuals.fig'));
end

function sampleIndex = resolveSampleIndex(clumped, S)
rawIndex = clumped.sample_index;
if isnumeric(rawIndex)
    sampleIndex = rawIndex;
else
    sampleIndex = str2double(string(rawIndex));
end

if all(isfinite(sampleIndex))
    sampleIndex = round(sampleIndex);
    if any(sampleIndex < 1 | sampleIndex > numel(S.sampleLon))
        error('Numeric sample_index values must refer to OPI sample rows.');
    end
    return
end

varNames = string(clumped.Properties.VariableNames);
if ~all(ismember(["lon", "lat"], varNames))
    error(['Non-numeric sample_index values require lon and lat columns ', ...
        'so samples can be matched to the nearest OPI sample location.']);
end

sampleIndex = nan(height(clumped), 1);
for i = 1:height(clumped)
    d = hypot(S.sampleLon(:) - clumped.lon(i), S.sampleLat(:) - clumped.lat(i));
    [~, sampleIndex(i)] = min(d);
end
end

function makeComparisonFigure(results, seasons, outputDir)
fig = figure('Color', 'w', 'Name', 'Clumped TF4 MAAT comparison');
tiledlayout(1, 1);
ax = nexttile;
set(ax, 'Color', 'w', 'XColor', 'k', 'YColor', 'k');
if isprop(ax, 'Toolbar') && ~isempty(ax.Toolbar)
    ax.Toolbar.Visible = 'off';
end
hold(ax, 'on');
colors = lines(numel(seasons));
for i = 1:numel(seasons)
    col = "MAAT_TF4_" + seasons(i) + "_C";
    scatter(ax, results.(col), results.OPI_surface_T_C, 48, colors(i, :), 'filled', ...
        'DisplayName', upper(seasons(i)));
end
allX = [];
for i = 1:numel(seasons)
    allX = [allX; results.("MAAT_TF4_" + seasons(i) + "_C")]; %#ok<AGROW>
end
lims = [min([allX; results.OPI_surface_T_C], [], 'omitnan'), ...
    max([allX; results.OPI_surface_T_C], [], 'omitnan')];
pad = max(1, (lims(2) - lims(1))*0.1);
lims = lims + [-pad, pad];
plot(ax, lims, lims, 'k--', 'DisplayName', '1:1');
xlim(ax, lims);
ylim(ax, lims);
grid(ax, 'on');
xlabel(ax, 'Clumped + Terrazas TF4 inferred MAAT (deg C)');
ylabel(ax, 'OPI precipitation-weighted surface T (deg C)');
lgd = legend(ax, 'Location', 'best');
set(lgd, 'Color', 'w', 'TextColor', 'k', 'EdgeColor', [0.3, 0.3, 0.3]);
title(ax, 'Clumped-temperature seasonal assumptions vs OPI', 'Color', 'k');
exportgraphics(fig, fullfile(outputDir, 'Fig_Clumped_TF4_vs_OPI_Temperature.png'), 'Resolution', 200);
savefig(fig, fullfile(outputDir, 'Fig_Clumped_TF4_vs_OPI_Temperature.fig'));
end

function makeResidualFigure(results, seasons, outputDir)
fig = figure('Color', 'w', 'Name', 'Clumped TF4 residuals');
ax = axes(fig);
set(ax, 'Color', 'w', 'XColor', 'k', 'YColor', 'k');
if isprop(ax, 'Toolbar') && ~isempty(ax.Toolbar)
    ax.Toolbar.Visible = 'off';
end
hold(ax, 'on');
means = nan(numel(seasons), 1);
stds = nan(numel(seasons), 1);
for i = 1:numel(seasons)
    col = "residual_OPI_minus_TF4_" + seasons(i) + "_C";
    means(i) = mean(results.(col), 'omitnan');
    stds(i) = std(results.(col), 'omitnan');
end
bar(ax, 1:numel(seasons), means);
errorbar(ax, 1:numel(seasons), means, stds, 'k.', 'LineWidth', 1.5);
yline(ax, 0, 'k--');
grid(ax, 'on');
xticks(ax, 1:numel(seasons));
xticklabels(ax, upper(seasons));
ylabel(ax, 'OPI minus clumped-TF4 MAAT residual (deg C)');
title(ax, 'Seasonal interpretation sensitivity', 'Color', 'k');
exportgraphics(fig, fullfile(outputDir, 'Fig_Clumped_TF4_Residuals.png'), 'Resolution', 200);
savefig(fig, fullfile(outputDir, 'Fig_Clumped_TF4_Residuals.fig'));
end

function makeHrenMaatFigure(results, seasons, outputDir)
fig = figure('Color', 'w', 'Name', 'Hren-Sheldon clumped MAAT comparison');
ax = axes(fig);
formatComparisonAxes(ax);
hold(ax, 'on');
colors = lines(numel(seasons));
allX = [];
for i = 1:numel(seasons)
    col = "MAAT_Hren2012_" + seasons(i) + "_C";
    sigmaCol = "sigma_MAAT_Hren2012_" + seasons(i) + "_C";
    errorbar(ax, results.(col), results.OPI_surface_T_C, ...
        zeros(height(results), 1), zeros(height(results), 1), ...
        results.(sigmaCol), results.(sigmaCol), ...
        'o', 'Color', colors(i, :), 'MarkerFaceColor', colors(i, :), ...
        'DisplayName', upper(seasons(i)));
    allX = [allX; results.(col) - results.(sigmaCol); ...
        results.(col) + results.(sigmaCol)]; %#ok<AGROW>
end
setOneToOneLimits(ax, allX, results.OPI_surface_T_C);
xlabel(ax, 'Clumped + Hren2012 inferred MAAT (deg C)');
ylabel(ax, 'OPI precipitation-weighted surface T (deg C)');
lgd = legend(ax, 'Location', 'best');
set(lgd, 'Color', 'w', 'TextColor', 'k', 'EdgeColor', [0.3, 0.3, 0.3]);
title(ax, 'Hren-Sheldon seasonal clumped-T interpretation vs OPI', 'Color', 'k');
exportgraphics(fig, fullfile(outputDir, 'Fig_Clumped_Hren2012_MAAT_vs_OPI.png'), 'Resolution', 200);
savefig(fig, fullfile(outputDir, 'Fig_Clumped_Hren2012_MAAT_vs_OPI.fig'));
end

function makeHrenMaatResidualFigure(results, seasons, outputDir)
fig = figure('Color', 'w', 'Name', 'Hren-Sheldon clumped MAAT residuals');
ax = axes(fig);
formatComparisonAxes(ax);
hold(ax, 'on');
means = nan(numel(seasons), 1);
stds = nan(numel(seasons), 1);
for i = 1:numel(seasons)
    col = "residual_OPI_minus_Hren2012_" + seasons(i) + "_C";
    means(i) = mean(results.(col), 'omitnan');
    stds(i) = std(results.(col), 'omitnan');
end
bar(ax, 1:numel(seasons), means);
errorbar(ax, 1:numel(seasons), means, stds, 'k.', 'LineWidth', 1.5);
yline(ax, 0, 'k--');
xticks(ax, 1:numel(seasons));
xticklabels(ax, upper(seasons));
ylabel(ax, 'OPI minus Hren2012-inferred MAAT (deg C)');
title(ax, 'Hren-Sheldon seasonal MAAT residuals', 'Color', 'k');
exportgraphics(fig, fullfile(outputDir, 'Fig_Clumped_Hren2012_MAAT_Residuals.png'), 'Resolution', 200);
savefig(fig, fullfile(outputDir, 'Fig_Clumped_Hren2012_MAAT_Residuals.fig'));
end

function makeHrenAirToWaterFigure(results, seasons, outputDir)
fig = figure('Color', 'w', 'Name', 'Hren-Sheldon OPI air-to-water comparison');
ax = axes(fig);
formatComparisonAxes(ax);
hold(ax, 'on');
colors = lines(numel(seasons));
allX = [];
allY = [results.T_clumped_C - results.sigma_T_C; ...
    results.T_clumped_C + results.sigma_T_C];
for i = 1:numel(seasons)
    col = "OPI_Hren2012_Tw_" + seasons(i) + "_C";
    sigmaCol = "sigma_OPI_Hren2012_Tw_" + seasons(i) + "_C";
    errorbar(ax, results.(col), results.T_clumped_C, ...
        results.sigma_T_C, results.sigma_T_C, ...
        results.(sigmaCol), results.(sigmaCol), ...
        'o', 'Color', colors(i, :), 'MarkerFaceColor', colors(i, :), ...
        'DisplayName', upper(seasons(i)));
    allX = [allX; results.(col) - results.(sigmaCol); ...
        results.(col) + results.(sigmaCol)]; %#ok<AGROW>
end
setOneToOneLimits(ax, allX, allY);
xlabel(ax, 'OPI + Hren2012 predicted lake-surface T (deg C)');
ylabel(ax, 'Observed clumped temperature (deg C)');
lgd = legend(ax, 'Location', 'best');
set(lgd, 'Color', 'w', 'TextColor', 'k', 'EdgeColor', [0.3, 0.3, 0.3]);
title(ax, 'Hren-Sheldon inverse transfer vs clumped temperature', 'Color', 'k');
exportgraphics(fig, fullfile(outputDir, 'Fig_OPI_Hren2012_Tw_vs_Clumped.png'), 'Resolution', 200);
savefig(fig, fullfile(outputDir, 'Fig_OPI_Hren2012_Tw_vs_Clumped.fig'));
end

function makeHrenAirToWaterResidualFigure(results, seasons, outputDir)
fig = figure('Color', 'w', 'Name', 'Hren-Sheldon air-to-water residuals');
ax = axes(fig);
formatComparisonAxes(ax);
hold(ax, 'on');
means = nan(numel(seasons), 1);
stds = nan(numel(seasons), 1);
for i = 1:numel(seasons)
    col = "residual_Tclumped_minus_OPI_Hren2012_Tw_" + seasons(i) + "_C";
    means(i) = mean(results.(col), 'omitnan');
    stds(i) = std(results.(col), 'omitnan');
end
bar(ax, 1:numel(seasons), means);
errorbar(ax, 1:numel(seasons), means, stds, 'k.', 'LineWidth', 1.5);
yline(ax, 0, 'k--');
xticks(ax, 1:numel(seasons));
xticklabels(ax, upper(seasons));
ylabel(ax, 'Clumped T minus OPI-Hren2012 lake T (deg C)');
title(ax, 'Hren-Sheldon air-to-water seasonal residuals', 'Color', 'k');
exportgraphics(fig, fullfile(outputDir, 'Fig_OPI_Hren2012_Tw_Residuals.png'), 'Resolution', 200);
savefig(fig, fullfile(outputDir, 'Fig_OPI_Hren2012_Tw_Residuals.fig'));
end

function formatComparisonAxes(ax)
set(ax, 'Color', 'w', 'XColor', 'k', 'YColor', 'k');
if isprop(ax, 'Toolbar') && ~isempty(ax.Toolbar)
    ax.Toolbar.Visible = 'off';
end
grid(ax, 'on');
end

function setOneToOneLimits(ax, xValues, yValues)
lims = [min([xValues(:); yValues(:)], [], 'omitnan'), ...
    max([xValues(:); yValues(:)], [], 'omitnan')];
pad = max(1, (lims(2) - lims(1))*0.1);
lims = lims + [-pad, pad];
plot(ax, lims, lims, 'k--', 'DisplayName', '1:1');
xlim(ax, lims);
ylim(ax, lims);
end
