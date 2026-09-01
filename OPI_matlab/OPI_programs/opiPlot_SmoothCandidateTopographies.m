function output = opiPlot_SmoothCandidateTopographies( ...
    experimentRoot, outputDir)
% Plot all 160 smooth candidate terrains with one common elevation scale.

if nargin < 1 || strlength(string(experimentRoot)) == 0
    scenarioRoot = fullfile(fileparts(mfilename('fullpath')), '..', ...
        'scenarios', ...
        'Qiangtang_30Ma_4000m_dH045_T290_fP030_Valid_platform_northsouth');
    experimentRoot = fullfile(scenarioRoot, ...
        'topography_north_south_grid_smooth');
end
if nargin < 2 || strlength(string(outputDir)) == 0
    outputDir = fullfile(experimentRoot, 'candidate_topography_atlas');
end
experimentRoot = char(string(experimentRoot));
outputDir = char(string(outputDir));
manifestFile = fullfile(experimentRoot, 'design', 'case_manifest.csv');
if ~isfile(manifestFile)
    error('Candidate terrain manifest not found: %s', manifestFile);
end

overviewDir = fullfile(outputDir, 'overview');
gangdeseDir = fullfile(outputDir, 'by_gangdese_height');
individualDir = fullfile(outputDir, 'individual_cases');
dataDir = fullfile(outputDir, 'plot_data');
folders = {outputDir, overviewDir, gangdeseDir, individualDir, dataDir};
for i = 1:numel(folders)
    if ~isfolder(folders{i})
        mkdir(folders{i});
    end
end

manifest = readtable(manifestFile, 'TextType', 'string');
required = ["case_id", "gangdese_target_m", "qiangtang_target_m", ...
    "valley_target_m"];
missing = setdiff(required, string(manifest.Properties.VariableNames));
if ~isempty(missing)
    error('Manifest is missing column(s): %s', strjoin(missing, ', '));
end
if height(manifest) ~= 160 || numel(unique(manifest.case_id)) ~= 160
    error('Expected exactly 160 unique smooth candidate terrains.');
end
manifest = sortrows(manifest, {'gangdese_target_m', ...
    'valley_target_m', 'qiangtang_target_m'});

[lon, lat, terrainKm] = loadTerrains(experimentRoot, manifest);
colorLimitsKm = [0, 6];
cmap = terrainColormap(256);
writePlotData(dataDir, manifest, lon, lat, terrainKm, individualDir);

makeOverview(manifest, lon, lat, terrainKm, colorLimitsKm, cmap, ...
    overviewDir);
gValues = unique(manifest.gangdese_target_m, 'sorted');
for i = 1:numel(gValues)
    makeGangdeseSheet(manifest, lon, lat, terrainKm, gValues(i), ...
        colorLimitsKm, cmap, gangdeseDir);
end
makeIndividualMaps(manifest, lon, lat, terrainKm, colorLimitsKm, ...
    cmap, individualDir);

output = struct;
output.status = "complete";
output.case_count = height(manifest);
output.output_directory = string(outputDir);
output.overview_png = string(fullfile(overviewDir, ...
    'all_160_candidate_topographies.png'));
output.plot_data_csv = string(fullfile(dataDir, ...
    'candidate_topography_plot_data.csv'));
save(fullfile(outputDir, 'candidate_topography_atlas.mat'), ...
    'output', 'manifest', 'lon', 'lat', 'colorLimitsKm', '-v7.3');
fprintf('Candidate topography atlas written for %d cases:\n%s\n', ...
    height(manifest), outputDir);
end

function [lon, lat, terrainKm] = loadTerrains(experimentRoot, manifest)
lon = [];
lat = [];
terrainKm = [];
for i = 1:height(manifest)
    topoFile = fullfile(experimentRoot, 'calc_only', ...
        manifest.case_id(i), 'Tibet_Eocene_30Ma_topo.mat');
    if ~isfile(topoFile)
        error('Missing candidate topography: %s', topoFile);
    end
    S = load(topoFile, 'lon', 'lat', 'hGrid');
    if i == 1
        lon = S.lon(:)';
        lat = S.lat(:);
        terrainKm = nan(numel(lat), numel(lon), height(manifest));
    elseif ~isequal(lon, S.lon(:)') || ~isequal(lat, S.lat(:))
        error('Candidate %s uses a different grid.', manifest.case_id(i));
    end
    if ~isequal(size(S.hGrid), [numel(lat), numel(lon)]) || ...
            any(~isfinite(S.hGrid), 'all')
        error('Candidate %s has an invalid elevation grid.', ...
            manifest.case_id(i));
    end
    terrainKm(:, :, i) = S.hGrid ./ 1000;
end
end

function writePlotData(dataDir, manifest, lon, lat, terrainKm, individualDir)
[lonGrid, latGrid] = meshgrid(lon, lat);
nGrid = numel(lonGrid);
nCase = height(manifest);
nRow = nGrid .* nCase;
caseId = strings(nRow, 1);
gangdeseM = nan(nRow, 1);
qiangtangM = nan(nRow, 1);
valleyM = nan(nRow, 1);
longitudeDegE = repmat(lonGrid(:), nCase, 1);
latitudeDegN = repmat(latGrid(:), nCase, 1);
elevationM = nan(nRow, 1);
for i = 1:nCase
    idx = (i - 1) .* nGrid + (1:nGrid);
    caseId(idx) = manifest.case_id(i);
    gangdeseM(idx) = manifest.gangdese_target_m(i);
    qiangtangM(idx) = manifest.qiangtang_target_m(i);
    valleyM(idx) = manifest.valley_target_m(i);
    values = terrainKm(:, :, i) .* 1000;
    elevationM(idx) = values(:);
end
plotData = table(caseId, gangdeseM, qiangtangM, valleyM, ...
    longitudeDegE, latitudeDegN, elevationM, ...
    'VariableNames', {'case_id', 'gangdese_target_m', ...
    'qiangtang_target_m', 'central_valley_target_m', ...
    'longitude_degE', 'latitude_degN', 'elevation_m'});
writetable(plotData, fullfile(dataDir, ...
    'candidate_topography_plot_data.csv'));

imageFile = strings(nCase, 1);
for i = 1:nCase
    imageFile(i) = string(fullfile(individualDir, ...
        char(manifest.case_id(i) + ".png")));
end
index = manifest(:, {'case_id', 'qiangtang_target_m', ...
    'valley_target_m', 'gangdese_target_m'});
index.individual_image_file = imageFile;
writetable(index, fullfile(dataDir, 'candidate_topography_image_index.csv'));
end

function makeOverview(manifest, lon, lat, terrainKm, limitsKm, cmap, outDir)
fig = figure('Visible', 'off', 'Color', 'w', ...
    'Position', [40, 40, 2200, 3100]);
cleanup = onCleanup(@() close(fig)); %#ok<NASGU>
layout = tiledlayout(fig, 16, 10, 'TileSpacing', 'compact', ...
    'Padding', 'compact');
gValues = unique(manifest.gangdese_target_m, 'sorted');
vValues = unique(manifest.valley_target_m, 'sorted');
qValues = unique(manifest.qiangtang_target_m, 'sorted');
for i = 1:height(manifest)
    gIndex = find(gValues == manifest.gangdese_target_m(i));
    vIndex = find(vValues == manifest.valley_target_m(i));
    qIndex = find(qValues == manifest.qiangtang_target_m(i));
    blockRow = floor((gIndex - 1) ./ 2);
    blockCol = mod(gIndex - 1, 2);
    tileRow = blockRow .* numel(vValues) + vIndex;
    tileCol = blockCol .* numel(qValues) + qIndex;
    tileIndex = (tileRow - 1) .* 10 + tileCol;
    ax = nexttile(layout, tileIndex);
    drawTerrain(ax, lon, lat, terrainKm(:, :, i), limitsKm, cmap);
    title(ax, sprintf('Q %.1f', manifest.qiangtang_target_m(i) / 1000), ...
        'FontSize', 6, 'FontWeight', 'normal');
    if qIndex == 1
        ylabel(ax, sprintf('G %.1f | V %.1f', ...
            manifest.gangdese_target_m(i) / 1000, ...
            manifest.valley_target_m(i) / 1000), 'FontSize', 6);
    else
        ax.YTickLabel = [];
    end
    if vIndex ~= numel(vValues)
        ax.XTickLabel = [];
    end
end
cb = colorbar;
cb.Layout.Tile = 'east';
cb.Label.String = 'Elevation (km)';
cb.FontSize = 12;
title(layout, ['All 160 smooth candidate topographies: ' ...
    'Qiangtang (Q), central valley (V), Gangdese (G)'], ...
    'FontSize', 20, 'FontWeight', 'normal');
xlabel(layout, 'Longitude (deg E)', 'FontSize', 14);
ylabel(layout, 'Latitude (deg N)', 'FontSize', 14);
base = fullfile(outDir, 'all_160_candidate_topographies');
exportgraphics(fig, [base, '.png'], 'Resolution', 140, ...
    'BackgroundColor', 'white');
exportgraphics(fig, [base, '.pdf'], 'ContentType', 'image', ...
    'Resolution', 200, 'BackgroundColor', 'white');
end

function makeGangdeseSheet(manifest, lon, lat, terrainKm, gValue, ...
    limitsKm, cmap, outDir)
keep = manifest.gangdese_target_m == gValue;
subset = manifest(keep, :);
sourceIndex = find(keep);
vValues = unique(subset.valley_target_m, 'sorted');
qValues = unique(subset.qiangtang_target_m, 'sorted');
fig = figure('Visible', 'off', 'Color', 'w', ...
    'Position', [40, 40, 2100, 2900]);
cleanup = onCleanup(@() close(fig)); %#ok<NASGU>
layout = tiledlayout(fig, numel(vValues), numel(qValues), ...
    'TileSpacing', 'compact', 'Padding', 'compact');
for i = 1:height(subset)
    vIndex = find(vValues == subset.valley_target_m(i));
    qIndex = find(qValues == subset.qiangtang_target_m(i));
    tileIndex = (vIndex - 1) .* numel(qValues) + qIndex;
    ax = nexttile(layout, tileIndex);
    drawTerrain(ax, lon, lat, terrainKm(:, :, sourceIndex(i)), ...
        limitsKm, cmap);
    title(ax, sprintf('Q %.1f km', subset.qiangtang_target_m(i) / 1000), ...
        'FontSize', 9, 'FontWeight', 'normal');
    if qIndex == 1
        ylabel(ax, sprintf('V %.1f km', ...
            subset.valley_target_m(i) / 1000), 'FontSize', 9);
    else
        ax.YTickLabel = [];
    end
    if vIndex ~= numel(vValues)
        ax.XTickLabel = [];
    end
end
cb = colorbar;
cb.Layout.Tile = 'east';
cb.Label.String = 'Elevation (km)';
title(layout, sprintf('Smooth candidate topographies | Gangdese %.1f km', ...
    gValue / 1000), 'FontSize', 18, 'FontWeight', 'normal');
xlabel(layout, 'Longitude (deg E)');
ylabel(layout, 'Latitude (deg N)');
base = fullfile(outDir, sprintf('candidate_topographies_G%d', gValue));
exportgraphics(fig, [base, '.png'], 'Resolution', 250, ...
    'BackgroundColor', 'white');
exportgraphics(fig, [base, '.pdf'], 'ContentType', 'image', ...
    'Resolution', 300, 'BackgroundColor', 'white');
end

function makeIndividualMaps(manifest, lon, lat, terrainKm, limitsKm, ...
    cmap, outDir)
for i = 1:height(manifest)
    imageFile = fullfile(outDir, char(manifest.case_id(i) + ".png"));
    if isfile(imageFile)
        continue
    end
    fig = figure('Visible', 'off', 'Color', 'w', ...
        'Position', [40, 40, 760, 650]);
    ax = axes(fig);
    drawTerrain(ax, lon, lat, terrainKm(:, :, i), limitsKm, cmap);
    xlabel(ax, 'Longitude (deg E)');
    ylabel(ax, 'Latitude (deg N)');
    title(ax, sprintf('%s | Q %.1f, V %.1f, G %.1f km', ...
        manifest.case_id(i), manifest.qiangtang_target_m(i) / 1000, ...
        manifest.valley_target_m(i) / 1000, ...
        manifest.gangdese_target_m(i) / 1000), ...
        'Interpreter', 'none', 'FontWeight', 'normal');
    cb = colorbar(ax);
    cb.Label.String = 'Elevation (km)';
    exportgraphics(fig, imageFile, 'Resolution', 180, ...
        'BackgroundColor', 'white');
    close(fig);
end
end

function drawTerrain(ax, lon, lat, valuesKm, limitsKm, cmap)
imagesc(ax, lon, lat, valuesKm);
set(ax, 'YDir', 'normal', 'CLim', limitsKm, 'Layer', 'top', ...
    'FontName', 'Arial', 'LineWidth', 0.6, 'TickDir', 'out');
axis(ax, 'image');
axis(ax, 'tight');
colormap(ax, cmap);
end

function cmap = terrainColormap(n)
position = [0, 0.18, 0.38, 0.58, 0.78, 1];
color = [0.90, 0.95, 0.84; ...
    0.55, 0.70, 0.40; ...
    0.58, 0.50, 0.32; ...
    0.72, 0.62, 0.48; ...
    0.86, 0.83, 0.76; ...
    0.98, 0.98, 0.97];
x = linspace(0, 1, n);
cmap = interp1(position, color, x, 'pchip');
cmap = min(max(cmap, 0), 1);
end
