function opiCreate_PlatformDivideFiles()
% opiCreate_PlatformDivideFiles rebuilds divide files for the platform topo
% scenario and writes north/south shifted variants.

scenarioDir = fullfile(fileparts(mfilename('fullpath')), '..', 'scenarios', ...
    'Qiangtang_30Ma_4000m_dH045_T290_fP030_Valid_platform_northsouth');
topoFile = fullfile(scenarioDir, 'Tibet_Eocene_30Ma_topo.mat');
S = load(topoFile, 'lon', 'lat', 'hGrid');

baseLat = chooseBaseDivideLatitude(S.lat, S.hGrid);
contDivideLon = S.lon(:);
contDivideLat = repmat(baseLat, numel(contDivideLon), 1);
save(fullfile(scenarioDir, 'Tibet_Eocene_30Ma_topo_divide_main.mat'), ...
    'contDivideLon', 'contDivideLat');
save(fullfile(scenarioDir, 'Tibet_Eocene_30Ma_topo_divide_secondary.mat'), ...
    'contDivideLon', 'contDivideLat');

shiftValues = [0.1, 0.2, 0.3];
for i = 1:numel(shiftValues)
    shift = shiftValues(i);
    writeShiftedDivide(scenarioDir, contDivideLon, contDivideLat, ...
        S.lat, shift, 'north');
    writeShiftedDivide(scenarioDir, contDivideLon, contDivideLat, ...
        S.lat, shift, 'south');
end

writeDivideSummary(scenarioDir, S.lat, S.hGrid, baseLat, shiftValues);
writeDividePlot(scenarioDir, S.lon, S.lat, S.hGrid, baseLat, shiftValues);

fprintf('Rebuilt divide files for platform scenario.\n');
fprintf('Base divide latitude: %.5f\n', baseLat);
end

function baseLat = chooseBaseDivideLatitude(lat, hGrid)
rowMean = mean(hGrid, 2, 'omitnan');
northMask = lat >= 32.5;
if ~any(northMask)
    error('Could not identify northern platform rows from latitude grid.');
end
candidateRows = find(northMask);
[~, relIdx] = max(rowMean(candidateRows));
baseLat = lat(candidateRows(relIdx));
end

function writeShiftedDivide(scenarioDir, contDivideLon, contDivideLat, latGrid, shift, direction)
switch direction
    case 'north'
        shiftedLat = contDivideLat + shift;
        suffix = sprintf('north%03d', round(shift * 100));
    case 'south'
        shiftedLat = contDivideLat - shift;
        suffix = sprintf('south%03d', round(shift * 100));
    otherwise
        error('Unknown divide-shift direction: %s', direction);
end

latMin = min(latGrid);
latMax = max(latGrid);
if any(shiftedLat < latMin | shiftedLat > latMax)
    error('Shifted divide %s exceeds topography latitude bounds.', suffix);
end

contDivideLat = shiftedLat; %#ok<NASGU>
save(fullfile(scenarioDir, ['Tibet_Eocene_30Ma_topo_divide_main_', suffix, '.mat']), ...
    'contDivideLon', 'contDivideLat');
end

function writeDivideSummary(scenarioDir, lat, hGrid, baseLat, shiftValues)
summaryFile = fullfile(scenarioDir, 'platform_divide_summary.txt');
fid = fopen(summaryFile, 'w');
if fid == -1
    error('Could not create divide summary file: %s', summaryFile);
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>

rowMean = mean(hGrid, 2, 'omitnan');
rowMax = max(hGrid, [], 2);
fprintf(fid, 'Platform-scenario divide summary\n');
fprintf(fid, 'Scenario directory: %s\n\n', scenarioDir);
fprintf(fid, 'Base divide latitude: %.5f\n', baseLat);
fprintf(fid, 'North shifts written: %.1f, %.1f, %.1f deg\n', shiftValues);
fprintf(fid, 'South shifts written: %.1f, %.1f, %.1f deg\n\n', shiftValues);
fprintf(fid, 'Latitude rows near northern platform\n');
for i = 1:numel(lat)
    if lat(i) >= 32.5
        fprintf(fid, '  %.5f : mean %.1f m, max %.1f m\n', ...
            lat(i), rowMean(i), rowMax(i));
    end
end
end

function writeDividePlot(scenarioDir, lon, lat, hGrid, baseLat, shiftValues)
fig = figure('Visible', 'off', 'Color', 'w');
imagesc(lon, lat, hGrid);
set(gca, 'YDir', 'normal');
hold on
plot(lon, baseLat .* ones(size(lon)), 'k-', 'LineWidth', 2);
colors = lines(numel(shiftValues));
for i = 1:numel(shiftValues)
    plot(lon, (baseLat + shiftValues(i)) .* ones(size(lon)), '--', ...
        'Color', colors(i, :), 'LineWidth', 1.2);
    plot(lon, (baseLat - shiftValues(i)) .* ones(size(lon)), ':', ...
        'Color', colors(i, :), 'LineWidth', 1.2);
end
xlabel('Longitude');
ylabel('Latitude');
title('Platform topography with rebuilt divide lines');
colorbar
exportgraphics(fig, fullfile(scenarioDir, 'platform_divide_lines.png'), ...
    'Resolution', 200);
close(fig);
end
