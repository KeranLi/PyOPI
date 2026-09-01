function opiCreate_PlatformScenario()
% opiCreate_PlatformScenario creates a derivative Qiangtang scenario whose
% southern and northern mountain belts use flattened plateau-style crests.

sourceScenario = fullfile(fileparts(mfilename('fullpath')), '..', 'scenarios', ...
    'Qiangtang_30Ma_4000m_dH045_T290_fP030_Valid');
targetName = 'Qiangtang_30Ma_4000m_dH045_T290_fP030_Valid_platform_northsouth';
targetScenario = fullfile(fileparts(sourceScenario), targetName);

if ~isfolder(targetScenario)
    mkdir(targetScenario);
end

params = defaultPlatformParameters();
copyScenarioSkeleton(sourceScenario, targetScenario);
buildPlatformTopography(sourceScenario, targetScenario, params);
rewriteRunFiles(targetScenario, params.sectionOrigin);
writeScenarioReadme(targetScenario, sourceScenario, params);

fprintf('Created platform scenario:\n%s\n', targetScenario);
end

function params = defaultPlatformParameters()
params = struct();
params.sectionOrigin = [87.2, 32.9];
params.south = struct( ...
    'latCenter', 29.15, ...
    'halfWidthDeg', 0.16, ...
    'transitionDeg', 0.18, ...
    'shoulderElevationM', 2350, ...
    'plateauElevationM', 2900);
params.north = struct( ...
    'latCenter', 33.20, ...
    'halfWidthDeg', 0.22, ...
    'transitionDeg', 0.20, ...
    'shoulderElevationM', 3200, ...
    'plateauElevationM', 4150);
end

function copyScenarioSkeleton(sourceScenario, targetScenario)
copyPlainFile(sourceScenario, targetScenario, 'Tibet_Eocene_30Ma_samples.xlsx');
copyPlainFile(sourceScenario, targetScenario, 'Tibet_Eocene_30Ma.run');
copyPlainFile(sourceScenario, targetScenario, 'Tibet_Eocene_30Ma_OxygenOnly.run');
copyPlainFile(sourceScenario, targetScenario, 'Tibet_Eocene_30Ma_OxygenOnly_Fit.run');

copyMatchingFiles(sourceScenario, targetScenario, 'Tibet_Eocene_30Ma_topo_divide*.mat');
copyMatchingFiles(sourceScenario, targetScenario, 'Tibet_Eocene_30Ma_topo_divide*.png');

copySubdirFiles(sourceScenario, targetScenario, 'proxy_clumped', {'*.xlsx', '*.csv'});
copySubdirFiles(sourceScenario, targetScenario, 'oxygen_clumped_ultra_aggressive', {'*.run'});
end

function copyPlainFile(sourceScenario, targetScenario, fileName)
copyfile(fullfile(sourceScenario, fileName), fullfile(targetScenario, fileName));
end

function copyMatchingFiles(sourceScenario, targetScenario, pattern)
files = dir(fullfile(sourceScenario, pattern));
for i = 1:numel(files)
    copyfile(fullfile(files(i).folder, files(i).name), ...
        fullfile(targetScenario, files(i).name));
end
end

function copySubdirFiles(sourceScenario, targetScenario, subdirName, patterns)
sourceDir = fullfile(sourceScenario, subdirName);
targetDir = fullfile(targetScenario, subdirName);
if ~isfolder(targetDir)
    mkdir(targetDir);
end
for i = 1:numel(patterns)
    files = dir(fullfile(sourceDir, patterns{i}));
    for j = 1:numel(files)
        copyfile(fullfile(files(j).folder, files(j).name), ...
            fullfile(targetDir, files(j).name));
    end
end
end

function buildPlatformTopography(sourceScenario, targetScenario, params)
sourceTopo = fullfile(sourceScenario, 'Tibet_Eocene_30Ma_topo.mat');
targetTopo = fullfile(targetScenario, 'Tibet_Eocene_30Ma_topo.mat');
S = load(sourceTopo, 'lon', 'lat', 'hGrid');
original = S.hGrid;

southWeight = bandWeight(S.lat, params.south.latCenter, ...
    params.south.halfWidthDeg, params.south.transitionDeg);
northWeight = bandWeight(S.lat, params.north.latCenter, ...
    params.north.halfWidthDeg, params.north.transitionDeg);

S.hGrid = applyPlateauBand(S.hGrid, southWeight, ...
    params.south.shoulderElevationM, params.south.plateauElevationM);
S.hGrid = applyPlateauBand(S.hGrid, northWeight, ...
    params.north.shoulderElevationM, params.north.plateauElevationM);
S.hGrid = max(S.hGrid, 0);

save(targetTopo, '-struct', 'S');
writeTopographySummary(targetScenario, S.lat, original, S.hGrid, params);
end

function weights = bandWeight(lat, center, halfWidth, transition)
distance = abs(lat(:) - center);
weights = zeros(size(distance));
weights(distance <= halfWidth) = 1;
mask = distance > halfWidth & distance < (halfWidth + transition);
weights(mask) = 1 - smoothstep((distance(mask) - halfWidth) ./ transition);
end

function hGridOut = applyPlateauBand(hGridIn, rowWeights, shoulderElevationM, plateauElevationM)
hGridOut = hGridIn;
for i = 1:size(hGridIn, 1)
    wLat = rowWeights(i);
    if wLat <= 0
        continue
    end
    row = hGridIn(i, :);
    t = clamp01((row - shoulderElevationM) ./ ...
        max(plateauElevationM - shoulderElevationM, eps));
    liftWeight = smoothstep(t);
    rowOut = row + wLat .* (plateauElevationM - row) .* liftWeight;
    rowOut = min(rowOut, row + wLat .* max(plateauElevationM - row, 0));
    rowOut = min(rowOut, plateauElevationM);
    hGridOut(i, :) = rowOut;
end
end

function rewriteRunFiles(targetScenario, sectionOrigin)
runFiles = {};
rootRuns = dir(fullfile(targetScenario, '*.run'));
for i = 1:numel(rootRuns)
    runFiles{end+1} = fullfile(rootRuns(i).folder, rootRuns(i).name); %#ok<AGROW>
end
subRuns = dir(fullfile(targetScenario, 'oxygen_clumped_ultra_aggressive', '*.run'));
for i = 1:numel(subRuns)
    runFiles{end+1} = fullfile(subRuns(i).folder, subRuns(i).name); %#ok<AGROW>
end

for i = 1:numel(runFiles)
    rewriteOneRunFile(runFiles{i}, targetScenario, sectionOrigin);
end
end

function rewriteOneRunFile(runFile, targetScenario, sectionOrigin)
lines = readlines(runFile, 'WhitespaceRule', 'preserve');
idx = findActiveLineIndices(lines);
lines(idx(3)) = string(targetScenario);
lines(idx(10)) = sprintf('%.5f, %.5f', sectionOrigin(1), sectionOrigin(2));
writelines(lines, runFile);
end

function idx = findActiveLineIndices(lines)
idx = [];
for i = 1:numel(lines)
    str = strip(string(lines(i)));
    if strlength(str) == 0 || startsWith(str, "%")
        continue
    end
    idx(end+1) = i; %#ok<AGROW>
end
if numel(idx) < 10
    error('Run file format did not match expected active-line count.');
end
end

function writeTopographySummary(targetScenario, lat, original, modified, params)
summaryFile = fullfile(targetScenario, 'platform_topography_summary.txt');
fid = fopen(summaryFile, 'w');
if fid == -1
    error('Could not create summary file: %s', summaryFile);
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>

rowMaxOriginal = max(original, [], 2);
rowMaxModified = max(modified, [], 2);
fprintf(fid, 'Platform-topography derivative scenario\n');
fprintf(fid, 'Target scenario: %s\n\n', targetScenario);
fprintf(fid, 'South platform\n');
fprintf(fid, '  latCenter = %.3f deg\n', params.south.latCenter);
fprintf(fid, '  halfWidthDeg = %.3f deg\n', params.south.halfWidthDeg);
fprintf(fid, '  transitionDeg = %.3f deg\n', params.south.transitionDeg);
fprintf(fid, '  shoulderElevationM = %.1f\n', params.south.shoulderElevationM);
fprintf(fid, '  plateauElevationM = %.1f\n\n', params.south.plateauElevationM);
fprintf(fid, 'North platform\n');
fprintf(fid, '  latCenter = %.3f deg\n', params.north.latCenter);
fprintf(fid, '  halfWidthDeg = %.3f deg\n', params.north.halfWidthDeg);
fprintf(fid, '  transitionDeg = %.3f deg\n', params.north.transitionDeg);
fprintf(fid, '  shoulderElevationM = %.1f\n', params.north.shoulderElevationM);
fprintf(fid, '  plateauElevationM = %.1f\n\n', params.north.plateauElevationM);
fprintf(fid, 'Original max elevation: %.1f m\n', max(original(:)));
fprintf(fid, 'Modified max elevation: %.1f m\n\n', max(modified(:)));
fprintf(fid, 'Row maxima by latitude (original -> modified)\n');
for i = 1:numel(lat)
    fprintf(fid, '  %.5f : %.1f -> %.1f\n', lat(i), rowMaxOriginal(i), rowMaxModified(i));
end
end

function writeScenarioReadme(targetScenario, sourceScenario, params)
readmeFile = fullfile(targetScenario, 'README_PLATFORM_TOPOGRAPHY.txt');
fid = fopen(readmeFile, 'w');
if fid == -1
    error('Could not create README: %s', readmeFile);
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>

fprintf(fid, 'Platform-topography scenario derived from:\n%s\n\n', sourceScenario);
fprintf(fid, 'Folder name marks this scenario as platform_northsouth.\n');
fprintf(fid, 'Both southern and northern mountain belts were reshaped toward flat-topped plateaus.\n');
fprintf(fid, 'Topography file: Tibet_Eocene_30Ma_topo.mat\n');
fprintf(fid, 'Fixed section origin retained for later sensitivity work: %.5f, %.5f\n\n', ...
    params.sectionOrigin(1), params.sectionOrigin(2));
fprintf(fid, 'Important note:\n');
fprintf(fid, '  Existing divide files were copied from the source scenario as placeholders.\n');
fprintf(fid, '  Before full divide-sensitivity experiments, regenerate divide files against the platform topo.\n');
end

function y = smoothstep(x)
x = clamp01(x);
y = x .* x .* (3 - 2 .* x);
end

function y = clamp01(x)
y = min(max(x, 0), 1);
end
