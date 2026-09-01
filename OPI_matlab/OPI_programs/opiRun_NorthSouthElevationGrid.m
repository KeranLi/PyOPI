function summary = opiRun_NorthSouthElevationGrid( ...
    rootScenario, experimentName, caseId)
% Run or resume fixed-parameter OPI calculations for the smooth terrain grid.


if nargin < 1 || strlength(string(rootScenario)) == 0
    rootScenario = fullfile(fileparts(mfilename('fullpath')), '..', ...
        'scenarios', ...
        'Qiangtang_30Ma_4000m_dH045_T290_fP030_Valid_platform_northsouth');
end
if nargin < 2 || strlength(string(experimentName)) == 0
    experimentName = 'topography_north_south_grid_smooth';
end
if nargin < 3
    caseId = "";
end
experimentRoot = fullfile(char(string(rootScenario)), ...
    char(string(experimentName)));
manifestFile = fullfile(experimentRoot, 'design', 'case_manifest.csv');
if ~isfile(manifestFile)
    error('North-south case manifest not found: %s', manifestFile);
end
manifest = readtable(manifestFile, 'TextType', 'string');
if strlength(string(caseId)) > 0
    manifest = manifest(manifest.case_id == string(caseId), :);
    if isempty(manifest)
        error('North-south grid case not found: %s', caseId);
    end
end
analysisRoot = fullfile(experimentRoot, 'analysis');
if ~isfolder(analysisRoot)
    mkdir(analysisRoot);
end

n = height(manifest);
weightedD18O = nan(n, 1);
runStatus = strings(n, 1);
message = strings(n, 1);
for i = 1:n
    thisCase = manifest.case_id(i);
    caseDir = fullfile(experimentRoot, 'calc_only', thisCase);
    resultFile = fullfile(caseDir, ...
        'opiCalc_TwoWinds_OxygenOnly_Results.mat');
    try
        if ~isfile(resultFile)
            bestRuns = dir(fullfile(caseDir, '*_Best.run'));
            if numel(bestRuns) ~= 1
                error('Expected one best run, found %d.', numel(bestRuns));
            end
            fprintf('North-south case %d/%d: %s\n', i, n, thisCase);
            opiCalc_TwoWinds_OxygenOnly( ...
                fullfile(bestRuns(1).folder, bestRuns(1).name));
        end
        weightedD18O(i) = summarizeLocalD18O(resultFile, 87.2, 32.9, 50);
        runStatus(i) = "complete";
        message(i) = "fixed calculation available";
    catch ME
        runStatus(i) = "failed";
        message(i) = string(ME.message);
        warning('North-south case failed: %s\n%s', thisCase, getReport(ME));
    end
end
summary = table(manifest.case_id, weightedD18O, ...
    manifest.gangdese_target_m, manifest.qiangtang_target_m, ...
    manifest.valley_mode, manifest.valley_target_m, runStatus, message, ...
    'VariableNames', {'case_id', 'weighted_d18O_50km_permil', ...
    'gangdese_target_m', 'qiangtang_target_m', 'valley_mode', ...
    'valley_target_m', 'run_status', 'message'});
if strlength(string(caseId)) == 0
    writetable(summary, fullfile(analysisRoot, ...
        'height_refinement_target_comparison.csv'));
    writetable(summary(:, {'case_id', 'run_status', 'message'}), ...
        fullfile(analysisRoot, 'case_status.csv'));
end
if any(summary.run_status ~= "complete")
    error('%d of %d north-south grid cases failed.', ...
        sum(summary.run_status ~= "complete"), height(summary));
end
fprintf('Completed %d north-south grid cases under:\n%s\n', ...
    height(summary), experimentRoot);
end

function value = summarizeLocalD18O(resultFile, lon0, lat0, radiusKm)
S = load(resultFile, 'lon', 'lat', 'd18OGrid', 'pGrid');
[lonGrid, latGrid] = meshgrid(S.lon, S.lat);
distanceKm = greatCircleDistanceKm(latGrid, lonGrid, lat0, lon0);
mask = distanceKm <= radiusKm & isfinite(S.pGrid) & S.pGrid > 0 & ...
    isfinite(S.d18OGrid);
if ~any(mask, 'all')
    error('No wet isotope cells within %.1f km.', radiusKm);
end
weight = S.pGrid(mask);
value = sum(weight .* S.d18OGrid(mask) .* 1e3) ./ sum(weight);
end

function distanceKm = greatCircleDistanceKm(lat1, lon1, lat2, lon2)
earthRadiusKm = 6371.0088;
lat1 = deg2rad(lat1);
lon1 = deg2rad(lon1);
lat2 = deg2rad(lat2);
lon2 = deg2rad(lon2);
dLat = lat1 - lat2;
dLon = lon1 - lon2;
a = sin(dLat ./ 2).^2 + cos(lat1) .* cos(lat2) .* ...
    sin(dLon ./ 2).^2;
distanceKm = 2 .* earthRadiusKm .* atan2(sqrt(a), sqrt(max(0, 1 - a)));
end
