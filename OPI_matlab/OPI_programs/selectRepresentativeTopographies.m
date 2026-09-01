function result = selectRepresentativeTopographies( ...
    posteriorFile, experimentRoot, outputRoot, hpdTarget, representativeCount)
% Select real terrain cases that represent the high-posterior ensemble.
%
% The smallest posterior-ranked prefix reaching the requested HPD mass is
% retained. Posterior-weighted k-medoids then selects representative cases
% using area-weighted RMS distance between their complete elevation grids.

if nargin < 1 || strlength(string(posteriorFile)) == 0
    archiveRoot = fullfile(getenv('HOME'), 'BaiduNetdiskSync', ...
        'posterior_OPI_topography_precip_isotope');
    posteriorFile = fullfile(archiveRoot, ...
        'assimilation_projection_smooth', ...
        'completed_assimilation_joint_case_posterior.csv');
end
if nargin < 2 || strlength(string(experimentRoot)) == 0
    assimilationRoot = fileparts(fileparts(mfilename('fullpath')));
    experimentRoot = fullfile(assimilationRoot, '..', 'OPI_matlab', ...
        'scenarios', ...
        'Qiangtang_30Ma_4000m_dH045_T290_fP030_Valid_platform_northsouth', ...
        'topography_north_south_grid_smooth');
end
if nargin < 3 || strlength(string(outputRoot)) == 0
    outputRoot = fullfile(fileparts(posteriorFile), ...
        'representative_topographies');
end
if nargin < 4 || isempty(hpdTarget)
    hpdTarget = 0.95;
end
if nargin < 5 || isempty(representativeCount)
    representativeCount = 4;
end
validateattributes(hpdTarget, {'numeric'}, ...
    {'scalar', 'finite', '>', 0, '<=', 1}, mfilename, 'hpdTarget');
validateattributes(representativeCount, {'numeric'}, ...
    {'scalar', 'integer', 'positive'}, mfilename, 'representativeCount');

posteriorFile = char(string(posteriorFile));
experimentRoot = char(string(experimentRoot));
outputRoot = char(string(outputRoot));
if ~isfile(posteriorFile)
    error('Posterior table not found: %s', posteriorFile);
end
if ~isfolder(experimentRoot)
    error('Terrain experiment not found: %s', experimentRoot);
end
if ~isfolder(outputRoot)
    mkdir(outputRoot);
end

posterior = readtable(posteriorFile, 'TextType', 'string');
posterior = standardizePosteriorSchema(posterior);
required = ["case_id", "qiangtang_elevation_m", ...
    "central_valley_zone_elevation_m", "gangdese_elevation_m", ...
    "posterior_probability"];
missing = setdiff(required, string(posterior.Properties.VariableNames));
if ~isempty(missing)
    error('Posterior table is missing column(s): %s', ...
        strjoin(missing, ', '));
end
if numel(unique(posterior.case_id)) ~= height(posterior)
    error('Posterior case IDs must be unique.');
end
probability = posterior.posterior_probability;
if any(~isfinite(probability)) || any(probability < 0) || sum(probability) <= 0
    error('Posterior probabilities must be finite and nonnegative.');
end
inputProbabilitySum = sum(probability);
posterior.posterior_probability = probability ./ inputProbabilitySum;
posterior = sortrows(posterior, ...
    {'posterior_probability', 'case_id'}, {'descend', 'ascend'});
posterior.posterior_rank = (1:height(posterior))';
posterior.cumulative_probability = cumsum( ...
    posterior.posterior_probability);

boundaryIndex = find(posterior.cumulative_probability >= hpdTarget, 1);
boundaryProbability = posterior.posterior_probability(boundaryIndex);
tieTolerance = max(eps(boundaryProbability) * 8, eps);
inHpd = posterior.posterior_probability >= ...
    boundaryProbability - tieTolerance;
posterior.in_hpd = inHpd;
hpd = posterior(inHpd, :);
hpdMass = sum(hpd.posterior_probability);
hpd.hpd_conditional_probability = ...
    hpd.posterior_probability ./ hpdMass;
if representativeCount > height(hpd)
    error('Cannot select %d representatives from %d HPD cases.', ...
        representativeCount, height(hpd));
end

writetable(posterior, fullfile(outputRoot, 'posterior_cases_ranked.csv'));
writetable(hpd, fullfile(outputRoot, 'hpd95_cases.csv'));

[lon, lat, terrainM] = loadTerrainEnsemble(experimentRoot, hpd.case_id);
distanceM = pairwiseAreaWeightedRms(terrainM, lat);
[medoidIndex, ~, objectiveM] = weightedPam( ...
    distanceM, hpd.hpd_conditional_probability, representativeCount);

% Present representative types from the lowest to highest central zone.
medoidParameters = [hpd.central_valley_zone_elevation_m(medoidIndex), ...
    hpd.qiangtang_elevation_m(medoidIndex), ...
    hpd.gangdese_elevation_m(medoidIndex)];
[~, medoidOrder] = sortrows(medoidParameters, [1, 2, 3]);
medoidIndex = medoidIndex(medoidOrder);
[~, assignment] = min(distanceM(:, medoidIndex), [], 2);

representativeId = "R" + string((1:representativeCount)');
representatives = buildRepresentativeTable(hpd, medoidIndex, assignment, ...
    distanceM, representativeId, experimentRoot);
assignments = buildAssignmentTable(hpd, medoidIndex, assignment, ...
    distanceM, representativeId);
writetable(representatives, fullfile(outputRoot, ...
    'representative_topographies.csv'));
writetable(assignments, fullfile(outputRoot, ...
    'hpd95_representative_assignments.csv'));

figureBase = fullfile(outputRoot, ...
    'Fig_HPD95_Four_Representative_Topographies');
plotRepresentatives(lon, lat, terrainM(:, :, medoidIndex), ...
    representatives, figureBase);
opiValidation = evaluateRepresentativeOpi(posterior, hpd, ...
    representatives, experimentRoot, outputRoot);
reportFile = fullfile(outputRoot, ...
    'representative_topography_selection_report.md');
writeReport(reportFile, posteriorFile, experimentRoot, hpdTarget, ...
    hpdMass, boundaryProbability, inputProbabilitySum, posterior, ...
    representatives, objectiveM, opiValidation);

save(fullfile(outputRoot, 'representative_topography_selection.mat'), ...
    'posteriorFile', 'experimentRoot', 'hpdTarget', 'hpdMass', ...
    'boundaryProbability', 'representativeCount', 'medoidIndex', ...
    'assignment', 'distanceM', 'objectiveM', 'representatives', ...
    'opiValidation', 'lon', 'lat', '-v7.3');

result = struct;
result.hpdTarget = hpdTarget;
result.hpdMass = hpdMass;
result.hpdCaseCount = height(hpd);
result.representatives = representatives;
result.assignments = assignments;
result.opiValidation = opiValidation;
result.outputRoot = string(outputRoot);
fprintf(['Retained %d of %d cases in the %.0f%% HPD set ' ...
    '(mass %.6f).\n'], height(hpd), height(posterior), ...
    100 * hpdTarget, hpdMass);
fprintf('Selected %d representative topographies under:\n%s\n', ...
    representativeCount, outputRoot);
end

function posterior = standardizePosteriorSchema(posterior)
names = string(posterior.Properties.VariableNames);
if ~ismember("case_id", names)
    error('Posterior table is missing column: case_id');
end

posterior.qiangtang_elevation_m = selectNumericColumn(posterior, ...
    ["qiangtang_elevation_m", "qiangtang_target_m"]);
posterior.gangdese_elevation_m = selectNumericColumn(posterior, ...
    ["gangdese_elevation_m", "gangdese_target_m"]);
if ismember("central_valley_zone_elevation_m", names)
    posterior.central_valley_zone_elevation_m = ...
        posterior.central_valley_zone_elevation_m;
elseif ismember("valley_target_m", names)
    posterior.central_valley_zone_elevation_m = posterior.valley_target_m;
elseif ismember("valley_mode", names)
    posterior.central_valley_zone_elevation_m = ...
        str2double(erase(string(posterior.valley_mode), "V"));
else
    error(['Posterior table must contain central_valley_zone_elevation_m, ' ...
        'valley_target_m, or valley_mode.']);
end
posterior.posterior_probability = selectNumericColumn(posterior, ...
    ["posterior_probability_normalized", "posterior_probability"]);

standardized = [posterior.qiangtang_elevation_m; ...
    posterior.central_valley_zone_elevation_m; ...
    posterior.gangdese_elevation_m; posterior.posterior_probability];
if any(~isfinite(standardized))
    error('Posterior terrain parameters and probabilities must be finite.');
end
end

function values = selectNumericColumn(T, candidates)
names = string(T.Properties.VariableNames);
i = find(ismember(candidates, names), 1, 'first');
if isempty(i)
    error('Posterior table is missing one of: %s', strjoin(candidates, ', '));
end
values = T.(candidates(i));
if ~isnumeric(values)
    values = str2double(string(values));
end
values = values(:);
end

function [lon, lat, terrainM] = loadTerrainEnsemble(experimentRoot, caseId)
lon = [];
lat = [];
terrainM = [];
for i = 1:numel(caseId)
    topoFile = fullfile(experimentRoot, 'calc_only', caseId(i), ...
        'Tibet_Eocene_30Ma_topo.mat');
    if ~isfile(topoFile)
        error('Missing terrain for posterior case %s: %s', ...
            caseId(i), topoFile);
    end
    S = load(topoFile, 'lon', 'lat', 'hGrid');
    if i == 1
        lon = S.lon(:)';
        lat = S.lat(:);
        terrainM = nan(numel(lat), numel(lon), numel(caseId));
    elseif ~isequal(lon, S.lon(:)') || ~isequal(lat, S.lat(:))
        error('Terrain %s does not use the common grid.', caseId(i));
    end
    if ~isequal(size(S.hGrid), [numel(lat), numel(lon)]) || ...
            any(~isfinite(S.hGrid), 'all')
        error('Terrain %s contains an invalid elevation grid.', caseId(i));
    end
    terrainM(:, :, i) = S.hGrid;
end
end

function distanceM = pairwiseAreaWeightedRms(terrainM, lat)
nCase = size(terrainM, 3);
grid = reshape(terrainM, [], nCase)';
cellWeight = repmat(cosd(lat(:)), size(terrainM, 2), 1);
cellWeight = cellWeight ./ sum(cellWeight);
weightedGrid = grid .* sqrt(cellWeight(:)');
squaredNorm = sum(weightedGrid.^2, 2);
distanceSquared = squaredNorm + squaredNorm' - ...
    2 .* (weightedGrid * weightedGrid');
distanceM = sqrt(max(distanceSquared, 0));
distanceM(1:nCase + 1:end) = 0;
end

function [medoids, assignment, objective] = weightedPam(distance, weight, k)
% Deterministic BUILD initialization followed by PAM swap refinement.
n = size(distance, 1);
weight = weight(:) ./ sum(weight);
firstCost = weight' * distance;
[~, medoids] = min(firstCost);
while numel(medoids) < k
    nearest = min(distance(:, medoids), [], 2);
    bestCost = inf;
    bestCandidate = nan;
    for candidate = 1:n
        if any(medoids == candidate)
            continue
        end
        candidateCost = sum(weight .* ...
            min(nearest, distance(:, candidate)));
        if candidateCost < bestCost
            bestCost = candidateCost;
            bestCandidate = candidate;
        end
    end
    medoids(end + 1) = bestCandidate; %#ok<AGROW>
end

objective = sum(weight .* min(distance(:, medoids), [], 2));
improved = true;
while improved
    improved = false;
    bestObjective = objective;
    bestMedoids = medoids;
    for position = 1:k
        for candidate = 1:n
            if any(medoids == candidate)
                continue
            end
            trial = medoids;
            trial(position) = candidate;
            trialObjective = sum(weight .* ...
                min(distance(:, trial), [], 2));
            if trialObjective < bestObjective - 1e-10
                bestObjective = trialObjective;
                bestMedoids = trial;
                improved = true;
            end
        end
    end
    medoids = bestMedoids;
    objective = bestObjective;
end
[~, assignment] = min(distance(:, medoids), [], 2);
end

function representatives = buildRepresentativeTable( ...
    hpd, medoidIndex, assignment, distanceM, representativeId, experimentRoot)
n = numel(medoidIndex);
caseId = hpd.case_id(medoidIndex);
posteriorRank = hpd.posterior_rank(medoidIndex);
caseProbability = hpd.posterior_probability(medoidIndex);
qiangtangM = hpd.qiangtang_elevation_m(medoidIndex);
centralM = hpd.central_valley_zone_elevation_m(medoidIndex);
gangdeseM = hpd.gangdese_elevation_m(medoidIndex);
clusterMemberCount = nan(n, 1);
clusterPosteriorMass = nan(n, 1);
clusterConditionalMass = nan(n, 1);
clusterMeanDistanceM = nan(n, 1);
clusterMaxDistanceM = nan(n, 1);
caseDirectory = strings(n, 1);
topographyFile = strings(n, 1);
fixedRunFile = strings(n, 1);
opiResultFile = strings(n, 1);
hpdMass = sum(hpd.posterior_probability);
for i = 1:n
    members = assignment == i;
    memberWeight = hpd.posterior_probability(members);
    memberDistance = distanceM(members, medoidIndex(i));
    clusterMemberCount(i) = sum(members);
    clusterPosteriorMass(i) = sum(memberWeight);
    clusterConditionalMass(i) = clusterPosteriorMass(i) ./ hpdMass;
    clusterMeanDistanceM(i) = sum(memberWeight .* memberDistance) ./ ...
        clusterPosteriorMass(i);
    clusterMaxDistanceM(i) = max(memberDistance);
    caseDirectory(i) = string(fullfile(experimentRoot, 'calc_only', caseId(i)));
    topographyFile(i) = fullfile(caseDirectory(i), ...
        'Tibet_Eocene_30Ma_topo.mat');
    fixedRunFile(i) = fullfile(caseDirectory(i), ...
        "Tibet_Eocene_30Ma_SmoothGrid_" + caseId(i) + "_Best.run");
    opiResultFile(i) = fullfile(caseDirectory(i), ...
        'opiCalc_TwoWinds_OxygenOnly_Results.mat');
end
representatives = table(representativeId, caseId, posteriorRank, ...
    caseProbability, qiangtangM, centralM, gangdeseM, ...
    clusterMemberCount, clusterPosteriorMass, clusterConditionalMass, ...
    clusterMeanDistanceM, clusterMaxDistanceM, caseDirectory, ...
    topographyFile, fixedRunFile, opiResultFile, ...
    'VariableNames', {'representative_id', 'case_id', 'posterior_rank', ...
    'case_posterior_probability', 'qiangtang_elevation_m', ...
    'central_valley_zone_elevation_m', 'gangdese_elevation_m', ...
    'hpd_cluster_member_count', 'hpd_cluster_posterior_mass', ...
    'hpd_cluster_conditional_mass', 'cluster_weighted_mean_rms_m', ...
    'cluster_max_rms_m', 'case_directory', 'topography_file', ...
    'fixed_opi_run_file', 'existing_opi_result_file'});
end

function assignments = buildAssignmentTable( ...
    hpd, medoidIndex, assignment, distanceM, representativeId)
assignedRepresentative = representativeId(assignment);
assignedCaseId = hpd.case_id(medoidIndex(assignment));
distanceToRepresentativeM = nan(height(hpd), 1);
for i = 1:height(hpd)
    distanceToRepresentativeM(i) = distanceM(i, medoidIndex(assignment(i)));
end
assignments = hpd(:, {'posterior_rank', 'case_id', ...
    'qiangtang_elevation_m', 'central_valley_zone_elevation_m', ...
    'gangdese_elevation_m', 'posterior_probability', ...
    'hpd_conditional_probability'});
assignments = addvars(assignments, assignedRepresentative, assignedCaseId, ...
    distanceToRepresentativeM, 'After', 'case_id', ...
    'NewVariableNames', {'representative_id', ...
    'representative_case_id', 'rms_distance_to_representative_m'});
assignments = sortrows(assignments, ...
    {'representative_id', 'posterior_rank'});
end

function plotRepresentatives(lon, lat, terrainM, representatives, baseFile)
fig = figure('Visible', 'off', 'Color', 'w', ...
    'Position', [50, 50, 1280, 980]);
cleanup = onCleanup(@() close(fig)); %#ok<NASGU>
layout = tiledlayout(fig, 2, 2, 'TileSpacing', 'compact', ...
    'Padding', 'compact');
for i = 1:height(representatives)
    ax = nexttile(layout, i);
    imagesc(ax, lon, lat, terrainM(:, :, i) ./ 1000);
    ax.YDir = 'normal';
    axis(ax, 'image');
    xlim(ax, [min(lon), max(lon)]);
    ylim(ax, [min(lat), max(lat)]);
    clim(ax, [0, 6]);
    title(ax, sprintf('%s | %s', ...
        representatives.representative_id(i), representatives.case_id(i)), ...
        'FontWeight', 'normal', 'FontSize', 12, ...
        'Interpreter', 'none', 'Color', 'k');
    subtitleText = subtitle(ax, sprintf(['Q %.1f | V %.1f | G %.1f km | ' ...
        'HPD mass %.1f%%'], ...
        representatives.qiangtang_elevation_m(i) / 1000, ...
        representatives.central_valley_zone_elevation_m(i) / 1000, ...
        representatives.gangdese_elevation_m(i) / 1000, ...
        100 * representatives.hpd_cluster_posterior_mass(i)), ...
        'FontSize', 9);
    subtitleText.Color = 'k';
    xlabel(ax, 'Longitude (deg E)', 'Color', 'k');
    ylabel(ax, 'Latitude (deg N)', 'Color', 'k');
    ax.TickDir = 'out';
    ax.Box = 'on';
    ax.XColor = 'k';
    ax.YColor = 'k';
end
colormap(fig, parula(256));
cb = colorbar;
cb.Layout.Tile = 'east';
cb.Label.String = 'Elevation (km)';
cb.Color = 'k';
cb.Label.Color = 'k';
layoutTitle = title(layout, ...
    'Four representative topographies within the 95% HPD set', ...
    'FontWeight', 'normal', 'FontSize', 16, 'Color', 'k'); %#ok<NASGU>
exportgraphics(fig, [baseFile, '.png'], 'Resolution', 250, ...
    'BackgroundColor', 'white');
exportgraphics(fig, [baseFile, '.pdf'], 'ContentType', 'image', ...
    'Resolution', 300, 'BackgroundColor', 'white');
savefig(fig, [baseFile, '.fig']);
end

function validation = evaluateRepresentativeOpi(posterior, hpd, ...
    representatives, experimentRoot, outputRoot)
[lon, lat, fields] = loadOpiEnsemble(experimentRoot, posterior.case_id);
fullWeight = posterior.posterior_probability ./ ...
    sum(posterior.posterior_probability);
[foundHpd, hpdIndex] = ismember(hpd.case_id, posterior.case_id);
[foundRepresentative, representativeIndex] = ismember( ...
    representatives.case_id, posterior.case_id);
if ~all(foundHpd) || ~all(foundRepresentative)
    error('Could not map all HPD and representative cases into the ensemble.');
end
hpdWeight = hpd.hpd_conditional_probability;
representativeWeight = representatives.hpd_cluster_conditional_mass;

formal = summarizeMeanFields(fields, fullWeight, (1:height(posterior))');
hpdMean = summarizeMeanFields(fields, hpdWeight, hpdIndex);
reduced = summarizeMeanFields(fields, representativeWeight, ...
    representativeIndex);
formalStd = summarizeStandardDeviationFields(fields, fullWeight, ...
    (1:height(posterior))', formal);
hpdStd = summarizeStandardDeviationFields(fields, hpdWeight, ...
    hpdIndex, hpdMean);
reducedStd = summarizeStandardDeviationFields(fields, ...
    representativeWeight, representativeIndex, reduced);

fieldNames = ["topography_m", "precipitation_mm_hr", "d18O_permil"];
units = ["m", "mm h^-1", "per mil VSMOW"];
comparisons = ["four_representatives_minus_full_posterior", ...
    "four_representatives_minus_HPD95_conditional"];
validation = table;
for iComparison = 1:numel(comparisons)
    if iComparison == 1
        reference = formal;
    else
        reference = hpdMean;
    end
    for iField = 1:numel(fieldNames)
        name = fieldNames(iField);
        difference = reduced.(name) - reference.(name);
        [rmse, bias, maxAbs, validFraction] = spatialDifferenceMetrics( ...
            difference, lat);
        fullPosteriorSdRms = spatialFieldRms(formalStd.(name), lat);
        rmseFractionOfPosteriorSd = rmse ./ fullPosteriorSdRms;
        row = table(name, units(iField), comparisons(iComparison), ...
            rmse, bias, maxAbs, validFraction, fullPosteriorSdRms, ...
            rmseFractionOfPosteriorSd, ...
            'VariableNames', {'field', 'unit', 'comparison', ...
            'area_weighted_RMSE', 'area_weighted_mean_difference', ...
            'maximum_absolute_difference', 'valid_grid_fraction', ...
            'full_posterior_SD_RMS', ...
            'RMSE_fraction_of_full_posterior_SD_RMS'});
        validation = [validation; row]; %#ok<AGROW>
    end
end

stdComparisons = ["four_representatives_SD_minus_full_posterior_SD", ...
    "four_representatives_SD_minus_HPD95_conditional_SD"];
for iComparison = 1:numel(stdComparisons)
    if iComparison == 1
        referenceStd = formalStd;
    else
        referenceStd = hpdStd;
    end
    for iField = 1:numel(fieldNames)
        name = fieldNames(iField);
        difference = reducedStd.(name) - referenceStd.(name);
        [rmse, bias, maxAbs, validFraction] = spatialDifferenceMetrics( ...
            difference, lat);
        fullPosteriorSdRms = spatialFieldRms(formalStd.(name), lat);
        rmseFractionOfPosteriorSd = rmse ./ fullPosteriorSdRms;
        outputName = name + "_standard_deviation";
        row = table(outputName, units(iField), ...
            stdComparisons(iComparison), rmse, bias, maxAbs, ...
            validFraction, fullPosteriorSdRms, ...
            rmseFractionOfPosteriorSd, ...
            'VariableNames', {'field', 'unit', 'comparison', ...
            'area_weighted_RMSE', 'area_weighted_mean_difference', ...
            'maximum_absolute_difference', 'valid_grid_fraction', ...
            'full_posterior_SD_RMS', ...
            'RMSE_fraction_of_full_posterior_SD_RMS'});
        validation = [validation; row]; %#ok<AGROW>
    end
end
writetable(validation, fullfile(outputRoot, ...
    'representative_opi_validation.csv'));

representativeFields = subsetFields(fields, representativeIndex); %#ok<NASGU>
save(fullfile(outputRoot, 'representative_opi_fields.mat'), ...
    'lon', 'lat', 'formal', 'hpdMean', 'reduced', 'formalStd', ...
    'hpdStd', 'reducedStd', ...
    'representativeFields', 'representatives', 'validation', '-v7.3');
plotRepresentativeOpiFields(lon, lat, representativeFields, ...
    representatives, fullfile(outputRoot, ...
    'Fig_HPD95_Four_Representative_OPI_Fields'));
end

function summary = summarizeStandardDeviationFields(fields, weight, index, means)
names = string(fieldnames(fields));
weight = weight(:) ./ sum(weight);
weightGrid = reshape(weight, 1, 1, []);
for i = 1:numel(names)
    values = fields.(names(i))(:, :, index);
    valid = isfinite(values);
    difference = values - means.(names(i));
    difference(~valid) = 0;
    denominator = sum(valid .* weightGrid, 3);
    variance = sum(difference .^ 2 .* weightGrid, 3) ./ denominator;
    standardDeviation = sqrt(max(variance, 0));
    standardDeviation(denominator <= 0) = nan;
    summary.(names(i)) = standardDeviation;
end
end

function [lon, lat, fields] = loadOpiEnsemble(experimentRoot, caseId)
nCase = numel(caseId);
lon = [];
lat = [];
fields = struct;
for i = 1:nCase
    resultFile = fullfile(experimentRoot, 'calc_only', caseId(i), ...
        'opiCalc_TwoWinds_OxygenOnly_Results.mat');
    topographyFile = fullfile(experimentRoot, 'calc_only', caseId(i), ...
        'Tibet_Eocene_30Ma_topo.mat');
    if ~isfile(resultFile)
        error('Missing OPI result for %s: %s', caseId(i), resultFile);
    end
    if ~isfile(topographyFile)
        error('Missing topography for %s: %s', caseId(i), topographyFile);
    end
    S = load(resultFile, 'lon', 'lat', 'pGrid', 'd18OGrid');
    T = load(topographyFile, 'lon', 'lat', 'hGrid');
    if ~isequal(S.lon(:)', T.lon(:)') || ~isequal(S.lat(:), T.lat(:))
        error('Topography and OPI grids differ for %s.', caseId(i));
    end
    if i == 1
        lon = S.lon(:)';
        lat = S.lat(:);
        shape = [numel(lat), numel(lon), nCase];
        fields.topography_m = nan(shape);
        fields.precipitation_mm_hr = nan(shape);
        fields.d18O_permil = nan(shape);
    elseif ~isequal(lon, S.lon(:)') || ~isequal(lat, S.lat(:))
        error('OPI result %s does not use the common grid.', caseId(i));
    end
    if ~isequal(size(T.hGrid), [numel(lat), numel(lon)]) || ...
            ~isequal(size(S.pGrid), [numel(lat), numel(lon)]) || ...
            ~isequal(size(S.d18OGrid), [numel(lat), numel(lon)])
        error('OPI result %s has invalid field dimensions.', caseId(i));
    end
    fields.topography_m(:, :, i) = T.hGrid;
    fields.precipitation_mm_hr(:, :, i) = max(S.pGrid, 0) .* 3.6e3;
    d18O = S.d18OGrid .* 1e3;
    d18O(S.pGrid <= 0 | ~isfinite(d18O)) = nan;
    fields.d18O_permil(:, :, i) = d18O;
end
end

function summary = summarizeMeanFields(fields, weight, index)
names = string(fieldnames(fields));
weight = weight(:) ./ sum(weight);
for i = 1:numel(names)
    values = fields.(names(i))(:, :, index);
    valid = isfinite(values);
    weightedValues = values;
    weightedValues(~valid) = 0;
    weightGrid = reshape(weight, 1, 1, []);
    denominator = sum(valid .* weightGrid, 3);
    numerator = sum(weightedValues .* weightGrid, 3);
    meanField = numerator ./ denominator;
    meanField(denominator <= 0) = nan;
    summary.(names(i)) = meanField;
end
end

function subset = subsetFields(fields, index)
names = string(fieldnames(fields));
for i = 1:numel(names)
    subset.(names(i)) = fields.(names(i))(:, :, index);
end
end

function [rmse, bias, maxAbs, validFraction] = ...
    spatialDifferenceMetrics(difference, lat)
valid = isfinite(difference);
cellWeight = repmat(cosd(lat(:)), 1, size(difference, 2));
cellWeight(~valid) = 0;
weightSum = sum(cellWeight, 'all');
if weightSum <= 0
    rmse = nan;
    bias = nan;
    maxAbs = nan;
else
    rmse = sqrt(sum(cellWeight .* difference .^ 2, 'all', ...
        'omitnan') ./ weightSum);
    bias = sum(cellWeight .* difference, 'all', 'omitnan') ./ weightSum;
    maxAbs = max(abs(difference), [], 'all', 'omitnan');
end
validFraction = mean(valid, 'all');
end

function rmsValue = spatialFieldRms(field, lat)
valid = isfinite(field);
cellWeight = repmat(cosd(lat(:)), 1, size(field, 2));
cellWeight(~valid) = 0;
weightSum = sum(cellWeight, 'all');
if weightSum <= 0
    rmsValue = nan;
else
    rmsValue = sqrt(sum(cellWeight .* field .^ 2, 'all', ...
        'omitnan') ./ weightSum);
end
end

function plotRepresentativeOpiFields(lon, lat, fields, representatives, baseFile)
fig = figure('Visible', 'off', 'Color', 'w', ...
    'Position', [50, 50, 1550, 1500]);
cleanup = onCleanup(@() close(fig)); %#ok<NASGU>
layout = tiledlayout(fig, height(representatives), 3, ...
    'TileSpacing', 'compact', 'Padding', 'compact');
plotNames = ["topography_m", "precipitation_mm_hr", "d18O_permil"];
columnTitles = ["Topography (km)", "Precipitation (mm h^{-1})", ...
    "Precipitation \delta^{18}O (per mil VSMOW)"];
scale = [1e-3, 1, 1];
colorMaps = {parula(256), turbo(256), turbo(256)};
limits = nan(3, 2);
for j = 1:3
    values = fields.(plotNames(j)) .* scale(j);
    limits(j, :) = finiteLimits(values);
end
for i = 1:height(representatives)
    for j = 1:3
        ax = nexttile(layout);
        values = fields.(plotNames(j))(:, :, i) .* scale(j);
        imagesc(ax, lon, lat, values);
        ax.YDir = 'normal';
        axis(ax, 'image');
        clim(ax, limits(j, :));
        colormap(ax, colorMaps{j});
        cb = colorbar(ax);
        cb.Color = 'k';
        if i == 1
            title(ax, columnTitles(j), 'FontWeight', 'normal', 'Color', 'k');
        end
        if j == 1
            ylabel(ax, sprintf('%s\n%s\nLat (deg N)', ...
                representatives.representative_id(i), ...
                representatives.case_id(i)), 'Interpreter', 'none', ...
                'Color', 'k');
        end
        if i == height(representatives)
            xlabel(ax, 'Longitude (deg E)', 'Color', 'k');
        end
        ax.Box = 'on';
        ax.TickDir = 'out';
        ax.XColor = 'k';
        ax.YColor = 'k';
    end
end
layoutTitle = title(layout, ...
    ['Four posterior-weighted representative OPI simulations ' ...
    'from the 95% HPD set'], 'FontWeight', 'normal', 'Color', 'k'); %#ok<NASGU>
exportgraphics(fig, [baseFile, '.png'], 'Resolution', 250, ...
    'BackgroundColor', 'white');
exportgraphics(fig, [baseFile, '.pdf'], 'ContentType', 'image', ...
    'Resolution', 300, 'BackgroundColor', 'white');
savefig(fig, [baseFile, '.fig']);
end

function limits = finiteLimits(values)
values = values(isfinite(values));
if isempty(values)
    limits = [0, 1];
elseif min(values) == max(values)
    limits = min(values) + [-0.5, 0.5];
else
    limits = [min(values), max(values)];
end
end

function writeReport(reportFile, posteriorFile, experimentRoot, hpdTarget, ...
    hpdMass, boundaryProbability, inputProbabilitySum, posterior, ...
    representatives, objectiveM, opiValidation)
fid = fopen(reportFile, 'w');
if fid == -1
    error('Could not create representative-terrain report: %s', reportFile);
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '# Representative Topographies from the 95%% HPD Set\n\n');
fprintf(fid, 'Source posterior: `%s`\n\n', posteriorFile);
fprintf(fid, 'Terrain ensemble: `%s`\n\n', experimentRoot);
fprintf(fid, '## HPD screening\n\n');
fprintf(fid, ['Cases were sorted by posterior probability. The smallest ' ...
    'ranked prefix reaching the target mass was retained; all exact ties ' ...
    'at the boundary would also be retained.\n\n']);
fprintf(fid, '- Candidate count: %d.\n', height(posterior));
fprintf(fid, '- Input probability sum before normalization: %.12g.\n', ...
    inputProbabilitySum);
fprintf(fid, '- HPD target: %.1f%%.\n', 100 * hpdTarget);
fprintf(fid, '- Retained cases: %d.\n', sum(posterior.in_hpd));
fprintf(fid, '- Retained posterior mass: %.8f.\n', hpdMass);
fprintf(fid, '- Boundary posterior probability: %.8g.\n', ...
    boundaryProbability);
fprintf(fid, ['- Posterior mass of the four highest-ranked cases alone: ' ...
    '%.8f.\n\n'], sum(posterior.posterior_probability(1:4)));
fprintf(fid, '## Representative selection\n\n');
fprintf(fid, ['Only HPD cases were eligible to be representatives. Four ' ...
    'real cases were selected with posterior-weighted k-medoids (PAM). ' ...
    'Distance is the cosine-latitude-area-weighted RMS difference between ' ...
    'the complete two-dimensional elevation grids. The final weighted ' ...
    'within-cluster RMS objective is %.2f m.\n\n'], objectiveM);
fprintf(fid, ['The four cases are a compact representation of posterior ' ...
    'uncertainty, not evidence for exactly four isolated posterior modes.\n\n']);
fprintf(fid, ['| ID | Case | Rank | Q m | V m | G m | Case p | ' ...
    'Cluster HPD p | HPD members | Mean RMS m |\n']);
fprintf(fid, '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n');
for i = 1:height(representatives)
    fprintf(fid, '| %s | %s | %d | %.0f | %.0f | %.0f | %.5f | %.5f | %d | %.1f |\n', ...
        representatives.representative_id(i), representatives.case_id(i), ...
        representatives.posterior_rank(i), ...
        representatives.qiangtang_elevation_m(i), ...
        representatives.central_valley_zone_elevation_m(i), ...
        representatives.gangdese_elevation_m(i), ...
        representatives.case_posterior_probability(i), ...
        representatives.hpd_cluster_posterior_mass(i), ...
        representatives.hpd_cluster_member_count(i), ...
        representatives.cluster_weighted_mean_rms_m(i));
end
fprintf(fid, '\n## Four-case OPI approximation validation\n\n');
fprintf(fid, ['The existing OPI outputs for the four medoids were combined ' ...
    'using their cluster HPD conditional masses. This reduced mixture was ' ...
    'compared with both the formal 160-case posterior mean and the ' ...
    '18-case HPD-conditional mean.\n\n']);
fprintf(fid, ['| Field | Comparison | Area-weighted RMSE | Mean difference | ' ...
    'Maximum absolute difference | RMSE / posterior SD RMS |\n']);
fprintf(fid, '|---|---|---:|---:|---:|---:|\n');
for i = 1:height(opiValidation)
    fprintf(fid, '| %s | %s | %.6g %s | %.6g %s | %.6g %s | %.3f |\n', ...
        opiValidation.field(i), opiValidation.comparison(i), ...
        opiValidation.area_weighted_RMSE(i), opiValidation.unit(i), ...
        opiValidation.area_weighted_mean_difference(i), ...
        opiValidation.unit(i), ...
        opiValidation.maximum_absolute_difference(i), ...
        opiValidation.unit(i), ...
        opiValidation.RMSE_fraction_of_full_posterior_SD_RMS(i));
end
fprintf(fid, '\n## OPI inputs\n\n');
for i = 1:height(representatives)
    fprintf(fid, '- %s `%s`: `%s`\n', ...
        representatives.representative_id(i), representatives.case_id(i), ...
        representatives.fixed_opi_run_file(i));
end
end
