function result = opiAssimilate_CollectedCarbonateProxies( ...
    experimentRoot, d18OFile, clumpedFile, configFile, outputDir)
% Reweight discrete OPI cases using multi-site carbonate proxy observations.

projectRoot = fileparts(fileparts(mfilename('fullpath')));
scenarioRoot = fullfile(projectRoot, 'scenarios', ...
    'Qiangtang_30Ma_4000m_dH045_T290_fP030_Valid_platform_northsouth');
if nargin < 1 || isempty(experimentRoot)
    experimentRoot = fullfile(scenarioRoot, ...
        'topography_qiangtang_height_refinement');
end
if nargin < 2 || isempty(d18OFile)
    d18OFile = fullfile(projectRoot, 'collected_data', 'd18O.xlsx');
end
if nargin < 3 || isempty(clumpedFile)
    clumpedFile = fullfile(projectRoot, 'collected_data', 'clump_temp.xlsx');
end
if nargin < 4 || isempty(configFile)
    configFile = fullfile(projectRoot, 'data', 'assimilation', ...
        'collected_carbonate_assimilation_config.csv');
end
if nargin < 5 || isempty(outputDir)
    outputDir = fullfile(experimentRoot, ...
        'assimilation_collected_carbonate');
end

manifestFile = fullfile(experimentRoot, 'design', 'case_manifest.csv');
requiredFiles = string({manifestFile, d18OFile, clumpedFile, configFile});
missing = requiredFiles(~isfile(requiredFiles));
if ~isempty(missing)
    error('Missing collected-proxy assimilation input(s):\n%s', ...
        strjoin(missing, newline));
end
if ~isfolder(outputDir)
    mkdir(outputDir);
end

config = readConfig(configFile);
manifest = readtable(manifestFile, 'TextType', 'string');
observations = buildObservations(d18OFile, clumpedFile);
numCases = height(manifest);
caseRows = repmat(emptyCaseRow(), numCases, 1);
predictionTables = cell(numCases, 1);

for i = 1:numCases
    caseId = manifest.case_id(i);
    resultFile = fullfile(experimentRoot, 'calc_only', caseId, ...
        'opiCalc_TwoWinds_OxygenOnly_Results.mat');
    if ~isfile(resultFile)
        error('Missing OPI result for case %s: %s', caseId, resultFile);
    end
    [caseRows(i), predictionTables{i}] = evaluateCase( ...
        caseId, manifest(i, :), resultFile, observations, config);
end

predictions = vertcat(predictionTables{:});
casePosterior = struct2table(caseRows);
casePosterior.prior_probability = assignCasePriors(casePosterior, config);
logWeight = casePosterior.log_likelihood_joint + ...
    log(casePosterior.prior_probability);
casePosterior.posterior_probability = normalizeLogWeights(logWeight);
casePosterior.carbonate_only_probability = normalizeLogWeights( ...
    casePosterior.log_likelihood_carbonate + ...
    log(casePosterior.prior_probability));
casePosterior.temperature_only_probability = normalizeLogWeights( ...
    casePosterior.log_likelihood_temperature + ...
    log(casePosterior.prior_probability));
[~, order] = sort(casePosterior.posterior_probability, 'descend');
casePosterior.posterior_rank = nan(numCases, 1);
casePosterior.posterior_rank(order) = (1:numCases)';
heightPosterior = aggregateHeightPosterior(casePosterior);

writetable(observations, fullfile(outputDir, ...
    'collected_proxy_observations.csv'));
writetable(predictions, fullfile(outputDir, ...
    'collected_proxy_predictions.csv'));
writetable(casePosterior, fullfile(outputDir, ...
    'collected_proxy_case_posterior.csv'));
writetable(heightPosterior, fullfile(outputDir, ...
    'collected_proxy_height_posterior.csv'));
copyfile(configFile, fullfile(outputDir, ...
    'collected_proxy_assimilation_config_snapshot.csv'));
makePosteriorFigure(casePosterior, heightPosterior, outputDir);
writeReport(fullfile(outputDir, ...
    'collected_proxy_assimilation_report.md'), observations, predictions, ...
    casePosterior, heightPosterior, config, configFile);

result = struct;
result.status = "provisional_multi_site_proxy_assimilation";
result.observations = observations;
result.predictions = predictions;
result.casePosterior = casePosterior;
result.heightPosterior = heightPosterior;
result.config = config;
result.outputDir = string(outputDir);
save(fullfile(outputDir, 'collected_proxy_assimilation_results.mat'), ...
    'result', '-v7');

fprintf('Wrote collected-proxy assimilation to:\n%s\n', outputDir);
fprintf('Highest-weight case: %s (posterior %.4f)\n', ...
    casePosterior.case_id(order(1)), ...
    casePosterior.posterior_probability(order(1)));
end

function observations = buildObservations(d18OFile, clumpedFile)
d18O = readtable(d18OFile, 'TextType', 'string', ...
    'VariableNamingRule', 'preserve');
clumped = readtable(clumpedFile, 'TextType', 'string', ...
    'VariableNamingRule', 'preserve');
checkColumns(d18O, ["d18O_carbonate", "Uncertainty_2sigma", ...
    "Type", "longitude", "latitude", "Source", "doi"], ...
    'carbonate d18O');
checkColumns(clumped, ["d47_temperature_carbonate", ...
    "Uncertainty_2sigma", "Type", "longitude", "latitude", ...
    "Source", "doi"], 'clumped temperature');

nD18O = height(d18O);
siteId = "site_" + compose('%02d', (1:nD18O)');
source = d18O.Source;
doi = d18O.doi;
proxyType = d18O.Type;
longitude = d18O.longitude;
latitude = d18O.latitude;
d18OVPDB = d18O.d18O_carbonate;
d18OSigma = d18O.Uncertainty_2sigma ./ 2;
clumpedTemperatureC = nan(nD18O, 1);
clumpedTemperatureSigmaC = nan(nD18O, 1);

coordinateTolerance = 1e-6;
matchedClumped = false(height(clumped), 1);
for i = 1:nD18O
    distance = hypot(clumped.longitude - longitude(i), ...
        clumped.latitude - latitude(i));
    sameSource = strcmpi(strtrim(clumped.Source), strtrim(source(i)));
    match = find(distance <= coordinateTolerance & sameSource);
    if numel(match) > 1
        error('Multiple clumped matches for carbonate observation row %d.', i);
    elseif isscalar(match)
        clumpedTemperatureC(i) = clumped.d47_temperature_carbonate(match);
        clumpedTemperatureSigmaC(i) = clumped.Uncertainty_2sigma(match) ./ 2;
        matchedClumped(match) = true;
    end
end

unmatched = find(~matchedClumped);
for j = unmatched'
    siteId(end+1, 1) = "site_" + compose('%02d', numel(siteId) + 1); %#ok<AGROW>
    source(end+1, 1) = clumped.Source(j); %#ok<AGROW>
    doi(end+1, 1) = clumped.doi(j); %#ok<AGROW>
    proxyType(end+1, 1) = clumped.Type(j); %#ok<AGROW>
    longitude(end+1, 1) = clumped.longitude(j); %#ok<AGROW>
    latitude(end+1, 1) = clumped.latitude(j); %#ok<AGROW>
    d18OVPDB(end+1, 1) = nan; %#ok<AGROW>
    d18OSigma(end+1, 1) = nan; %#ok<AGROW>
    clumpedTemperatureC(end+1, 1) = ...
        clumped.d47_temperature_carbonate(j); %#ok<AGROW>
    clumpedTemperatureSigmaC(end+1, 1) = ...
        clumped.Uncertainty_2sigma(j) ./ 2; %#ok<AGROW>
end

hasD18O = isfinite(d18OVPDB) & isfinite(d18OSigma) & d18OSigma > 0;
hasClumpedTemperature = isfinite(clumpedTemperatureC) & ...
    isfinite(clumpedTemperatureSigmaC) & clumpedTemperatureSigmaC > 0;
proxyTypeLower = lower(strtrim(proxyType));
usesLakeOperator = ~contains(proxyTypeLower, "paleosol");
observations = table(siteId, source, doi, proxyType, longitude, latitude, ...
    d18OVPDB, d18OSigma, clumpedTemperatureC, ...
    clumpedTemperatureSigmaC, hasD18O, hasClumpedTemperature, ...
    usesLakeOperator, ...
    'VariableNames', {'site_id', 'source', 'doi', 'proxy_type', ...
    'longitude', 'latitude', 'd18O_carbonate_VPDB_permil', ...
    'd18O_carbonate_1sigma_permil', 'clumped_temperature_C', ...
    'clumped_temperature_1sigma_C', 'has_d18O', ...
    'has_clumped_temperature', 'uses_lake_operator'});
end

function [caseRow, prediction] = evaluateCase( ...
    caseId, manifestRow, resultFile, observations, config)
S = load(resultFile, 'lon', 'lat', 'd18OGrid', 'pGrid', ...
    'pGrid_1', 'pGrid_2', 'T_1', 'T_2', 'gammaSat_1', ...
    'gammaSat_2', 'topoFile', 'dataPath');
topographyFile = fullfile(S.dataPath, S.topoFile);
P = load(topographyFile, 'hGrid');
hGrid = P.hGrid;

temperature1C = S.T_1(1) - S.gammaSat_1(1) .* hGrid - 273.15;
temperature2C = S.T_2(1) - S.gammaSat_2(1) .* hGrid - 273.15;
airMAATGridC = (S.pGrid_1 .* temperature1C + ...
    S.pGrid_2 .* temperature2C) ./ S.pGrid;
airMAATGridC(~isfinite(airMAATGridC)) = nan;

[lonGrid, latGrid] = meshgrid(S.lon, S.lat);
nObs = height(observations);
insideDomain = observations.longitude >= min(S.lon) & ...
    observations.longitude <= max(S.lon) & ...
    observations.latitude >= min(S.lat) & ...
    observations.latitude <= max(S.lat);
waterD18OPred = nan(nObs, 1);
airMAATPred = nan(nObs, 1);
elevationPredM = nan(nObs, 1);
nSpatialCells = zeros(nObs, 1);
status = repmat("not_evaluated", nObs, 1);

for j = 1:nObs
    if ~observations.uses_lake_operator(j) && ...
            ~logical(config.include_non_lacustrine)
        status(j) = "unsupported_non_lacustrine_proxy";
        continue
    end
    if ~insideDomain(j)
        status(j) = "outside_model_domain";
        continue
    end
    distanceKm = greatCircleDistanceKm(latGrid, lonGrid, ...
        observations.latitude(j), observations.longitude(j));
    inRadius = distanceKm <= config.spatial_radius_km;
    temperatureMask = inRadius & isfinite(S.pGrid) & S.pGrid > 0 & ...
        isfinite(airMAATGridC) & isfinite(hGrid);
    if ~any(temperatureMask, 'all')
        status(j) = "no_valid_temperature_cells";
        continue
    end
    temperatureWeight = S.pGrid(temperatureMask);
    airMAATPred(j) = sum(temperatureWeight .* ...
        airMAATGridC(temperatureMask)) ./ sum(temperatureWeight);
    elevationPredM(j) = sum(temperatureWeight .* hGrid(temperatureMask)) ./ ...
        sum(temperatureWeight);
    nSpatialCells(j) = sum(temperatureMask, 'all');

    if observations.has_d18O(j)
        isotopeMask = inRadius & isfinite(S.pGrid) & S.pGrid > 0 & ...
            isfinite(S.d18OGrid);
        if ~any(isotopeMask, 'all')
            status(j) = "no_valid_isotope_cells";
            continue
        end
        isotopeWeight = S.pGrid(isotopeMask);
        waterD18OPred(j) = sum(isotopeWeight .* ...
            S.d18OGrid(isotopeMask) .* 1e3) ./ sum(isotopeWeight);
    end
    status(j) = "evaluated";
end

lakeWarmestPredC = nan(nObs, 1);
sigmaLakeTransferC = nan(nObs, 1);
canPredictTemperature = insideDomain & isfinite(airMAATPred);
if any(canPredictTemperature)
    [lakeWarmestPredC(canPredictTemperature), ...
        sigmaLakeTransferC(canPredictTemperature)] = ...
        lakeTransferAirToLake_Terrazas2025( ...
        airMAATPred(canPredictTemperature), ...
        observations.latitude(canPredictTemperature), ...
        elevationPredM(canPredictTemperature) ./ 1e3, "warmest");
end

carbonatePredVPDB = nan(nObs, 1);
carbonateResidual = nan(nObs, 1);
carbonateSigmaTotal = nan(nObs, 1);
carbonateZ = nan(nObs, 1);
temperatureResidual = nan(nObs, 1);
temperatureSigmaTotal = nan(nObs, 1);
temperatureZ = nan(nObs, 1);
logLikelihoodCarbonate = nan(nObs, 1);
logLikelihoodTemperature = nan(nObs, 1);
logLikelihoodJoint = zeros(nObs, 1);
isUsed = false(nObs, 1);

for j = 1:nObs
    if status(j) ~= "evaluated" || ~isfinite(lakeWarmestPredC(j))
        continue
    end
    sigmaTemperatureModel = hypot(sigmaLakeTransferC(j), ...
        config.temperature_model_discrepancy_C);
    hasCarbonate = observations.has_d18O(j) & isfinite(waterD18OPred(j));
    hasTemperature = observations.has_clumped_temperature(j);

    if hasCarbonate
        [carbonatePredVPDB(j), ~, derivativeTemperature, ...
            derivativeWater] = kimONeil1997CalciteWater( ...
            waterD18OPred(j), lakeWarmestPredC(j), ...
            "water_vsmow_to_carbonate_vpdb");
        carbonateResidual(j) = ...
            observations.d18O_carbonate_VPDB_permil(j) - ...
            carbonatePredVPDB(j);
        carbonateVariance = ...
            observations.d18O_carbonate_1sigma_permil(j).^2 + ...
            (derivativeTemperature .* sigmaTemperatureModel).^2 + ...
            (derivativeWater .* ...
            config.water_d18O_model_discrepancy_permil).^2 + ...
            config.carbonate_fractionation_discrepancy_permil.^2;
        carbonateSigmaTotal(j) = sqrt(carbonateVariance);
        carbonateZ(j) = carbonateResidual(j) ./ carbonateSigmaTotal(j);
        logLikelihoodCarbonate(j) = normalLogLikelihood( ...
            carbonateResidual(j), carbonateSigmaTotal(j));
    end

    if hasTemperature
        temperatureResidual(j) = observations.clumped_temperature_C(j) - ...
            lakeWarmestPredC(j);
        temperatureVariance = ...
            observations.clumped_temperature_1sigma_C(j).^2 + ...
            sigmaTemperatureModel.^2;
        temperatureSigmaTotal(j) = sqrt(temperatureVariance);
        temperatureZ(j) = temperatureResidual(j) ./ temperatureSigmaTotal(j);
        logLikelihoodTemperature(j) = normalLogLikelihood( ...
            temperatureResidual(j), temperatureSigmaTotal(j));
    end

    if hasCarbonate && hasTemperature
        sharedTemperatureVariance = sigmaTemperatureModel.^2;
        covariance = derivativeTemperature .* sharedTemperatureVariance;
        covarianceMatrix = [carbonateSigmaTotal(j).^2, covariance; ...
            covariance, temperatureSigmaTotal(j).^2];
        residual = [carbonateResidual(j); temperatureResidual(j)];
        logLikelihoodJoint(j) = multivariateNormalLogLikelihood( ...
            residual, covarianceMatrix);
    elseif hasCarbonate
        logLikelihoodJoint(j) = logLikelihoodCarbonate(j);
    elseif hasTemperature
        logLikelihoodJoint(j) = logLikelihoodTemperature(j);
    else
        continue
    end
    isUsed(j) = true;
end

prediction = table(repmat(string(caseId), nObs, 1), ...
    observations.site_id, observations.source, observations.longitude, ...
    observations.latitude, observations.has_d18O, ...
    observations.has_clumped_temperature, insideDomain, isUsed, status, ...
    repmat(config.spatial_radius_km, nObs, 1), nSpatialCells, ...
    elevationPredM, airMAATPred, lakeWarmestPredC, sigmaLakeTransferC, ...
    waterD18OPred, carbonatePredVPDB, ...
    observations.d18O_carbonate_VPDB_permil, carbonateResidual, ...
    carbonateSigmaTotal, carbonateZ, observations.clumped_temperature_C, ...
    temperatureResidual, temperatureSigmaTotal, temperatureZ, ...
    logLikelihoodCarbonate, ...
    logLikelihoodTemperature, logLikelihoodJoint, ...
    'VariableNames', {'case_id', 'site_id', 'source', 'longitude', ...
    'latitude', 'has_d18O', 'has_clumped_temperature', ...
    'inside_model_domain', 'is_used', 'status', 'spatial_radius_km', ...
    'n_spatial_cells', 'predicted_elevation_m', ...
    'predicted_air_MAAT_C', 'predicted_lake_warmest_C', ...
    'sigma_lake_transfer_C', 'predicted_water_d18O_VSMOW_permil', ...
    'predicted_carbonate_d18O_VPDB_permil', ...
    'observed_carbonate_d18O_VPDB_permil', ...
    'carbonate_residual_permil', 'carbonate_sigma_total_permil', ...
    'carbonate_z', ...
    'observed_clumped_temperature_C', 'temperature_residual_C', ...
    'temperature_sigma_total_C', 'temperature_z', ...
    'log_likelihood_carbonate', ...
    'log_likelihood_temperature', 'log_likelihood_joint'});

caseRow = emptyCaseRow();
caseRow.case_id = string(caseId);
caseRow.gangdese_target_m = manifestRow.gangdese_target_m;
caseRow.qiangtang_target_m = manifestRow.qiangtang_target_m;
caseRow.valley_mode = manifestRow.valley_mode;
caseRow.n_observations_total = nObs;
caseRow.n_observations_used = sum(isUsed);
caseRow.n_carbonate_used = sum(isUsed & observations.has_d18O);
caseRow.n_temperature_used = sum(isUsed & ...
    observations.has_clumped_temperature);
caseRow.log_likelihood_carbonate = sum( ...
    logLikelihoodCarbonate(isUsed & observations.has_d18O));
caseRow.log_likelihood_temperature = sum( ...
    logLikelihoodTemperature(isUsed & ...
    observations.has_clumped_temperature));
caseRow.log_likelihood_joint = sum(logLikelihoodJoint(isUsed));
caseRow.mean_abs_carbonate_residual_permil = mean( ...
    abs(carbonateResidual(isUsed & observations.has_d18O)), 'omitnan');
caseRow.mean_abs_temperature_residual_C = mean( ...
    abs(temperatureResidual(isUsed & ...
    observations.has_clumped_temperature)), 'omitnan');
caseRow.max_abs_carbonate_z = max(abs( ...
    carbonateZ(isUsed & observations.has_d18O)), [], 'omitnan');
caseRow.n_carbonate_abs_z_gt3 = sum(abs( ...
    carbonateZ(isUsed & observations.has_d18O)) > 3);
caseRow.max_abs_temperature_z = max(abs( ...
    temperatureZ(isUsed & observations.has_clumped_temperature)), ...
    [], 'omitnan');
caseRow.result_file = string(resultFile);
end

function config = readConfig(configFile)
importOptions = detectImportOptions(configFile, 'TextType', 'string');
importOptions = setvartype(importOptions, {'key', 'value'}, 'string');
T = readtable(configFile, importOptions);
checkColumns(T, ["key", "value"], 'assimilation config');
config = struct;
for i = 1:height(T)
    key = char(T.key(i));
    numericValue = str2double(T.value(i));
    if isfinite(numericValue)
        config.(key) = numericValue;
    else
        config.(key) = T.value(i);
    end
end
end

function checkColumns(T, required, label)
missing = setdiff(required, string(T.Properties.VariableNames));
if ~isempty(missing)
    error('Missing %s column(s): %s', label, strjoin(missing, ', '));
end
end

function distanceKm = greatCircleDistanceKm(lat, lon, centerLat, centerLon)
radiusEarthKm = 6371;
dLat = deg2rad(lat - centerLat);
dLon = deg2rad(lon - centerLon);
a = sin(dLat ./ 2).^2 + cosd(lat) .* cosd(centerLat) .* ...
    sin(dLon ./ 2).^2;
distanceKm = 2 .* radiusEarthKm .* asin(min(1, sqrt(a)));
end

function value = normalLogLikelihood(residual, sigma)
value = -0.5 .* (residual ./ sigma).^2 - log(sigma) - 0.5 .* log(2*pi);
end

function value = multivariateNormalLogLikelihood(residual, covariance)
[R, flag] = chol(covariance);
if flag ~= 0
    error('Proxy likelihood covariance is not positive definite.');
end
standardized = R' \ residual;
value = -0.5 .* (standardized' * standardized) - ...
    sum(log(diag(R))) - numel(residual) ./ 2 .* log(2*pi);
end

function prior = assignCasePriors(casePosterior, config)
n = height(casePosterior);
prior = zeros(n, 1);
switch lower(string(config.case_prior_type))
    case {"uniform", "uniform_case"}
        prior(:) = 1 ./ n;
    case "uniform_qiangtang_height_then_case"
        heights = unique(casePosterior.qiangtang_target_m);
        for i = 1:numel(heights)
            keep = casePosterior.qiangtang_target_m == heights(i);
            prior(keep) = 1 ./ numel(heights) ./ sum(keep);
        end
    otherwise
        error('Unsupported case prior: %s', config.case_prior_type);
end
end

function weight = normalizeLogWeights(logWeight)
maximum = max(logWeight);
weight = exp(logWeight - maximum);
weight = weight ./ sum(weight);
end

function heightPosterior = aggregateHeightPosterior(casePosterior)
[heightM, ~, group] = unique(casePosterior.qiangtang_target_m);
priorProbability = accumarray(group, casePosterior.prior_probability, ...
    [numel(heightM), 1], @sum, 0);
posteriorProbability = accumarray(group, ...
    casePosterior.posterior_probability, [numel(heightM), 1], @sum, 0);
carbonateOnlyProbability = accumarray(group, ...
    casePosterior.carbonate_only_probability, ...
    [numel(heightM), 1], @sum, 0);
temperatureOnlyProbability = accumarray(group, ...
    casePosterior.temperature_only_probability, ...
    [numel(heightM), 1], @sum, 0);
heightPosterior = table(heightM, priorProbability, posteriorProbability, ...
    carbonateOnlyProbability, temperatureOnlyProbability, ...
    'VariableNames', {'qiangtang_height_m', 'prior_probability', ...
    'posterior_probability', 'carbonate_only_probability', ...
    'temperature_only_probability'});
heightPosterior = sortrows(heightPosterior, 'qiangtang_height_m');
end

function row = emptyCaseRow()
row = struct('case_id', "", 'gangdese_target_m', nan, ...
    'qiangtang_target_m', nan, 'valley_mode', "", ...
    'n_observations_total', nan, 'n_observations_used', nan, ...
    'n_carbonate_used', nan, 'n_temperature_used', nan, ...
    'log_likelihood_carbonate', nan, ...
    'log_likelihood_temperature', nan, ...
    'log_likelihood_joint', nan, ...
    'mean_abs_carbonate_residual_permil', nan, ...
    'mean_abs_temperature_residual_C', nan, ...
    'max_abs_carbonate_z', nan, 'n_carbonate_abs_z_gt3', nan, ...
    'max_abs_temperature_z', nan, 'result_file', "", ...
    'prior_probability', nan, 'posterior_probability', nan, ...
    'carbonate_only_probability', nan, ...
    'temperature_only_probability', nan, 'posterior_rank', nan);
end

function makePosteriorFigure(casePosterior, heightPosterior, outputDir)
fig = figure('Color', 'w', 'Name', 'Collected carbonate proxy assimilation');
tiledlayout(fig, 1, 2);
ax1 = nexttile;
bar(ax1, categorical(casePosterior.case_id), ...
    casePosterior.posterior_probability, 'FaceColor', [0.15, 0.48, 0.65]);
grid(ax1, 'on');
ylabel(ax1, 'Posterior probability');
title(ax1, 'Topography cases');
ax1.XTickLabelRotation = 60;

ax2 = nexttile;
plot(ax2, heightPosterior.qiangtang_height_m, ...
    heightPosterior.prior_probability, '--o', 'LineWidth', 1.5, ...
    'DisplayName', 'Prior');
hold(ax2, 'on');
plot(ax2, heightPosterior.qiangtang_height_m, ...
    heightPosterior.posterior_probability, '-s', 'LineWidth', 2, ...
    'DisplayName', 'Posterior');
plot(ax2, heightPosterior.qiangtang_height_m, ...
    heightPosterior.carbonate_only_probability, ':^', 'LineWidth', 1.4, ...
    'DisplayName', 'Carbonate only');
plot(ax2, heightPosterior.qiangtang_height_m, ...
    heightPosterior.temperature_only_probability, '-.d', 'LineWidth', 1.4, ...
    'DisplayName', 'Temperature only');
grid(ax2, 'on');
xlabel(ax2, 'Qiangtang height (m)');
ylabel(ax2, 'Probability');
title(ax2, 'Height posterior');
legend(ax2, 'Location', 'best');
exportgraphics(fig, fullfile(outputDir, ...
    'Fig_CollectedProxy_Assimilation.png'), 'Resolution', 220);
savefig(fig, fullfile(outputDir, 'Fig_CollectedProxy_Assimilation.fig'));
close(fig);
end

function writeReport(reportFile, observations, predictions, ...
    casePosterior, heightPosterior, config, configFile)
fid = fopen(reportFile, 'w');
if fid == -1
    error('Could not create assimilation report: %s', reportFile);
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
[~, best] = max(casePosterior.posterior_probability);
bestPredictions = predictions(predictions.case_id == ...
    casePosterior.case_id(best), :);

fprintf(fid, '# Collected Carbonate Proxy Assimilation\n\n');
fprintf(fid, '> Status: provisional discrete-ensemble data assimilation.\n\n');
fprintf(fid, 'The observation operator maps OPI precipitation d18O and ');
fprintf(fid, 'surface-air MAAT to warmest lake temperature and then to ');
fprintf(fid, 'carbonate d18O using Kim and O''Neil (1997).\n\n');
fprintf(fid, '## Observation operator\n\n');
fprintf(fid, '- Spatial support radius: %.1f km, precipitation weighted.\n', ...
    config.spatial_radius_km);
fprintf(fid, '- Water d18O model discrepancy: %.2f per mil.\n', ...
    config.water_d18O_model_discrepancy_permil);
fprintf(fid, '- Carbonate fractionation discrepancy: %.2f per mil.\n', ...
    config.carbonate_fractionation_discrepancy_permil);
fprintf(fid, '- Temperature model discrepancy: %.2f C.\n\n', ...
    config.temperature_model_discrepancy_C);

fprintf(fid, '## Observations\n\n');
fprintf(fid, '- Total proxy sites: %d.\n', height(observations));
fprintf(fid, '- Carbonate d18O sites: %d.\n', sum(observations.has_d18O));
fprintf(fid, '- Clumped-temperature sites: %d.\n', ...
    sum(observations.has_clumped_temperature));
fprintf(fid, '- Sites assigned to the lake operator: %d.\n', ...
    sum(observations.uses_lake_operator));
fprintf(fid, '- Best-case observations used: %d.\n\n', ...
    sum(bestPredictions.is_used));

fprintf(fid, '## Posterior\n\n');
fprintf(fid, '- Highest-weight case: `%s`.\n', casePosterior.case_id(best));
fprintf(fid, '- Posterior probability: %.4f.\n', ...
    casePosterior.posterior_probability(best));
fprintf(fid, '- Mean absolute carbonate residual: %.3f per mil.\n', ...
    casePosterior.mean_abs_carbonate_residual_permil(best));
fprintf(fid, '- Mean absolute temperature residual: %.3f C.\n\n', ...
    casePosterior.mean_abs_temperature_residual_C(best));
fprintf(fid, '- Maximum absolute carbonate z score: %.2f.\n', ...
    casePosterior.max_abs_carbonate_z(best));
fprintf(fid, '- Carbonate observations beyond 3 sigma: %d.\n', ...
    casePosterior.n_carbonate_abs_z_gt3(best));
fprintf(fid, '- Maximum absolute temperature z score: %.2f.\n\n', ...
    casePosterior.max_abs_temperature_z(best));
if casePosterior.n_carbonate_abs_z_gt3(best) > 0
    fprintf(fid, ['The posterior ranks the tested cases, but the best case ' ...
        'does not provide an adequate absolute fit to every carbonate ' ...
        'observation. Inspect the site diagnostics before interpreting ' ...
        'the posterior as a validated elevation estimate.\n\n']);
end

fprintf(fid, '| Height m | Prior | Joint | Carbonate only | Temperature only |\n');
fprintf(fid, '|---:|---:|---:|---:|---:|\n');
for i = 1:height(heightPosterior)
    fprintf(fid, '| %.0f | %.4f | %.4f | %.4f | %.4f |\n', ...
        heightPosterior.qiangtang_height_m(i), ...
        heightPosterior.prior_probability(i), ...
        heightPosterior.posterior_probability(i), ...
        heightPosterior.carbonate_only_probability(i), ...
        heightPosterior.temperature_only_probability(i));
end
fprintf(fid, '\n## Best-case site diagnostics\n\n');
fprintf(fid, '| Source | Used | Water d18O | Carbonate residual | Carbonate z | T residual | T z | Status |\n');
fprintf(fid, '|---|---:|---:|---:|---:|---:|---:|---|\n');
for i = 1:height(bestPredictions)
    fprintf(fid, '| %s | %d | %.3f | %.3f | %.3f | %.3f | %.3f | %s |\n', ...
        bestPredictions.source(i), bestPredictions.is_used(i), ...
        bestPredictions.predicted_water_d18O_VSMOW_permil(i), ...
        bestPredictions.carbonate_residual_permil(i), ...
        bestPredictions.carbonate_z(i), ...
        bestPredictions.temperature_residual_C(i), ...
        bestPredictions.temperature_z(i), ...
        bestPredictions.status(i));
end
fprintf(fid, '\nConfiguration source: `%s`.\n', configFile);
end
