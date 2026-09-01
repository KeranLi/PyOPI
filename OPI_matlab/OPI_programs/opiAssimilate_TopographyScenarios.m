function result = opiAssimilate_TopographyScenarios( ...
    experimentRoot, clumpedFile, configFile, outputDir)
% opiAssimilate_TopographyScenarios reweights discrete OPI topographies.
%
% This first-stage assimilation marginalizes explicit warmest-month climate,
% lapse-rate, lake-geometry, and source-isotope nuisance parameters. It uses
% raw clumped lake temperature in a forward Terrazas ML likelihood and the
% independent 50 km precipitation d18O reconstruction in a second likelihood.

projectRoot = fileparts(fileparts(mfilename('fullpath')));
scenarioRoot = fullfile(projectRoot, 'scenarios', ...
    'Qiangtang_30Ma_4000m_dH045_T290_fP030_Valid_platform_northsouth');
if nargin < 1 || isempty(experimentRoot)
    experimentRoot = fullfile(scenarioRoot, ...
        'topography_qiangtang_height_refinement');
end
if nargin < 2 || isempty(clumpedFile)
    clumpedFile = fullfile(scenarioRoot, 'proxy_clumped', ...
        'clumped_temperature.xlsx');
end
if nargin < 3 || isempty(configFile)
    configFile = fullfile(projectRoot, 'data', 'assimilation', ...
        'qiangtang_assimilation_config.csv');
end
if nargin < 4 || isempty(outputDir)
    outputDir = fullfile(experimentRoot, 'assimilation');
end

requiredFiles = string({ ...
    fullfile(experimentRoot, 'design', 'case_manifest.csv'), ...
    fullfile(experimentRoot, 'analysis', ...
        'height_refinement_target_comparison.csv'), ...
    clumpedFile, configFile});
missingFiles = requiredFiles(~isfile(requiredFiles));
if ~isempty(missingFiles)
    error('Missing assimilation input(s):\n%s', strjoin(missingFiles, newline));
end
if ~isfolder(outputDir)
    mkdir(outputDir);
end

config = readConfig(configFile);
manifest = readtable(requiredFiles(1), 'TextType', 'string');
d18Summary = readtable(requiredFiles(2), 'TextType', 'string');
clumped = readtable(clumpedFile, 'TextType', 'string', ...
    'VariableNamingRule', 'preserve');
validateInputs(manifest, d18Summary, clumped);

obs = aggregateClumpedObservation(clumped, config);
draws = sampleNuisancePriors(config);
numCases = height(manifest);
rows = repmat(emptyCaseRow(), numCases, 1);

for i = 1:numCases
    caseId = manifest.case_id(i);
    j = find(d18Summary.case_id == caseId, 1);
    if isempty(j)
        error('No d18O summary row for case: %s', caseId);
    end
    resultFile = fullfile(experimentRoot, 'calc_only', caseId, ...
        'opiCalc_TwoWinds_OxygenOnly_Results.mat');
    if ~isfile(resultFile)
        error('Missing OPI result for case %s: %s', caseId, resultFile);
    end

    state = readCaseState(resultFile, obs.longitude, obs.latitude);
    geometry = struct( ...
        'gangdeseM', manifest.gangdese_target_m(i), ...
        'qiangtangM', manifest.qiangtang_target_m(i), ...
        'valleyMode', manifest.valley_mode(i));
    d18Base = d18Summary.weighted_d18O_50km_permil(j);
    rows(i) = evaluateCase(caseId, geometry, state, d18Base, ...
        obs, draws, config, resultFile);
end

casePosterior = struct2table(rows);
casePosterior.prior_weight = assignCasePriors(casePosterior, config);
valid = casePosterior.prior_weight > 0 & isfinite(casePosterior.log_evidence_joint);
logWeight = casePosterior.log_evidence_joint + log(casePosterior.prior_weight);
logWeight(~valid) = -inf;
casePosterior.posterior_probability = normalizeLogWeights(logWeight);
[~, rankOrder] = sort(casePosterior.posterior_probability, 'descend');
casePosterior.posterior_rank = nan(numCases, 1);
casePosterior.posterior_rank(rankOrder) = (1:numCases)';

heightPosterior = aggregateHeightPosterior(casePosterior);
writetable(casePosterior, fullfile(outputDir, ...
    'assimilation_case_posterior.csv'));
writetable(heightPosterior, fullfile(outputDir, ...
    'assimilation_qiangtang_height_posterior.csv'));
writetable(struct2table(obs), fullfile(outputDir, ...
    'assimilation_observation_summary.csv'));
copyfile(configFile, fullfile(outputDir, ...
    'assimilation_config_snapshot.csv'));
makePosteriorFigure(casePosterior, heightPosterior, outputDir);
writeAssimilationReport(fullfile(outputDir, 'assimilation_report.md'), ...
    casePosterior, heightPosterior, obs, config, configFile);

result = struct;
result.status = "provisional_internal_assimilation";
result.casePosterior = casePosterior;
result.heightPosterior = heightPosterior;
result.observation = obs;
result.config = config;
result.outputDir = string(outputDir);
save(fullfile(outputDir, 'assimilation_results.mat'), 'result', '-v7');

fprintf('Wrote provisional topography assimilation to:\n%s\n', outputDir);
fprintf('Highest-weight case: %s (posterior %.3f)\n', ...
    casePosterior.case_id(rankOrder(1)), ...
    casePosterior.posterior_probability(rankOrder(1)));
end

function config = readConfig(configFile)
importOptions = detectImportOptions(configFile, 'TextType', 'string');
importOptions = setvartype(importOptions, {'key', 'value'}, 'string');
T = readtable(configFile, importOptions);
required = ["key", "value"];
if ~all(ismember(required, string(T.Properties.VariableNames)))
    error('Assimilation config requires key and value columns: %s', configFile);
end
if numel(unique(T.key)) ~= height(T)
    error('Assimilation config contains duplicate keys: %s', configFile);
end
config = struct;
for i = 1:height(T)
    key = char(T.key(i));
    numericValue = str2double(T.value(i));
    if strcmpi(T.value(i), "NaN")
        config.(key) = nan;
    elseif isfinite(numericValue)
        config.(key) = numericValue;
    else
        config.(key) = T.value(i);
    end
end
end

function validateInputs(manifest, d18Summary, clumped)
manifestRequired = ["case_id", "gangdese_target_m", ...
    "qiangtang_target_m", "valley_mode"];
d18Required = ["case_id", "weighted_d18O_50km_permil"];
clumpedRequired = ["lon", "lat", "T_clumped_C", "sigma_T_C"];
checkColumns(manifest, manifestRequired, 'case manifest');
checkColumns(d18Summary, d18Required, 'd18O summary');
checkColumns(clumped, clumpedRequired, 'clumped observations');
end

function checkColumns(T, required, label)
missing = setdiff(required, string(T.Properties.VariableNames));
if ~isempty(missing)
    error('Missing %s column(s): %s', label, strjoin(missing, ', '));
end
end

function obs = aggregateClumpedObservation(T, config)
good = isfinite(T.T_clumped_C) & isfinite(T.sigma_T_C) & T.sigma_T_C > 0;
if ~any(good)
    error('No finite clumped observations are available.');
end
lakeTemperature = T.T_clumped_C(good) - config.dolomite_offset_C;
sigma = sqrt(T.sigma_T_C(good).^2 + config.sigma_dolomite_offset_C.^2);
weight = 1 ./ sigma.^2;
obs = struct;
obs.n_clumped = sum(good);
if config.observation_age_min_Ma > config.observation_age_max_Ma || ...
        (isfinite(config.observation_age_mean_Ma) && ...
        (config.observation_age_mean_Ma < config.observation_age_min_Ma || ...
        config.observation_age_mean_Ma > config.observation_age_max_Ma))
    error('Invalid local-observation age interval in assimilation config.');
end
obs.age_mean_Ma = config.observation_age_mean_Ma;
obs.age_min_Ma = config.observation_age_min_Ma;
obs.age_max_Ma = config.observation_age_max_Ma;
obs.age_basis = string(config.observation_age_basis);
obs.chronology_resolution = string(config.observation_chronology_resolution);
obs.lake_temperature_warmest_C = sum(weight .* lakeTemperature) ./ sum(weight);
obs.sigma_analytical_mean_C = sqrt(1 ./ sum(weight));
obs.longitude = sum(weight .* T.lon(good)) ./ sum(weight);
obs.latitude = sum(weight .* T.lat(good)) ./ sum(weight);
obs.dolomite_offset_C = config.dolomite_offset_C;
obs.sigma_dolomite_offset_C = config.sigma_dolomite_offset_C;
obs.target_d18O_permil = config.target_d18O_permil;
obs.sigma_target_d18O_permil = config.sigma_target_d18O_permil;
end

function draws = sampleNuisancePriors(config)
n = round(config.num_nuisance_samples);
if n < 100
    error('num_nuisance_samples must be at least 100.');
end
rng(config.random_seed, 'twister');
draws = struct;
draws.T0WarmestC = samplePrior(n, config.T0_warmest_prior_type, ...
    config.T0_warmest_mean_C, config.T0_warmest_sigma_C, ...
    config.T0_warmest_min_C, config.T0_warmest_max_C);
draws.lapseRateCPerKm = samplePrior(n, config.lapse_rate_prior_type, ...
    config.lapse_rate_mean_C_per_km, config.lapse_rate_sigma_C_per_km, ...
    config.lapse_rate_min_C_per_km, config.lapse_rate_max_C_per_km);
draws.sourceOffsetPermil = samplePrior(n, config.source_d18O_prior_type, ...
    config.source_d18O_mean_permil, config.source_d18O_sigma_permil, ...
    config.source_d18O_min_permil, config.source_d18O_max_permil);
draws.lakeAreaKm2 = exp(log(config.lake_area_median_km2) + ...
    config.lake_area_log_sigma .* randn(n, 1));
draws.lakeDepthM = exp(log(config.lake_depth_median_m) + ...
    config.lake_depth_log_sigma .* randn(n, 1));
if isfinite(config.observation_age_mean_Ma)
    draws.ageMa = repmat(config.observation_age_mean_Ma, n, 1);
else
    draws.ageMa = config.observation_age_min_Ma + ...
        (config.observation_age_max_Ma - config.observation_age_min_Ma) .* ...
        rand(n, 1);
end
durationMyr = config.westerhold_old_age_Ma - ...
    config.westerhold_young_age_Ma;
temperatureChangeC = truncatedNormal(n, ...
    config.westerhold_temperature_change_C, ...
    config.westerhold_temperature_change_1sigma_C, 0, inf);
regionalResponse = truncatedNormal(n, ...
    config.westerhold_regional_response_mean, ...
    config.westerhold_regional_response_1sigma, 0, inf);
ageOffsetMyr = draws.ageMa - config.model_time_slice_Ma;
draws.temperatureTimeOffsetC = regionalResponse .* ...
    temperatureChangeC ./ durationMyr .* ageOffsetMyr + ...
    config.temporal_temperature_rate_C_per_Myr .* abs(ageOffsetMyr) .* ...
    randn(n, 1);
draws.waterTimeOffsetPermil = ...
    config.temporal_water_d18O_rate_permil_per_Myr .* ...
    abs(ageOffsetMyr) .* randn(n, 1);
end

function value = samplePrior(n, priorType, mu, sigma, lowerBound, upperBound)
priorType = lower(string(priorType));
switch priorType
    case "uniform"
        value = lowerBound + (upperBound - lowerBound) .* rand(n, 1);
    case "normal"
        value = truncatedNormal(n, mu, sigma, lowerBound, upperBound);
    otherwise
        error('Unsupported nuisance prior type: %s', priorType);
end
end

function value = truncatedNormal(n, mu, sigma, lowerBound, upperBound)
if sigma <= 0
    if mu < lowerBound || mu > upperBound
        error('Degenerate normal prior mean lies outside its bounds.');
    end
    value = repmat(mu, n, 1);
    return
end
value = nan(n, 1);
remaining = true(n, 1);
while any(remaining)
    candidate = mu + sigma .* randn(sum(remaining), 1);
    accept = candidate >= lowerBound & candidate <= upperBound;
    idx = find(remaining);
    value(idx(accept)) = candidate(accept);
    remaining(idx(accept)) = false;
end
end

function state = readCaseState(resultFile, proxyLon, proxyLat)
S = load(resultFile, 'sampleLon', 'sampleLat', 'elevationPred', ...
    'T_1', 'T_2', 'gammaSat_1', 'gammaSat_2');
[~, k] = min(hypot(S.sampleLon(:) - proxyLon, S.sampleLat(:) - proxyLat));
zM = S.elevationPred(k);
state = struct;
state.sampleIndex = k;
state.sampleLongitude = S.sampleLon(k);
state.sampleLatitude = S.sampleLat(k);
state.elevationM = zM;
state.opiState1SurfaceC = S.T_1(1) - S.gammaSat_1(1).*zM - 273.15;
state.opiState2SurfaceC = S.T_2(1) - S.gammaSat_2(1).*zM - 273.15;
end

function row = evaluateCase(caseId, geometry, state, d18Base, ...
    obs, draws, config, resultFile)
n = numel(draws.T0WarmestC);
airWarmestC = draws.T0WarmestC - ...
    draws.lapseRateCPerKm .* (state.elevationM ./ 1e3) + ...
    draws.temperatureTimeOffsetC;
d18Pred = d18Base + config.source_d18O_response .* ...
    draws.sourceOffsetPermil + draws.waterTimeOffsetPermil;
[logLikelihoodJoint, likelihood] = opiAssimilationLikelihood( ...
    d18Pred, airWarmestC, obs.latitude, obs, ...
    'LakeAreaKm2', draws.lakeAreaKm2, ...
    'LakeDepthM', draws.lakeDepthM, ...
    'D18OModelDiscrepancyPermil', config.d18O_model_discrepancy_permil, ...
    'TemperatureModelDiscrepancyC', ...
    config.temperature_model_discrepancy_C);
logLikelihoodTemperature = likelihood.logLikelihoodTemperature;
logLikelihoodD18 = likelihood.logLikelihoodD18O;
lakePredC = likelihood.lakePredictedC;
temperatureResidualC = likelihood.temperatureResidualC;
d18Residual = likelihood.d18OResidualPermil;
transferInfo = likelihood.transferInfo;

conditionalWeight = normalizeLogWeights(logLikelihoodJoint);
priorWeight = double(geometry.qiangtangM >= config.qiangtang_min_m & ...
    geometry.qiangtangM <= config.qiangtang_max_m & ...
    geometry.gangdeseM >= config.gangdese_min_m);

row = emptyCaseRow();
row.case_id = string(caseId);
row.gangdese_target_m = geometry.gangdeseM;
row.qiangtang_target_m = geometry.qiangtangM;
row.valley_mode = string(geometry.valleyMode);
row.proxy_elevation_m = state.elevationM;
row.opi_state1_surface_C = state.opiState1SurfaceC;
row.opi_state2_surface_C = state.opiState2SurfaceC;
row.d18O_50km_base_permil = d18Base;
row.log_evidence_d18O = logMeanExp(logLikelihoodD18);
row.log_evidence_temperature = logMeanExp(logLikelihoodTemperature);
row.log_evidence_joint = logMeanExp(logLikelihoodJoint);
row.prior_weight = priorWeight;
row.posterior_mean_T0_warmest_C = sum(conditionalWeight .* draws.T0WarmestC);
row.posterior_mean_lapse_C_per_km = ...
    sum(conditionalWeight .* draws.lapseRateCPerKm);
row.posterior_mean_source_offset_permil = ...
    sum(conditionalWeight .* draws.sourceOffsetPermil);
row.posterior_mean_air_warmest_C = sum(conditionalWeight .* airWarmestC);
row.posterior_mean_lake_predicted_C = sum(conditionalWeight .* lakePredC);
row.posterior_mean_temperature_residual_C = ...
    sum(conditionalWeight .* temperatureResidualC);
row.posterior_mean_d18O_residual_permil = ...
    sum(conditionalWeight .* d18Residual);
row.fraction_outside_global_ML_range = mean( ...
    transferInfo.outsideGlobalTrainingRange);
row.fraction_outside_high_elevation_ML_range = mean( ...
    transferInfo.outsideHighElevationTrainingRange);
row.n_nuisance_samples = n;
row.result_file = string(resultFile);
end

function value = logMeanExp(logValues)
maximum = max(logValues);
value = maximum + log(mean(exp(logValues - maximum)));
end

function weight = normalizeLogWeights(logWeight)
weight = zeros(size(logWeight));
finite = isfinite(logWeight);
if ~any(finite)
    return
end
maximum = max(logWeight(finite));
weight(finite) = exp(logWeight(finite) - maximum);
weight = weight ./ sum(weight);
end

function priorWeight = assignCasePriors(casePosterior, config)
eligible = casePosterior.prior_weight > 0;
priorWeight = zeros(height(casePosterior), 1);
priorType = lower(string(config.case_prior_type));
switch priorType
    case {"uniform", "uniform_case"}
        priorWeight(eligible) = 1 ./ sum(eligible);
    case "uniform_qiangtang_height_then_case"
        heights = unique(casePosterior.qiangtang_target_m(eligible));
        for i = 1:numel(heights)
            keep = eligible & casePosterior.qiangtang_target_m == heights(i);
            priorWeight(keep) = 1 ./ numel(heights) ./ sum(keep);
        end
    otherwise
        error('Unsupported case_prior_type: %s', priorType);
end
end

function row = emptyCaseRow()
row = struct( ...
    'case_id', "", 'gangdese_target_m', nan, 'qiangtang_target_m', nan, ...
    'valley_mode', "", 'proxy_elevation_m', nan, ...
    'opi_state1_surface_C', nan, 'opi_state2_surface_C', nan, ...
    'd18O_50km_base_permil', nan, 'log_evidence_d18O', nan, ...
    'log_evidence_temperature', nan, 'log_evidence_joint', nan, ...
    'prior_weight', nan, 'posterior_mean_T0_warmest_C', nan, ...
    'posterior_mean_lapse_C_per_km', nan, ...
    'posterior_mean_source_offset_permil', nan, ...
    'posterior_mean_air_warmest_C', nan, ...
    'posterior_mean_lake_predicted_C', nan, ...
    'posterior_mean_temperature_residual_C', nan, ...
    'posterior_mean_d18O_residual_permil', nan, ...
    'fraction_outside_global_ML_range', nan, ...
    'fraction_outside_high_elevation_ML_range', nan, ...
    'n_nuisance_samples', nan, 'result_file', "", ...
    'posterior_probability', nan, 'posterior_rank', nan);
end

function heightPosterior = aggregateHeightPosterior(casePosterior)
[height, ~, group] = unique(casePosterior.qiangtang_target_m);
probability = accumarray(group, casePosterior.posterior_probability, ...
    [numel(height), 1], @sum, 0);
priorProbability = accumarray(group, casePosterior.prior_weight, ...
    [numel(height), 1], @sum, 0);
d18Only = normalizeLogWeights(groupLogEvidence( ...
    group, casePosterior.log_evidence_d18O, casePosterior.prior_weight));
temperatureOnly = normalizeLogWeights(groupLogEvidence( ...
    group, casePosterior.log_evidence_temperature, casePosterior.prior_weight));
heightPosterior = table(height, priorProbability, probability, d18Only, ...
    temperatureOnly, ...
    'VariableNames', {'qiangtang_height_m', 'prior_probability', 'joint_probability', ...
    'd18O_only_probability', 'temperature_only_probability'});
heightPosterior = sortrows(heightPosterior, 'qiangtang_height_m');
end

function grouped = groupLogEvidence(group, logEvidence, priorWeight)
nGroup = max(group);
grouped = -inf(nGroup, 1);
for g = 1:nGroup
    keep = group == g & priorWeight > 0 & isfinite(logEvidence);
    if any(keep)
        values = logEvidence(keep) + log(priorWeight(keep));
        maximum = max(values);
        grouped(g) = maximum + log(sum(exp(values - maximum)));
    end
end
end

function makePosteriorFigure(casePosterior, heightPosterior, outputDir)
fig = figure('Color', 'w', 'Name', 'Provisional topography assimilation');
tiledlayout(fig, 1, 2);
ax1 = nexttile;
if isprop(ax1, 'Toolbar') && ~isempty(ax1.Toolbar)
    ax1.Toolbar.Visible = 'off';
end
bar(ax1, categorical(casePosterior.case_id), ...
    casePosterior.posterior_probability, 'FaceColor', [0.2, 0.45, 0.7]);
grid(ax1, 'on');
ylabel(ax1, 'Posterior probability');
title(ax1, 'Discrete topography cases');
ax1.XTickLabelRotation = 60;

ax2 = nexttile;
if isprop(ax2, 'Toolbar') && ~isempty(ax2.Toolbar)
    ax2.Toolbar.Visible = 'off';
end
plot(ax2, heightPosterior.qiangtang_height_m, ...
    heightPosterior.joint_probability, '-o', 'LineWidth', 2, ...
    'DisplayName', 'Joint');
hold(ax2, 'on');
plot(ax2, heightPosterior.qiangtang_height_m, ...
    heightPosterior.d18O_only_probability, '--s', 'LineWidth', 1.5, ...
    'DisplayName', 'd18O only');
plot(ax2, heightPosterior.qiangtang_height_m, ...
    heightPosterior.temperature_only_probability, ':^', 'LineWidth', 1.5, ...
    'DisplayName', 'Temperature only');
grid(ax2, 'on');
xlabel(ax2, 'Qiangtang target height (m)');
ylabel(ax2, 'Aggregated probability');
title(ax2, 'Height contribution');
legend(ax2, 'Location', 'best');
if isprop(ax1, 'Toolbar') && ~isempty(ax1.Toolbar)
    ax1.Toolbar.Visible = 'off';
end
if isprop(ax2, 'Toolbar') && ~isempty(ax2.Toolbar)
    ax2.Toolbar.Visible = 'off';
end
exportgraphics(fig, fullfile(outputDir, ...
    'Fig_Provisional_Topography_Assimilation.png'), 'Resolution', 200);
savefig(fig, fullfile(outputDir, ...
    'Fig_Provisional_Topography_Assimilation.fig'));
close(fig);
end

function writeAssimilationReport(reportFile, casePosterior, ...
    heightPosterior, obs, config, configFile)
fid = fopen(reportFile, 'w');
if fid == -1
    error('Could not create assimilation report: %s', reportFile);
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
[~, best] = max(casePosterior.posterior_probability);

fprintf(fid, '# Provisional Qiangtang Topography Assimilation\n\n');
fprintf(fid, '> Status: internal calibration framework, not external validation.\n\n');
fprintf(fid, 'The calculation uses a forward Terrazas warmest-lake model and marginalizes nuisance parameters sampled from the configuration snapshot.\n\n');
fprintf(fid, '## Observations\n\n');
fprintf(fid, ['- Measured sample age support: %.1f-%.1f Ma; all five ' ...
    'samples lie in this interval, with no sample-level age assignment.\n'], ...
    obs.age_min_Ma, obs.age_max_Ma);
fprintf(fid, '- Weighted warmest lake temperature: %.3f +/- %.3f C (n=%d).\n', ...
    obs.lake_temperature_warmest_C, obs.sigma_analytical_mean_C, ...
    obs.n_clumped);
fprintf(fid, '- Meteoric-water d18O: %.3f +/- %.3f per mil.\n', ...
    obs.target_d18O_permil, obs.sigma_target_d18O_permil);
fprintf(fid, '- Dolomite offset: %.3f +/- %.3f C.\n\n', ...
    obs.dolomite_offset_C, obs.sigma_dolomite_offset_C);
fprintf(fid, ['- Chronology resolution: `%s`; a latent group age is ' ...
    'marginalized across the full interval because sample-level ages are ' ...
    'unavailable.\n'], obs.chronology_resolution);
fprintf(fid, ['- Westerhold background cooling: %.2f +/- %.2f C from ' ...
    '%.1f to %.1f Ma; regional response %.2f +/- %.2f.\n\n'], ...
    config.westerhold_temperature_change_C, ...
    config.westerhold_temperature_change_1sigma_C, ...
    config.westerhold_old_age_Ma, config.westerhold_young_age_Ma, ...
    config.westerhold_regional_response_mean, ...
    config.westerhold_regional_response_1sigma);
fprintf(fid, '## Provisional result\n\n');
fprintf(fid, '- Highest-weight case: `%s` (probability %.4f).\n', ...
    casePosterior.case_id(best), casePosterior.posterior_probability(best));
fprintf(fid, '- Its nuisance-conditional mean T0 warmest: %.2f C.\n', ...
    casePosterior.posterior_mean_T0_warmest_C(best));
fprintf(fid, '- Its nuisance-conditional mean lapse rate: %.2f C/km.\n\n', ...
    casePosterior.posterior_mean_lapse_C_per_km(best));
fprintf(fid, 'These values are conditional on provisional priors and must not be interpreted as externally validated paleoclimate estimates.\n\n');
fprintf(fid, '## Identifiability diagnostics\n\n');
fprintf(fid, '- Joint information gain over the height prior: %.4f nats.\n', ...
    discreteKLDivergence(heightPosterior.joint_probability, ...
    heightPosterior.prior_probability));
fprintf(fid, '- d18O-only information gain: %.4f nats.\n', ...
    discreteKLDivergence(heightPosterior.d18O_only_probability, ...
    heightPosterior.prior_probability));
fprintf(fid, '- Temperature-only information gain: %.4f nats.\n', ...
    discreteKLDivergence(heightPosterior.temperature_only_probability, ...
    heightPosterior.prior_probability));
fprintf(fid, '- Best case nuisance draws outside the high-elevation ML range: %.1f%%.\n\n', ...
    100 .* casePosterior.fraction_outside_high_elevation_ML_range(best));
fprintf(fid, 'A near-zero temperature information gain means that broad T0/lapse priors allow the temperature observation to be matched across all tested heights.\n\n');
fprintf(fid, '## Qiangtang height weights\n\n');
fprintf(fid, '| Height m | Prior | Joint | d18O only | Temperature only |\n');
fprintf(fid, '|---:|---:|---:|---:|---:|\n');
for i = 1:height(heightPosterior)
    fprintf(fid, '| %.0f | %.4f | %.4f | %.4f | %.4f |\n', ...
        heightPosterior.qiangtang_height_m(i), ...
        heightPosterior.prior_probability(i), ...
        heightPosterior.joint_probability(i), ...
        heightPosterior.d18O_only_probability(i), ...
        heightPosterior.temperature_only_probability(i));
end
fprintf(fid, '\n## Required updates\n\n');
fprintf(fid, '- Freeze newly collected external lakes as a validation set before retraining.\n');
fprintf(fid, '- Replace broad T0-warmest and lapse-rate priors with independent constraints.\n');
fprintf(fid, '- Estimate modern-to-Eocene model discrepancy after external validation.\n');
fprintf(fid, '- Add paleolake area and depth constraints when available.\n\n');
fprintf(fid, 'Configuration source: `%s`. Nuisance samples: %d.\n', ...
    configFile, round(config.num_nuisance_samples));
end

function value = discreteKLDivergence(posterior, prior)
keep = posterior > 0 & prior > 0;
value = sum(posterior(keep) .* log(posterior(keep) ./ prior(keep)));
end
