function audit = opiAudit_D18OAssimilation(scenarioRoot, configFile, outputRoot)
% opiAudit_D18OAssimilation compares legacy and probabilistic d18O constraints.
%
% Existing OPI forward results are preserved. The audit replaces binary
% target-interval acceptance with likelihood weights after marginalizing the
% state-1 moisture-source isotope uncertainty.

projectRoot = fileparts(fileparts(mfilename('fullpath')));
if nargin < 1 || isempty(scenarioRoot)
    scenarioRoot = fullfile(projectRoot, 'scenarios', ...
        'Qiangtang_30Ma_4000m_dH045_T290_fP030_Valid_platform_northsouth');
end
if nargin < 2 || isempty(configFile)
    configFile = fullfile(projectRoot, 'data', 'assimilation', ...
        'qiangtang_assimilation_config.csv');
end
if nargin < 3 || isempty(outputRoot)
    outputRoot = fullfile(scenarioRoot, 'assimilation_v1', 'd18O_audit');
end
if ~isfile(configFile)
    error('Assimilation config not found: %s', configFile);
end
if ~isfolder(outputRoot)
    mkdir(outputRoot);
end

config = readConfig(configFile);
sourceCalibration = loadSourceCalibration(scenarioRoot, config);
catalog = buildCatalog(scenarioRoot);
experiments = unique(catalog.experiment, 'stable');

allCases = table();
allGroups = table();
diagnostics = table();
for i = 1:numel(experiments)
    name = experiments(i);
    cases = catalog(catalog.experiment == name, :);
    [cases, groups, diagnostic] = auditExperiment( ...
        cases, name, sourceCalibration, config);
    experimentDir = fullfile(outputRoot, name);
    if ~isfolder(experimentDir)
        mkdir(experimentDir);
    end
    writetable(cases, fullfile(experimentDir, 'case_constraint_comparison.csv'));
    writetable(groups, fullfile(experimentDir, 'group_posterior.csv'));
    allCases = [allCases; cases]; %#ok<AGROW>
    allGroups = [allGroups; groups]; %#ok<AGROW>
    diagnostics = [diagnostics; diagnostic]; %#ok<AGROW>
end

writetable(allCases, fullfile(outputRoot, 'all_case_constraint_comparison.csv'));
writetable(allGroups, fullfile(outputRoot, 'all_group_posteriors.csv'));
writetable(diagnostics, fullfile(outputRoot, 'experiment_diagnostics.csv'));
writetable(sourceCalibration, fullfile(outputRoot, ...
    'source_response_calibration.csv'));
copyfile(configFile, fullfile(outputRoot, 'assimilation_config_snapshot.csv'));
makeAuditFigure(diagnostics, outputRoot);
writeAuditReport(fullfile(outputRoot, 'd18O_assimilation_audit_report.md'), ...
    diagnostics, allCases, config);

audit = struct;
audit.status = "provisional_internal_assimilation_audit";
audit.cases = allCases;
audit.groups = allGroups;
audit.diagnostics = diagnostics;
audit.sourceCalibration = sourceCalibration;
audit.outputRoot = string(outputRoot);
save(fullfile(outputRoot, 'd18O_assimilation_audit.mat'), 'audit', '-v7');

fprintf('Wrote d18O assimilation audit to:\n%s\n', outputRoot);
for i = 1:height(diagnostics)
    fprintf('%s: legacy best %s; assimilation best %s; KL %.4f nats\n', ...
        diagnostics.experiment(i), diagnostics.legacy_best_case(i), ...
        diagnostics.assimilation_best_case(i), ...
        diagnostics.information_gain_nats(i));
end
end

function catalog = buildCatalog(root)
catalog = emptyCatalog();
catalog = [catalog; standardCatalog(root, ...
    "height_refinement", 'topography_qiangtang_height_refinement', ...
    'analysis/height_refinement_target_comparison.csv')];
catalog = [catalog; standardCatalog(root, ...
    "valley_height", 'topography_valley_height_sensitivity', ...
    'analysis/valley_target_comparison.csv')];
catalog = [catalog; standardCatalog(root, ...
    "farnsworth", 'topography_farnsworth_band', ...
    'analysis/fixed_precipitation_d18O_summary.csv')];
catalog = [catalog; sourceCrossCatalog(root)];
catalog = [catalog; bandMorphologyCatalog(root)];
end

function T = standardCatalog(root, experimentName, folderName, summaryRelative)
experimentRoot = fullfile(root, folderName);
manifestFile = fullfile(experimentRoot, 'design', 'case_manifest.csv');
summaryFile = fullfile(experimentRoot, summaryRelative);
requireFiles([string(manifestFile), string(summaryFile)]);
M = readCommaCsv(manifestFile);
S = readCommaCsv(summaryFile);
n = height(M);
T = emptyCatalog(n);
T.experiment(:) = experimentName;
T.case_id = M.case_id;
T.topography_id = M.case_id;
T.gangdese_m = numericColumn(M, 'gangdese_target_m', nan(n, 1));
T.qiangtang_m = numericColumn(M, 'qiangtang_target_m', nan(n, 1));
T.primary_height_m = T.qiangtang_m;
T.valley_mode = stringColumn(M, 'valley_mode', repmat("", n, 1));
T.valley_m = numericColumn(M, 'valley_target_m', nan(n, 1));
T.case_dir = fullfile(string(experimentRoot), 'calc_only', T.case_id);
for i = 1:n
    j = find(S.case_id == T.case_id(i), 1);
    if isempty(j)
        error('Missing 50 km d18O summary for %s/%s.', experimentName, T.case_id(i));
    end
    T.d18O_50km_permil(i) = S.weighted_d18O_50km_permil(j);
end
end

function T = sourceCrossCatalog(root)
experimentName = "source_cross";
experimentRoot = fullfile(root, 'source_d18O_topography_cross');
manifestFile = fullfile(experimentRoot, 'design', 'case_manifest.csv');
summaryFile = fullfile(experimentRoot, 'analysis', ...
    'source_topography_target_comparison.csv');
requireFiles([string(manifestFile), string(summaryFile)]);
M = readCommaCsv(manifestFile);
S = readCommaCsv(summaryFile);
n = height(M);
T = emptyCatalog(n);
T.experiment(:) = experimentName;
T.case_id = M.case_id;
T.topography_id = M.topography_id;
T.source_offset_permil = M.d18O0_1_offset_permil;
T.case_dir = fullfile(string(experimentRoot), 'calc_only', T.case_id);
for i = 1:n
    [T.gangdese_m(i), T.qiangtang_m(i), T.valley_mode(i), T.valley_m(i)] = ...
        parseTopographyId(T.topography_id(i));
    T.primary_height_m(i) = T.qiangtang_m(i);
    j = find(S.case_id == T.case_id(i), 1);
    if isempty(j)
        error('Missing source-cross d18O summary for %s.', T.case_id(i));
    end
    T.d18O_50km_permil(i) = S.weighted_d18O_50km_permil(j);
end
end

function T = bandMorphologyCatalog(root)
experimentName = "band_morphology";
experimentRoot = fullfile(root, 'topography_sensitivity_clumped_band');
manifestFile = fullfile(experimentRoot, 'design', 'case_manifest.csv');
requireFiles(string(manifestFile));
M = readCommaCsv(manifestFile);
if ismember('stage', M.Properties.VariableNames)
    M = M(M.stage == "calc_only", :);
end
n = height(M);
T = emptyCatalog(n);
T.experiment(:) = experimentName;
T.case_id = M.case_id;
T.topography_id = M.case_id;
T.primary_height_m = M.height_target_m;
T.normalization_mode = M.normalization_mode;
T.pattern_id = M.pattern_id;
T.case_dir = fullfile(string(experimentRoot), 'calc_only', T.case_id);
for i = 1:n
    resultFile = fullfile(T.case_dir(i), ...
        'opiCalc_TwoWinds_OxygenOnly_Results.mat');
    T.d18O_50km_permil(i) = calculate50kmD18O(resultFile, 87.2, 32.9);
end
end

function [cases, groups, diagnostic] = auditExperiment( ...
    cases, experimentName, sourceCalibration, config)
n = height(cases);
target = config.target_d18O_permil;
sigma = sqrt(config.sigma_target_d18O_permil.^2 + ...
    config.d18O_model_discrepancy_permil.^2);
cases.d18O_residual_permil = cases.d18O_50km_permil - target;
cases.legacy_in_target_interval = abs(cases.d18O_residual_permil) <= ...
    config.sigma_target_d18O_permil;
[~, legacyOrder] = sort(abs(cases.d18O_residual_permil));
cases.legacy_absolute_residual_rank = nan(n, 1);
cases.legacy_absolute_residual_rank(legacyOrder) = (1:n)';

if experimentName == "source_cross"
    cases.source_response_used = nan(n, 1);
    cases.source_response_origin(:) = "explicit_source_case";
    cases.log_d18O_evidence = normalLogLikelihood( ...
        target - cases.d18O_50km_permil, sigma);
    cases.posterior_mean_source_offset_permil = cases.source_offset_permil;
    cases.prior_weight = sourceCrossPrior(cases, config);
else
    rng(config.random_seed, 'twister');
    sourceDraws = sampleSourcePrior(config);
    cases.prior_weight = designPrior(cases, experimentName, config);
    cases.log_d18O_evidence = nan(n, 1);
    cases.posterior_mean_source_offset_permil = nan(n, 1);
    for i = 1:n
        [response, origin] = lookupSourceResponse( ...
            cases.topography_id(i), sourceCalibration, config);
        cases.source_response_used(i) = response;
        cases.source_response_origin(i) = origin;
        predicted = cases.d18O_50km_permil(i) + response .* sourceDraws;
        logLikelihood = normalLogLikelihood(target - predicted, sigma);
        cases.log_d18O_evidence(i) = logMeanExp(logLikelihood);
        conditional = normalizeLogWeights(logLikelihood);
        cases.posterior_mean_source_offset_permil(i) = ...
            sum(conditional .* sourceDraws);
    end
end

logPosterior = cases.log_d18O_evidence + log(cases.prior_weight);
logPosterior(cases.prior_weight <= 0) = -inf;
cases.assimilation_probability = normalizeLogWeights(logPosterior);
[~, posteriorOrder] = sort(cases.assimilation_probability, 'descend');
cases.assimilation_rank = nan(n, 1);
cases.assimilation_rank(posteriorOrder) = (1:n)';
legacyCount = sum(cases.legacy_in_target_interval);
if legacyCount > 0
    cases.legacy_binary_probability = ...
        double(cases.legacy_in_target_interval) ./ legacyCount;
else
    cases.legacy_binary_probability = zeros(n, 1);
end
cases.rank_change_legacy_minus_assimilation = ...
    cases.legacy_absolute_residual_rank - cases.assimilation_rank;

caseOrder = ["experiment", "case_id", "topography_id", ...
    "gangdese_m", "qiangtang_m", "primary_height_m", "valley_mode", ...
    "valley_m", "normalization_mode", "pattern_id", ...
    "source_offset_permil", "d18O_50km_permil", ...
    "d18O_residual_permil", "legacy_in_target_interval", ...
    "legacy_absolute_residual_rank", "legacy_binary_probability", ...
    "source_response_used", "source_response_origin", ...
    "posterior_mean_source_offset_permil", "log_d18O_evidence", ...
    "prior_weight", "assimilation_probability", "assimilation_rank", ...
    "rank_change_legacy_minus_assimilation", "case_dir"];
cases = cases(:, caseOrder);

groups = aggregateExperimentGroups(cases, experimentName);
prior = cases.prior_weight ./ sum(cases.prior_weight);
[~, legacyBest] = min(abs(cases.d18O_residual_permil));
[~, assimilationBest] = max(cases.assimilation_probability);
diagnostic = table(experimentName, n, sum(cases.prior_weight > 0), ...
    legacyCount, cases.case_id(legacyBest), ...
    cases.d18O_residual_permil(legacyBest), ...
    cases.case_id(assimilationBest), ...
    cases.d18O_residual_permil(assimilationBest), ...
    sum(cases.assimilation_probability(cases.legacy_in_target_interval)), ...
    1 ./ sum(cases.assimilation_probability.^2), ...
    discreteKLDivergence(cases.assimilation_probability, prior), ...
    cases.case_id(legacyBest) ~= cases.case_id(assimilationBest), ...
    'VariableNames', {'experiment', 'n_cases', 'n_prior_eligible', ...
    'n_legacy_accepted', 'legacy_best_case', ...
    'legacy_best_residual_permil', 'assimilation_best_case', ...
    'assimilation_best_residual_permil', ...
    'posterior_mass_in_legacy_interval', 'effective_case_count', ...
    'information_gain_nats', 'best_case_changed'});
end

function prior = designPrior(cases, experimentName, config)
eligible = true(height(cases), 1);
if experimentName ~= "band_morphology"
    hasQ = isfinite(cases.qiangtang_m);
    eligible(hasQ) = cases.qiangtang_m(hasQ) >= config.qiangtang_min_m & ...
        cases.qiangtang_m(hasQ) <= config.qiangtang_max_m;
    hasG = isfinite(cases.gangdese_m);
    eligible(hasG) = eligible(hasG) & ...
        cases.gangdese_m(hasG) >= config.gangdese_min_m;
end
prior = zeros(height(cases), 1);
switch experimentName
    case "height_refinement"
        labels = "Q" + string(cases.qiangtang_m);
    case "valley_height"
        labels = cases.valley_mode;
    case "band_morphology"
        labels = string(cases.primary_height_m);
    otherwise
        labels = cases.case_id;
end
levels = unique(labels(eligible));
for i = 1:numel(levels)
    keep = eligible & labels == levels(i);
    prior(keep) = 1 ./ numel(levels) ./ sum(keep);
end
end

function prior = sourceCrossPrior(cases, config)
prior = zeros(height(cases), 1);
topographies = unique(cases.topography_id);
eligibleTopography = false(numel(topographies), 1);
for i = 1:numel(topographies)
    first = find(cases.topography_id == topographies(i), 1);
    eligibleTopography(i) = cases.qiangtang_m(first) >= config.qiangtang_min_m & ...
        cases.qiangtang_m(first) <= config.qiangtang_max_m & ...
        cases.gangdese_m(first) >= config.gangdese_min_m;
end
validTopographies = topographies(eligibleTopography);
for i = 1:numel(validTopographies)
    keep = cases.topography_id == validTopographies(i);
    offset = cases.source_offset_permil(keep);
    sourceMass = sourcePriorMass(offset, config);
    sourceMass = sourceMass ./ sum(sourceMass);
    prior(keep) = sourceMass ./ numel(validTopographies);
end
end

function mass = sourcePriorMass(offset, config)
priorType = lower(string(config.source_d18O_prior_type));
switch priorType
    case "normal"
        mass = exp(-0.5 .* ((offset - config.source_d18O_mean_permil) ./ ...
            config.source_d18O_sigma_permil).^2);
    case "uniform"
        mass = double(offset >= config.source_d18O_min_permil & ...
            offset <= config.source_d18O_max_permil);
    otherwise
        error('Unsupported source prior: %s', priorType);
end
end

function groups = aggregateExperimentGroups(cases, experimentName)
switch experimentName
    case "height_refinement"
        label = "Q" + string(cases.qiangtang_m);
    case "valley_height"
        label = cases.valley_mode;
    case "source_cross"
        label = cases.topography_id;
    case "band_morphology"
        label = string(cases.normalization_mode) + "|H" + ...
            string(cases.primary_height_m) + "|" + cases.pattern_id;
    otherwise
        label = cases.case_id;
end
[groupLabel, ~, group] = unique(label, 'stable');
prior = accumarray(group, cases.prior_weight, [numel(groupLabel), 1], @sum, 0);
posterior = accumarray(group, cases.assimilation_probability, ...
    [numel(groupLabel), 1], @sum, 0);
legacy = accumarray(group, cases.legacy_binary_probability, ...
    [numel(groupLabel), 1], @sum, 0);
groups = table(repmat(experimentName, numel(groupLabel), 1), groupLabel, ...
    prior, legacy, posterior, ...
    'VariableNames', {'experiment', 'group_label', 'prior_probability', ...
    'legacy_binary_probability', 'assimilation_probability'});
end

function calibration = loadSourceCalibration(root, config)
fileName = fullfile(root, 'source_d18O_topography_cross', 'analysis', ...
    'source_topography_target_comparison.csv');
requireFiles(string(fileName));
T = readCommaCsv(fileName);
topographies = unique(T.topography_id, 'stable');
calibration = table('Size', [numel(topographies), 5], ...
    'VariableTypes', {'string', 'double', 'double', 'double', 'double'}, ...
    'VariableNames', {'topography_id', 'response_slope', 'intercept', ...
    'n_source_cases', 'fallback_response'});
for i = 1:numel(topographies)
    keep = T.topography_id == topographies(i);
    x = T.d18O0_1_offset_permil(keep);
    y = T.weighted_d18O_50km_permil(keep);
    coefficients = polyfit(x, y, 1);
    calibration.topography_id(i) = topographies(i);
    calibration.response_slope(i) = coefficients(1);
    calibration.intercept(i) = coefficients(2);
    calibration.n_source_cases(i) = sum(keep);
    calibration.fallback_response(i) = config.source_d18O_response;
end
end

function [response, origin] = lookupSourceResponse(topographyId, calibration, config)
j = find(calibration.topography_id == topographyId, 1);
if isempty(j)
    response = config.source_d18O_response;
    origin = "configured_fallback";
else
    response = calibration.response_slope(j);
    origin = "source_cross_case_specific";
end
end

function sourceDraws = sampleSourcePrior(config)
n = round(config.num_nuisance_samples);
rng(config.random_seed, 'twister');
priorType = lower(string(config.source_d18O_prior_type));
switch priorType
    case "uniform"
        sourceDraws = config.source_d18O_min_permil + ...
            (config.source_d18O_max_permil - config.source_d18O_min_permil) .* ...
            rand(n, 1);
    case "normal"
        sourceDraws = truncatedNormal(n, config.source_d18O_mean_permil, ...
            config.source_d18O_sigma_permil, ...
            config.source_d18O_min_permil, config.source_d18O_max_permil);
    otherwise
        error('Unsupported source prior: %s', priorType);
end
end

function value = truncatedNormal(n, mu, sigma, lowerBound, upperBound)
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

function d18O = calculate50kmD18O(resultFile, centerLon, centerLat)
if ~isfile(resultFile)
    error('Missing band-morphology OPI result: %s', resultFile);
end
S = load(resultFile, 'lon', 'lat', 'd18OGrid', 'pGrid');
d = S.d18OGrid .* 1e3;
p = S.pGrid;
[xx, yy] = meshgrid(S.lon, S.lat);
distanceKm = hypot((xx - centerLon) .* 100.5, (yy - centerLat) .* 111.1);
keep = distanceKm <= 50 & isfinite(d) & isfinite(p) & p > 0;
if ~any(keep, 'all')
    d18O = nan;
else
    d18O = sum(d(keep) .* p(keep), 'omitnan') ./ sum(p(keep), 'omitnan');
end
end

function [gangdese, qiangtang, valleyMode, valleyM] = parseTopographyId(id)
tokens = regexp(char(id), '^G(\d+)_Q(\d+)_V(.+)$', 'tokens', 'once');
if isempty(tokens)
    error('Could not parse topography id: %s', id);
end
gangdese = str2double(tokens{1});
qiangtang = str2double(tokens{2});
valleyMode = "V" + string(tokens{3});
if valleyMode == "Vnone"
    valleyM = nan;
else
    valleyM = str2double(extractAfter(valleyMode, 1));
end
end

function config = readConfig(configFile)
options = detectImportOptions(configFile, 'TextType', 'string');
options = setvartype(options, {'key', 'value'}, 'string');
T = readtable(configFile, options);
config = struct;
for i = 1:height(T)
    numericValue = str2double(T.value(i));
    key = char(T.key(i));
    if isfinite(numericValue)
        config.(key) = numericValue;
    else
        config.(key) = T.value(i);
    end
end
end

function T = readCommaCsv(fileName)
% Explicit delimiter avoids misdetecting underscore-rich case IDs as headers.
T = readtable(fileName, 'Delimiter', ',', 'TextType', 'string', ...
    'VariableNamingRule', 'preserve');
end

function T = emptyCatalog(n)
if nargin < 1, n = 0; end
T = table('Size', [n, 14], ...
    'VariableTypes', {'string', 'string', 'string', 'double', 'double', ...
    'double', 'string', 'double', 'string', 'string', 'double', 'double', ...
    'string', 'string'}, ...
    'VariableNames', {'experiment', 'case_id', 'topography_id', ...
    'gangdese_m', 'qiangtang_m', 'primary_height_m', 'valley_mode', ...
    'valley_m', 'normalization_mode', 'pattern_id', ...
    'source_offset_permil', 'd18O_50km_permil', 'case_dir', ...
    'source_response_origin'});
end

function values = numericColumn(T, name, defaultValues)
if ismember(name, T.Properties.VariableNames)
    values = T.(name);
else
    values = defaultValues;
end
end

function values = stringColumn(T, name, defaultValues)
if ismember(name, T.Properties.VariableNames)
    values = string(T.(name));
else
    values = defaultValues;
end
end

function requireFiles(files)
missing = files(~isfile(files));
if ~isempty(missing)
    error('Missing d18O audit input(s):\n%s', strjoin(missing, newline));
end
end

function logLikelihood = normalLogLikelihood(residual, sigma)
logLikelihood = -0.5 .* (residual ./ sigma).^2 - log(sigma) ...
    - 0.5 .* log(2*pi);
end

function value = logMeanExp(logValues)
maximum = max(logValues);
value = maximum + log(mean(exp(logValues - maximum)));
end

function weight = normalizeLogWeights(logWeight)
weight = zeros(size(logWeight));
finite = isfinite(logWeight);
if ~any(finite), return; end
maximum = max(logWeight(finite));
weight(finite) = exp(logWeight(finite) - maximum);
weight = weight ./ sum(weight);
end

function value = discreteKLDivergence(posterior, prior)
keep = posterior > 0 & prior > 0;
value = sum(posterior(keep) .* log(posterior(keep) ./ prior(keep)));
end

function makeAuditFigure(diagnostics, outputRoot)
fig = figure('Color', 'w', 'Name', 'd18O assimilation audit');
tiledlayout(fig, 1, 2);
ax1 = nexttile;
bar(ax1, categorical(diagnostics.experiment), ...
    diagnostics.information_gain_nats, 'FaceColor', [0.15, 0.45, 0.7]);
grid(ax1, 'on');
ylabel(ax1, 'Information gain (nats)');
title(ax1, 'Probabilistic d18O constraint');
ax1.XTickLabelRotation = 35;
ax2 = nexttile;
bar(ax2, categorical(diagnostics.experiment), ...
    [diagnostics.n_legacy_accepted, diagnostics.effective_case_count]);
grid(ax2, 'on');
ylabel(ax2, 'Number of cases');
title(ax2, 'Hard acceptance vs effective posterior cases');
legend(ax2, {'Legacy accepted', 'Assimilation effective'}, 'Location', 'best');
ax2.XTickLabelRotation = 35;
if isprop(ax1, 'Toolbar') && ~isempty(ax1.Toolbar), ax1.Toolbar.Visible = 'off'; end
if isprop(ax2, 'Toolbar') && ~isempty(ax2.Toolbar), ax2.Toolbar.Visible = 'off'; end
exportgraphics(fig, fullfile(outputRoot, ...
    'Fig_D18O_Assimilation_Audit.png'), 'Resolution', 220);
savefig(fig, fullfile(outputRoot, 'Fig_D18O_Assimilation_Audit.fig'));
close(fig);
end

function writeAuditReport(reportFile, diagnostics, allCases, config)
fid = fopen(reportFile, 'w');
if fid == -1
    error('Could not create d18O assimilation audit report: %s', reportFile);
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '# Precipitation d18O Assimilation Audit\n\n');
fprintf(fid, '> Status: provisional internal comparison, not external validation.\n\n');
fprintf(fid, 'Legacy analysis accepts cases when the absolute 50 km precipitation d18O residual is no more than %.2f per mil. The assimilation assigns continuous likelihood weights and marginalizes source-water isotope uncertainty.\n\n', ...
    config.sigma_target_d18O_permil);
effectiveSigma = sqrt(config.sigma_target_d18O_permil.^2 + ...
    config.d18O_model_discrepancy_permil.^2 + ...
    (config.source_d18O_response .* config.source_d18O_sigma_permil).^2);
fprintf(fid, ['With the configured source response (%.3f) and source-isotope ' ...
    'prior sigma (%.2f per mil), the approximate marginalized d18O ' ...
    'constraint width is %.2f per mil before case-specific response ' ...
    'adjustments.\n\n'], config.source_d18O_response, ...
    config.source_d18O_sigma_permil, effectiveSigma);
fprintf(fid, '## Cross-experiment diagnostics\n\n');
fprintf(fid, ['| Experiment | Cases | Legacy accepted | Legacy best | ' ...
    'Assimilation best | Posterior mass in legacy interval | Effective ' ...
    'cases | KL nats | Best changed |\n']);
fprintf(fid, '|---|---:|---:|---|---|---:|---:|---:|---:|\n');
for i = 1:height(diagnostics)
    fprintf(fid, '| %s | %d | %d | %s | %s | %.3f | %.2f | %.4f | %d |\n', ...
        diagnostics.experiment(i), diagnostics.n_cases(i), ...
        diagnostics.n_legacy_accepted(i), diagnostics.legacy_best_case(i), ...
        diagnostics.assimilation_best_case(i), ...
        diagnostics.posterior_mass_in_legacy_interval(i), ...
        diagnostics.effective_case_count(i), ...
        diagnostics.information_gain_nats(i), diagnostics.best_case_changed(i));
end
fprintf(fid, '\n## Interpretation rules\n\n');
fprintf(fid, '- `Legacy accepted` is a hard interval count and gives all accepted cases equal status.\n');
fprintf(fid, '- `Assimilation probability` preserves relative support inside and outside that interval.\n');
fprintf(fid, '- `Posterior mass in legacy interval` shows how much probabilistic support the old pass/fail rule retained; one minus this value is supported outside the old interval.\n');
fprintf(fid, '- `Effective cases` is 1/sum(p^2); a smaller value means stronger discrimination.\n');
fprintf(fid, '- `KL nats` measures information gained relative to the registered design prior.\n\n');

experiments = unique(allCases.experiment, 'stable');
for i = 1:numel(experiments)
    E = allCases(allCases.experiment == experiments(i), :);
    [~, order] = sort(E.assimilation_probability, 'descend');
    order = order(1:min(3, numel(order)));
    fprintf(fid, '## %s top cases\n\n', experiments(i));
    fprintf(fid, '| Case | d18O 50 km | Residual | Legacy accepted | Probability | Source response |\n');
    fprintf(fid, '|---|---:|---:|---:|---:|---:|\n');
    for j = order'
        fprintf(fid, '| %s | %.3f | %.3f | %d | %.4f | %.3f |\n', ...
            E.case_id(j), E.d18O_50km_permil(j), ...
            E.d18O_residual_permil(j), E.legacy_in_target_interval(j), ...
            E.assimilation_probability(j), E.source_response_used(j));
    end
    fprintf(fid, '\n');
end
fprintf(fid, 'The source-cross experiment uses its explicit source-isotope cases. Other experiments use case-specific response slopes where available and the configured %.3f fallback elsewhere.\n', ...
    config.source_d18O_response);
end
