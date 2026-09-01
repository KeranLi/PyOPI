function influence = analyzeExternalSiteInfluence( ...
    ownCaseFile, externalPredictionFile, weightFile, outputDir)
% Quantify each external site's influence using leave-one-site-out posteriors.


assimilationRoot = fileparts(fileparts(mfilename('fullpath')));
if nargin < 1 || strlength(string(ownCaseFile)) == 0
    ownCaseFile = fullfile(assimilationRoot, '..', 'OPI_matlab', ...
        'scenarios', ...
        'Qiangtang_30Ma_4000m_dH045_T290_fP030_Valid_platform_northsouth', ...
        'topography_qiangtang_height_3500_5500', 'assimilation', ...
        'assimilation_case_posterior.csv');
end
externalRoot = fullfile(assimilationRoot, 'results', ...
    'topography_qiangtang_height_3500_5500');
if nargin < 2 || strlength(string(externalPredictionFile)) == 0
    externalPredictionFile = fullfile(externalRoot, ...
        'collected_proxy_predictions.csv');
end
if nargin < 3 || strlength(string(weightFile)) == 0
    weightFile = fullfile(assimilationRoot, 'config', ...
        'assimilation_block_weights.csv');
end
if nargin < 4 || strlength(string(outputDir)) == 0
    outputDir = fullfile(externalRoot, 'combined_blocks');
end
if ~isfolder(outputDir)
    mkdir(outputDir);
end

own = readtable(ownCaseFile, 'TextType', 'string');
prediction = readtable(externalPredictionFile, 'TextType', 'string');
weights = readtable(weightFile, 'TextType', 'string');
requiredOwn = ["case_id", "qiangtang_target_m", ...
    "log_evidence_joint", "prior_weight"];
requiredPrediction = ["case_id", "site_id", "source", ...
    "temporal_status", "is_temporally_eligible", "is_used", ...
    "log_likelihood_joint"];
checkColumns(own, requiredOwn, 'local measured-data block');
checkColumns(prediction, requiredPrediction, 'external predictions');
checkColumns(weights, ["external_weight", "status"], 'block weights');
primary = find(lower(weights.status) == "primary");
if ~isscalar(primary)
    error('Exactly one external block weight must have primary status.');
end
externalWeight = weights.external_weight(primary);

caseIds = own.case_id;
nCase = numel(caseIds);
allSiteIds = unique(prediction.site_id, 'stable');
nAllSite = numel(allSiteIds);
allSource = strings(nAllSite, 1);
allTemporalStatus = strings(nAllSite, 1);
active = false(nAllSite, 1);
for j = 1:nAllSite
    rows = prediction.site_id == allSiteIds(j);
    siteRows = prediction(rows, :);
    [found, order] = ismember(caseIds, siteRows.case_id);
    if ~all(found) || height(siteRows) ~= nCase
        error('Site %s must have one row for every case.', allSiteIds(j));
    end
    sourceValues = unique(siteRows.source);
    temporalValues = unique(siteRows.temporal_status);
    if numel(sourceValues) ~= 1 || numel(temporalValues) ~= 1
        error('Site %s has inconsistent source or chronology labels.', ...
            allSiteIds(j));
    end
    allSource(j) = sourceValues;
    allTemporalStatus(j) = temporalValues;
    eligible = siteRows.is_temporally_eligible(order);
    used = siteRows.is_used(order);
    if all(eligible) && all(used)
        active(j) = true;
    elseif any(eligible) || any(used)
        error('Site %s is only partially active across cases.', allSiteIds(j));
    end
end
deferred = table(allSiteIds(~active), allSource(~active), ...
    allTemporalStatus(~active), ...
    'VariableNames', {'site_id', 'source', 'temporal_status'});
siteIds = allSiteIds(active);
nSite = numel(siteIds);
siteLogLikelihood = nan(nCase, nSite);
source = allSource(active);
for j = 1:nSite
    rows = prediction.site_id == siteIds(j);
    siteRows = prediction(rows, :);
    [found, order] = ismember(caseIds, siteRows.case_id);
    if ~all(found) || height(siteRows) ~= nCase || ...
            ~all(siteRows.is_used(order))
        error('Site %s must have one used row for every case.', siteIds(j));
    end
    siteLogLikelihood(:, j) = siteRows.log_likelihood_joint(order);
end
if any(~isfinite(siteLogLikelihood), 'all')
    error('External site likelihood matrix contains nonfinite values.');
end

externalLogLikelihood = sum(siteLogLikelihood, 2);
prior = own.prior_weight;
baseline = weightedBlockPosterior(prior, ...
    [own.log_evidence_joint, externalLogLikelihood], [1; externalWeight]);
[baselineHeight, baselineHeightProbability] = aggregateHeight( ...
    own.qiangtang_target_m, baseline);
baselineMeanHeightM = sum(baselineHeight .* baselineHeightProbability);

highestProbabilityCase = strings(nSite, 1);
highestProbabilityHeightM = nan(nSite, 1);
posteriorMeanHeightM = nan(nSite, 1);
deltaMeanHeightM = nan(nSite, 1);
totalVariationDistance = nan(nSite, 1);
siteOnlyHighestHeightM = nan(nSite, 1);
logLikelihoodRange = nan(nSite, 1);
for j = 1:nSite
    looExternal = externalLogLikelihood - siteLogLikelihood(:, j);
    loo = weightedBlockPosterior(prior, ...
        [own.log_evidence_joint, looExternal], [1; externalWeight]);
    [~, caseIndex] = max(loo);
    highestProbabilityCase(j) = caseIds(caseIndex);
    [heightM, probability] = aggregateHeight(own.qiangtang_target_m, loo);
    [~, heightIndex] = max(probability);
    highestProbabilityHeightM(j) = heightM(heightIndex);
    posteriorMeanHeightM(j) = sum(heightM .* probability);
    deltaMeanHeightM(j) = posteriorMeanHeightM(j) - baselineMeanHeightM;
    totalVariationDistance(j) = 0.5 .* sum(abs(loo - baseline));

    siteOnly = weightedBlockPosterior(prior, ...
        [own.log_evidence_joint, siteLogLikelihood(:, j)], [0; 1]);
    [siteHeight, siteProbability] = aggregateHeight( ...
        own.qiangtang_target_m, siteOnly);
    [~, siteHeightIndex] = max(siteProbability);
    siteOnlyHighestHeightM(j) = siteHeight(siteHeightIndex);
    logLikelihoodRange(j) = max(siteLogLikelihood(:, j)) - ...
        min(siteLogLikelihood(:, j));
end

influence = table(siteIds, source, repmat(externalWeight, nSite, 1), ...
    highestProbabilityCase, highestProbabilityHeightM, ...
    posteriorMeanHeightM, deltaMeanHeightM, totalVariationDistance, ...
    siteOnlyHighestHeightM, logLikelihoodRange, ...
    'VariableNames', {'site_id', 'source', 'external_weight', ...
    'loo_highest_probability_case', 'loo_highest_probability_height_m', ...
    'loo_posterior_mean_height_m', 'loo_minus_full_mean_height_m', ...
    'loo_total_variation_distance', 'site_only_highest_height_m', ...
    'site_log_likelihood_range'});
influence = sortrows(influence, 'loo_total_variation_distance', 'descend');
writetable(influence, fullfile(outputDir, ...
    'external_site_leave_one_out_influence.csv'));
writetable(deferred, fullfile(outputDir, ...
    'external_sites_deferred_by_chronology.csv'));
writeReport(fullfile(outputDir, 'external_site_influence_report.md'), ...
    influence, deferred, externalWeight, baselineMeanHeightM);
fprintf('Wrote external-site influence diagnostics to:\n%s\n', outputDir);
end

function [heightM, probability] = aggregateHeight(caseHeightM, caseProbability)
[heightM, ~, group] = unique(caseHeightM);
probability = accumarray(group, caseProbability, [], @sum);
end

function writeReport(reportFile, influence, deferred, ...
    externalWeight, baselineMeanHeightM)
fid = fopen(reportFile, 'w');
if fid == -1
    error('Could not create site-influence report: %s', reportFile);
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '# External Site Influence\n\n');
fprintf(fid, ['Each row removes one complete literature site from the ' ...
    'external block and recomputes the combined posterior. The local ' ...
    'measured-data block remains at weight 1.\n\n']);
fprintf(fid, '- External block weight: %.2f.\n', externalWeight);
fprintf(fid, '- Full-data posterior mean height: %.0f m.\n\n', ...
    baselineMeanHeightM);
fprintf(fid, ['| Removed source | LOO best height m | LOO mean height m | ' ...
    'Mean shift m | TV distance | Site-only best height m | ' ...
    'LogL range |\n']);
fprintf(fid, '|---|---:|---:|---:|---:|---:|---:|\n');
for i = 1:height(influence)
    fprintf(fid, '| %s | %.0f | %.0f | %+.0f | %.3f | %.0f | %.2f |\n', ...
        influence.source(i), ...
        influence.loo_highest_probability_height_m(i), ...
        influence.loo_posterior_mean_height_m(i), ...
        influence.loo_minus_full_mean_height_m(i), ...
        influence.loo_total_variation_distance(i), ...
        influence.site_only_highest_height_m(i), ...
        influence.site_log_likelihood_range(i));
end
fprintf(fid, '\n## Deferred chronology bins\n\n');
if isempty(deferred)
    fprintf(fid, 'None.\n');
else
    fprintf(fid, '| Source | Status |\n');
    fprintf(fid, '|---|---|\n');
    for i = 1:height(deferred)
        fprintf(fid, '| %s | %s |\n', deferred.source(i), ...
            deferred.temporal_status(i));
    end
end
end

function checkColumns(T, required, label)
missing = setdiff(required, string(T.Properties.VariableNames));
if ~isempty(missing)
    error('Missing %s column(s): %s', label, strjoin(missing, ', '));
end
end
