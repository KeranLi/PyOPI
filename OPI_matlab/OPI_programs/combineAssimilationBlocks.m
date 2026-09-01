function result = combineAssimilationBlocks( ...
    ownCaseFile, externalCaseFile, weightFile, outputDir)
% Fuse local measured-data and external-literature likelihood blocks.


assimilationRoot = fileparts(fileparts(mfilename('fullpath')));
if nargin < 1 || strlength(string(ownCaseFile)) == 0
    ownCaseFile = fullfile(assimilationRoot, '..', 'OPI_matlab', ...
        'scenarios', ...
        'Qiangtang_30Ma_4000m_dH045_T290_fP030_Valid_platform_northsouth', ...
        'topography_qiangtang_height_3500_5500', 'assimilation', ...
        'assimilation_case_posterior.csv');
end
if nargin < 2 || strlength(string(externalCaseFile)) == 0
    externalCaseFile = fullfile(assimilationRoot, 'results', ...
        'topography_qiangtang_height_3500_5500', ...
        'collected_proxy_case_posterior.csv');
end
if nargin < 3 || strlength(string(weightFile)) == 0
    weightFile = fullfile(assimilationRoot, 'config', ...
        'assimilation_block_weights.csv');
end
if nargin < 4 || strlength(string(outputDir)) == 0
    outputDir = fullfile(assimilationRoot, 'results', ...
        'topography_qiangtang_height_3500_5500', 'combined_blocks');
end
requiredFiles = string({ownCaseFile, externalCaseFile, weightFile});
missing = requiredFiles(~isfile(requiredFiles));
if ~isempty(missing)
    error('Missing assimilation block input(s):\n%s', strjoin(missing, newline));
end
if ~isfolder(outputDir)
    mkdir(outputDir);
end

own = readtable(ownCaseFile, 'TextType', 'string');
external = readtable(externalCaseFile, 'TextType', 'string');
weights = readtable(weightFile, 'TextType', 'string');
checkColumns(own, ["case_id", "gangdese_target_m", ...
    "qiangtang_target_m", "valley_mode", "log_evidence_joint", ...
    "prior_weight"], 'local measured-data block');
checkColumns(external, ["case_id", "gangdese_target_m", ...
    "qiangtang_target_m", "valley_mode", "log_likelihood_joint", ...
    "prior_probability"], 'external literature block');
checkColumns(weights, ["external_weight", "status"], 'block weights');

[found, externalIndex] = ismember(own.case_id, external.case_id);
if ~all(found) || height(own) ~= height(external) || ...
        numel(unique(own.case_id)) ~= height(own)
    error('Local and external blocks must contain the same unique cases.');
end
external = external(externalIndex, :);
sameGeometry = own.gangdese_target_m == external.gangdese_target_m & ...
    own.qiangtang_target_m == external.qiangtang_target_m & ...
    own.valley_mode == external.valley_mode;
if ~all(sameGeometry)
    error('Assimilation blocks disagree on case geometry.');
end
prior = own.prior_weight;
if max(abs(prior - external.prior_probability)) > 1e-12
    error('Assimilation blocks use different case priors.');
end

externalWeights = weights.external_weight;
if any(~isfinite(externalWeights)) || any(externalWeights < 0) || ...
        numel(unique(externalWeights)) ~= numel(externalWeights)
    error('External block weights must be unique finite nonnegative values.');
end
primary = find(lower(weights.status) == "primary");
if ~isscalar(primary)
    error('Exactly one external block weight must have primary status.');
end

logBlocks = [own.log_evidence_joint, external.log_likelihood_joint];
ownProbability = weightedBlockPosterior(prior, logBlocks, [1; 0]);
externalProbability = weightedBlockPosterior(prior, logBlocks, [0; 1]);
casePosterior = table(own.case_id, own.gangdese_target_m, ...
    own.qiangtang_target_m, own.valley_mode, prior, ...
    own.log_evidence_joint, external.log_likelihood_joint, ...
    ownProbability, externalProbability, ...
    'VariableNames', {'case_id', 'gangdese_target_m', ...
    'qiangtang_target_m', 'valley_mode', 'prior_probability', ...
    'log_likelihood_own', 'log_likelihood_external', ...
    'own_only_probability', 'external_only_probability'});

probabilityNames = strings(numel(externalWeights), 1);
for i = 1:numel(externalWeights)
    probabilityNames(i) = "combined_external_w" + ...
        replace(compose('%.2f', externalWeights(i)), '.', 'p');
    casePosterior.(probabilityNames(i)) = weightedBlockPosterior( ...
        prior, logBlocks, [1; externalWeights(i)]);
end
casePosterior.combined_primary_probability = ...
    casePosterior.(probabilityNames(primary));
[~, order] = sort(casePosterior.combined_primary_probability, 'descend');
casePosterior.combined_primary_rank = nan(height(casePosterior), 1);
casePosterior.combined_primary_rank(order) = (1:height(casePosterior))';

heightPosterior = aggregateByHeight(casePosterior, probabilityNames);
sensitivity = makeSensitivitySummary( ...
    casePosterior, heightPosterior, externalWeights, probabilityNames);
writetable(casePosterior, fullfile(outputDir, ...
    'combined_block_case_posterior.csv'));
writetable(heightPosterior, fullfile(outputDir, ...
    'combined_block_height_posterior.csv'));
writetable(sensitivity, fullfile(outputDir, ...
    'combined_block_weight_sensitivity.csv'));
copyfile(weightFile, fullfile(outputDir, ...
    'assimilation_block_weights_snapshot.csv'));
makeFigure(heightPosterior, externalWeights, probabilityNames, outputDir);
writeReport(fullfile(outputDir, 'combined_block_report.md'), ...
    casePosterior, heightPosterior, sensitivity, ...
    externalWeights(primary), ownCaseFile, externalCaseFile);

result = struct;
result.status = "provisional_weighted_likelihood_block_fusion";
result.casePosterior = casePosterior;
result.heightPosterior = heightPosterior;
result.sensitivity = sensitivity;
result.externalWeights = externalWeights;
result.primaryExternalWeight = externalWeights(primary);
result.outputDir = string(outputDir);
save(fullfile(outputDir, 'combined_block_results.mat'), 'result', '-v7');
fprintf('Wrote weighted assimilation-block fusion to:\n%s\n', outputDir);
fprintf('Primary external weight: %.2f; best case: %s\n', ...
    externalWeights(primary), casePosterior.case_id(order(1)));
end

function heightPosterior = aggregateByHeight(casePosterior, probabilityNames)
[heightM, ~, group] = unique(casePosterior.qiangtang_target_m);
heightPosterior = table(heightM, accumarray(group, ...
    casePosterior.prior_probability, [], @sum), ...
    accumarray(group, casePosterior.own_only_probability, [], @sum), ...
    accumarray(group, casePosterior.external_only_probability, [], @sum), ...
    'VariableNames', {'qiangtang_height_m', 'prior_probability', ...
    'own_only_probability', 'external_only_probability'});
for i = 1:numel(probabilityNames)
    heightPosterior.(probabilityNames(i)) = accumarray(group, ...
        casePosterior.(probabilityNames(i)), [], @sum);
end
heightPosterior = sortrows(heightPosterior, 'qiangtang_height_m');
end

function sensitivity = makeSensitivitySummary( ...
    casePosterior, heightPosterior, externalWeights, probabilityNames)
n = numel(externalWeights);
bestCase = strings(n, 1);
bestHeightM = nan(n, 1);
meanHeightM = nan(n, 1);
probabilityAtUpperEdge = nan(n, 1);
for i = 1:n
    [~, caseIndex] = max(casePosterior.(probabilityNames(i)));
    [~, heightIndex] = max(heightPosterior.(probabilityNames(i)));
    bestCase(i) = casePosterior.case_id(caseIndex);
    bestHeightM(i) = heightPosterior.qiangtang_height_m(heightIndex);
    meanHeightM(i) = sum(heightPosterior.qiangtang_height_m .* ...
        heightPosterior.(probabilityNames(i)));
    probabilityAtUpperEdge(i) = heightPosterior.(probabilityNames(i))(end);
end
sensitivity = table(externalWeights, bestCase, bestHeightM, meanHeightM, ...
    probabilityAtUpperEdge, 'VariableNames', {'external_weight', ...
    'highest_probability_case', 'highest_probability_height_m', ...
    'posterior_mean_height_m', 'probability_at_5500m'});
end

function makeFigure(heightPosterior, externalWeights, probabilityNames, outputDir)
fig = figure('Color', 'w', 'Name', 'Weighted assimilation blocks', ...
    'Position', [100, 100, 1000, 650]);
ax = axes(fig);
plot(ax, heightPosterior.qiangtang_height_m, ...
    heightPosterior.prior_probability, '--o', 'LineWidth', 1.2, ...
    'DisplayName', 'Prior');
hold(ax, 'on');
plot(ax, heightPosterior.qiangtang_height_m, ...
    heightPosterior.own_only_probability, '-s', 'LineWidth', 2, ...
    'DisplayName', 'Local measured data');
plot(ax, heightPosterior.qiangtang_height_m, ...
    heightPosterior.external_only_probability, ':^', 'LineWidth', 1.5, ...
    'DisplayName', 'External only');
for i = 1:numel(externalWeights)
    plot(ax, heightPosterior.qiangtang_height_m, ...
        heightPosterior.(probabilityNames(i)), 'LineWidth', 1.5, ...
        'DisplayName', sprintf('Combined, external w=%.2f', ...
        externalWeights(i)));
end
grid(ax, 'on');
xlabel(ax, 'Qiangtang target height (m)');
ylabel(ax, 'Posterior probability');
title(ax, 'Local-data anchor and external-weight sensitivity');
legend(ax, 'Location', 'best');
if isprop(ax, 'Toolbar') && ~isempty(ax.Toolbar)
    ax.Toolbar.Visible = 'off';
end
exportgraphics(fig, fullfile(outputDir, ...
    'Fig_Weighted_Assimilation_Blocks.png'), 'Resolution', 220);
savefig(fig, fullfile(outputDir, 'Fig_Weighted_Assimilation_Blocks.fig'));
close(fig);
end

function writeReport(reportFile, casePosterior, heightPosterior, ...
    sensitivity, primaryExternalWeight, ownCaseFile, externalCaseFile)
fid = fopen(reportFile, 'w');
if fid == -1
    error('Could not create weighted-block report: %s', reportFile);
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
primaryName = "combined_external_w" + ...
    replace(compose('%.2f', primaryExternalWeight), '.', 'p');
[~, bestCase] = max(casePosterior.(primaryName));
[~, bestHeight] = max(heightPosterior.(primaryName));
fprintf(fid, '# Weighted Assimilation Blocks\n\n');
fprintf(fid, '> Status: provisional power-likelihood sensitivity analysis.\n\n');
fprintf(fid, ['The local measured-data block has fixed power 1.0. It ' ...
    'contains the five clumped-temperature measurements and independent ' ...
    'triple-oxygen meteoric-water reconstruction at 87.2 E, 32.9 N. ' ...
    'All five samples are constrained only to the interval 25.1-33.7 Ma; ' ...
    'no sample-level or mean age is inferred. The external likelihood ' ...
    'retains all supported literature sites and increases their proxy ' ...
    'uncertainty continuously with distance from the 30 Ma model age.\n\n']);
fprintf(fid, ['The 3.5-5.5 km interval defines the searched ensemble and ' ...
    'is not multiplied as a second likelihood from the same local data.\n\n']);
fprintf(fid, '## Primary result\n\n');
fprintf(fid, '- External block weight: %.2f.\n', primaryExternalWeight);
fprintf(fid, '- Highest-probability case: `%s`.\n', ...
    casePosterior.case_id(bestCase));
fprintf(fid, '- Highest-probability Qiangtang height: %.0f m.\n\n', ...
    heightPosterior.qiangtang_height_m(bestHeight));
fprintf(fid, '## Weight sensitivity\n\n');
fprintf(fid, '| External weight | Best case | Best height m | Mean height m | P(5500 m) |\n');
fprintf(fid, '|---:|---|---:|---:|---:|\n');
for i = 1:height(sensitivity)
    fprintf(fid, '| %.2f | %s | %.0f | %.0f | %.4g |\n', ...
        sensitivity.external_weight(i), ...
        sensitivity.highest_probability_case(i), ...
        sensitivity.highest_probability_height_m(i), ...
        sensitivity.posterior_mean_height_m(i), ...
        sensitivity.probability_at_5500m(i));
end
fprintf(fid, '\nLocal block source: `%s`.\n\n', ownCaseFile);
fprintf(fid, 'External block source: `%s`.\n', externalCaseFile);
end

function checkColumns(T, required, label)
missing = setdiff(required, string(T.Properties.VariableNames));
if ~isempty(missing)
    error('Missing %s column(s): %s', label, strjoin(missing, ', '));
end
end
