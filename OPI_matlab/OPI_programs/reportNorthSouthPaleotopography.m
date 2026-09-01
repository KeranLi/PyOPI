function result = reportNorthSouthPaleotopography( ...
    outputRoot, ownCaseFile, weightFile, manifestFile)
% Report the rate-marginalized north-to-south paleotopography posterior.


assimilationRoot = fileparts(fileparts(mfilename('fullpath')));
opiRoot = fullfile(assimilationRoot, '..', 'OPI_matlab');
experimentRoot = fullfile(opiRoot, 'scenarios', ...
    'Qiangtang_30Ma_4000m_dH045_T290_fP030_Valid_platform_northsouth', ...
    'topography_qiangtang_height_3500_5500');
if nargin < 1 || strlength(string(outputRoot)) == 0
    outputRoot = fullfile(assimilationRoot, 'results', ...
        'topography_qiangtang_height_3500_5500', ...
        'temporal_uncertainty_sensitivity');
end
if nargin < 2 || strlength(string(ownCaseFile)) == 0
    ownCaseFile = fullfile(experimentRoot, 'assimilation', ...
        'assimilation_case_posterior.csv');
end
if nargin < 3 || strlength(string(weightFile)) == 0
    weightFile = fullfile(assimilationRoot, 'config', ...
        'assimilation_block_weights.csv');
end
if nargin < 4 || strlength(string(manifestFile)) == 0
    manifestFile = fullfile(experimentRoot, 'design', 'case_manifest.csv');
end

sensitivityFile = fullfile(outputRoot, ...
    'temporal_uncertainty_sensitivity_summary.csv');
requiredFiles = string({ownCaseFile, weightFile, manifestFile, sensitivityFile});
missing = requiredFiles(~isfile(requiredFiles));
if ~isempty(missing)
    error('Missing north-south posterior input(s):\n%s', ...
        strjoin(missing, newline));
end
own = readtable(ownCaseFile, 'TextType', 'string');
weights = readtable(weightFile, 'TextType', 'string');
manifest = readtable(manifestFile, 'TextType', 'string');
sensitivity = readtable(sensitivityFile, 'TextType', 'string');
primaryWeight = weights.external_weight(lower(weights.status) == "primary");
if ~isscalar(primaryWeight)
    error('Exactly one primary external weight is required.');
end

scale = sensitivity.temporal_rate_scale;
nScale = numel(scale);
jointLogWeight = nan(height(own), nScale);
for i = 1:nScale
    label = "scale_" + replace(compose('%.1f', scale(i)), '.', 'p');
    externalFile = fullfile(outputRoot, label, ...
        'collected_proxy_case_posterior.csv');
    external = readtable(externalFile, 'TextType', 'string');
    [found, order] = ismember(own.case_id, external.case_id);
    if ~all(found)
        error('Case IDs differ for temporal rate scale %.1f.', scale(i));
    end
    jointLogWeight(:, i) = log(own.prior_weight) + ...
        own.log_evidence_joint + primaryWeight .* ...
        external.log_likelihood_joint(order) - log(nScale);
end
maximum = max(jointLogWeight, [], 'all');
jointProbability = exp(jointLogWeight - maximum);
jointProbability = jointProbability ./ sum(jointProbability, 'all');
caseProbability = sum(jointProbability, 2);

[found, order] = ismember(own.case_id, manifest.case_id);
if ~all(found)
    error('Case manifest does not contain every posterior case.');
end
manifest = manifest(order, :);
centralTargetM = manifest.valley_target_m;
ramp = ~isfinite(centralTargetM);
centralTargetM(ramp) = 0.5 .* (manifest.qiangtang_target_m(ramp) + ...
    manifest.gangdese_target_m(ramp));
casePosterior = table(own.case_id, manifest.qiangtang_target_m, ...
    centralTargetM, manifest.gangdese_target_m, manifest.valley_mode, ...
    caseProbability, 'VariableNames', {'case_id', ...
    'qiangtang_elevation_m', 'central_valley_zone_elevation_m', ...
    'gangdese_elevation_m', 'central_morphology', ...
    'posterior_probability'});
writetable(casePosterior, fullfile(outputRoot, ...
    'north_south_case_posterior.csv'));

unit = [repmat("Qiangtang", height(own), 1); ...
    repmat("Central_valley_zone", height(own), 1); ...
    repmat("Gangdese", height(own), 1)];
elevationM = [manifest.qiangtang_target_m; centralTargetM; ...
    manifest.gangdese_target_m];
probability = repmat(caseProbability, 3, 1);
marginal = table(unit, elevationM, probability);
marginal = groupsummary(marginal, {'unit', 'elevationM'}, 'sum', 'probability');
marginal.GroupCount = [];
marginal.Properties.VariableNames{'sum_probability'} = 'posterior_probability';
marginal = sortrows(marginal, {'unit', 'elevationM'});
writetable(marginal, fullfile(outputRoot, ...
    'north_south_unit_elevation_posterior.csv'));

unitOrder = ["Qiangtang"; "Central_valley_zone"; "Gangdese"];
modeM = nan(3, 1);
meanM = nan(3, 1);
p16M = nan(3, 1);
p84M = nan(3, 1);
for i = 1:3
    rows = marginal.unit == unitOrder(i);
    z = marginal.elevationM(rows);
    p = marginal.posterior_probability(rows);
    [z, order] = sort(z);
    p = p(order) ./ sum(p);
    [~, modeIndex] = max(p);
    modeM(i) = z(modeIndex);
    meanM(i) = sum(z .* p);
    cdf = cumsum(p);
    p16M(i) = z(find(cdf >= 0.16, 1));
    p84M(i) = z(find(cdf >= 0.84, 1));
end
summary = table((1:3)', unitOrder, modeM, meanM, p16M, p84M, ...
    'VariableNames', {'north_to_south_order', 'unit', ...
    'highest_probability_elevation_m', 'posterior_mean_elevation_m', ...
    'p16_elevation_m', 'p84_elevation_m'});
writetable(summary, fullfile(outputRoot, ...
    'north_south_paleotopography_summary.csv'));

isLowValley = manifest.valley_mode == "V1500";
lowValleyPrior = sum(own.prior_weight(isLowValley));
lowValleyPosterior = sum(caseProbability(isLowValley));
reportFile = fullfile(outputRoot, 'north_south_paleotopography_report.md');
fid = fopen(reportFile, 'w');
if fid == -1
    error('Could not create north-south report: %s', reportFile);
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '# North-to-South Paleotopography Posterior\n\n');
fprintf(fid, ['This posterior marginalizes %d tested residual temporal-error ' ...
    'rate scales after applying the directional Westerhold temperature ' ...
    'trend and age-interval quadrature. It retains the local-data power ' ...
    '1.0 and literature-data power %.2f.\n\n'], nScale, primaryWeight);
fprintf(fid, '| North-south order | Unit | Best elevation m | Mean m | 16th-84th m |\n');
fprintf(fid, '|---:|---|---:|---:|---:|\n');
for i = 1:height(summary)
    fprintf(fid, '| %d | %s | %.0f | %.0f | %.0f-%.0f |\n', ...
        summary.north_to_south_order(i), summary.unit(i), ...
        summary.highest_probability_elevation_m(i), ...
        summary.posterior_mean_elevation_m(i), ...
        summary.p16_elevation_m(i), summary.p84_elevation_m(i));
end
fprintf(fid, '\n## Central morphology\n\n');
fprintf(fid, '- Prior probability of a 1500 m central valley: %.4f.\n', ...
    lowValleyPrior);
fprintf(fid, '- Posterior probability of a 1500 m central valley: %.4g.\n\n', ...
    lowValleyPosterior);
if all(isfinite(manifest.valley_target_m))
    valleyValues = unique(manifest.valley_target_m);
    gangdeseValues = unique(manifest.gangdese_target_m);
    valleyRows = marginal.unit == "Central_valley_zone";
    upperValley = max(marginal.elevationM(valleyRows));
    upperProbability = marginal.posterior_probability( ...
        valleyRows & marginal.elevationM == upperValley);
    valleyProbability = marginal.posterior_probability(valleyRows);
    valleyElevation = marginal.elevationM(valleyRows);
    [~, valleyModeIndex] = max(valleyProbability);
    valleyMode = valleyElevation(valleyModeIndex);
    fprintf(fid, ['The central valley is independently sampled at %d ' ...
        'elevations from %.0f to %.0f m. '], numel(valleyValues), ...
        min(valleyValues), max(valleyValues));
    if valleyMode == upperValley
        fprintf(fid, ['The upper-edge %.0f m state retains posterior ' ...
            'probability %.4f, so the central estimate remains ' ...
            'boundary-limited. '], upperValley, upperProbability);
    else
        fprintf(fid, ['The posterior mode is internal at %.0f m and the ' ...
            'upper-edge %.0f m probability is %.4f, so the estimate is ' ...
            'no longer upper-boundary limited, although its interval ' ...
            'remains broad. '], valleyMode, upperValley, upperProbability);
    end
    fprintf(fid, ['Gangdese is independently sampled at %d elevations ' ...
        'from %.0f to %.0f m.\n'], numel(gangdeseValues), ...
        min(gangdeseValues), max(gangdeseValues));
else
    fprintf(fid, ['The central estimate is not an independently sampled ' ...
        'valley-height inversion. Vnone cases use the modeled linear ramp ' ...
        'between Qiangtang and Gangdese, evaluated at 31.3 N; only explicit ' ...
        'V cases test a low valley.\n']);
end

result = struct;
result.casePosterior = casePosterior;
result.marginal = marginal;
result.summary = summary;
result.lowValleyPrior = lowValleyPrior;
result.lowValleyPosterior = lowValleyPosterior;
result.outputRoot = string(outputRoot);
fprintf('Wrote north-to-south paleotopography report to:\n%s\n', outputRoot);
end
