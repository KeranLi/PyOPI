function [model, validation] = opiTrain_TerrazasWarmestML(dataCsv, outputDir, varargin)
% opiTrain_TerrazasWarmestML trains a toolbox-free warmest-season model.
%
% The primary forward model predicts warmest lake-surface temperature from
% warmest-month air temperature, latitude, lake area, and lake depth. The
% companion inverse model provides an independent lake-to-air check. Model
% selection uses held-out geographic blocks rather than random rows.

projectRoot = fileparts(fileparts(mfilename('fullpath')));
if nargin < 1 || isempty(dataCsv)
    dataCsv = fullfile(projectRoot, 'data', 'reference', ...
        'LakeTransferFunction', 'ERA5_LakeTemp.csv');
end
if nargin < 2 || isempty(outputDir)
    outputDir = fullfile(projectRoot, 'data', 'derived', ...
        'LakeTransferFunction', 'TerrazasWarmestML');
end

p = inputParser;
addParameter(p, 'NumFolds', 5, @(x) isscalar(x) && x >= 2);
addParameter(p, 'BlockLonDegrees', 15, @(x) isscalar(x) && x > 0);
addParameter(p, 'BlockLatDegrees', 10, @(x) isscalar(x) && x > 0);
addParameter(p, 'NumBootstrap', 200, @(x) isscalar(x) && x >= 0);
addParameter(p, 'RandomSeed', 20250722, @(x) isscalar(x) && isfinite(x));
parse(p, varargin{:});
opts = p.Results;

if ~isfile(dataCsv)
    error('Terrazas lake dataset not found: %s', dataCsv);
end
if ~isfolder(outputDir)
    mkdir(outputDir);
end

T = readtable(dataCsv, 'VariableNamingRule', 'preserve');
required = ["center_long", "center_lat", "abs_lat", "elevation_km", ...
    "Lake_area", "Depth_avg", "lswt_warmest_avg", ...
    "tas_warmest_month", "tas_ann_avg"];
missing = setdiff(required, string(T.Properties.VariableNames));
if ~isempty(missing)
    error('Missing required Terrazas column(s): %s', strjoin(missing, ', '));
end

good = isfinite(T.center_long) & isfinite(T.center_lat) & ...
    isfinite(T.abs_lat) & isfinite(T.elevation_km) & ...
    isfinite(T.Lake_area) & T.Lake_area > 0 & ...
    isfinite(T.Depth_avg) & T.Depth_avg > 0 & ...
    isfinite(T.lswt_warmest_avg) & isfinite(T.tas_warmest_month) & ...
    isfinite(T.tas_ann_avg);
T = T(good, :);

logArea = log(T.Lake_area);
logDepth = log(T.Depth_avg);
XForward = [T.tas_warmest_month, T.abs_lat, logArea, logDepth];
XInverse = [T.lswt_warmest_avg, T.abs_lat, logArea, logDepth];

[fold, blockGroup, blockTable] = makeSpatialFolds( ...
    T.center_long, T.center_lat, T.elevation_km, opts);
lambdaGrid = [0, logspace(-4, 4, 17)];

rng(opts.RandomSeed, 'twister');
forward = trainRidgeModel(XForward, T.lswt_warmest_avg, fold, ...
    blockGroup, lambdaGrid, opts.NumBootstrap);
forward.direction = "air_to_lake";
forward.response = "lswt_warmest_avg";
forward.predictorNames = ["tas_warmest_month", "abs_lat", ...
    "log_lake_area", "log_depth"];
forward.predictorMin = min(XForward, [], 1);
forward.predictorMax = max(XForward, [], 1);

inverse = trainRidgeModel(XInverse, T.tas_warmest_month, fold, ...
    blockGroup, lambdaGrid, opts.NumBootstrap);
inverse.direction = "lake_to_air";
inverse.response = "tas_warmest_month";
inverse.predictorNames = ["lswt_warmest_avg", "abs_lat", ...
    "log_lake_area", "log_depth"];
inverse.predictorMin = min(XInverse, [], 1);
inverse.predictorMax = max(XInverse, [], 1);

linearForward = spatialLinearPrediction( ...
    T.tas_warmest_month, T.lswt_warmest_avg, fold);
linearInverse = spatialLinearPrediction( ...
    T.lswt_warmest_avg, T.tas_warmest_month, fold);

high = T.elevation_km > 3;
if ~any(high)
    error('No lakes above 3 km are available for paleolake defaults.');
end
forward.highElevationPredictorMin = min(XForward(high, :), [], 1);
forward.highElevationPredictorMax = max(XForward(high, :), [], 1);
inverse.highElevationPredictorMin = min(XInverse(high, :), [], 1);
inverse.highElevationPredictorMax = max(XInverse(high, :), [], 1);
modelForward = forward;
modelInverse = inverse;

model = struct;
model.schemaVersion = 1;
model.name = "TerrazasWarmestML_toolbox_free_ridge";
model.createdAt = string(datetime('now', 'TimeZone', 'local'));
model.reference = "Terrazas et al. (2025) modern lake dataset";
model.dataCsv = string(dataCsv);
model.nTraining = height(T);
model.featureDefinition = ["intercept", "x1", "x2", "x3", "x4", ...
    "x1_squared", "x2_squared", "x3_squared", "x4_squared", ...
    "x1_x2", "x1_x3", "x1_x4", "x2_x3", "x2_x4", "x3_x4"];
model.includesElevationPredictor = false;
model.forward = modelForward;
model.inverse = modelInverse;
model.defaults = struct( ...
    'lakeAreaKm2', median(T.Lake_area(high), 'omitnan'), ...
    'lakeDepthM', median(T.Depth_avg(high), 'omitnan'), ...
    'sourceSubset', "Terrazas lakes above 3 km", ...
    'nSource', sum(high));
model.crossValidation = struct( ...
    'method', "geographic block holdout", ...
    'numFolds', opts.NumFolds, ...
    'blockLonDegrees', opts.BlockLonDegrees, ...
    'blockLatDegrees', opts.BlockLatDegrees, ...
    'numBlocks', height(blockTable), ...
    'fold', fold, ...
    'blockGroup', blockGroup, ...
    'blockTable', blockTable);
model.bootstrap = struct( ...
    'method', "geographic block resampling", ...
    'numReplicates', opts.NumBootstrap, ...
    'randomSeed', opts.RandomSeed);
model.toolboxDependencies = "base MATLAB only";

validation = buildValidationTable(T, forward.oofPrediction, ...
    inverse.oofPrediction, linearForward, linearInverse);
model.validation = validation;

modelFile = fullfile(outputDir, 'TerrazasWarmestML_model.mat');
save(modelFile, 'model', '-v7');
writetable(validation, fullfile(outputDir, ...
    'TerrazasWarmestML_validation_metrics.csv'));

predictions = table(T.center_long, T.center_lat, T.abs_lat, ...
    T.elevation_km, T.Lake_area, T.Depth_avg, T.tas_warmest_month, ...
    T.tas_ann_avg, T.lswt_warmest_avg, fold, blockGroup, ...
    forward.oofPrediction, inverse.oofPrediction, linearForward, ...
    linearInverse, 'VariableNames', {'longitude', 'latitude', 'abs_lat', ...
    'elevation_km', 'lake_area_km2', 'lake_depth_m', ...
    'tas_warmest_month_C', 'tas_annual_C', 'lswt_warmest_C', ...
    'spatial_fold', 'spatial_block', 'ml_forward_oof_C', ...
    'ml_inverse_oof_C', 'linear_forward_oof_C', 'linear_inverse_oof_C'});
writetable(predictions, fullfile(outputDir, ...
    'TerrazasWarmestML_cv_predictions.csv'));
writeReport(fullfile(outputDir, 'TerrazasWarmestML_report.md'), ...
    model, validation);

fprintf('Trained Terrazas warmest-season ML model with %d lakes.\n', height(T));
fprintf('Model: %s\n', modelFile);
fprintf('Validation: %s\n', fullfile(outputDir, ...
    'TerrazasWarmestML_validation_metrics.csv'));
end

function [fold, blockGroup, blockTable] = makeSpatialFolds(lon, lat, elevKm, opts)
lonBin = floor((lon + 180) ./ opts.BlockLonDegrees);
latBin = floor((lat + 90) ./ opts.BlockLatDegrees);
[blockPairs, ~, blockGroup] = unique([lonBin, latBin], 'rows');
nBlocks = size(blockPairs, 1);
counts = accumarray(blockGroup, 1, [nBlocks, 1]);
maxElev = accumarray(blockGroup, elevKm, [nBlocks, 1], @max, nan);
hasHigh = maxElev > 3;

% Put high-elevation blocks into separate folds first, then balance row counts.
[~, order] = sortrows([-double(hasHigh), -maxElev, -counts, (1:nBlocks)'], ...
    [1, 2, 3, 4]);
blockFold = zeros(nBlocks, 1);
foldCounts = zeros(opts.NumFolds, 1);
for i = 1:nBlocks
    b = order(i);
    candidates = find(foldCounts == min(foldCounts));
    f = candidates(1);
    blockFold(b) = f;
    foldCounts(f) = foldCounts(f) + counts(b);
end
fold = blockFold(blockGroup);

blockTable = table((1:nBlocks)', blockPairs(:, 1), blockPairs(:, 2), ...
    counts, maxElev, hasHigh, blockFold, ...
    'VariableNames', {'block_id', 'longitude_bin', 'latitude_bin', ...
    'n_lakes', 'max_elevation_km', 'contains_lake_above_3km', 'fold'});
end

function submodel = trainRidgeModel(X, y, fold, blockGroup, lambdaGrid, nBoot)
nLambda = numel(lambdaGrid);
cvPred = nan(numel(y), nLambda);
for j = 1:nLambda
    for f = unique(fold(:))'
        train = fold ~= f;
        test = fold == f;
        [mu, sigma] = scaling(X(train, :));
        phiTrain = featureMatrix((X(train, :) - mu) ./ sigma);
        beta = ridgeSolve(phiTrain, y(train), lambdaGrid(j));
        phiTest = featureMatrix((X(test, :) - mu) ./ sigma);
        cvPred(test, j) = phiTest * beta;
    end
end
cvRmse = sqrt(mean((cvPred - y).^2, 1, 'omitnan'));
[~, best] = min(cvRmse);

[mu, sigma] = scaling(X);
phi = featureMatrix((X - mu) ./ sigma);
beta = ridgeSolve(phi, y, lambdaGrid(best));
trainPrediction = phi * beta;

bootstrapBeta = nan(size(phi, 2), nBoot);
nBlocks = max(blockGroup);
blockRows = cell(nBlocks, 1);
for b = 1:nBlocks
    blockRows{b} = find(blockGroup == b);
end
for i = 1:nBoot
    sampledBlocks = randi(nBlocks, nBlocks, 1);
    rows = vertcat(blockRows{sampledBlocks});
    bootstrapBeta(:, i) = ridgeSolve(phi(rows, :), y(rows), lambdaGrid(best));
end

submodel = struct;
submodel.mu = mu;
submodel.sigma = sigma;
submodel.beta = beta;
submodel.lambda = lambdaGrid(best);
submodel.lambdaGrid = lambdaGrid;
submodel.cvRmseByLambda = cvRmse;
submodel.oofPrediction = cvPred(:, best);
submodel.trainingPrediction = trainPrediction;
submodel.trainingRmse = sqrt(mean((trainPrediction - y).^2, 'omitnan'));
submodel.sigmaResidual = cvRmse(best);
submodel.bootstrapBeta = bootstrapBeta;
end

function prediction = spatialLinearPrediction(x, y, fold)
prediction = nan(size(y));
for f = unique(fold(:))'
    train = fold ~= f;
    test = fold == f;
    Xtrain = [ones(sum(train), 1), x(train)];
    beta = Xtrain \ y(train);
    prediction(test) = [ones(sum(test), 1), x(test)] * beta;
end
end

function [mu, sigma] = scaling(X)
mu = mean(X, 1, 'omitnan');
sigma = std(X, 0, 1, 'omitnan');
sigma(~isfinite(sigma) | sigma == 0) = 1;
end

function phi = featureMatrix(X)
phi = [ones(size(X, 1), 1), X, X.^2, ...
    X(:, 1).*X(:, 2), X(:, 1).*X(:, 3), X(:, 1).*X(:, 4), ...
    X(:, 2).*X(:, 3), X(:, 2).*X(:, 4), X(:, 3).*X(:, 4)];
end

function beta = ridgeSolve(phi, y, lambda)
penalty = eye(size(phi, 2));
penalty(1, 1) = 0;
beta = pinv(phi' * phi + lambda .* penalty) * (phi' * y);
end

function validation = buildValidationTable(T, forwardML, inverseML, ...
    forwardLinear, inverseLinear)
validation = table();
subsets = {true(height(T), 1), T.abs_lat >= 25 & T.abs_lat <= 40, ...
    T.elevation_km > 3, T.elevation_km > 4};
subsetNames = ["all", "latitude_25_to_40", "elevation_above_3km", ...
    "elevation_above_4km"];

for i = 1:numel(subsets)
    keep = subsets{i};
    validation = [validation; metricRow("warmest_air_to_lake", ...
        "ML_quadratic_ridge_spatial_CV", subsetNames(i), ...
        T.lswt_warmest_avg(keep), forwardML(keep))]; %#ok<AGROW>
    validation = [validation; metricRow("warmest_air_to_lake", ...
        "linear_air_temperature_spatial_CV", subsetNames(i), ...
        T.lswt_warmest_avg(keep), forwardLinear(keep))]; %#ok<AGROW>
    validation = [validation; metricRow("warmest_lake_to_air", ...
        "ML_quadratic_ridge_spatial_CV", subsetNames(i), ...
        T.tas_warmest_month(keep), inverseML(keep))]; %#ok<AGROW>
    validation = [validation; metricRow("warmest_lake_to_air", ...
        "linear_lake_temperature_spatial_CV", subsetNames(i), ...
        T.tas_warmest_month(keep), inverseLinear(keep))]; %#ok<AGROW>

    tf3 = lakeTransfer_Terrazas2025(T.lswt_warmest_avg(keep), ...
        T.abs_lat(keep), T.elevation_km(keep), "warmest", "TF3");
    tf4 = lakeTransfer_Terrazas2025(T.lswt_warmest_avg(keep), ...
        T.abs_lat(keep), T.elevation_km(keep), "warmest", "TF4");
    validation = [validation; metricRow("published_lake_to_MAAT", ...
        "published_TF3", subsetNames(i), T.tas_ann_avg(keep), tf3)]; %#ok<AGROW>
    validation = [validation; metricRow("published_lake_to_MAAT", ...
        "published_TF4", subsetNames(i), T.tas_ann_avg(keep), tf4)]; %#ok<AGROW>
end
end

function row = metricRow(task, modelName, subset, observed, predicted)
good = isfinite(observed) & isfinite(predicted);
observed = observed(good);
predicted = predicted(good);
residual = predicted - observed;
if isempty(observed)
    rmse = nan; mae = nan; bias = nan; r2 = nan;
else
    rmse = sqrt(mean(residual.^2));
    mae = mean(abs(residual));
    bias = mean(residual);
    denom = sum((observed - mean(observed)).^2);
    if denom > 0
        r2 = 1 - sum(residual.^2) ./ denom;
    else
        r2 = nan;
    end
end
row = table(task, modelName, subset, numel(observed), rmse, mae, bias, r2, ...
    'VariableNames', {'task', 'model', 'subset', 'n', 'rmse_C', ...
    'mae_C', 'bias_pred_minus_obs_C', 'R2'});
end

function writeReport(reportFile, model, validation)
fid = fopen(reportFile, 'w');
if fid == -1
    error('Could not create ML report: %s', reportFile);
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>

fprintf(fid, '# Terrazas Warmest-Season MATLAB Model\n\n');
fprintf(fid, 'Created: %s\n\n', model.createdAt);
fprintf(fid, 'This model uses base MATLAB only. It does not require the Statistics and Machine Learning Toolbox.\n\n');
fprintf(fid, '## Scientific role\n\n');
fprintf(fid, '- Forward: warmest-month air temperature to warmest lake-surface temperature.\n');
fprintf(fid, '- Inverse check: warmest lake-surface temperature to warmest-month air temperature.\n');
fprintf(fid, '- Elevation is deliberately excluded from the ML predictors so a separately specified warm-season lapse rate can carry the elevation effect.\n');
fprintf(fid, '- Published TF3/TF4 metrics retain their original target, mean annual air temperature, and are reported as a separate task rather than as warmest-air competitors.\n\n');
fprintf(fid, '## Validation design\n\n');
fprintf(fid, '- %d lakes in %d geographic blocks.\n', ...
    model.nTraining, model.crossValidation.numBlocks);
fprintf(fid, '- %d-fold geographic-block holdout; rows from one block never occur in both training and validation.\n', ...
    model.crossValidation.numFolds);
fprintf(fid, '- Ridge lambda: %.6g (forward), %.6g (inverse).\n', ...
    model.forward.lambda, model.inverse.lambda);
fprintf(fid, '- Predictive sigma uses spatial-CV RMSE plus geographic-block bootstrap model spread.\n\n');
fprintf(fid, '## Paleolake defaults\n\n');
fprintf(fid, 'When lake area or depth is unavailable, prediction uses medians from %s (n=%d):\n\n', ...
    model.defaults.sourceSubset, model.defaults.nSource);
fprintf(fid, '- area: %.3f km2\n', model.defaults.lakeAreaKm2);
fprintf(fid, '- depth: %.3f m\n\n', model.defaults.lakeDepthM);
fprintf(fid, '## Validation metrics\n\n');
fprintf(fid, '| Task | Model | Subset | n | RMSE C | MAE C | Bias C | R2 |\n');
fprintf(fid, '|---|---|---:|---:|---:|---:|---:|---:|\n');
for i = 1:height(validation)
    fprintf(fid, '| %s | %s | %s | %d | %.3f | %.3f | %.3f | %.3f |\n', ...
        validation.task(i), validation.model(i), validation.subset(i), ...
        validation.n(i), validation.rmse_C(i), validation.mae_C(i), ...
        validation.bias_pred_minus_obs_C(i), validation.R2(i));
end
fprintf(fid, '\n## Use in OPI\n\n');
fprintf(fid, 'The OPI warmest clumped objective calls `lakeTransferML_TerrazasWarmest` in the forward direction. Annual and JJA sensitivity modes remain on their existing seasonal transfer path.\n');
end
