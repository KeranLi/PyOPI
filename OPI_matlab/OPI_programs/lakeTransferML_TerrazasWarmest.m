function [temperatureOut, sigmaOut, info] = lakeTransferML_TerrazasWarmest( ...
    temperatureIn, latitude, direction, varargin)
% lakeTransferML_TerrazasWarmest applies the Terrazas warmest-season model.
%
% Examples
%   [Tw, sigmaTw] = lakeTransferML_TerrazasWarmest( ...
%       TairWarmest, latitude, "air_to_lake");
%   [Tair, sigmaTair] = lakeTransferML_TerrazasWarmest( ...
%       TwWarmest, latitude, "lake_to_air", ...
%       'LakeAreaKm2', 200, 'LakeDepthM', 20);
%
% Lake area and depth default to medians of modern Terrazas lakes above
% 3 km. info records every row for which a default was used.

if nargin < 3 || isempty(direction)
    direction = "air_to_lake";
end

p = inputParser;
addParameter(p, 'LakeAreaKm2', []);
addParameter(p, 'LakeDepthM', []);
addParameter(p, 'Model', []);
addParameter(p, 'ModelFile', "");
addParameter(p, 'IncludeResidual', true, ...
    @(x) islogical(x) || (isnumeric(x) && isscalar(x)));
parse(p, varargin{:});
opts = p.Results;

[model, modelSource] = resolveModel(opts.Model, opts.ModelFile);
direction = lower(string(direction));
switch direction
    case {"air_to_lake", "forward"}
        submodel = model.forward;
        direction = "air_to_lake";
    case {"lake_to_air", "inverse"}
        submodel = model.inverse;
        direction = "lake_to_air";
    otherwise
        error('Unknown Terrazas ML direction: %s', direction);
end

[temperatureIn, latitude, area, depth] = normalizeInputs( ...
    temperatureIn, latitude, opts.LakeAreaKm2, opts.LakeDepthM);
latitude = abs(latitude);

usedDefaultArea = ~isfinite(area) | area <= 0;
usedDefaultDepth = ~isfinite(depth) | depth <= 0;
area(usedDefaultArea) = model.defaults.lakeAreaKm2;
depth(usedDefaultDepth) = model.defaults.lakeDepthM;

X = [temperatureIn, latitude, log(area), log(depth)];
phi = featureMatrix((X - submodel.mu) ./ submodel.sigma);
temperatureOut = phi * submodel.beta;

if isempty(submodel.bootstrapBeta)
    sigmaModel = zeros(size(temperatureOut));
else
    bootstrapPrediction = phi * submodel.bootstrapBeta;
    sigmaModel = std(bootstrapPrediction, 0, 2, 'omitnan');
end
sigmaResidual = repmat(submodel.sigmaResidual, size(temperatureOut));
if opts.IncludeResidual
    sigmaOut = sqrt(sigmaResidual.^2 + sigmaModel.^2);
else
    sigmaOut = sigmaModel;
end

invalid = ~isfinite(temperatureIn) | ~isfinite(latitude);
temperatureOut(invalid) = nan;
sigmaOut(invalid) = nan;

info = struct;
info.direction = direction;
info.modelName = model.name;
info.modelSource = string(modelSource);
info.modelSchemaVersion = model.schemaVersion;
info.lakeAreaKm2 = area;
info.lakeDepthM = depth;
info.usedDefaultArea = usedDefaultArea;
info.usedDefaultDepth = usedDefaultDepth;
info.defaultSource = model.defaults.sourceSubset;
info.sigmaModel = sigmaModel;
info.sigmaResidual = sigmaResidual;
info.sigmaIncludesResidual = logical(opts.IncludeResidual);
info.elevationUsedAsPredictor = model.includesElevationPredictor;
info.outsideGlobalTrainingRange = any( ...
    X < submodel.predictorMin | X > submodel.predictorMax, 2);
info.outsideHighElevationTrainingRange = any( ...
    X < submodel.highElevationPredictorMin | ...
    X > submodel.highElevationPredictorMax, 2);
end

function [model, source] = resolveModel(modelInput, modelFile)
if ~isempty(modelInput)
    if ~isstruct(modelInput) || ~isfield(modelInput, 'forward')
        error('Model must be a TerrazasWarmestML model struct.');
    end
    model = modelInput;
    source = "provided model struct";
    return
end

if strlength(string(modelFile)) == 0
    projectRoot = fileparts(fileparts(mfilename('fullpath')));
    modelFile = fullfile(projectRoot, 'data', 'derived', ...
        'LakeTransferFunction', 'TerrazasWarmestML', ...
        'TerrazasWarmestML_model.mat');
end
modelFile = char(string(modelFile));
if ~isfile(modelFile)
    error(['Terrazas warmest ML model not found: %s\n', ...
        'Train it with opiTrain_TerrazasWarmestML().'], modelFile);
end

persistent cachedFile cachedStamp cachedModel
fileInfo = dir(modelFile);
stamp = fileInfo.datenum;
if isempty(cachedModel) || ~strcmp(cachedFile, modelFile) || cachedStamp ~= stamp
    S = load(modelFile, 'model');
    if ~isfield(S, 'model')
        error('Model file does not contain a model struct: %s', modelFile);
    end
    cachedModel = S.model;
    cachedFile = modelFile;
    cachedStamp = stamp;
end
model = cachedModel;
source = modelFile;
end

function [temperature, latitude, area, depth] = normalizeInputs( ...
    temperature, latitude, area, depth)
n = max([numel(temperature), numel(latitude), numel(area), numel(depth), 1]);
temperature = expandInput(temperature, n, 'temperatureIn', nan);
latitude = expandInput(latitude, n, 'latitude', nan);
area = expandInput(area, n, 'LakeAreaKm2', nan);
depth = expandInput(depth, n, 'LakeDepthM', nan);
end

function value = expandInput(value, n, label, emptyValue)
if isempty(value)
    value = repmat(emptyValue, n, 1);
elseif isscalar(value)
    value = repmat(double(value), n, 1);
elseif numel(value) == n
    value = double(value(:));
else
    error('%s must be scalar or have %d elements.', label, n);
end
end

function phi = featureMatrix(X)
phi = [ones(size(X, 1), 1), X, X.^2, ...
    X(:, 1).*X(:, 2), X(:, 1).*X(:, 3), X(:, 1).*X(:, 4), ...
    X(:, 2).*X(:, 3), X(:, 2).*X(:, 4), X(:, 3).*X(:, 4)];
end
