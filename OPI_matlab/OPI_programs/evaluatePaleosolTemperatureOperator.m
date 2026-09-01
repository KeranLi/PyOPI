function diagnostic = evaluatePaleosolTemperatureOperator( ...
    operatorName, observedTemperatureC, observedSigmaC, elevationM, config)
% Evaluate a named paleosol-temperature operator in temperature space.

% The likelihood marginalizes formation-season branches instead of first
% converting the observed temperature to a single paleoelevation estimate.


operatorName = lower(strtrim(string(operatorName)));
switch operatorName
    case "xiong2022_chattian_paleosol"
        [branchTemperatureC, branchSigmaModelC, model] = ...
            paleosolTemperature_Xiong2022(elevationM, config);
    otherwise
        error('Unsupported paleosol temperature operator: %s', operatorName);
end

if ~isscalar(observedTemperatureC) || ~isfinite(observedTemperatureC) || ...
        ~isscalar(observedSigmaC) || ~isfinite(observedSigmaC) || ...
        observedSigmaC <= 0 || size(branchTemperatureC, 1) ~= 1
    error('Paleosol likelihood inputs must be finite scalar observations.');
end

branchWeight = model.weight ./ sum(model.weight);
if any(~isfinite(branchWeight)) || any(branchWeight < 0) || ...
        sum(branchWeight) <= 0
    error('Paleosol branch weights must be finite and nonnegative.');
end
branchSigmaTotalC = hypot(observedSigmaC, branchSigmaModelC);
branchResidualC = observedTemperatureC - branchTemperatureC;
branchLogLikelihood = -0.5 .* (branchResidualC ./ branchSigmaTotalC).^2 - ...
    log(branchSigmaTotalC) - 0.5 .* log(2*pi) + log(branchWeight);
maximum = max(branchLogLikelihood);
logLikelihood = maximum + log(sum(exp(branchLogLikelihood - maximum)));

predictedMeanC = sum(branchWeight .* branchTemperatureC);
predictedVarianceC = sum(branchWeight .* (branchSigmaTotalC.^2 + ...
    (branchTemperatureC - predictedMeanC).^2));

diagnostic = struct;
diagnostic.operator = operatorName;
diagnostic.model = model;
diagnostic.branchTemperatureC = branchTemperatureC;
diagnostic.branchSigmaTotalC = branchSigmaTotalC;
diagnostic.branchWeight = branchWeight;
diagnostic.branchResidualC = branchResidualC;
diagnostic.branchLogLikelihood = branchLogLikelihood;
diagnostic.logLikelihood = logLikelihood;
diagnostic.predictedMeanC = predictedMeanC;
diagnostic.residualC = observedTemperatureC - predictedMeanC;
diagnostic.sigmaTotalC = sqrt(predictedVarianceC);
diagnostic.z = diagnostic.residualC ./ diagnostic.sigmaTotalC;
end
