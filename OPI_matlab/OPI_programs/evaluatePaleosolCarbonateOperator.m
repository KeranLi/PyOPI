function diagnostic = evaluatePaleosolCarbonateOperator( ...
    operatorName, observedCarbonateVPDB, observedSigmaVPDB, ...
    waterD18OVSMOW, elevationM, config)
% Evaluate paleosol carbonate d18O by marginalizing formation seasons.


operatorName = lower(strtrim(string(operatorName)));
switch operatorName
    case "xiong2022_chattian_paleosol"
        [branchTemperatureC, branchSigmaModelC, model] = ...
            paleosolTemperature_Xiong2022(elevationM, config);
    otherwise
        error('Unsupported paleosol carbonate operator: %s', operatorName);
end

if ~isscalar(observedCarbonateVPDB) || ...
        ~isfinite(observedCarbonateVPDB) || ...
        ~isscalar(observedSigmaVPDB) || ~isfinite(observedSigmaVPDB) || ...
        observedSigmaVPDB <= 0 || ~isscalar(waterD18OVSMOW) || ...
        ~isfinite(waterD18OVSMOW) || size(branchTemperatureC, 1) ~= 1
    error('Paleosol carbonate likelihood inputs must be finite scalars.');
end

branchWeight = model.weight ./ sum(model.weight);
if any(~isfinite(branchWeight)) || any(branchWeight < 0) || ...
        sum(branchWeight) <= 0
    error('Paleosol branch weights must be finite and nonnegative.');
end
[branchCarbonateVPDB, ~, derivativeTemperature, derivativeWater] = ...
    kimONeil1997CalciteWater(waterD18OVSMOW, branchTemperatureC, ...
    "water_vsmow_to_carbonate_vpdb");
waterSigma = hypot(config.water_d18O_model_discrepancy_permil, ...
    config.paleosol_soil_water_discrepancy_permil);
branchVariance = observedSigmaVPDB.^2 + ...
    (derivativeTemperature .* branchSigmaModelC).^2 + ...
    (derivativeWater .* waterSigma).^2 + ...
    config.carbonate_fractionation_discrepancy_permil.^2;
branchSigmaTotalVPDB = sqrt(branchVariance);
branchResidualVPDB = observedCarbonateVPDB - branchCarbonateVPDB;
branchLogLikelihood = -0.5 .* ...
    (branchResidualVPDB ./ branchSigmaTotalVPDB).^2 - ...
    log(branchSigmaTotalVPDB) - 0.5 .* log(2*pi) + log(branchWeight);
maximum = max(branchLogLikelihood);
logLikelihood = maximum + log(sum(exp(branchLogLikelihood - maximum)));

predictedMeanVPDB = sum(branchWeight .* branchCarbonateVPDB);
predictedVarianceVPDB = sum(branchWeight .* (branchSigmaTotalVPDB.^2 + ...
    (branchCarbonateVPDB - predictedMeanVPDB).^2));

diagnostic = struct;
diagnostic.operator = operatorName;
diagnostic.model = model;
diagnostic.branchTemperatureC = branchTemperatureC;
diagnostic.branchCarbonateVPDB = branchCarbonateVPDB;
diagnostic.branchSigmaTotalVPDB = branchSigmaTotalVPDB;
diagnostic.branchWeight = branchWeight;
diagnostic.branchResidualVPDB = branchResidualVPDB;
diagnostic.branchLogLikelihood = branchLogLikelihood;
diagnostic.logLikelihood = logLikelihood;
diagnostic.predictedMeanVPDB = predictedMeanVPDB;
diagnostic.residualVPDB = observedCarbonateVPDB - predictedMeanVPDB;
diagnostic.sigmaTotalVPDB = sqrt(predictedVarianceVPDB);
diagnostic.z = diagnostic.residualVPDB ./ diagnostic.sigmaTotalVPDB;
end
