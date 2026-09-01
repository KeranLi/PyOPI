function tests = testPaleosolTemperatureOperator
tests = functiontests(localfunctions);
end

function testXiongTableOneEndpoints(testCase)
config = makeConfig();
[temperatureC, sigmaModelC, model] = ...
    paleosolTemperature_Xiong2022([2500; 4500], config);

verifyEqual(testCase, temperatureC, [24, 22; 14, 10], ...
    'AbsTol', 1e-12);
verifyEqual(testCase, sigmaModelC, 3 .* ones(2), 'AbsTol', 1e-12);
verifyEqual(testCase, model.ageRangeMa, [29, 37]);
end

function testBranchLikelihoodIsMarginalized(testCase)
config = makeConfig();
diagnostic = evaluatePaleosolTemperatureOperator( ...
    "xiong2022_chattian_paleosol", 12.2, 1.6, 4500, config);

branchTemperatureC = [14, 10];
sigmaTotalC = hypot(1.6, 3.0);
expectedDensity = 0.5 .* exp(-0.5 .* ...
    ((12.2 - branchTemperatureC) ./ sigmaTotalC).^2) ./ ...
    (sqrt(2*pi) .* sigmaTotalC);
verifyEqual(testCase, diagnostic.logLikelihood, log(sum(expectedDensity)), ...
    'AbsTol', 1e-12);
verifyEqual(testCase, diagnostic.predictedMeanC, 12, 'AbsTol', 1e-12);
end

function testPaleosolCarbonateUsesBothTemperatureBranches(testCase)
config = makeConfig();
config.water_d18O_model_discrepancy_permil = 1.0;
config.paleosol_soil_water_discrepancy_permil = 1.0;
config.carbonate_fractionation_discrepancy_permil = 0.5;
diagnostic = evaluatePaleosolCarbonateOperator( ...
    "xiong2022_chattian_paleosol", -14.4, 0.3, -17.0, 3000, config);

verifySize(testCase, diagnostic.branchCarbonateVPDB, [1, 2]);
verifyEqual(testCase, diagnostic.branchTemperatureC, [21.5, 19], ...
    'AbsTol', 1e-12);
verifyTrue(testCase, isfinite(diagnostic.logLikelihood));
verifyGreaterThan(testCase, diagnostic.sigmaTotalVPDB, 0.3);
end

function config = makeConfig()
config = struct;
config.paleosol_temperature_model_discrepancy_C = 3.0;
config.xiong2022_early_summer_intercept_C = 36.5;
config.xiong2022_early_summer_lapse_C_per_km = -5.0;
config.xiong2022_september_intercept_C = 37.0;
config.xiong2022_september_lapse_C_per_km = -6.0;
config.xiong2022_early_summer_weight = 0.5;
config.xiong2022_september_weight = 0.5;
end
