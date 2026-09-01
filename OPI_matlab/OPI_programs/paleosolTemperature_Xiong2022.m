function [temperatureC, sigmaModelC, model] = ...
    paleosolTemperature_Xiong2022(elevationM, config)
% Predict 29-37 Ma central-Tibet paleosol formation temperature.
%
% The two branches are linear relations between prescribed elevation and
% modeled carbonate-formation-period dry-bulb temperature from the
% Chattian valley and plateau end members in Xiong et al. (2022), Table 1.
% They represent April-June/May-June and September formation windows.
%
% DOI: 10.1126/sciadv.abj0944

elevationKm = elevationM(:) ./ 1e3;
earlySummerC = config.xiong2022_early_summer_intercept_C + ...
    config.xiong2022_early_summer_lapse_C_per_km .* elevationKm;
septemberC = config.xiong2022_september_intercept_C + ...
    config.xiong2022_september_lapse_C_per_km .* elevationKm;
temperatureC = [earlySummerC, septemberC];
sigmaModelC = repmat(config.paleosol_temperature_model_discrepancy_C, ...
    size(temperatureC));

if any(~isfinite(temperatureC), 'all') || ...
        ~isfinite(config.paleosol_temperature_model_discrepancy_C) || ...
        config.paleosol_temperature_model_discrepancy_C < 0
    error('Invalid Xiong 2022 paleosol operator input or configuration.');
end

model = struct;
model.name = "Xiong2022_Chattian_paleosol_Td_elevation_mixture";
model.ageRangeMa = [29, 37];
model.region = "central_Tibet";
model.branch = ["early_summer", "september"];
model.weight = [config.xiong2022_early_summer_weight, ...
    config.xiong2022_september_weight];
model.temperatureC = temperatureC;
model.sigmaModelC = sigmaModelC;
model.sourceDoi = "10.1126/sciadv.abj0944";
end
