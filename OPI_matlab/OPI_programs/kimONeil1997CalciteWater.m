function [valueOut, alpha, derivativeTemperature, derivativeIsotope] = ...
    kimONeil1997CalciteWater(valueIn, temperatureC, direction)
% kimONeil1997CalciteWater applies the Kim and O'Neil (1997) fractionation.

temperatureK = temperatureC + 273.15;
if any(~isfinite(temperatureK) | temperatureK <= 0)
    error('Kim-O''Neil temperature must be finite and above absolute zero.');
end

thousandLnAlpha = 18.03 .* (1000 ./ temperatureK) - 32.42;
alpha = exp(thousandLnAlpha ./ 1000);
direction = lower(string(direction));

switch direction
    case {"water_vsmow_to_carbonate_vpdb", "forward"}
        carbonateVSMOW = (valueIn + 1000) .* alpha - 1000;
        valueOut = (carbonateVSMOW - 30.91) ./ 1.03091;
        derivativeTemperature = -(carbonateVSMOW + 1000) .* 18.03 ./ ...
            temperatureK.^2 ./ 1.03091;
        derivativeIsotope = alpha ./ 1.03091;
    case {"carbonate_vpdb_to_water_vsmow", "inverse"}
        carbonateVSMOW = 1.03091 .* valueIn + 30.91;
        valueOut = (carbonateVSMOW + 1000) ./ alpha - 1000;
        derivativeTemperature = (valueOut + 1000) .* 18.03 ./ ...
            temperatureK.^2;
        derivativeIsotope = 1.03091 ./ alpha;
    otherwise
        error('Unknown Kim-O''Neil conversion direction: %s', direction);
end
end
