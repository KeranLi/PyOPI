function MAAT = lakeTransfer_Terrazas2025(Tw, lat, elevKm, season, modelType)
% lakeTransfer_Terrazas2025 estimates MAAT from lake temperature.
%
% Tw is lake surface water temperature in deg C. lat is absolute latitude in
% degrees. elevKm is elevation in km. season can be annual, ao, amj, jja, or
% warmest. modelType can be TF3 or TF4.

if nargin < 5 || isempty(modelType)
    modelType = 'TF4';
end

season = lower(string(season));
modelType = upper(string(modelType));
lat = abs(lat);

switch season
    case {"annual", "ann", "malswt"}
        c3 = [-0.0402, 2.5413, -0.0024, 0, -14.2560];
        c4 = [-0.0403, 2.3890, -0.0767, -0.7038, -8.8227];
    case {"ao", "spring_summer", "spring-through-summer"}
        c3 = [-0.0141, 1.6034, -0.1099, 0, -8.7553];
        c4 = [-0.0162, 1.4369, -0.2282, -1.5486, 0.4960];
    case {"amj", "spring"}
        c3 = [-0.0167, 1.4745, -0.1068, 0, -3.0564];
        c4 = [-0.0172, 1.2746, -0.2331, -1.6609, 6.1134];
    case {"jja", "summer"}
        c3 = [-0.0015, 0.9189, -0.2738, 0, 1.4836];
        c4 = [-0.0043, 0.7775, -0.3937, -2.3094, 12.0188];
    case {"warmest", "warmest_month"}
        c3 = [-0.0067, 1.1595, -0.3147, 0, -0.9535];
        c4 = [-0.0055, 0.8336, -0.4307, -2.4042, 11.8427];
    otherwise
        error('Unknown season: %s', season);
end

switch modelType
    case "TF3"
        c = c3;
    case "TF4"
        c = c4;
    otherwise
        error('Unknown modelType: %s', modelType);
end

MAAT = c(1).*Tw.^2 + c(2).*Tw + c(3).*lat + c(4).*elevKm + c(5);

end
