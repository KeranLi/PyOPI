function [Tw, sigmaTw, model] = lakeTransferAirToLake_Terrazas2025(MAAT, lat, elevKm, season, dataCsv)
% lakeTransferAirToLake_Terrazas2025 predicts LSWT from air temperature.
%
% This is an independently refit air-to-lake model using the Terrazas 2025
% modern lake dataset. It is not the algebraic inverse of the published
% lake-to-air transfer functions.

if nargin < 5 || isempty(dataCsv)
    thisFile = mfilename('fullpath');
    projectRoot = fileparts(fileparts(fileparts(thisFile)));
    dataCsv = fullfile(projectRoot, 'data', 'reference', ...
        'LakeTransferFunction', 'ERA5_LakeTemp.csv');
end
if ~isfile(dataCsv)
    error('Terrazas lake-transfer CSV not found: %s', dataCsv);
end

season = lower(string(season));
[lswtCol, seasonLabel] = seasonToLswtColumn(season);

T = readtable(dataCsv, 'VariableNamingRule', 'preserve');
required = ["tas_ann_avg", "abs_lat", "elevation_km", lswtCol];
missing = setdiff(required, string(T.Properties.VariableNames));
if ~isempty(missing)
    error('Missing required Terrazas CSV column(s): %s', strjoin(missing, ', '));
end

xMaat = T.tas_ann_avg;
xLat = T.abs_lat;
xElev = T.elevation_km;
y = T.(lswtCol);
iGood = isfinite(xMaat) & isfinite(xLat) & isfinite(xElev) & isfinite(y);
xMaat = xMaat(iGood);
xLat = xLat(iGood);
xElev = xElev(iGood);
y = y(iGood);

X = [ones(size(xMaat)), xMaat, xMaat.^2, xLat, xElev];
beta = X \ y;
yHat = X * beta;
residuals = y - yHat;
n = size(X, 1);
p = size(X, 2);
sigmaResidual = sqrt(sum(residuals.^2) / max(1, n - p));
covBeta = sigmaResidual.^2 .* pinv(X' * X);

MAAT = MAAT(:);
lat = abs(lat(:));
elevKm = elevKm(:);
X0 = [ones(size(MAAT)), MAAT, MAAT.^2, lat, elevKm];
Tw = X0 * beta;
sigmaMean = sqrt(max(0, sum((X0 * covBeta) .* X0, 2)));
sigmaTw = sqrt(sigmaResidual.^2 + sigmaMean.^2);

model = struct;
model.name = "Terrazas2025_air_to_lake_" + seasonLabel;
model.season = seasonLabel;
model.response = lswtCol;
model.predictors = ["constant", "tas_ann_avg", "tas_ann_avg_squared", "abs_lat", "elevation_km"];
model.beta = beta;
model.n = n;
model.p = p;
model.sigmaResidual = sigmaResidual;
model.rmse = sqrt(mean(residuals.^2));
model.dataCsv = dataCsv;

end

function [lswtCol, seasonLabel] = seasonToLswtColumn(season)
switch season
    case {"annual", "ann", "malswt"}
        lswtCol = "lswt_ann_avg";
        seasonLabel = "annual";
    case {"ao", "spring_summer", "spring-through-summer"}
        lswtCol = "lswt_ao_avg";
        seasonLabel = "ao";
    case {"amj", "spring"}
        lswtCol = "lswt_amj_avg";
        seasonLabel = "amj";
    case {"jja", "summer"}
        lswtCol = "lswt_jja_avg";
        seasonLabel = "jja";
    case {"warmest", "warmest_month"}
        lswtCol = "lswt_warmest_avg";
        seasonLabel = "warmest";
    otherwise
        error('Unknown season: %s', season);
end
end
