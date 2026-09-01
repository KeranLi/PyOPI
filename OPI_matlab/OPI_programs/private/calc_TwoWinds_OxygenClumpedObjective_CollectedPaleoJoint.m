function [chiR2Total, nuTotal, detail] = calc_TwoWinds_OxygenClumpedObjective_CollectedPaleoJoint( ...
    beta, fC, hR, x, y, lat, lat0, hGrid, bMWLSample, ijCatch, ptrCatch, ...
    sampleD2H, sampleD18O, cov, nParametersFree, isFit, ...
    clumped, sampleLon, sampleLat, dolomiteOffsetC, sigmaOffsetC, season, external)
% calc_TwoWinds_OxygenClumpedObjective combines d18O and dolomite clumped T.

TC2K = 273.15;
if nargin < 22
    external = table();
end

try
    [chiR2O, nuO, ~, ...
        ~, T_1, ~, gammaSat_1, ~, ...
        ~, ~, ~, ~, ...
        ~, ~, ~, pGrid_1, ~, ~, ...
        ~, ~, ~, ~, ~, ...
        ~, T_2, ~, gammaSat_2, ~, ...
        ~, ~, ~, ~, ...
        ~, ~, ~, pGrid_2, ~, ~, ...
        ~, ~, ~, ~, ~, ...
        pGrid, ~, d18OGrid, ~, ~, ~, ~] = ...
        calc_TwoWinds_OxygenOnly(beta, fC, hR, ...
        x, y, lat, lat0, hGrid, bMWLSample, ijCatch, ptrCatch, ...
        sampleD2H, sampleD18O, cov, nParametersFree, isFit);
catch ME
    if isFit
        chiR2Total = realmax;
        nuTotal = 1;
        if nargout > 2
            detail = struct('error', string(ME.message));
        end
        return
    else
        rethrow(ME)
    end
end

TGrid1_C = T_1(1) - gammaSat_1(1).*hGrid - TC2K;
TGrid2_C = T_2(1) - gammaSat_2(1).*hGrid - TC2K;
TGridCombined_C = (pGrid_1.*TGrid1_C + pGrid_2.*TGrid2_C)./pGrid;
TGridCombined_C(~isfinite(TGridCombined_C)) = nan;

nClumped = height(clumped);
opiT_C = nan(nClumped, 1);
opiElevM = nan(nClumped, 1);
for i = 1:nClumped
    k = clumped.OPI_sample_index(i);
    ij = catchmentIndices(k, ijCatch, ptrCatch);
    wt = pGrid(ij);
    if sum(wt, 'omitnan') > 0
        wt = wt ./ sum(wt, 'omitnan');
        opiT_C(i) = sum(wt .* TGridCombined_C(ij), 'omitnan');
        opiElevM(i) = sum(wt .* hGrid(ij), 'omitnan');
    else
        opiT_C(i) = interp2(x, y, TGridCombined_C, sampleLon(k), sampleLat(k));
        opiElevM(i) = interp2(x, y, hGrid, sampleLon(k), sampleLat(k));
    end
end

season = lower(string(season));
if ismember(season, ["warmest", "warmest_month"])
    [lakeAreaKm2, lakeDepthM] = optionalLakeProperties(clumped);
    [twPred, sigmaTw, transferInfo] = lakeTransferML_TerrazasWarmest( ...
        opiT_C, clumped.lat, "air_to_lake", ...
        'LakeAreaKm2', lakeAreaKm2, 'LakeDepthM', lakeDepthM);
    transferModel = string(transferInfo.modelName);
else
    [twPred, sigmaTw] = lakeTransferAirToLake_Terrazas2025( ...
        opiT_C, clumped.lat, opiElevM./1e3, season);
    transferInfo = struct;
    transferModel = "Terrazas2025_legacy_seasonal_refit";
end
tdolPred = twPred + dolomiteOffsetC;
sigmaTotal = sqrt(clumped.sigma_T_C.^2 + sigmaTw.^2 + sigmaOffsetC.^2);
rT = clumped.T_clumped_C - tdolPred;
zT = rT ./ sigmaTotal;
chi2T = sum(zT(isfinite(zT)).^2);
nT = sum(isfinite(zT));

[chi2External, nExternal, externalDetail] = evaluateExternal( ...
    external, lonFromGrid(x, lon0FromLat(lat0)), lat, pGrid, d18OGrid, TGridCombined_C, hGrid);

chi2O = chiR2O * nuO;
nuTotal = max(1, nuO + nT + nExternal);
chiR2Total = (chi2O + chi2T + chi2External) / nuTotal;

if nargout > 2
    detail = struct;
    detail.chiR2O = chiR2O;
    detail.nuO = nuO;
    detail.chi2T = chi2T;
    detail.nT = nT;
    detail.meanResidualT_C = mean(rT, 'omitnan');
    detail.meanZT = mean(zT, 'omitnan');
    detail.opiT_C = opiT_C;
    detail.twPred = twPred;
    detail.tdolPred = tdolPred;
    detail.sigmaTotal = sigmaTotal;
    detail.transferModel = transferModel;
    detail.transferInfo = transferInfo;
    detail.chi2External = chi2External;
    detail.nExternal = nExternal;
    detail.external = externalDetail;
end

function lon = lonFromGrid(x, lon0)
lon = lon0 + x ./ (pi * 6371e3 / 180 * cosd(32.9));
end

function lon0 = lon0FromLat(~)
lon0 = 87.2;
end

function [chi2, nUsed, detail] = evaluateExternal(T, lon, lat, pGrid, d18OGrid, airGrid, hGrid)
chi2 = 0; nUsed = 0; detail = table();
if isempty(T) || height(T) == 0, return; end
names = string(T.Properties.VariableNames);
required = ["longitude","latitude","has_d18O","has_clumped_temperature", ...
    "uses_lake_operator","d18O_carbonate_VPDB_permil", ...
    "d18O_carbonate_1sigma_permil","clumped_temperature_C", ...
    "clumped_temperature_1sigma_C"];
if ~all(ismember(required,names)), return; end
[LON,LAT] = meshgrid(lon,lat);
predC = nan(height(T),1); predT = nan(height(T),1);
zC = nan(height(T),1); zT = nan(height(T),1);
for j = 1:height(T)
    if T.longitude(j)<min(lon) || T.longitude(j)>max(lon) || ...
            T.latitude(j)<min(lat) || T.latitude(j)>max(lat), continue; end
    distance = greatCircleDistanceKm(LAT,LON,T.latitude(j),T.longitude(j));
    wet = distance <= 50 & isfinite(pGrid) & pGrid > 0 & isfinite(airGrid);
    if ~any(wet,'all'), continue; end
    w = pGrid(wet); air = sum(w .* airGrid(wet),'omitnan') / sum(w,'omitnan');
    isotope = wet & isfinite(d18OGrid);
    water = sum(pGrid(isotope) .* d18OGrid(isotope),'omitnan') / ...
        sum(pGrid(isotope),'omitnan') * 1e3;
    elev = sum(w .* hGrid(wet),'omitnan') / sum(w,'omitnan');
    if T.uses_lake_operator(j)
        [lakeT, sigmaLake] = lakeTransferAirToLake_Terrazas2025(air, T.latitude(j), elev/1e3, 'warmest');
        if T.has_d18O(j) && isfinite(water)
            predC(j) = kimONeil1997CalciteWater(water, lakeT, 'water_vsmow_to_carbonate_vpdb');
            zC(j) = (T.d18O_carbonate_VPDB_permil(j)-predC(j)) / ...
                hypot(T.d18O_carbonate_1sigma_permil(j), 1.5);
        end
        if T.has_clumped_temperature(j)
            predT(j) = lakeT;
            zT(j) = (T.clumped_temperature_C(j)-lakeT) / ...
                hypot(T.clumped_temperature_1sigma_C(j), sigmaLake);
        end
    elseif T.has_clumped_temperature(j)
        predT(j) = air;
        zT(j) = (T.clumped_temperature_C(j)-air) / ...
            hypot(T.clumped_temperature_1sigma_C(j), 4.0);
    end
end
chi2 = sum(zC(isfinite(zC)).^2) + sum(zT(isfinite(zT)).^2);
nUsed = sum(isfinite(zC) | isfinite(zT));
detail = table(T.longitude, T.latitude, predC, predT, zC, zT, ...
    'VariableNames', {'longitude','latitude','predicted_carbonate', ...
    'predicted_temperature','z_carbonate','z_temperature'});
end

function d = greatCircleDistanceKm(lat1, lon1, lat2, lon2)
R = 6371.0088;
lat1=deg2rad(lat1); lon1=deg2rad(lon1); lat2=deg2rad(lat2); lon2=deg2rad(lon2);
a=sin((lat1-lat2)./2).^2+cos(lat1).*cos(lat2).*sin((lon1-lon2)./2).^2;
d=2*R*atan2(sqrt(a),sqrt(max(0,1-a)));
end

end

function [lakeAreaKm2, lakeDepthM] = optionalLakeProperties(clumped)
names = string(clumped.Properties.VariableNames);
if ismember("lake_area_km2", names)
    lakeAreaKm2 = clumped.lake_area_km2;
else
    lakeAreaKm2 = [];
end
if ismember("lake_depth_m", names)
    lakeDepthM = clumped.lake_depth_m;
else
    lakeDepthM = [];
end
end
