function [chiR2Total, nuTotal, detail] = calc_TwoWinds_OxygenClumpedObjective( ...
    beta, fC, hR, x, y, lat, lat0, hGrid, bMWLSample, ijCatch, ptrCatch, ...
    sampleD2H, sampleD18O, cov, nParametersFree, isFit, ...
    clumped, sampleLon, sampleLat, dolomiteOffsetC, sigmaOffsetC, season)
% calc_TwoWinds_OxygenClumpedObjective combines d18O and dolomite clumped T.

TC2K = 273.15;

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
        pGrid, ~, ~, ~, ~, ~, ~] = ...
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

chi2O = chiR2O * nuO;
nuTotal = max(1, nuO + nT);
chiR2Total = (chi2O + chi2T) / nuTotal;

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
