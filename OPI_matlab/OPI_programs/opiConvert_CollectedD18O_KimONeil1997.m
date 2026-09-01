function result = opiConvert_CollectedD18O_KimONeil1997(dataDir)
% Convert paired carbonate d18O and clumped T to water d18O.

if nargin < 1 || isempty(dataDir)
    projectRoot = fileparts(fileparts(mfilename('fullpath')));
    dataDir = fullfile(projectRoot, 'collected_data');
end

d18OFile = fullfile(dataDir, 'd18O.xlsx');
temperatureFile = fullfile(dataDir, 'clump_temp.xlsx');
if ~isfile(d18OFile) || ~isfile(temperatureFile)
    error('Missing collected-data input file in %s.', dataDir);
end

d18O = readtable(d18OFile, 'TextType', 'string', ...
    'VariableNamingRule', 'preserve');
temperature = readtable(temperatureFile, 'TextType', 'string', ...
    'VariableNamingRule', 'preserve');

requiredD18O = ["d18O_carbonate", "Uncertainty_2sigma", "Type", ...
    "longitude", "latitude", "Source", "doi"];
requiredTemperature = ["d47_temperature_carbonate", ...
    "Uncertainty_2sigma", "longitude", "latitude", "Source"];
checkColumns(d18O, requiredD18O, 'carbonate d18O');
checkColumns(temperature, requiredTemperature, 'clumped temperature');

n = height(d18O);
matchedTemperatureRow = nan(n, 1);
temperatureC = nan(n, 1);
temperature2SigmaC = nan(n, 1);
carbonateVSMOW = nan(n, 1);
alphaCalciteWater = nan(n, 1);
waterVSMOW = nan(n, 1);
water1Sigma = nan(n, 1);
water2Sigma = nan(n, 1);
status = repmat("unpaired_no_colocated_temperature", n, 1);

coordinateToleranceDegrees = 1e-6;
for i = 1:n
    distance = hypot(temperature.longitude - d18O.longitude(i), ...
        temperature.latitude - d18O.latitude(i));
    sameSource = strcmpi(strtrim(temperature.Source), ...
        strtrim(d18O.Source(i)));
    match = find(distance <= coordinateToleranceDegrees & sameSource);
    if isempty(match)
        continue
    end
    if numel(match) > 1
        error('Multiple clumped-temperature matches for d18O row %d.', i);
    end

    j = match(1);
    matchedTemperatureRow(i) = j;
    temperatureC(i) = temperature.d47_temperature_carbonate(j);
    temperature2SigmaC(i) = temperature.Uncertainty_2sigma(j);

    [waterVSMOW(i), alphaCalciteWater(i), derivativeTemperature, ...
        derivativeVPDB] = kimONeil1997CalciteWater( ...
        d18O.d18O_carbonate(i), temperatureC(i), ...
        "carbonate_vpdb_to_water_vsmow");
    carbonateVSMOW(i) = 1.03091 .* d18O.d18O_carbonate(i) + 30.91;

    sigmaCarbonateVPDB = d18O.Uncertainty_2sigma(i) ./ 2;
    sigmaTemperatureC = temperature2SigmaC(i) ./ 2;
    water1Sigma(i) = hypot(derivativeVPDB .* sigmaCarbonateVPDB, ...
        derivativeTemperature .* sigmaTemperatureC);
    water2Sigma(i) = 2 .* water1Sigma(i);
    status(i) = "converted_assuming_equilibrium_calcite";
end

result = table((1:n)', matchedTemperatureRow, d18O.longitude, d18O.latitude, ...
    d18O.Source, d18O.doi, d18O.Type, d18O.d18O_carbonate, ...
    d18O.Uncertainty_2sigma, temperatureC, temperature2SigmaC, ...
    carbonateVSMOW, alphaCalciteWater, waterVSMOW, water1Sigma, ...
    water2Sigma, status, ...
    'VariableNames', {'d18O_row', 'clumped_temperature_row', 'longitude', ...
    'latitude', 'Source', 'doi', 'Type', ...
    'd18O_carbonate_VPDB_permil', 'd18O_carbonate_2sigma_permil', ...
    'clumped_temperature_C', 'clumped_temperature_2sigma_C', ...
    'd18O_carbonate_VSMOW_permil', 'alpha_calcite_water', ...
    'd18O_water_VSMOW_permil', 'd18O_water_1sigma_permil', ...
    'd18O_water_2sigma_permil', 'conversion_status'});

csvFile = fullfile(dataDir, 'd18O_water_KimONeil1997.csv');
xlsxFile = fullfile(dataDir, 'd18O_water_KimONeil1997.xlsx');
writetable(result, csvFile);
writetable(result, xlsxFile, 'Sheet', 'converted');

fprintf('Converted %d of %d carbonate d18O rows.\n', ...
    sum(isfinite(waterVSMOW)), n);
fprintf('CSV: %s\n', csvFile);
fprintf('XLSX: %s\n', xlsxFile);
end

function checkColumns(T, required, label)
missing = setdiff(required, string(T.Properties.VariableNames));
if ~isempty(missing)
    error('Missing %s column(s): %s', label, strjoin(missing, ', '));
end
end
