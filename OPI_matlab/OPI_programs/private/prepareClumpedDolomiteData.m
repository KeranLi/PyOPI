function clumped = prepareClumpedDolomiteData(clumpedFile, sampleLon, sampleLat)
% prepareClumpedDolomiteData reads clumped data and maps rows to OPI samples.

clumped = readtable(clumpedFile, 'TextType', 'string', 'VariableNamingRule', 'preserve');
required = ["sample_index", "lon", "lat", "T_clumped_C", "sigma_T_C"];
missing = setdiff(required, string(clumped.Properties.VariableNames));
if ~isempty(missing)
    error('Missing clumped column(s): %s', strjoin(missing, ', '));
end

sampleIndex = str2double(string(clumped.sample_index));
if all(isfinite(sampleIndex))
    sampleIndex = round(sampleIndex);
else
    sampleIndex = nan(height(clumped), 1);
    for i = 1:height(clumped)
        d = hypot(sampleLon(:) - clumped.lon(i), sampleLat(:) - clumped.lat(i));
        [~, sampleIndex(i)] = min(d);
    end
end

if any(sampleIndex < 1 | sampleIndex > numel(sampleLon))
    error('Clumped sample mapping produced invalid OPI sample indices.');
end
clumped.OPI_sample_index = sampleIndex;

end
