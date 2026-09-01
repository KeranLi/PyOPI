function out = opiCompare_AridityCases(modernResultFile, ancientResultFiles, varargin)
% Compare Relative P/PET for a modern and two ancient OPI result files.
% Produces 3 Relative P/PET maps, plus difference and ratio maps for each
% ancient case (seven PNG figures total).

p = inputParser;
addRequired(p, 'modernResultFile', @(x) ischar(x) || isstring(x));
addRequired(p, 'ancientResultFiles', @(x) iscell(x) && numel(x)==2);
addParameter(p, 'OutputDir', '', @(x) ischar(x) || isstring(x));
addParameter(p, 'PrecipitationScale', 0.03, @(x) isnumeric(x)&&isscalar(x)&&x>0);
addParameter(p, 'RelativeCLim', [0 8], @(x) isnumeric(x)&&numel(x)==2&&x(2)>x(1));
addParameter(p, 'DifferenceCLim', [-6 6], @(x) isnumeric(x)&&numel(x)==2&&x(2)>x(1));
addParameter(p, 'RatioCLim', [0 10], @(x) isnumeric(x)&&numel(x)==2&&x(2)>x(1));
parse(p, modernResultFile, ancientResultFiles, varargin{:});
o = p.Results;
if isempty(o.OutputDir)
    o.OutputDir = fullfile(fileparts(char(o.modernResultFile)), 'aridity_comparison');
end
if ~isfolder(o.OutputDir), mkdir(o.OutputDir); end

files = [{char(o.modernResultFile)}, cellfun(@char, o.ancientResultFiles, 'UniformOutput', false)];
names = {'modern_Tibet','Q4_V35_G4','Q4_V25_G4'};
R = cell(1,3);
for k=1:3
    R{k} = opiEstimate_Aridity(files{k}, 'OutputDir', fullfile(o.OutputDir,names{k}), ...
        'PrecipitationScales', o.PrecipitationScale, 'MakeFigure', false);
end

% Use one common modern denominator for cross-case comparison. The
% within-case aridityRelative fields are intentionally not comparable.
for k=1:3
    [~,idxScale] = min(abs(R{k}.precipitationScales - o.PrecipitationScale));
    R{k}.comparisonRelative = R{k}.aridityAbsolute(:,:,idxScale);
end
modernValid = isfinite(R{1}.comparisonRelative);
modernDenominator = median(R{1}.comparisonRelative(modernValid));
for k=1:3
    R{k}.comparisonRelative = R{k}.comparisonRelative ./ modernDenominator;
end

% Require identical grids so differences are spatially meaningful.
for k=2:3
    if ~isequal(R{1}.lon, R{k}.lon) || ~isequal(R{1}.lat, R{k}.lat)
        error('Modern and ancient results must use identical lon/lat grids.');
    end
end
modern = R{1}.comparisonRelative;
for k=1:3
    writeMap(R{k}.lon, R{k}.lat, R{k}.comparisonRelative, ...
        sprintf('%s_Relative_P_PET.png', names{k}), o.OutputDir, 'Relative P/PET', o.RelativeCLim);
end

summary = table(string(names(:)), nan(3,1), nan(3,1), ...
    'VariableNames', {'case','medianRelativeP_PET','validCells'});
for k=1:3
    x=R{k}.comparisonRelative; summary.medianRelativeP_PET(k)=median(x(isfinite(x))); summary.validCells(k)=sum(isfinite(x),'all');
end
for k=2:3
    d=R{k}.comparisonRelative-modern; q=R{k}.comparisonRelative./modern;
    d(~isfinite(d))=nan; q(~isfinite(q))=nan;
    writeMap(R{k}.lon,R{k}.lat,d,sprintf('%s_minus_modern.png',names{k}),o.OutputDir,'Ancient minus modern Relative P/PET', o.DifferenceCLim);
    writeMap(R{k}.lon,R{k}.lat,q,sprintf('%s_div_modern.png',names{k}),o.OutputDir,'Ancient / modern Relative P/PET', o.RatioCLim);
end
writetable(summary, fullfile(o.OutputDir,'relative_p_pet_summary.csv'));
out = struct('outputDir',string(o.OutputDir),'summary',summary,'results',{R});
fprintf('Wrote seven maps and summary to %s\n',o.OutputDir);
end

function writeMap(lon,lat,z,fileName,outDir,ttl,clim)
f=figure('Visible','off','Color','w'); imagesc(lon,lat,z); set(gca,'YDir','normal'); axis tight; caxis(clim); colorbar; xlabel('Longitude'); ylabel('Latitude'); title(ttl); exportgraphics(f,fullfile(outDir,fileName),'Resolution',180); close(f);
end
