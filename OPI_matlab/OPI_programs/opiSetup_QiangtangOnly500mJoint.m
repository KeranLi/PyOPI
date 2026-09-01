function opiSetup_QiangtangOnly500mJoint()
% Create independent joint-fit cases that retain only the Qiangtang band.
%
% Each case preserves its existing terrain within 32.75--33.65 N and sets
% all other grid cells to 500 m.  The original topographies are unchanged.

scenarioRoot = fullfile(fileparts(mfilename('fullpath')), '..', 'scenarios', ...
    'Qiangtang_30Ma_4000m_dH045_T290_fP030_Valid_platform_northsouth', ...
    'complete_joint_simulation');
scenarios = {'Q4_V35_G4', 'Q4_V25_G4'};
qiangtangCenterLat = 33.20;
qiangtangHalfWidthDeg = 0.45;
backgroundElevationM = 500;

for i = 1:numel(scenarios)
    scenarioName = scenarios{i};
    sourceDir = fullfile(scenarioRoot, scenarioName);
    caseDir = fullfile(sourceDir, 'qiangtang_only_500m');
    if isfolder(caseDir)
        error('Refusing to overwrite existing case: %s', caseDir);
    end
    mkdir(caseDir);

    S = load(fullfile(sourceDir, 'Tibet_Eocene_30Ma_topo.mat'), ...
        'lon', 'lat', 'hGrid');
    keepRows = abs(S.lat(:) - qiangtangCenterLat) <= qiangtangHalfWidthDeg;
    originalHGrid = S.hGrid;
    S.hGrid(:) = backgroundElevationM;
    S.hGrid(keepRows, :) = originalHGrid(keepRows, :);
    lon = S.lon; %#ok<NASGU>
    lat = S.lat; %#ok<NASGU>
    hGrid = S.hGrid; %#ok<NASGU>
    save(fullfile(caseDir, 'Tibet_Eocene_30Ma_topo.mat'), 'lon', 'lat', 'hGrid', '-v7.3');

    copyfile(fullfile(sourceDir, 'Tibet_Eocene_30Ma_samples.xlsx'), caseDir);
    copyfile(fullfile(sourceDir, 'Tibet_Eocene_30Ma_topo_divide_main.mat'), caseDir);
    copyfile(fullfile(sourceDir, 'collected_proxy_observations.csv'), caseDir);
    copyfile(fullfile(sourceDir, 'collected_carbonate_assimilation_config.csv'), caseDir);
    copyfile(fullfile(sourceDir, 'proxy_clumped'), fullfile(caseDir, 'proxy_clumped'));

    runFile = fullfile(caseDir, sprintf( ...
        'Tibet_Eocene_30Ma_%s_QiangtangOnly500m_fit.run', scenarioName));
    writeNorthSourceRun(sourceDir, caseDir, runFile, scenarioName);
    writeReadme(caseDir, scenarioName, qiangtangCenterLat, ...
        qiangtangHalfWidthDeg, backgroundElevationM, keepRows, originalHGrid, hGrid);
end
end

function writeNorthSourceRun(sourceDir, caseDir, runFile, scenarioName)
template = fullfile(sourceDir, 'north_source_state2', ...
    sprintf('Tibet_Eocene_30Ma_%s_northsource_fit.run', scenarioName));
if ~isfile(template)
    error('North-source template is missing: %s', template);
end
lines = readlines(template, 'WhitespaceRule', 'preserve');
idx = activeLines(lines);
lines(idx(1)) = string(scenarioName) + ...
    " Qiangtang-only terrain (500 m outside) joint fit";
lines(idx(3)) = string(caseDir);
writelines(lines, runFile);
end

function writeReadme(caseDir, scenarioName, centerLat, halfWidthDeg, backgroundM, ...
    keepRows, originalHGrid, hGrid)
fid = fopen(fullfile(caseDir, 'README.txt'), 'w', 'native', 'UTF-8');
if fid == -1
    error('Could not write case README.');
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, 'Scenario: %s\n', scenarioName);
fprintf(fid, 'Terrain design: retain only the Qiangtang latitude band.\n');
fprintf(fid, 'Retained band: %.2f to %.2f deg N.\n', centerLat-halfWidthDeg, centerLat+halfWidthDeg);
fprintf(fid, 'All cells outside retained band: %.0f m.\n', backgroundM);
fprintf(fid, 'Transition: none; the requested change is a discrete terrain contrast.\n');
fprintf(fid, 'State 1 azimuth constraint: 150 to 180 deg (southern source).\n');
fprintf(fid, 'State 2 azimuth constraint: -45 to 0 deg (northwest through north).\n');
fprintf(fid, 'Retained rows: %d of %d.\n', sum(keepRows), numel(keepRows));
fprintf(fid, 'Original retained-band mean elevation: %.1f m.\n', ...
    mean(originalHGrid(keepRows, :), 'all', 'omitnan'));
fprintf(fid, 'Final grid minimum/maximum elevation: %.1f / %.1f m.\n', ...
    min(hGrid, [], 'all'), max(hGrid, [], 'all'));
end

function idx = activeLines(lines)
idx = [];
for i = 1:numel(lines)
    value = strip(string(lines(i)));
    if strlength(value) > 0 && ~startsWith(value, "%")
        idx(end+1) = i; %#ok<AGROW>
    end
end
if numel(idx) < 10
    error('Run file does not contain expected active lines.');
end
end
