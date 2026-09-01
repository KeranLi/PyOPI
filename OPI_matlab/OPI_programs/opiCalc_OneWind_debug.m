function opiCalc_OneWind_debug(runFile)
% 调试版本 - 在 catchmentNodes 调用前停止

%% Initialize
close all; clc;
if nargin < 1
    runFile = 'Tibet_Eocene_30Ma_OneWind.run';
end

%% Constants
TC2K = 273.15;
startTimeOpiCalc = datetime; 
radiusEarth = 6371e3; 
mPerDegree = pi*radiusEarth/180;
hR = 540;
sdResRatio = 28.3;

%% Get run file information
if nargin < 1
    [runPath, runFile, runTitle, ~, dataPath, ...
        topoFile, rTukey, sampleFile, contDivideFile, ~, ...
        mapLimits, sectionLon0, sectionLat0, ~, ~, ...
        ~, ~, lB, uB, beta] ...
        = getRunFile;
else
    [runPath, runFile, runTitle, ~, dataPath, ...
        topoFile, rTukey, sampleFile, contDivideFile, ~, ...
        mapLimits, sectionLon0, sectionLat0, ~, ~, ...
        ~, ~, lB, uB, beta] ...
        = getRunFile(runFile);
end

if isempty(beta)
    error('opiCalc requires that the run file include an OPI solution at the end of the run file.')
end
if length(beta)~=9
    error('Number of parameters is incorrect for this program')
end

%% Get input data
[lon, lat, x, y, hGrid, lon0, lat0, ...
    sampleLine, sampleLon, sampleLat, sampleX, sampleY, ...
    sampleD2H, sampleD18O, sampleDExcess, sampleLC, ...
    sampleLineAlt, sampleLonAlt, sampleLatAlt, sampleXAlt, sampleYAlt, ...
    sampleD2HAlt, sampleD18OAlt, sampleDExcessAlt, sampleLCAlt, ...
    bMWLSample, sdDataMin, sdDataMax, cov, fC] ...
    = getInput(dataPath, topoFile, rTukey, sampleFile, sdResRatio);

%% DEBUG: 检查变量
fprintf('=== DEBUG INFO ===\n');
fprintf('样品数量: %d\n', length(sampleX));
fprintf('样品X范围: %.2f to %.2f\n', min(sampleX), max(sampleX));
fprintf('样品Y范围: %.2f to %.2f\n', min(sampleY), max(sampleY));
fprintf('网格X范围: %.2f to %.2f\n', min(x), max(x));
fprintf('网格Y范围: %.2f to %.2f\n', min(y), max(y));
fprintf('样品LC: %s\n', sampleLC);
fprintf('==================\n');

[m, n] = size(hGrid);
fprintf('网格大小: %d x %d\n', m, n);

% 测试第一个样品的插值
for k = 1:min(3, length(sampleX))
    rowIdx = round(interp1(y, 1:m, sampleY(k), 'linear', 'extrap'));
    colIdx = round(interp1(x, 1:n, sampleX(k), 'linear', 'extrap'));
    fprintf('样品 %d: Y=%.2f -> row=%.1f, X=%.2f -> col=%.1f\n', ...
        k, sampleY(k), rowIdx, sampleX(k), colIdx);
    rowIdx = max(1, min(m, rowIdx));
    colIdx = max(1, min(n, colIdx));
    fprintf('  限制后: row=%d, col=%d\n', rowIdx, colIdx);
end

%% 保存变量到工作区供检查
assignin('base', 'debug_sampleX', sampleX);
assignin('base', 'debug_sampleY', sampleY);
assignin('base', 'debug_sampleLC', sampleLC);
assignin('base', 'debug_x', x);
assignin('base', 'debug_y', y);
assignin('base', 'debug_hGrid', hGrid);
assignin('base', 'debug_m', m);
assignin('base', 'debug_n', n);

fprintf('\n变量已保存到工作区，前缀为 debug_\n');
fprintf('现在可以手动测试: row = round(interp1(debug_y, 1:debug_m, debug_sampleY(1), ''linear'', ''extrap''))\n');

%% 尝试调用 catchmentNodes
fprintf('\n尝试调用 catchmentNodes...\n');
try
    [ijCatch, ptrCatch] = catchmentNodes(sampleX, sampleY, sampleLC, x, y, hGrid);
    fprintf('成功!\n');
catch ME
    fprintf('错误: %s\n', ME.message);
    fprintf('发生在: %s 第 %d 行\n', ME.stack(1).name, ME.stack(1).line);
end

end
