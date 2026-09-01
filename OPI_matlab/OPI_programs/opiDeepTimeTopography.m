function opiDeepTimeTopography
% opiDeepTimeTopography - 生成深时（Deep Time）古地貌
%
% 结合构造抬升、气候侵蚀、海平面变化生成地质历史时期的古地貌
%
% 输出: 可直接用于OPI计算的 .mat 地形文件
%
% 作者: AI Assistant
% 日期: 2026

%% 初始化
close all; clc;
dbstop if error

%% ============ 1. 定义模拟参数 ============
% 地理范围（以某点为中心，单位：度）
lon0 = 90;      % 中心经度（例如：青藏高原）
lat0 = 30;      % 中心纬度
Lx = 10;        % 经度范围（°）
Ly = 8;         % 纬度范围（°）

% 网格分辨率
dLon = 0.02;    % 经度分辨率（约2km在30°N）
dLat = 0.02;    % 纬度分辨率

% 地质时间（Ma = 百万年前）
targetAge = 50;  % 目标年代：50 Ma（始新世）

%% ============ 2. 创建坐标网格 ============
lon = linspace(lon0 - Lx/2, lon0 + Lx/2, ceil(Lx/dLon) + 1);
lat = linspace(lat0 - Ly/2, lat0 + Ly/2, ceil(Ly/dLat) + 1);
[lonGrid, latGrid] = meshgrid(lon, lat);

% 转换为米制坐标（用于物理计算）
x = (lon - lon0) * 111320 * cosd(lat0);  % 1°经度 ≈ 111km * cos(lat)
y = (lat - lat0) * 110540;               % 1°纬度 ≈ 110km
[X, Y] = meshgrid(x, y);

fprintf('网格大小: %d x %d (%.1f km x %.1f km)\n', ...
    length(lon), length(lat), max(x)/1e3, max(y)/1e3);

%% ============ 3. 构造地貌模型 ============
% 方法A: 多期构造抬升（叠加多个高斯隆起）
hTectonic = zeros(size(X));

% 定义构造单元（可添加多个）
structures = [
    % [X中心(km), Y中心(km), 宽度X(km), 宽度Y(km), 高度(km), 起始时间(Ma), 结束时间(Ma)]
    0,      0,    200,   150,   5.0,   65,   45;   % 主隆起（65-45Ma）
    -150,   50,   100,   80,    3.0,   55,   40;   % 次级隆起
    100,   -80,   120,   100,   2.5,   50,   30;   % 第三期
];

for i = 1:size(structures, 1)
    s = structures(i, :);
    % 计算该时期抬升量（线性插值）
    if targetAge <= s(6) && targetAge >= s(7)
        upliftRatio = (s(6) - targetAge) / (s(6) - s(7));
    elseif targetAge < s(7)
        upliftRatio = 1;  % 已完成抬升
    else
        upliftRatio = 0;  % 尚未开始
    end
    
    % 高斯形隆起
    hLocal = s(5) * upliftRatio * exp(-((X-s(1)*1e3).^2)/(2*(s(3)*1e3)^2) ...
                                      - ((Y-s(2)*1e3).^2)/(2*(s(4)*1e3)^2));
    hTectonic = hTectonic + hLocal;
end

%% ============ 4. 侵蚀模型（可选）============
% 简单的坡度依赖侵蚀
applyErosion = true;

if applyErosion
    % 计算坡度
    [dhdx, dhdy] = gradient(hTectonic, dLon*111e3, dLat*110e3);
    slope = sqrt(dhdx.^2 + dhdy.^2);
    
    % 侵蚀系数（与坡度、时间相关）
    K_erosion = 5e-6;  % 侵蚀系数 (m/yr)
    erosionTime = 20e6;  % 侵蚀持续时间 (yr)
    
    % 计算侵蚀量
    erosion = K_erosion * slope * erosionTime;
    hEroded = hTectonic - erosion;
    hEroded(hEroded < 0) = 0;  % 不低于海平面
else
    hEroded = hTectonic;
end

%% ============ 5. 添加细节地形（分形噪声）============
addFractalNoise = true;

if addFractalNoise
    % 生成多尺度分形噪声
    noise = zeros(size(X));
    amplitudes = [500, 200, 50];  % 不同尺度的振幅（米）
    wavelengths = [50, 10, 2];    % 波长（km）
    
    for i = 1:length(amplitudes)
        % 使用正弦波组合模拟地形起伏
        kx = 2*pi / (wavelengths(i) * 1e3);
        ky = 2*pi / (wavelengths(i) * 1e3);
        phase = rand() * 2*pi;
        noise = noise + amplitudes(i) * sin(kx*X + ky*Y + phase);
    end
    
    hFinal = hEroded + noise;
else
    hFinal = hEroded;
end

% 确保无负值（高于海平面）
hFinal(hFinal < 0) = 0;

%% ============ 6. 可视化 ============
figure('Position', [100 100 1200 400]);

subplot(1,3,1);
pcolor(lon, lat, hTectonic/1e3);
shading interp; colorbar;
title(sprintf('构造地貌 (%.0f Ma)', targetAge));
xlabel('经度 (°)'); ylabel('纬度 (°)');
clabel = colorbar; ylabel(clabel, '高程 (km)');

subplot(1,3,2);
pcolor(lon, lat, hEroded/1e3);
shading interp; colorbar;
title('侵蚀后地貌');
xlabel('经度 (°)'); ylabel('纬度 (°)');
clabel = colorbar; ylabel(clabel, '高程 (km)');

subplot(1,3,3);
pcolor(lon, lat, hFinal/1e3);
shading interp; colorbar;
title('最终古地貌（含噪声）');
xlabel('经度 (°)'); ylabel('纬度 (°)');
clabel = colorbar; ylabel(clabel, '高程 (km)');

%% ============ 7. 保存为OPI可用格式 ============
hGrid = hFinal;  % 重命名为OPI标准变量名

% 确保格式符合gridRead要求
% lon: 行向量, lat: 列向量, hGrid: (lat, lon)
if size(hGrid, 1) ~= length(lat) || size(hGrid, 2) ~= length(lon)
    error('网格维度不匹配！');
end

% 保存
outputName = sprintf('DeepTime_Topo_%dMa_%.1fE_%.1fN.mat', ...
    targetAge, lon0, lat0);

save(['../data/' outputName], 'lon', 'lat', 'hGrid', '-v7.3');

fprintf('\n=== 保存成功 ===\n');
fprintf('文件名: %s\n', outputName);
fprintf('最大高程: %.1f m\n', max(hGrid(:)));
fprintf('平均高程: %.1f m\n', mean(hGrid(:)));
fprintf('网格点数: %d x %d\n', length(lon), length(lat));

%% ============ 8. 生成配套的 .run 文件模板 ============
runFileName = sprintf('template_%dMa.run', targetAge);
fid = fopen(['../runs/' runFileName], 'w');

fprintf(fid, '%% OPI Run File for Deep-Time Topography\n');
fprintf(fid, '%% Generated for %.0f Ma paleogeography\n\n', targetAge);

fprintf(fid, 'runTitle = Paleotopography %.0f Ma\n', targetAge);
fprintf(fid, 'runPath = ../runs/\n');
fprintf(fid, 'dataPath = ../data/\n');
fprintf(fid, 'topoFile = %s\n', outputName);
fprintf(fid, 'sampleFile = \n');  % 需要用户添加
fprintf(fid, 'rTukey = 0.0\n');
fprintf(fid, 'lon0 = %.2f\n', lon0);
fprintf(fid, 'lat0 = %.2f\n', lat0);

fclose(fid);

fprintf('Run模板: %s\n', runFileName);

end
