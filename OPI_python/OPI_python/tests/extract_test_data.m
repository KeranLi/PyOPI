% extract_test_data.m
% 从 MATLAB OPI 运行中提取测试数据，用于验证 Python 实现
% 运行此脚本需要先将工作目录切换到 OPI programs 文件夹
%
% 使用方法:
% 1. 在 MATLAB 中打开 OPI programs 文件夹
% 2. 运行此脚本
% 3. 选择示例运行文件（如 OPI Example Gaussian Mountain Range/runFileOneWindExample.run）
% 4. 脚本将保存测试数据到 OPI_python/tests/matlab_reference_data/

function extract_test_data()
    % 创建输出目录
    outputDir = fullfile(fileparts(mfilename('fullpath')), 'matlab_reference_data');
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end
    
    fprintf('========================================\n');
    fprintf('OPI Test Data Extraction\n');
    fprintf('========================================\n\n');
    
    % 获取运行文件
    try
        [runPath, runName] = getRunFile();
    catch ME
        fprintf('Error: %s\n', ME.message);
        fprintf('请确保您在 OPI programs 文件夹中运行此脚本\n');
        return;
    end
    
    fprintf('运行文件: %s\n', fullfile(runPath, runName));
    
    % 加载运行文件数据
    runData = getRunFile(fullfile(runPath, runName));
    fprintf('运行标题: %s\n', runData.runTitle);
    
    % 加载输入数据
    fprintf('\n加载输入数据...\n');
    inputData = getInput(runData.runPath, runData.topoFile, runData.rTukey, ...
        runData.sampleFile, runData.sdResRatio);
    
    % 保存输入数据
    fprintf('保存输入数据...\n');
    save(fullfile(outputDir, 'input_data.mat'), '-struct', 'inputData');
    
    % 提取关键变量
    x = inputData.x;
    y = inputData.y;
    hGrid = inputData.hGrid;
    lat = inputData.lat;
    lat0 = inputData.lat0;
    fC = inputData.fC;
    hR = inputData.hR;
    
    % 使用合成参数（与 Python 示例一致）
    beta = [10.0; 90.0; 290.0; 0.25; 0.0; 1000.0; -5.0e-3; -2.0e-3; 0.7];
    U = beta(1);
    azimuth = beta(2);
    T0 = beta(3);
    M = beta(4);
    kappa = beta(5);
    tauC = beta(6);
    d2H0 = beta(7);
    dD2H0_dLat = beta(8);
    fP0 = beta(9);
    
    % 计算浮力频率 NM
    NM = 0.01; % 典型值，约 0.01 rad/s
    
    fprintf('\n========================================\n');
    fprintf('测试参数:\n');
    fprintf('  U = %.2f m/s\n', U);
    fprintf('  azimuth = %.2f degrees\n', azimuth);
    fprintf('  T0 = %.2f K\n', T0);
    fprintf('  NM = %.4f rad/s\n', NM);
    fprintf('  fC = %.6f rad/s\n', fC);
    fprintf('  hRho = %.2f m\n', hR);
    fprintf('========================================\n\n');
    
    %% 测试 1: windGrid
    fprintf('测试 1: windGrid...\n');
    [Sxy, Txy, s, t, Xst, Yst] = windGrid(x, y, azimuth);
    save(fullfile(outputDir, 'test_windgrid.mat'), 'Sxy', 'Txy', 's', 't', 'Xst', 'Yst', 'x', 'y', 'azimuth');
    fprintf('  完成: s 向量长度 = %d, t 向量长度 = %d\n', length(s), length(t));
    
    %% 测试 2: fourierSolution
    fprintf('测试 2: fourierSolution...\n');
    [s_f, t_f, Sxy_f, Txy_f, hWind, kS, kT, hHat, kZ] = ...
        fourierSolution(x, y, hGrid, U, azimuth, NM, fC, hR);
    save(fullfile(outputDir, 'test_fourier.mat'), 's_f', 't_f', 'Sxy_f', 'Txy_f', ...
        'hWind', 'kS', 'kT', 'hHat', 'kZ', 'x', 'y', 'hGrid', 'U', 'azimuth', 'NM', 'fC', 'hR');
    fprintf('  完成: hHat 大小 = %s, kZ 大小 = %s\n', mat2str(size(hHat)), mat2str(size(kZ)));
    
    %% 测试 3: baseState
    fprintf('测试 3: baseState...\n');
    [zBar, T, gammaEnv, gammaSat, gammaRatio, rhoS0, hS, rho0, hRho] = baseState(NM, T0);
    save(fullfile(outputDir, 'test_basestate.mat'), 'zBar', 'T', 'gammaEnv', 'gammaSat', ...
        'gammaRatio', 'rhoS0', 'hS', 'rho0', 'hRho', 'NM', 'T0');
    fprintf('  完成: zBar 范围 = %.0f 到 %.0f m\n', min(zBar), max(zBar));
    
    %% 测试 4: precipitationGrid (简化参数)
    fprintf('测试 4: precipitationGrid...\n');
    % 注意: precipitationGrid 需要很多参数，这里使用简化版本
    % 实际测试时需要完整的 calc_one_wind 调用
    
    %% 保存参数摘要
    fprintf('\n保存参数摘要...\n');
    params.U = U;
    params.azimuth = azimuth;
    params.T0 = T0;
    params.M = M;
    params.kappa = kappa;
    params.tauC = tauC;
    params.d2H0 = d2H0;
    params.dD2H0_dLat = dD2H0_dLat;
    params.fP0 = fP0;
    params.NM = NM;
    params.fC = fC;
    params.hR = hR;
    params.beta = beta;
    save(fullfile(outputDir, 'test_params.mat'), '-struct', 'params');
    
    fprintf('\n========================================\n');
    fprintf('测试数据提取完成!\n');
    fprintf('输出目录: %s\n', outputDir);
    fprintf('========================================\n');
    
    % 列出保存的文件
    files = dir(fullfile(outputDir, '*.mat'));
    fprintf('\n保存的文件:\n');
    for i = 1:length(files)
        fprintf('  - %s (%.2f KB)\n', files(i).name, files(i).bytes/1024);
    end
end

% 如果直接运行此脚本
if ~isdeployed && ~exist('extract_test_data', 'file')
    extract_test_data();
end
