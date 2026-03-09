OPI (Orographic Precipitation Isotopes) MATLAB 复现指南
=========================================================

项目简介
--------
本仓库包含 OPI 3.7 模型的 MATLAB 实现，用于模拟地形降水同位素分布。
支持单风场(OneWind)和双风场(TwoWinds)两种模式。

文件结构
--------
OPI-Orographic-Precipitation-and-Isotopes/
├── OPI programs/              # 主程序目录
│   ├── opiCalc_TwoWinds.m     # 核心计算程序
│   ├── opiMaps_TwoWinds.m     # 可视化程序
│   ├── opiCalc_OneWind.m      # 单风场计算
│   ├── opiMaps_OneWind.m      # 单风场可视化
│   └── private/               # 辅助函数
│       ├── getRunFile.m       # 配置文件读取
│       ├── getInput.m         # 数据输入处理
│       ├── isotopeGrid.m      # 同位素网格计算
│       ├── calc_TwoWinds.m    # 双风场计算核心
│       └── 其他物理计算函数
├── data/                      # 数据文件夹
│   ├── topography.mat         # 地形数据（DEM）
│   ├── samples.xlsx           # 水同位素样本
│   └── divide.mat             # 分水岭数据（双风场必需）
└── runs/                      # 运行结果文件夹
    └── run001/
        ├── run001.run         # 运行配置文件
        ├── opiCalc_TwoWinds_Results.mat  # 计算结果
        └── opiMaps_TwoWinds_Log.txt      # 运行日志

环境要求
--------
- MATLAB R2019b 或更高版本
- Mapping Toolbox（地理坐标转换）
- Optimization Toolbox（优化算法）
- 内存：至少 8GB RAM（推荐 16GB+）

关键修复说明（原版代码 Bug 修复）
-----------------------------------

1. getRunFile.m 修复
   位置: private/getRunFile.m 第 48 行附近
   问题: 字符串比较逻辑错误，且缺少并行选项处理
   修复:
   
   将字符串转换为数字（处理 '0' 或 '1' 字符串）
   parallelOption = str2double(str);
   
   检查有效性
   if parallelOption ~= 0 && parallelOption ~= 1
       error('Serial vs. parallel option must be either 0 or 1.')
   end

2. getInput.m 修复
   位置: private/getInput.m
   修复内容:
   
   第 50 行: 文件检查逻辑修正 ~isfile(...)
   第 69 行: 保留所有列 X(iData,:)（不要删除第一列）
   第 82-99 行: Excel 数据读取改为兼容数字和文本格式：
   
   sampleLon = str2double(string(X(:,1)));
   sampleLat = str2double(string(X(:,2)));
   sampleElev = str2double(string(X(:,3)));
   sampleD2H = str2double(string(X(:,4)))*1e-3;
   sampleD18O = str2double(string(X(:,5)))*1e-3;
   sampleLC = char(upper(string(X(:,6))));

3. isotopeGrid.m 修复
   位置: private/isotopeGrid.m 第 197 行
   问题: 纬度向量维度不匹配（行向量 vs 列向量）
   修复:
   
   确保 lat 是列向量
   if isrow(lat)
       lat = lat.';
   end
   d2HGrid = (1 + d2H0 + dDH0dLat*(abs(lat) - abs(lat0))).*F(Sxy, Txy) - 1;

快速开始
--------

1. 准备数据

   地形数据 (topography.mat)
   必须包含以下变量：
   - lon: [1×nx] 经度向量（如 56.5°E - 114.5°E）
   - lat: [1×ny] 纬度向量（如 11°N - 43°N）
   - hGrid: [ny×nx] 高程矩阵（米）

   样本数据 (samples.xlsx)
   Excel 文件，包含 6 列：
   1. Longitude: 经度（度）
   2. Latitude: 纬度（度）
   3. Elevation: 海拔（米）
   4. d2H: 氢同位素（‰，如 -100）
   5. d18O: 氧同位素（‰，如 -15）
   6. Type: 类型（'L'=Local, 'C'=Catchment, 'A'=Altered）

   分水岭数据 (divide.mat) - 双风场必需
   必须包含：
   - contDivideLon: [n×1] 分水岭经度（列向量）
   - contDivideLat: [n×1] 分水岭纬度（列向量）
   
   注意: 分水岭线必须与地图边界有至少 2 个交点。
   简单对角线示例：
   
   contDivideLon = [56.5; 114.5];  % 列向量
   contDivideLat = [43; 11];       % 列向量
   save('divide.mat', 'contDivideLon', 'contDivideLat');

2. 创建运行配置文件 (.run)

   文件名如 run001.run，格式如下（严格按行顺序）：
   
   第1行: 运行标题（任意字符串）
   Himalaya Test Run 001
   
   第2行: 并行选项（0=串行，1=并行）
   0
   
   第3-4行: 数据路径（两个路径，第二个可为 no）
   F:\...\OPI-Orographic-Precipitation-and-Isotopes\data
   no
   
   第5行: 地形文件名
   topography.mat
   
   第6行: Tukey窗比例（0-1，推荐0.5）
   0.5
   
   第7行: 样本文件名（no表示无样本）
   samples.xlsx
   
   第8行: 分水岭文件名（双风场必需，不能为空）
   divide.mat
   
   第9行: 地图范围 [minLon, maxLon, minLat, maxLat]
   77.5, 97.5, 20.0, 36.0
   
   第10行: 剖面原点 [lon, lat] 或 map
   90.0, 29.0
   
   第11行: CRS参数 [mu, epsilon0]
   20, 1e-4
   
   第12行: 重启文件名（no表示无）
   no
   
   第13行: 参数标签（19个，用|分隔，双风场）
   U|Azimuth|T0|M|kappa|tauC|d2H0|dLat|fP|fraction|U2|Az2|T02|M2|kappa2|tauC2|d2H0_2|dLat2|fP2
   
   第14行: 指数缩放（19个0）
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
   
   第15行: 下限 lB（19个值）
   0.1, -90, 270, 0, 0, 0, -0.060, -0.015, 0, 0, 0.1, 0, 270, 0, 0, 0, -0.060, -0.015, 0
   
   第16行: 上限 uB（19个值）
   20, 90, 300, 1, 1e6, 2500, 0, 0, 1, 1, 20, 180, 300, 1, 1e6, 2500, 0, 0, 1
   
   第17行: 初始猜测值 beta（19个值，可选但建议提供）
   6.6, 33.8, 293.5, 0.34, 233368, 1296, -0.0026, -0.000533, 1.0, 0.52, 16.0, 125.7, 294.0, 0.693, 722120, 2450, -0.0001, -0.000007, 1.0

3. 运行模拟

   在MATLAB命令窗口：
   
   设置路径
   addpath('F:\...\OPI programs');
   addpath('F:\...\OPI programs\private');
   
   运行核心计算
   opiCalc_TwoWinds
   % 选择 run001.run 文件，等待计算完成（显示迭代信息）
   
   运行可视化（生成20个PDF图）
   opiMaps_TwoWinds
   % 选择生成的 opiCalc_TwoWinds_Results.mat 文件

参数说明（双风场）
------------------
beta 向量（19个参数）：
- 1-9: 状态1参数 [U, Azimuth, T0, M, kappa, tauC, d2H0, dLat, fP]
- 10: 状态1占比 fraction (0-1)
- 11-19: 状态2参数 [U2, Az2, T02, M2, kappa2, tauC2, d2H0_2, dLat2, fP2]

物理意义:
- U: 风速 (m/s)
- Azimuth: 风向角度 (度，0=北，90=东)
- T0: 海平面温度 (K)
- M: 山高度数 (无量纲，<1为线性区)
- kappa: 水平涡动扩散系数 (m²/s)
- tauC: 云水停留时间 (s)
- d2H0: 基础降水δ²H (小数，如-0.0026 = -2.6‰)
- dLat: δ²H纬度梯度 (‰/度)
- fP: 蒸发残留比例 (0-1)

常见问题
--------
1. "Sample file not found"
   检查 getInput.m 第50行是否已修复为：
   if ~isfile([dataPath, '/', sampleFile]), ...

2. "Continental divide polyline has less than 2 intersections"
   确保分水岭线与地图边界相交。使用简单对角线：
   contDivideLon = [minLon; maxLon];
   contDivideLat = [maxLat; minLat];

3. "Array dimensions mismatch" in isotopeGrid
   确保 lat 是列向量（ny×1），修复见上文。

4. 内存不足
   减小地形网格尺寸或增加虚拟内存。

输出结果
--------
运行完成后生成：
- opiCalc_TwoWinds_Results.mat: 包含所有网格结果和参数
- opiMaps_TwoWinds_Log.txt: 运行日志
- Fig01-20.pdf: 20张可视化图表（地形、降水、同位素、流线等）

引用
----
如需引用 OPI 模型，请参考：
- Brandon, M.T. (2022). Orographic Precipitation Isotopes (OPI) model v3.7
- Smith, R.B. & Barstad, I. (2004). Linear theory of orographic precipitation

最后更新
--------
2026-03-09
复现状态: 成功运行（MATLAB R2023a）