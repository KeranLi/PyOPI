function [d2HGrid, d18OGrid, ....
    evapD2HGrid, uEvapD2HGrid, evapD18OGrid, uEvapD18OGrid] = ...
    isotopeGrid( ...
    s, t, Sxy, Txy, lat, lat0, hWind, fMWind, rHWind, fPWind, ...
    z223Wind, z258Wind, tauF, ...
    U, T, gammaSat, hS, hR, d2H0, d18O0, dDH0dLat, dD18O0_dLat, isFit)
% isotopeGrid：根据温度、地形抬升、降水下落和水汽损失，
% 计算风向网格及地理网格上的 d2H 和 d18O 降水同位素组成。
% d2H0、d18O0 是参考纬度 lat0 处的基础降水同位素组成；
% dDH0dLat、dD18O0_dLat 是随绝对纬度变化的区域梯度。
% Calculate isotope grid as a function of temperature, and the dlnFM_dS 
% grid, which is the log derivative of the precipitation field in 
% the wind direction.
% The calculation includes three steps:
% 1) The fractionation associated with the creation and fall of the 
% precipitation is determined by a weighted vertical average of the
% fractionation factor along the fall path, where the weighting accounts
% for the production rate of precipitation as a function of height. 
% This vertical averaging is done along a fall path, and starts
% with calculation the fractionation factor as a function of temperature and 
% phase (ice, water) along the fall path. This initial fractionation factor
% is adjusted to account for isotopic resetting of rain drops with vapor 
% as they fall from the freezing surface to the ground.
% 2) A fraction of the fallen precipitation, equal to (1-fP), is returned
% back to the atmosphere as water vaport produced by evaporation, where
% fP fraction of the precipitation that leaves through the base of the
% model. fP is assumed to a constant that applies across the model domain.
% 3) The vertical averaged fractionation factors are then integrated
% along wind path to get the isotopic composition of the precipitation
% and the evaporated vapor at the base of the model.
% The calculation accounts for ice and water, as determined by the 
% freezing surface is set to 258 K, which is the midpoint of 
% the 268 - 248 K range used by Cias and Jouzel (1994) to represent 
% the Bergeron-Findeisen zone. Temperatures are calculated using the
% LTOP solution. 
% The regional variation in isotopic composition is defined by a
% "regional" composition defined by a linear relationship in latitude,
% relative to the sample centroid. Thus, the final calculated 
% precipitation isotope fields are a sum of regional and orographic 
% contributions. 

% Mark Brandon, Yale University, 2016-2020

%% Initialize system
warning('off', 'MATLAB:griddedInterpolant:MeshgridEval2DWarnId');

%% Initialize variables
%... Shear for bringing fall path to vertical, where the shear ratio
% is the horizontal over vertical distances for fall of precipitation.
% tauF是降水平均下落时间，乘以风速U表示沿着风向移动的距离
% 除以水汽密度尺度高度表示水汽相对于纵向移动比值
shear = U*tauF/hS;
% U*tauF 是降水下落期间被风水平输送的距离；shear 用于把倾斜下落路径拉直。
%... Constants for fractionation due to evaporation
% Diffusivity ratios from Merlivat, 1978, with the rare isotopologue
% in the numerator
DRatio2H = 0.9755;
% D原子相对于H原子的扩散能力
DRatio18O = 0.9723;
% 18O原子相对于16O原子的扩散能力
% Exponent, n, for fractionation due to evaporation.
% Recommended value: n = 1
% This exponent has a potential range from about 0.5 to 1. 
% My experience with OPI is that specific values in this range 
% have little influence on the best-fit solution. 
% Consider as well the following published estimates:
% Stewart, 1975 estimates n = 0.58 for evaporation of falling water drops. 
% Criss, 1999, p. 175 recommends n = 1 for evaporation from soils.
n = 1;
% n 是蒸发动力学分馏指数；这里采用推荐值 1。

%% Set up wind-direction grid
%... Parameters for wind grid for topographic data
dS = s(2)-s(1); % 沿风向的网格间距，单位 m，用于后续沿风向积分
[~, nT] = size(hWind); % 获得横风向网格点数量

% Horizontal shear needed to transform to vertical paths to land surface
sSurfaceShearWind = s + shear*hWind;
% sSurfaceShearWind：每个地面网格点在剪切坐标中的沿风向位置。
% 把空气沿横向移动的距离投影到风向上

%... Construct grid for temperature at land surface
% TLSWind：风向网格中的地表温度，单位 K。
% 当前模型使用海平面处的 gammaSat(1) 作为全高度范围的常数近似：
% 地形越高，估算的地表温度越低。由于 gammaSat 通常随高度变化，
% 对特别高的地形，这种近似可能高估地表温度（使结果偏暖）。
TLSWind = T(1) - gammaSat(1)*hWind;

% 更严格的替代方案：沿 baseState 的饱和绝热递减率垂直积分，
% 再将累计降温插值到每个地形高度。以下代码目前只作为记录，
% 保持注释状态以避免改变原模型结果。
% deltaTSat = cumtrapz(zBar, gammaSat);
% TLSWind = T(1) - interp1(zBar, deltaTSat, hWind, 'linear', 'extrap');

% Horizontal shear of isothermal surface, to make fall paths vertical.
sShearWind = s + shear*z223Wind; % 这里是在计算在风向上移动的雨雪在风向上的移动距离
for j = 1:nT % 这个循环是要处理每条风向上的路径
    % Use first crossing where surface is steeper than fall path
    iMonotonic = sShearWind(:,j) ...
        >cummax([sShearWind(1,j)-1; sShearWind(1:end-1,j)]);
    % iMonotonic的计算逻辑是基于循环迭代不断寻找风向坐标中不断递增的点作为风向传播方向
    % 基于新的风向差值得到当前风向传播路径上223K的等温面
    z223Wind(:,j) = interp1(sShearWind(iMonotonic,j), ...
        z223Wind(iMonotonic,j), sSurfaceShearWind(:,j), ...
        'linear', z223Wind(1,j));
end
% 插值后，z223Wind 表示最终落到各地面点的降水沿下落路径经过的 223 K 高度。
% Finalize by calculating height of isothermal surface above 
% land surface along fall path. 
zBar223Wind = z223Wind - hWind;
% zBar223Wind：223 K 等温面距当地地面的高度。
clear z223Wind

% Horizontal shear of isothermal surface, to make fall paths vertical.
sShearWind = s + shear*z258Wind;
for j = 1:nT
    % Use first crossing where surface is steeper than fall path
    iMonotonic = sShearWind(:,j) ...
        >cummax([sShearWind(1,j)-1; sShearWind(1:end-1,j)]);
    z258Wind(:,j) = interp1(sShearWind(iMonotonic,j), ...
        z258Wind(iMonotonic,j), sSurfaceShearWind(:,j), ...
        'linear', z258Wind(1,j));
end
% 同样将 258 K 等温面校正到各降水落点对应的倾斜下落路径。
clear sShearWind
% Finalize by calculating height of isothermal surface above 
% land surface along fall path. 
zBar258Wind = z258Wind - hWind;
% zBar258Wind：258 K 等温面距当地地面的高度。
clear z258Wind

% Calculate zBarFSWind, the height of freezing surface above land surface.
% Set to zero where freezing surface is below land surface. 
zBarFSWind = zBar258Wind;
zBarFSWind(zBarFSWind<0) = 0;
% 这一部分是在将258K等温度面高度转换为距离地表的高度，小于0就截断为0
% 由于z258Wind已经考虑了沿风向下落的移动距离
% 实际表示的物理意义是降落的水汽经过258K等温面的平均海拔

% Differentiate ln(fM) in wind direction
% fMWind是风向网格中当前总水分柱质量相对于初始背景水汽柱质量的比例
% gradient是对相邻两个网格s间的fMWind的变化率
% 得到dLnFM_dSWind，表示总水分比例在对数坐标下沿风向变化的空间梯度
[~, dLnFM_dSWind] = gradient(log(fMWind), dS);
% dLnFM_dSWind：总水分比例自然对数沿风向的空间梯度，供同位素积分使用。

% Calculate hydrogen-isotope grid
% Get specific equilibrium factors as required for averaging calculation.
% 计算氢同位素平衡分馏系数在"地表至 258 K"和"258 K 至 223 K"两个垂直区间内的平均高度梯
aLSWind = fractionationHydrogen(TLSWind);
a258 = fractionationHydrogen(258);
a223 = fractionationHydrogen(223);
% 三个 alpha 分别对应地表温度、258 K 和 223 K 下的氢同位素平衡分馏系数。
% Calculate fractionation factors by vertical averaging.
% Subscripts A and B refer to above and below 258 K point.
bA = (a223 - a258)./(zBar223Wind - zBar258Wind);
bA(isnan(bA)) = 0;
bB = (a258 - aLSWind)./zBar258Wind;
bB(isnan(bB)) = 0;
% Fractionation factor for precipitation, equal to R_prec/R_vapor.
% aLSWind表示当地地表温度对应的氢同位素平衡分馏系
% a258表示表示258K时的氢同位素平衡分馏系数
% bA表示在258K到223K的高空区域中，氢同位素平衡分馏系数平均每升高1m的变化量
% bB表示从地表到258K等温面的区域中，氢同位素平衡分馏系数平均每升高1m的变化
% zBarFSWind表示258K冻结参考面距离当地地面的有效高度
% 如果258K等温面位于地表海拔之下，该值被设为0
% hR是降水粒子在下降过程中发生同位素再平衡或"重置"的尺度高度
% 可以直观理解为降水粒子从高空向下落时，它原来携带的高空同位素特征能够保存多久
aPrecWind = ...
    ( (a258 + bA.*(hS + zBarFSWind - zBar258Wind)).*exp(-zBarFSWind./hR) ...
    + aLSWind.*(1 - exp(-zBarFSWind./hR)) ).*exp(-zBarFSWind./hS) ...
    + bB.*(hR^2*hS/(hS + hR)^2) ...
    .*(1 - (1 + (1/hS + 1/hR).*zBarFSWind) ...
    .*exp(-(1/(hS + 1/hR).*zBarFSWind))) + aLSWind.*(1 - exp(-zBarFSWind./hS));
% aPrecWind：综合垂直降水生成、冰/水相分馏及下落重置后的降水分馏系数。
clear bA bB cA cB

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %... Initial water vapor composition for d2H (no evaporative recycling)
% aPrec0 = mean(aPrecWind(1,:));
% d2H0_Vapor = (1 + d2H0)/aPrec0 - 1;
% fprintf('\n\nd2H0 for initial precipitation (per mil): %.1f\n', d2H0*1e3)
% fprintf('d2H0 for initial water vapor (per mil): %.1f\n', d2H0_Vapor*1e3)
% fprintf('alpha for initial precipitation: %.7f\n\n', aPrec0)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Next calculate fractionation due to evaporative recycling,
% using method from Criss, 1999 (p. 154-155, and 175).
% Get equilibrium fractionation factor at the evaporation temperature,
% which is defined as the temperature at the surface of the evaporating 
% water. We follow the general practice, which is to approximate 
% this temperature using the surface air temperature. Note this
% is the only place where temperature affects isotope fractionation 
% associated with evaporation.
a1EvapWind = fractionationHydrogen(TLSWind);
% a1EvapWind：地表温度、饱和条件下的平衡蒸发分馏系数。
% Calculate fractionation factor for evaporation at rH = 0.
a0EvapWind = a1EvapWind.*DRatio2H^-n;
% a0EvapWind：加入重/轻水分子扩散差异后的干燥条件蒸发分馏系数。
% Calculate fractionation factor for evaporation process (R_evap/R_vapor).
aEvapWind = a1EvapWind.*rHWind./(1 - a0EvapWind.*(1 - rHWind));
% aEvapWind：当前相对湿度 rHWind 下，蒸发水汽相对液态水的分馏系数。
% Calculate exponent for integration of evaporative fractionation.
uEvap_d2HWind = 1./(a0EvapWind.*(1-rHWind)) - 1;
uEvap_d2HWind(rHWind==1) = 0;
% 饱和时没有净蒸发，将指数设为 0 以避免除零并表示无蒸发分馏。
% Combine to get fractionation factor for residual precipitation
% relative to atmospheric water vapor (R_residual/R_vapor).
aResidualVaporWind = fPWind.^uEvap_d2HWind.*aPrecWind ...
    + (1 - fPWind.^uEvap_d2HWind).*aEvapWind;
% 将未蒸发的原始降水分馏与蒸发返回水汽分馏混合。
clear a0Evap a1Evap

% If opiFit, then remove unneeded arrays
if isFit==true, clear aEvap_d2HWind uEvap_d2HWind, end

% Integrate fractionation along the wind direction (down the columns)
% Result is the isotope ratio for the precipitation.
R_PrecWind = aPrecWind./aPrecWind(1,:) ...
    .*exp(cumtrapz((aResidualVaporWind - 1).*dLnFM_dSWind).*dS);
% R_PrecWind：沿风向积分得到的相对同位素比值，以上风边界为归一化基准。
clear aPrec aResidual_Vapor

% If opiCalc, then calculate d2HEvapGrid and uEvap_d2HGrid
evapD2HGrid = [];
uEvapD2HGrid = [];
if isFit==falsew
    % Calculate d2HEvapWind and the convert to geographic grid
    d2HEvapWind = (aEvapWind./ aPrecWind).*R_PrecWind - 1;
    F = griddedInterpolant({s, t}, d2HEvapWind, 'linear', 'none');
    clear d2HEvapWind
    evapD2HGrid = F(Sxy, Txy);
    % Convert uEvap_d2HWind back to geographic grid
    F = griddedInterpolant({s, t}, uEvap_d2HWind, 'linear', 'none');
    clear uEvap_d2HWind
    uEvapD2HGrid = F(Sxy, Txy);
    clear F
end

%... Finalize d2H calculation
% 1) Transform d2H back to geographic grid
% 2) Converts from isotope ratio, RP, to to delta, d, representation.
% 3) Account for in regional isotopic composition of precipitation, 
% with d2H0 for value at centroid, and dD2H0dLat as the latitudinal
% gradient, which is relative to the absolute value of latitude. 
F = griddedInterpolant({s, t}, R_PrecWind, 'linear', 'none');
clear R_PrecWind
%d2HGrid = ... fond a bug here by Keran Li, 0310, 2026
    %(1 + d2H0 + dDH0dLat*(abs(lat) - abs(lat0))).*F(Sxy, Txy) - 1;
    % 确保 lat 是列向量以便广播到整个网格
    if isrow(lat)
        lat = lat.';
    end
    d2HGrid = (1 + d2H0 + dDH0dLat*(abs(lat) - abs(lat0))).*F(Sxy, Txy) - 1;
    % 将相对同位素比值乘以区域纬度背景，并转换为 delta 表示。

%% Calculate oxygen-isotope grid
% 以下重复同样的流程计算氧同位素，但使用 18O 专用分馏函数和扩散系数比。
% Get specific equilibrium factors as required for averaging calculation.
aLSWind = fractionationOxygen(TLSWind);
a258 = fractionationOxygen(258);
a223 = fractionationOxygen(223);
% 三个 alpha 分别对应地表温度、258 K 和 223 K 下的氧同位素平衡分馏系数。
% Calculate fractionation factors by vertical averaging.
% Subscripts A and B refer to above and below freezing surface.
bA = (a223 - a258)./(zBar223Wind - zBar258Wind);
bA(isnan(bA)) = 0;
bB = (a258 - aLSWind)./zBar258Wind;
bB(isnan(bB)) = 0;
clear TLS zBar223Wind
% Fractionation factor for precipitation, equal to R_prec/R_vapor.
aPrecWind = ...
    ( (a258 + bA.*(hS + zBarFSWind - zBar258Wind)).*exp(-zBarFSWind./hR) ...
    + aLSWind.*(1 - exp(-zBarFSWind./hR)) ).*exp(-zBarFSWind./hS) ...
    + bB.*(hR^2*hS/(hS + hR)^2) ...
    .*(1 - (1 + (1/hS + 1/hR).*zBarFSWind) ...
    .*exp(-(1/(hS + 1/hR).*zBarFSWind))) + aLSWind.*(1 - exp(-zBarFSWind./hS));
clear bA bB cA cB zBar258Wind

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %... Initial water vapor composition for d2H (no evaporative recycling)
% aPrec0 = mean(aPrecWind(1,:));
% d18O0_Vapor = (1 + d18O0)/aPrec0 - 1;
% fprintf('\n\nd18O0 for initial precipitation (per mil): %.1f\n', d18O0*1e3)
% fprintf('d18O0 for initial water vapor (per mil): %.1f\n', d18O0_Vapor*1e3)
% fprintf('alpha for initial precipitation: %.7f\n\n', aPrec0)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Next calculate fractionation due to evaporative recycling,
% using method from Criss, 1999 (p. 154-155, and 175).
% Get equilibrium fractionation factor at evaporation temperature.
% (See note above about approximation of this temperature.)
a1EvapWind = fractionationOxygen(TLSWind);
% Calculate fractionation factor for evaporation at rH = 0.
a0EvapWind = a1EvapWind.*DRatio18O^-n;
% Calculate fractionation factor for evaporation process (R_evap/R_vapor).
aEvapWind = a1EvapWind.*rHWind./(1 - a0EvapWind.*(1 - rHWind));
% Calculate exponent for integration of evaporative fractionation.
uEvap_d18OWind = 1./(a0EvapWind.*(1-rHWind)) - 1;
uEvap_d18OWind(rHWind==1) = 0;
% Combine to get fractionation factor for residual precipitation
% relative to atmospheric water vapor (R_residual/R_vapor).
aResidualVaporWind = fPWind.^uEvap_d18OWind.*aPrecWind ...
    + (1 - fPWind.^uEvap_d18OWind).*aEvapWind;
% 得到考虑蒸发再循环后的氧同位素有效分馏系数。
% relative to atmospheric water vapor (R_residual/R_vapor).
clear a0Evap a1Evap
% If opiFit, then remove unneeded arrays
if isFit==true, clear aEvap_d18OWind uEvap_d18OWind, end

% Integrate fractionation along the wind direction (down the columns)
% Result is the isotope ratio for the precipitation.
R_PrecWind = aPrecWind./aPrecWind(1,:) ...
    .*exp(cumtrapz((aResidualVaporWind-1).*dLnFM_dSWind).*dS);
% 沿风向积分得到氧同位素相对比值。
clear aPrec aResidual_Vapor dLnFM_dSWind

% If opiCalc, then calculate d18OEvapGrid and uEvap_d18OGrid
evapD18OGrid = [];
uEvapD18OGrid = [];
if isFit==false
    % Calculate d18OEvapWind and the convert to geographic grid
    d18OEvapWind = (aEvapWind./ aPrecWind).*R_PrecWind - 1;
    F = griddedInterpolant({s, t}, d18OEvapWind, 'linear', 'none');
    clear d18OEvapWind
    evapD18OGrid = F(Sxy, Txy);
    % Convert uEvap_d18OWind back to geographic grid
    F = griddedInterpolant({s, t}, uEvap_d18OWind, 'linear', 'none');
    clear uEvap_d18OWind
    uEvapD18OGrid = F(Sxy, Txy);
    clear F
end

%... Finalize d18O precipitation calculation
% 1) Transform d18O back to geographic grid
% 2) Converts from isotope ratio, RP, to to delta, d, representation.
% 3) Account for in regional isotopic composition of precipitation, 
% with d18O0 for value at centroid, and dd18O0dLat as the latitudinal
% gradient, which is relative to the absolute value of latitude. 
F = griddedInterpolant({s, t}, R_PrecWind, 'linear', 'none');
clear R_PrecWind
d18OGrid = ...
    (1 + d18O0 + dD18O0_dLat*(abs(lat) - abs(lat0))).*F(Sxy, Txy) - 1;
% 将氧同位素相对比值转换为地理网格上的 delta18O。

end
