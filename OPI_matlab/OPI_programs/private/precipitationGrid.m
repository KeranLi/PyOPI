function [s, t, Sxy, Txy, pGrid, hWind, fMWind, rHWind, fPWind, ...
    z223Wind, z258Wind, tauF] = precipitationGrid ...
    (x, y,  hGrid, U, azimuth, NM, fC, kappa, tauC, hRho, zBar, ...
    T, gammaEnv, gammaSat, gammaRatio, rhoS0, hS, fP0)
% DEBUG: Check input values
if hS <= 0 || isnan(hS) || isinf(hS)
% 检查水汽密度尺度高度 hS，要求它是有限的正数。
    fid = fopen('debug_precipitationGrid.txt', 'a');
    fprintf(fid, 'ERROR: invalid hS = %g\n', hS);
    fprintf(fid, '  U=%g, azimuth=%g, NM=%g\n', U, azimuth, NM);
    fprintf(fid, '  kappa=%g, tauC=%g, hRho=%g\n', kappa, tauC, hRho);
    fclose(fid);
    error('precipitationGrid: invalid hS = %g', hS);
end
if U <= 0 || isnan(U) || isinf(U)
% 检查水平风速 U，要求它是有限的正数。
    error('precipitationGrid: invalid U = %g', U);
end
if tauC <= 0 || isnan(tauC) || isinf(tauC)
% 检查云水凝结时间 tauC，要求它是有限的正数。
    error('precipitationGrid: invalid tauC = %g', tauC);
end
%... precipitationGrid calculates grids for precipitation rate 
% (kg m^-2 s^-1) and moisture ratio (dimensionless), using a modified 
% version of the LTOP algorithm of Smith and Barstad (2004). 
% The modifications include Coriolis forcing, and moisture balance. 
% The function assumes that all grids are in the "wind grid" format, 
% with coordinates s,t corresponding to the 1st and 2nd dimensions 
% of the grid, respectively. The +s axis points in the downwind 
% direction, and the +t axis is 90 degrees clockwise for the +s axis.
%
% Feb 2021: Added moisture balance including evaporative recycling,
% where fP is the fraction of precipitation that leaves the 
% base of the model.

% August 8, 2021: Evaporation calculation had an error, now fixed, where fP 
% was left everywhere equal to 1, rather than being set to the candidate 
% value, fP0, prescribed for each step of the search. This error has 
% likely been active since March 23, 2021 (OPI 3.5).

% Mark Brandon, Yale University, 2016-2021

%% Initialize system
warning('off', 'MATLAB:griddedInterpolant:MeshgridEval2DWarnId');

%% Constants
% Dry adiabatic lapse rate (K/m)
gammaDry = 0.009754;

%% Compute reference solution
%... Get Fourier solution for Euler equations
% 将地形旋转并插值到风向坐标，再把二维地形分解为不同水平波数的地形波。
 [s, t, Sxy, Txy, hWind, kS, kT, hHat, kZ] = ...
    fourierSolution(x, y, hGrid, U, azimuth, NM, fC, hRho);

%... Parameters for wind grids
dS = s(2)-s(1);
nS = length(s);
nT = length(t);

%... Calculate height array for the freezing surface as needed to 
% calculate tauF. The freezing surface is defined by the 
% 258 K isosurface, which marks the midpoint of the 268 - 248 K range
% for freezing in the atmosphere (WBF zone, Cias and Jouzel, 1994).
% 计算二维高度场：每个风向网格位置上，258 K 等温面相对海平面的高度。
z258Wind = isotherm(258, zBar, T, gammaEnv, gammaSat, hRho, nS, nT, hHat, kZ);

%... Calculate mean fall time
% Velocities wR and wS (m/s) for rain and snow, respectively.
% References: Langleben, 1954; White et al., 2002; Yuter and Houze, 2003;
% Barstad and Schuller, 2011.
wFSnow = -1; % 雪下落的速度，z轴上规定向上为正，那么降落就为负
wFRain = -6; % 雨滴下落的速度
% Mean fall time
% hWind 是旋转、插值到风向坐标后的二维地形高度场。
% z258Wind 和 hWind 的尺寸均为 nS×nT；比较二者会生成二维逻辑掩码。
% 当 258 K 等温面不高于地面时，整个平均下落过程按雪速处理。
% 当 258 K 等温面高于地面时，公式综合雪区和雨区的平均下落时间。
tauF_calc = (z258Wind<=hWind).* -hS/wFSnow ...
    + (z258Wind>hWind).* ...
    -((z258Wind - hWind)/wFRain + hS*exp(-(z258Wind - hWind)./hS)/wFSnow);
tauF = mean(tauF_calc, 'all');
if isnan(tauF) || isinf(tauF)
    % Write debug info to file for parallel debugging
    fid = fopen('debug_tauF_error.txt', 'a');
    fprintf(fid, '=== tauF Error Debug ===\n');
    fprintf(fid, 'hS = %g (wFSnow=%g, wFRain=%g)\n', hS, wFSnow, wFRain);
    fprintf(fid, 'z258Wind: min=%g, max=%g, anyNaN=%d, anyInf=%d\n', ...
        min(z258Wind, [], 'all'), max(z258Wind, [], 'all'), ...
        any(isnan(z258Wind), 'all'), any(isinf(z258Wind), 'all'));
    fprintf(fid, 'hWind: min=%g, max=%g, anyNaN=%d, anyInf=%d\n', ...
        min(hWind, [], 'all'), max(hWind, [], 'all'), ...
        any(isnan(hWind), 'all'), any(isinf(hWind), 'all'));
    fprintf(fid, 'tauF_calc: min=%g, max=%g, anyNaN=%d, anyInfPos=%d, anyInfNeg=%d\n', ...
        min(tauF_calc, [], 'all'), max(tauF_calc, [], 'all'), ...
        any(isnan(tauF_calc), 'all'), any(isinf(tauF_calc)&tauF_calc>0, 'all'), ...
        any(isinf(tauF_calc)&tauF_calc<0, 'all'));
    fprintf(fid, 'Base state: zBar(1)=%g, T(1)=%g\n', zBar(1), T(1));
    fclose(fid);
    error('tauF is NaN or Inf in precipitationGrid (see debug_tauF_error.txt)')
end
clear z258Wind

%... Calculate grid z223Wind, which is elevation relative to sea level of the 
% 223 K isothermal surface. Used by isotopeGrid function. 
% 223 K 是高空冰相同位素分馏的低温参考端点；它与 258 K 一起用于
% isotopeGrid 中近似分馏系数在高空冰相区域随高度的变化。
z223Wind = ...
    isotherm(223, zBar, T, gammaEnv, gammaSat, hRho, nS, nT, hHat, kZ);

%... Calculate grid z258Wind, which is elevation relative to sea level of the 
% 258 K isothermal surface. Used by isotopeGrid function. 
% 这里是之前计算的258Wind已经被清除了
z258Wind = ...
    isotherm(258, zBar, T, gammaEnv, gammaSat, hRho, nS, nT, hHat, kZ);

%... Calculate PStarHat, reference precipitation rate for wave domain
% gammaRatio 是饱和绝热递减率与环境递减率之比的水汽加权平均。
% GSHat 是地形抬升产生水汽源的响应，GCHat 是云水形成响应，
% GFHat 是降水下落响应，kappa 是水平扩散系数。
% pStarHat 是综合上述响应后得到的傅立叶域参考降水率。
GSHat = gammaRatio*rhoS0*1i*kS.*U./(1 - hS*(1i*kZ + 1/(2*hRho)));
GCHat = 1./(tauC*(kappa*(kS.^2 + kT.^2) + 1i*kS*U) + 1);
GFHat = 1./(tauF*(kappa*(kS.^2 + kT.^2) + 1i*kS*U) + 1);
pStarHat = GSHat.*GCHat.*GFHat.*hHat;

%... Transform back to space domain, remove padding, and 
% set negative values in pStarPosWind to zero. The 'symmetric" option
% for ifft2 indicates that PStarHat is conjugate symmetric, which ensures
% that pStarWind is returned as a real-valued grid.
pStarPosWind = ifft2(pStarHat, 'symmetric');
% 逆傅立叶变换把各个波数分量重新叠加为空间中的参考降水率。
clear pStarHat
pStarPosWind = pStarPosWind(1:nS, 1:nT);
% fourierSolution 为傅立叶计算扩展了网格，这里裁回实际的 nS×nT 区域。
pStarPosWind(pStarPosWind<0) = 0;
% 线性波动解可以产生负的降水扰动，但实际降水率不能为负，因此截为零。

%% Calculate vapor ratio and moisture-corrected precipitation rate
%... Calculate column-density fields for cloud water QC, 
% falling precipitation QF, and the total moisture QT. These
% fields are for the reference solution, and are truncated  to 
% postive values, as required for the moisture-balance calculation.
% 将垂直方向积分为单位地面面积上的水分柱质量。
% QCStarPosWind 是云水柱质量，QFStarPosWind 是正在下落的降水柱质量；
% QTStarPosWind 还包括背景水汽柱质量 rhoS0*hS。
% 这些量先在傅立叶域求解，再逆变换、裁剪，并将非物理负值截为零。
QCStarPosWind = ifft2(tauC*GSHat.*GCHat.*hHat, 'symmetric');
QCStarPosWind = QCStarPosWind(1:nS, 1:nT);
QCStarPosWind(QCStarPosWind<0) = 0;

QFStarPosWind = ifft2(tauF*GSHat.*GCHat.*GFHat.*hHat, 'symmetric');
QFStarPosWind = QFStarPosWind(1:nS, 1:nT);
QFStarPosWind(QFStarPosWind<0) = 0;
clear GSHat GCHat GFHat

QTStarPosWind = rhoS0*hS + QCStarPosWind + QFStarPosWind;
clear QCStarPosWind QFStarPosWind

%... Account for evaporatiive recycling
if fP0==1
        %... No evaporative recycling
        rHWind = 1;
        fPWind = 1;
else
        %... Evaporative recycling is restricted to areas where the air
        % at the base of the model is undersaturated, which is usually
        % above the lee slopes of the topography.
        % Start by calculating the cloud-water density, rhoC, at zBar = 0.
        % Note that rhoC is calculated in the LTOP model assuming 
        % that a parcel follows a moist-adiabat, with rhoC>=1 when
        % saturated, and rhoC<1 when undersaturated. 
        rhoCStarHat = (gammaRatio*rhoS0.*1i.*kS*U/hS).*hHat ...
            ./((kappa*(kS.^2 + kT.^2) + 1i*kS*U) + 1/tauC);
        rhoCStarWind = ifft2(rhoCStarHat, 'symmetric');
        rhoCStarWind = rhoCStarWind(1:nS, 1:nT);
        clear rhoCStarHat
        % Calculate relative humidity, with range 0 to 1. 
        % The ratio gammaDry/gammaSat(1) corrects for the fact that
        % rhoC follows the dry adiabat when undersaturated. Also note 
        % that rhoC and rhoS0 are both scaled by fV, so fV cancels out.
        rHWind = 1 + (gammaDry/gammaSat(1))*rhoCStarWind./rhoS0;
        rHWind(rHWind>1) = 1;
        % clear rhoCStarWind 
        % The residual precipitation grid, fPWind, is set to 1
        % where rHWind=1 (saturated), and to the specified fP0 
        % where rhWind<1.
        fPWind = ones(nS,nT);
        fPWind(rHWind<1) = fP0;
        
%         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         % For setting leeside relative humidity to a constant value.
%         % Chirp sound and print output warn that this kludge is active. 
%         S = load('chirp.mat'); sound(S.y)
%         rHWind0 = 1;
%         rHWind(rHWind<1) = rHWind0;
%         fprintf('\n>>> precipitationGrid: Manually set rHWind in leeside regions. <<<\n')
%         fprintf('rHWind0 = %g\n', rHWind0)
%         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
end

%... Integrate along the columns of the s,t grid (+s direction) to
% calculate the water-vapor ratio, fV.
% 计算空气沿着风向移动到某个位置状态的相对水汽保留比例
% cumtrapz 表示 cumulative trapezoidal integration，即"累计梯形积分"
% 计算空气从上风边界移动到当前网格点过程中累积损失的水汽；dS/U
% 把沿风向的空间积分换算为随空气移动时间的积分。
fVWind = (rhoS0*hS./QTStarPosWind) ...
    .*exp(-(1/U)*cumtrapz(fPWind.*pStarPosWind.*dS./QTStarPosWind));

%... Calculate PGrid, moisture-corrected precipitation-rate field
% 用剩余水汽比例修正参考降水率，得到风向网格中的实际降水率。
pWind = fVWind.*pStarPosWind;
clear pStarPosWind

%... Transform precipitation rate back to geographic grid
% 将风向坐标 (s,t) 中的降水率插值回原始地理坐标 (x,y)。
F = griddedInterpolant({s, t}, pWind, 'linear', 'none');
pGrid = F(Sxy, Txy);
clear pWind

%... Calculate fM, moisture-ratio field
% 计算当前实际总水分柱质量相对于初始背景水汽柱质量的比例。
fMWind = fVWind.*QTStarPosWind/(rhoS0*hS);
clear QTStarPosWind fVWind

end
