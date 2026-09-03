function [s, t, Sxy, Txy, hWind, kS, kT, hHat, kZ] = ...
    fourierSolution(x, y, hGrid, U, azimuth, NM, fC, hRho)
% Calculates the Fourier solution for the linearized Euler equations
% for flow of air over topography. The solution is based on saturated
% base state with a uniform buoyancy frequency, NM, and also uses the
% anelastic approximation, which allows for vertical variation in density
% in the base state. The solution comes from Durran and Klemp, 1982.
% Input arguments:
% x, y = grid vectors indicating coordinates for geographic grids
%   (e.g. hGrid), with x and y oriented in the east and north directions,
%   respectively (vectors with lengths nX and nY, m).
% hGrid = geographic grid for topography, with x (east) in the row
%   direction, and y (north) in the column direction (matrix, nY x nX, m).
% U = base-state wind speed (scalar, m/s).
% azimuth = base-state wind direction (down wind) (scalar, degrees).
% NM = saturation buoyancy frequency (scalar, rad/s).
% fC = Coriolis frequency (scalar, rad/s).
% hRho = scale height for density (scalar, m).
% Output arguments:
% s, t = grid vectors indicating coordinates for wind grids (e.g., hWind),
%   with +s oriented in the down-wind direction, and +t oriented 
%   90 degrees counterclockwise (s,t,z is right handed).
%   (vectors with lengths nS and nt, m).
% Sxy, Txy = grids containing the s and t coordinates, respectively, for
%   nodes in a geographic grid. These grids provide the basis for 
%   interpolating field variables from the wind-grid solution, back onto 
%   grid nodes associated with the original geographic grid
%    (matrices, nY x nX, m).
% hWind = wind grid for topography (matrix, nY x nX, m).
% kS, kT = grid vectors indicating wavenumber coordinates for the
%   Fourier coefficients in hHat, which has a wind-grid orientation
%   (vectors with lengths nS x nT, rad/m).
% hHat = grid with Fourier coefficients for topography (matrix, 
%   complex, nSPad x nTPad, m).
% kZ = grid of vertical wavenumbers for the Fourier solution of the
%   Euler equations, in wind-grid orientation
%   (matrix, complex, nSPad x nTPad, rad/m).
%
% Mark Brandon, Yale University, 2016-2020

%% Initialize system
warning('off', 'MATLAB:griddedInterpolant:MeshgridEval2DWarnId');

%% Compute
%... Transform topography to wind coordinates (s,t,z is right handed)
% 这里Sxy和Txy是从(x,y)的坐标场向(s,t)的风场变换查询方式
% 这里Xst和Yst是从(s,t)的风场坐标向(x,y)的空间场坐标的查询方式
% 注意(s,t)的风场坐标遵循右手坐标系，即掏出右手，面向手心，大拇指为s，四指为t，手背到手心即为z
[Sxy, Txy, s, t, Xst, Yst] = windGrid(x, y, azimuth);
% Note: griddedInterpolant uses the ndgrid format, so that the order
% of the grid vectors, x and y, are reversed to account for the
% meshgrid format for hGrid. Also note that grid vectors must be
% specified as cell variables. The setting 'none' causes extrapolated
% values to be set to nans, which are then found and set to zero.
F = griddedInterpolant({y, x}, hGrid, 'linear', 'none');
% 这里F的作用在原有空间场的基础上，插值找到对应的风场坐标，是一个插值器
hWind = F(Yst, Xst);
hWind(isnan(hWind)) = 0;
% 再把风场坐标在空间坐标的映射投在风场坐标，建立风场状态下的海拔场
clear Xst Yst

%... Parameters for wind grid for topography
dS = s(2)-s(1);
dT = t(2)-t(1);
[nS, nT] = size(hWind);

%... Add zero padding around topography, to account for the full
% response caused by lifting over the topography. For example, an
% increase NM or U, or a decrease tauC or tauF will cause lifting
% and precipitatin to shift upwind, and vice versa. If the grid is
% too small to contain these responses, then the features will appear
% on the other size of the grid (a phenomenon called "wraparound").
% In my experience, zero padding in the wind direction is helpful.
% The recommendation is nSPad = 2*nS, nTPad = nT.
% Fourier 变换默认采用周期边界。地形抬升、云水凝结和降水下落
% 会使响应沿风向延伸；若计算区域过小，响应可能从另一侧周期性绕回，产生 wraparound 伪影。
% 因此在风向 s 上增加零高度填充区域，以减小周期边界影响。
% 经验建议：沿风向扩展为原网格的 2 倍，横风向保持原大小
nSPad = 2*nS;
nTPad = nT;

%... Calculate wavenumber grids for topography (rad/m)
% Wavenumbers for s direction (wind direction)
% ceil函数的作用向上取整，i_kSmostNeg
% nSpand已经在风向傅立叶场里面了
i_kSMostNeg = ceil(nSPad/2)+1;
kS = (0:nSPad-1)'/nSPad;
kS(i_kSMostNeg:nSPad) = kS(i_kSMostNeg:nSPad)-1;
kS = 2*pi*kS/dS;
dKS = kS(2) - kS(1);
% 这段代码把沿风向的离散空间网格转换成与FFT 结果对应的正、负波数网格kS
% 计算波数分辨率dKS，供后续地形波方程使用
% Wavenumbers for t direction
i_kTMostNeg = ceil(nTPad/2)+1;
kT = (0:nTPad-1)/nTPad;
kT(i_kTMostNeg:nTPad) = kT(i_kTMostNeg:nTPad)-1;
kT = 2*pi*kT/dT;

%... Calculate fourier transform of topography
hHat = fft2(hWind, nSPad, nTPad);

%... Calculate denominator for kZ equation
% This step is used to avoid singularies where abs(U*kS)==abs(fC).
% My method, from Queney, 1947, p. 46-48, uses these corrections:
% hHat((abs(denominator)<(U*dkS/2)^2) = 0, and
% demoninator(denominator==0) = eps. The justification is that, at
% this singularity, the vertical velocities go to zero.
% The modification to hHat removes the excitation at the singularity, and
% the modification to the demoninator avoids errors due to irrelevant nans.
% Note that denominator is a nSPad-length column vector, but is implicitly 
% expanded to a nSPad x nTPad matrix for multiplication in the kZ equation.
% dKS是沿风向波数网格的间隔。
% denominator是kZ公式中的分母
% 计算kZ方程中的分母D=(U*kS)^2-fC^2。
% 当|U*kS|接近|fC|时，D接近0，直接代入后续公式会产生Inf或NaN
% 根据 Queney (1947) 的处理，将奇异波数附近的地形Fourier振幅设为0
% 表示这些波数不激发有效垂直速度
% 同时把严格为0的分母替换为 eps，避免后续除零
denominator = (U*kS).^2 - fC^2;
iZero = abs(denominator)<(U*dKS/2)^2;
hHat(iZero) = 0;
clear iZero
denominator(denominator==0) = eps;

%... Calculate kZ, vertical wave number grid
% kS沿风向的水平波数
% kT横风向的水平波数
% kZ竖直方向的波数
% 波数用来描述地形波在空气中向上或向下传播时变化得有多快
kZ2 = (kS.^2 + kT.^2).*((NM.^2 - (U*kS).^2)./denominator) - 1/(4*hRho^2);
kZ = sqrt(kZ2);
%... Assign appropriate roots for sqrt(kZ2)
% If kZ2>0, then sqrt(kZ2) is real (propagating wave).
% The sign of real kZ values is set to match the sign of its associated 
% kS value, which ensures that the wave propagates upward for both 
% positive and negative wavenumbers for kS. 
% Note that the logical calculation for iNeg does an implicit expansion 
% of the nSPad-length row vector on the right.
% kZ是由大气波动方程确定地形扰动在竖直方向的波形
% 当kZ为实数时，扰动以传播波形式向高处延伸
% 当kZ为虚数时，扰动随高度衰减
iNeg = (kZ2>0) & (kS<0);
kZ(iNeg) = -kZ(iNeg);

end