function [colorMap1, ticks, tickLabels] = cmapscale(z, colorMap0, varargin)
%CMAPSCALE   Rescales color map to better match the cumulative 
%   distribution of the data variable, z. The results assume that
%   clim for the figure has been set to the min and max limits for z. 
%
%   colorMap1 = cmapscale(z, colorMap0) rescales the specified colormap,
%   colorMap0, and outputs the result as a new colormap, colorMap1.
%
%   colorMap1 = cmapscale(z, colorMap0, factor) Factor adjusts the 
%   contrast (range: 0<= factor <=1). Factor = 0 gives linear mapping 
%   (no change), and factor = 1 gives uniform mapping (maximum contrast,
%   equivalent to histogram equalization).
%
%   colorMap1 = cmapscale(z, colorMap0, factor, z0) rescales so that the
%   value z0 lies at the center of the colormap, which is assumed to 
%   be a centered color sequence. If z0<=z, then the upper half of 
%   colorMap0 is used. If z0>=z, then the lower half of colorMap0 is used. 
%   Default is no centering.
%
%   [colorMap1, ticks, tickLabels] = ...
%      cmapscale(z, colorMap0, factor, z0, nTicks, nRound) 
%   This option avoids saturation of the colorbar by using the 
%   initial colormap, colorMap0. nTicks indicates the number of ticks
%   for the colorbar, and nRound specifies decimal position for rounding
%   the tick labels. The output arguments ticks and tickLabels are 
%   used to assign the correct positions and labels for the colorbar ticks.
%
%   EXAMPLE: Map of the world with uniform mapping, with
%   colormap set to "cool", factor = 1, and centered on 
%   sea level (z0 = 0). The output tics defines 11 tic labels for
%   a colobar using the initial colormap, and rounding to decimal.
%
%     clear
%     load topo
%     image([0 360],[-90 90], flip(topo), 'CDataMapping', 'scaled');
%     hA = gca;
%     hCB = colorbar;
%     cmap0 = colormap(coolwarm);
%     factor = 1;
%     z0 = 0;
%     nTicks = 11;
%     nRound = 0;
%     [cmap1, hCB.Ticks, hCB.TickLabels] = ...
%         cmapscale(topo, cmap0, factor, z0, nTicks, nRound);
%     colormap(hA, cmap1)
%     colormap(hCB, cmap0)
%
%    By Mark Brandon, Yale University, 2012-2022.
%    Inspired by Eisemann, M., Albuquerque, G., and Magnor, M., 2011, 
%       Data driven colormapping: Proceedings EuroVA.

%% Start function
%... Check number of total number of arguments
narginchk(2, 6);
%... Check colormap0
if size(colorMap0,2) ~= 3
    error('colormap0 must have 3 columns to be a proper colormap.')
end

%... Check for contrast factor
factor = 1;
if ~isempty(varargin)
    factor = cell2mat(varargin(1));
    if factor<0 || factor>1
        error('Factor must be in the range 0 to 1');
    end
end
%... Check for centering value z0
z0 = [];
if length(varargin) > 1
    z0 = cell2mat(varargin(2));
    % Adjust z0 to be consistent with range for z 
    if z0 <= min(z(:))
        mMap = size(colorMap0,1);
        colorMap0 = interp1(0:mMap-1, colorMap0, ...
            linspace((mMap-1)/2, mMap-1, mMap) );
        z0 = [];
    end
    if z0 >= max(z(:))
        mMap = size(colorMap0,1);
        colorMap0 = interp1(0:mMap-1, colorMap0, ...
            linspace(0, (mMap-1)/2, mMap) );
        z0 = [];
    end
end
%... Parse parameters, if present, for tick calculation 
nTicks = [];
if length(varargin) > 2
    nTicks = cell2mat(varargin(3));
    if nTicks < 1 || fix(nTicks)~=nTicks
        error('Number of ticks, nTic,s must be positive integer value.');
    end
end
if length(varargin) > 3
    nRound = cell2mat(varargin(4));
    if round(nRound)~=nRound
        error('Rounding parameter, nRound, much be integer value.')
    end
end

%... Recast contrast factor as a stretching parameter s, and limit
% its range to avoid numerical problems.
s = tan(pi*factor/2);
if s>1e4, s = 1e4; end

%... Remove nans from z and reshape as a column vector
z = z(~isnan(z));
%... Calculate min and max for z
zMin = min(z);
zMax = max(z);

%... Check for uniform z values
if (zMax-zMin)<10*eps
    fprintf('cmapscale: z values are equal. Returning the input color map.\n')
    colorMap1 = colorMap0;
    ticks = [];
    tickLabels = [];
    return
end

%... Create linear probability scale pL and uniform probability 
% scale pU, and also correct for duplicates in the z sequence.
% The result are strictly monotonic sequences, as required 
% for use of interp1 (see below) for interpolation. 
% Duplicates are determined using a relative tolerance of 1e-7,
% (see uniquetol), and are replaced by median values (see accumarray).
%... Create linear probability scale pL
pL = (z - zMin)/(zMax - zMin);
[pL, i, j] = uniquetol(pL, 1e-7);
m = length(pL);
%... Create uniform probability scale pU
pU = (j - 1)./(m - 1);
pU = accumarray(j, pU, [], @min);
pU = sort(pU);
%... Assemble unique version of z vector
z = z(i);

%... Create stretched probability scale pS using weighted sum of pL and pU
if isempty(z0)
    %... Color scale with no specified center point
    pS = (pL + s^2*pU)./(1 + s^2);
else
    %... Color scale with center point at z0
    pL0 = interp1(z, pL, z0);
    pU0 = interp1(z, pU, z0);
    i0 = round(interp1(z, 1:m, z0));
    pL(i0) = pL0;
    pU(i0) = pU0;
    
    % Project lower and upper halves of data distributions onto
    % new scale pS. This is done in two steps to maintain z0 at center.
    % Project lower part onto 0 < pS < 0.5
    pS = zeros(m,1);
    pS(1:i0) = 0.5*(pL(1:i0)/pL0 + s^2*pU(1:i0)/pU0)/(1 + s^2);
    % Project upper half  onto 0.5 < pS < 1
    pS(i0:m) = 0.5 + 0.5* ...
        ((pL(i0:m)-pL0)/(1-pL0) + s^2*(pU(i0:m)-pU0)/(1-pU0))/(1 + s^2);
end

%... Renormalize to one, to account for round-off errors
pL = pL/pL(m);
pS = pS/pS(m);

%... Interpolate new colors for colormap
mMap = size(colorMap0, 1);
pLMap = (0:mMap-1)'/(mMap-1);
pSMap = interp1(pL, pS, pLMap);
colorMap1 = interp1(pLMap, colorMap0, pSMap);

%... Calculate tick values
if ~isempty(nTicks)
    % The color bar has a linear scale, ranging from zMin to zMax.
    % The goal here is to find z values in the stretched distribution
    % that correspond to evenly spaced tick locations along the linear
    % colorbar scale. 
    pSTicks = (0:nTicks-1)'/(nTicks-1);
    ticks = interp1([0,1], [zMin,zMax], pSTicks);
    % The tickLabels determined by finding the corresponding z values 
    % in the stretched distribution. Note that pSTicks corresponds
    % to probabilities for the stretch distribution. 
    % Thus, the next step is to map back to probabilities in the 
    % linear distribution, and to z values for that distribution.
    pLTicks = interp1(pS, pL, pSTicks);
    tickLabels = interp1([0,1], [zMin,zMax], pLTicks);
    % Round tick labels to specified decimal position
    tickLabels = round(tickLabels, nRound);
    % Fix "negative zero" values
    tickLabels(tickLabels==0) = 0;
    format = sprintf('%s%d%s', '%0.', nRound, 'f'); 
    tickLabels = compose(format, tickLabels);
end

