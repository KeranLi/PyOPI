function [xPath, yPath, sPath, sLimits] = ...
    windPath(xPath0, yPath0, azimuth, x, y, varargin)
% windPath finds the wind path passing through a reference point
% Input arguments
% xPath0, yPath0: reference point used to define specific wind path. 
%    [length] (scalar) (All length variables must use the same units.) 
% azimuth: wind direction [degrees] (scalar)
% x, y: grid vectors for topography grid (hGrid) [length] (vector)
% Additional input argument (varagin)
% xyLimits: optional argument with 4 components defining a smaller 
%    domain for display purposes, with values: xMin, xMax, yMin, yMax.
%    If not defined, xyLimits is set to limits implied by x and y. 
%    [length] (vector)
% Output arguments
% xPath, yPath: easting and northing for path, as bounded by 
%    the limits indicated by the grid vectors x, y. [length] (vectors)
% sPath: distance (meters) along path, with x0,y0 point set to zero,
%    and distance increasing in wind direction. Step size along path 
%    is adjusted to match step size for hSGrid. [length] (vector)
% sLimits: vector with sMin, sMax values for the sPath values where
% xyLimits intersects the specified wind path. [length] (vector)
%    
% Mark Brandon, Yale University, 2020

%% Initialize variables
%... Set xyLimits
if ~isempty(varargin)
    % Set to input argument if present
    xyLimits = varargin{1};
else    
    % Set to limits implied by x and y if not present
    xyLimits(1) = min(x);
    xyLimits(2) = max(x);
    xyLimits(3) = min(y);
    xyLimits(4) = max(y);
end
%... Check content of xyLimits
if length(xyLimits)~=4
    error('Input argument xyLimits must be a 4 component vector.')
end

%% Compute wind path
%... Calculate step size dS as equal to grid size along wind path.
% dSPath is calculated using ellipse equation:
% 1/dS.^2 = (sind(azimuth)/dX).^2 + (cosd(azimuth/dY).^2 
dX = x(2)-x(1);
dY = y(2)-y(1);
dSPath = sqrt(1./((sind(azimuth)/dX).^2 + (cosd(azimuth)/dY).^2));
%... Find maximum distance between reference point and grid corners
xBox = x([1 end end 1 1]');
yBox = y([1 1 end end 1]');
dMax = max(sqrt(xBox - xPath0).^2 + (yBox - yPath0).^2);
% Construct path that extends beyond the bounding box
xPath = xPath0 + sind(azimuth).*1.01*[-dMax dMax]';
yPath = yPath0 + cosd(azimuth).*1.01*[-dMax dMax]';
[xPath, yPath] = polyxpoly(xPath, yPath, xBox, yBox);
% Throw error if only one intersection between bounding box and path.
if length(xPath)==1
    error('Specified path origin must lie inside model domain.')
end
% Ensure that xPath, yPath are ordered from start to finish 
if dot([xPath(2) - xPath(1), yPath(2) - yPath(1)], ...
        [sind(azimuth), cosd(azimuth)])<0
    xPath = xPath([2,1]);
    yPath = yPath([2,1]);
end
% Calculate distance along path, relative to x0, y0, and 
% positive in wind direction.
if abs(xPath(2) - xPath(1))>abs(yPath(2) - yPath(1))
    sPath = (xPath - xPath0)./sind(azimuth); 
else  
    sPath = (yPath - yPath0)./cosd(azimuth);
end

% Set nodes with a spacing close to dS, and with first and last nodes 
% located on the windward and leeward sides of the grid.
nS = floor((sPath(2) - sPath(1))/dSPath) + 1;
%... Calculate x,y coordinates for nodes along wind path
xPath = linspace(xPath(1), xPath(2), nS)';
yPath = linspace(yPath(1), yPath(2), nS)';
%... sPath coordinates along wind path, with sPath = 0 at reference point
sPath = linspace(sPath(1), sPath(2), nS)';
%... Calculate sLimits
% Consider case where xyLimits is specified    
xBox = [xyLimits(1), xyLimits(2), xyLimits(2), xyLimits(1), xyLimits(1)]';
yBox = [xyLimits(3), xyLimits(3), xyLimits(4), xyLimits(4), xyLimits(3)]';
[xLimits, yLimits] = polyxpoly(xPath, yPath, xBox, yBox);
if length(xLimits)<2
    % No intersection with xyLimits box, so set sLimits to ends of sPath
    sLimits = [sPath(1), sPath(end)];
    return
end
if abs(diff(xLimits))>abs(diff(yLimits))
    sLimits = interp1(xPath, sPath, xLimits);
else
    sLimits = interp1(yPath, sPath, yLimits);
end
if diff(sLimits)<0, sLimits = sLimits([2,1]); end
end
