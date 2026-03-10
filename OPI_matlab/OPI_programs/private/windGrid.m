function [Sxy, Txy, s, t, Xst, Yst] = windGrid(x, y, azimuth)
% WINDGRID Conversion between geographic and wind grids.
% Sets up the transformation of a geographic grid, with x,y grid vectors,
% to a wind grid, with s,t grid vectors. Geographic grids use the
% meshgrid format, and wind grids use the ndgrid format. In other words,
% for geographic grids, x and y correspond to the 2nd and 1st dimensions
% of the grid, and for wind grids, s and t correspond to the 1st and 
% 2nd dimensions of the grid. 
% Note that x,y,z and s,t,z are both right-handed coordinate frames, as 
% required to ensure correct calculation of the Coriolis effect. 
% Input arguments:
% x, y = grid vectors indicating coordinates for geographic grids
%   (e.g. hGrid), with x and y oriented in the east and north directions,
%   respectively (vectors with lengths nX and nY, m).
% Output arguments:
% s, t = grid vectors indicating coordinates for wind grids (e.g., hWind),
%   with s oriented in the down-wind direction, and t oriented 90
%   degrees counterclockwise from the s direction (s,t,z is right handed)
%   (vectors with lengths nS and nt, m).
% Sxy, Txy = grids containing the s and t coordinates, respectively, for
%   nodes in a geographic grid. These grids provide the basis for 
%   interpolating field variables from the wind-grid solution, back onto 
%   grid nodes associated with the original geographic grid
%    (matrices, nY x nX, m).
% Xst, Yst = grids containing the x and y coordinates, respectively, for
%   nodes in a wind grid. (matrices, nY x nX, m).

% Mark Brandon, Yale University, 2016-2020

%% Set up wind-direction grid
dX = x(2) - x(1);
dY = y(2) - y(1);

%... Calculate s,t coordinates for the x,y nodes of geographic grid
% (s,t,z is right handed).
[X, Y] = meshgrid(x, y);
Sxy = X*sind(azimuth) + Y*cosd(azimuth);
Txy = -X*cosd(azimuth) + Y*sind(azimuth);
clear X Y

%... Calculate initial estimate for node spacing for the new grid
% The ellipse equation is used to find a nodal spacing for the new grid
% that matches the nodal spacing of the original grid given the
% new azimuth direction, where:
% 1/dS.^2 = (sind(azimuth)/dX).^2 + (cosd(azimuth/dY).^2 
dS = sqrt(1./((sind(azimuth)/dX).^2 + (cosd(azimuth)/dY).^2));
dT = sqrt(1./((sind(azimuth+90)/dX).^2 + (cosd(azimuth+90)/dY).^2));

%... Calculate s and t vectors for wind grid
sMin = min(Sxy, [], 'all');
sMax = max(Sxy, [], 'all');
tMin = min(Txy, [], 'all');
tMax = max(Txy, [], 'all');
dS = (sMax - sMin)/ceil((sMax - sMin)/dS);
dT = (tMax - tMin)/ceil((tMax - tMin)/dT);
s = (sMin:dS:sMax)';
t = tMin:dT:tMax;

%... Calculate x,y coordinates for the s,t nodes of wind grid
% (x,y,z and s,t,z are right-handed).
[S, T] = ndgrid(s, t);
Xst = S*sind(azimuth) - T*cosd(azimuth);
Yst = S*cosd(azimuth) + T*sind(azimuth);
clear S Y

end
