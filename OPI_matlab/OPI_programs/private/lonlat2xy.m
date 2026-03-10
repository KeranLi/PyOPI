function [x, y] = lonlat2xy(lon, lat, lon0, lat0)
% Convert from geographic coordinates to projected coordinates
% The equirectangular projection is used here, which works well for
% maps that span less than about 1000 km in the zonal direction.
% Input arguments
% lon, lat: longitude and latitude (degrees) (scalar, vector, or matrix)
% lon0: central meridian [degrees] (scalar)
% lat0: standard parallel [degrees] (scalar)
% Note: The origin x,y = 0 lies at lon0, lat0.
% Output arguments
% x, y: easting, northing coordinates [meters]

% Mark Brandon, Yale University, 2020

%% Constants
%... Mean radius of the Earth (m)
radiusEarth = 6371e3;
%... Meters per arc degree for the Earth's surface
mPerDegree = pi*radiusEarth/180;

%% Compute
%... Reverse projection
x = (lon - lon0).*(mPerDegree*cosd(lat0));
y = (lat -lat0).*mPerDegree;

end

