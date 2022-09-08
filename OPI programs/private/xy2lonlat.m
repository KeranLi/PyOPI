function [lon, lat] = xy2lonlat(x, y, lon0, lat0)
% Convert from projected coordinates, x,y, to geographic coordinates
% The equirectangular projection is used here, which works well for
% maps that are smaller than about 1000 km.
% Input arguments
% x, y: easting, northing coordinates [meters] (scalar, vector, or matrix)
% lon0: central meridian [degrees] (scalar)
% lat0: standard parallel [degrees] (scalar)
% Note: The origin x,y = 0 lies at lon0, lat0.
% Output arguments
% lon, lat: longitude and latitude (degrees)

% Mark Brandon, Yale University, 2020

%% Constants
%... Mean radius of the Earth (m)
radiusEarth = 6371e3;
%... Meters per arc degree for the Earth's surface
mPerDegree = pi*radiusEarth/180;

%% Compute
%... Forward projection
lon = lon0 + x./(mPerDegree*cosd(lat0));
lat = lat0 + y./mPerDegree;

end

