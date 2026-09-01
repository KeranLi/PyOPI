function opiSyntheticTopography
% Create synthetic topography and associated mat file for opi. 

%% Initialize system
close all
clc
dbstop if error

%% Define parameters for topography
%... Longitude and latitude at centroid (degrees)
lon0 = 0;
lat0 = 0;
%... Target grid spacing in easting (x) and northing (y) directions (m)
dX = 1000; 
dY = 1000;
%... Size of grid in the easting (x) and northing (y) directions (m)
lX = 700e3;
lY = 700e3;
%... Calculate longitude and latitude grid vectors
nX = ceil(lX/dX) + 1;
dX = lX/(nX - 1);
xMin = -lX/2;
x = xMin + (0:nX-1)*dX;
nY = ceil(lY/dY) + 1;
dY = lY/(nY - 1);
yMin = -lY/2;
y = yMin + (0:nY-1)*dY;
[lon, lat] = xy2lonlat(x, y, lon0, lat0);
%... gridRead requires that lon be a row vector, 
% and lat be a column vector.
lon = lon(:)';
lat = lat(:);

%% Calculate topography
%... Model topography: North-directed sinusoids at equator
% Wavelength, amplitude (trough to crest), and offset from start (m)
wavelength = 100e3;
amplitude = 3e3;
offsetHorizontal = 100e3;
offsetVertical = 1;
hGrid = amplitude* ...
    (1 + sin(2*pi*(y - (yMin + offsetHorizontal))/wavelength - pi/2))/2;
hGrid(y<(yMin + offsetHorizontal)) = 0;
hGrid = hGrid + offsetVertical;
hGrid = repmat(hGrid(:), 1, nX);
% save('data/NorthDirectedSinusoidalTopography_3km height_lat0N.mat', ...
%     'lon', 'lat', 'hGrid', '-v7.3')

figure(1)
pcolor(lon, lat, hGrid)
shading interp
title('North-Directed Sinusoids')
xlabel('Longitude (degrees)')
ylabel('Latitude (degrees)')

%... Model topography: East-directed sinusoids at equator
% Wavelength, amplitude (trough to crest), and offset from start (m)
wavelength = 100e3;
amplitude = 3e3;
offsetHorizontal = 100e3;
offsetVertical = 1;
hGrid = amplitude* ...
    (1 + sin(2*pi*(x - (xMin + offsetHorizontal))/wavelength - pi/2))/2;
hGrid(x<(xMin + offsetHorizontal)) = 0;
hGrid = hGrid + offsetVertical;
hGrid = repmat(hGrid(:)', nY, 1);
% save('data/EastDirectedSinusoidalTopography_3km height_lat0N.mat', ...
%     'lon', 'lat', 'hGrid', '-v7.3')

figure(2)
pcolor(lon, lat, hGrid)
shading interp
title('East-Directed Sinusoids')
xlabel('Longitude (degrees)')
ylabel('Latitude (degrees)')

%... Model topography: East-directed Gaussian at equator
% Sigma, amplitude (trough to crest) (m)
sigma = 100e3/3;
amplitude = 3e3;
offsetVertical = 1;
hGrid = amplitude*exp(-0.5*(x/sigma).^2);
hGrid = hGrid + offsetVertical;
hGrid = repmat(hGrid(:)', nY, 1);
% save('data/EastDirectedGaussianTopography_3km height_lat0N.mat', ...
%     'lon', 'lat', 'hGrid', '-v7.3')

figure(3)
pcolor(lon, lat, hGrid)
shading interp
title('East-Directed Gaussian')
xlabel('Longitude (degrees)')
ylabel('Latitude (degrees)')

%... Model topography: East-directed Gaussian at 45 degrees north
% Create new grid vectors for hGrid
lon0 = 0;
lat0 = 45;
[lon, lat] = xy2lonlat(x, y, lon0, lat0);
%... gridRead requires that lon be a row vector, 
% and lat be a column vector.
lon = lon(:)';
lat = lat(:);

% Sigma, amplitude (trough to crest) (m)
sigma = 100e3/3;
amplitude = 3e3;
offsetVertical = 1;
hGrid = amplitude*exp(-0.5*(x/sigma).^2);
hGrid = hGrid + offsetVertical;
hGrid = repmat(hGrid(:)', nY, 1);
% save('data/EastDirectedGaussianTopography_3km height_lat45N.mat', ...
%     'lon', 'lat', 'hGrid', '-v7.3')

figure(4)
pcolor(lon, lat, hGrid)
shading interp
title('East-Directed Gaussian')
xlabel('Longitude (degrees)')
ylabel('Latitude (degrees)')

end