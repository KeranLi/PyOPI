function [liftMaxPred, elevationPred, subsidencePred] = ...
    lifting(x, y, hGrid, pGrid, azimuth, U, tauF, ijCatch, ptrCatch)
%... Calculate maximum lifting for sample locations. 
% The vector liftMaxPred gives is the maximum "lifting" along the wind path 
% upwind of each sample point, as represented in ijCatch and ptrCatch.
% The maximum lifting is calculated at the sample point or as a 
% precipitation-weighted average for the catchment above the sample point,
% depending on whether the sample is "local" or "catchment" (type L or C). 
% The vector elevationPred is provided for comparison, and it contains
% either local elevation of the precipitation-weighted average elevation 
% the upstream catchment, depending on whether the sample location is 
% type L or C. 
% This calculation accounts for the average horizontal offset of 
% during fall of precipitation to the ground. This is done by adjusting 
% Xst and Yst from the windGrid function to locations upwind by U*tauF. 
% The vector elevationPred gives the precipitation-weighted average
% for catchment above the specified sample site. If the sample location
% is type L, then the local elevation of the sample location is returned. 
% The vector subsidencePred gives the difference between the maximum
% upwind elevation and the elevation of the sample, with averaging
% determined by the sample type, either L or C.
%
% Mark Brandon, Yale University, 2016-2021

%% Initialize variables
nSamples = length(ptrCatch);
liftMaxPred = nan(nSamples, 1);
elevationPred = nan(nSamples, 1);
subsidencePred = nan(nSamples, 1);
if nSamples==0, return, end

%% Start calculation
%... Transform topography to wind coordinates (s,t,z is right handed)
[Sxy, Txy, s, t, Xst, Yst] = windGrid(x, y, azimuth);

%... Offset topography upwind to account for downwind transport
% during fallout of the precipitation. The upwind offset is accomplished
% by subtracting the offset from the grid coordinates Xst and Yst.
Xst = Xst - U*sind(azimuth)*tauF;
Yst = Yst - U*cosd(azimuth)*tauF;
% Note: griddedInterpolant uses the ndgrid format, so that the order
% of the grid vectors, x and y, are reversed to account for the
% meshgrid format for hGrid. Also note that grid vectors must be
% specified as cell variables. The setting 'none' causes extrapolated
% values to be set to nans, which are then found and set to zero.
F = griddedInterpolant({y, x}, hGrid, 'linear', 'none');
hWind = F(Yst, Xst);
clear F
hWind(isnan(hWind)) = 0;

%... Calculate maximum upwind lifting for points in each column
liftMaxWind = cummax(hWind);
clear hWind
%... Transform liftMax back to geographic coordinates
F = griddedInterpolant({s, t}, liftMaxWind, 'linear', 'none');
clear liftMaxWind 
liftMaxGrid = F(Sxy, Txy);

%... Calculate subsidence, using a grid of maximum upwind elevation
% with no offset.
% Note: griddedInterpolant uses the ndgrid format, so that the order
% of the grid vectors, x and y, are reversed to account for the
% meshgrid format for hGrid. Also note that grid vectors must be
% specified as cell variables. The setting 'none' causes extrapolated
% values to be set to nans, which are then found and set to zero.
F = griddedInterpolant({y, x}, hGrid, 'linear', 'none');
hWind = F(Yst, Xst);
clear F
hWind(isnan(hWind)) = 0;
%... Calculate maximum upwind elevation for points in each column
elevationMaxWind = cummax(hWind);
clear hWind
%... Transform elevationMax back to geographic coordinates
F = griddedInterpolant({s, t}, elevationMaxWind, 'linear', 'none');
clear elevationMaxWind
elevationMaxGrid = F(Sxy, Txy);
clear s t Sxy Txy F

%... Calculate averages, and account for both type L and C sample locations
for k = 1:nSamples
    % Extract indices for sample catchment
    ij = catchmentIndices(k, ijCatch, ptrCatch);
    % Calculate sum of precipitation rates for catchment
    pSum = sum(pGrid(ij));
    if pSum>0
        % Calculate precipitation-weighted maximum lifting
        liftMaxPred(k) = ...
            sum(pGrid(ij).*liftMaxGrid(ij)) / pSum;
        % Calculate precipitation-weighted elevation
        elevationPred(k) = ...
            sum(pGrid(ij).*hGrid(ij)) / pSum;
        % Calculate precipitation-weighted maximum elevation
        subsidencePred(k) = ...
            sum(pGrid(ij).*elevationMaxGrid(ij)) / pSum ...
            - elevationPred(k);        
    else
        % No precipitation, so use uniform weighting
        liftMaxPred(k) = mean(liftMaxGrid(ij)); 
        elevationPred(k) = mean(hGrid(ij));
        subsidencePred(k) = mean(elevationMaxGrid(ij)) ...
            - elevationPred(k);        
    end
end

end