function [x, y, z] = gridRead(matFileName)
% Extracts grid variables from a mat file.
% The function looks for and extracts an mxn numerical array
% (class single or double), and two matching vectors
% with lengths m and n, which correspond to the y and x grid vectors. 
% Note that x and y correspond to column and row directions in the array.
% The mat file can contain other non-numeric data, such as a string
% array with descriptive information. 
% 
% Mark Brandon, Yale University, January, 2019.

%% Compute
%... Get information about mat file
try
    W = whos('-file', matFileName);
catch ME
    if strcmp(ME.identifier, 'MATLAB:whos:fileIsNotThere')
        error('Select mat file does not exist.')
    end
end
%... Test to see if grid and two vectors are present
wSize = vertcat(W(:).size);
isClass = (strcmp('single', {W.class}) | strcmp('double', {W.class}))';
wSize(~isClass) = 0;
if (mod(length([W(isClass).size]),3)~=0 ...
    || sum(unique([W(isClass).size])~=1)>2)        
    error('%s\n%s', ...
        'Mat file must have only three numerical variables that are an mxn',...
        'matrix, and two matching grid vectors of length m and n.');    
end
%... Find grid and define grid size
[~, iGrid] = max(wSize(:,1).*wSize(:,2));
%... Find grid vectors
wSize(iGrid,:) = 0;
[~, iX] = max(wSize(:,2));
[~, iY] = max(wSize(:,1));
%... Load mat file, and extract grid vectors and grid
S = load(matFileName);
x = S.(W(iX).name);
y = S.(W(iY).name);
z = S.(W(iGrid).name);

end


