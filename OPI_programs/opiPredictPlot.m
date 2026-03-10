function opiPredictPlot
%... Construct figures for Malargue results based on analysis 
% produced by opiPredictCalc. 

% Mark Brandon, Yale University, 2021-2022

%% Initialize system
close all
clc
dbstop if error

%% Initialize variables
%... Start time for calculation
startTime = datetime;
% Kelvin to Celsius
TC2K = 273.15;

%... Load results from opiPredictCalc
matPath = fnPersistentPath;
[matFilename, matPath] = uigetfile([matPath,'/*.mat']);
if matFilename==0, error('opiPrecit matfile not found'), end
%... Remove terminal slash, if present
if matPath(end)=='/' || matPath(end)=='\'
    matPath = matPath(1:end-1);
end
fnPersistentPath(matPath);

% 'matfilesPath_Climate', 'matFileBenthicForamClimate', 'matFileMEBM', ...
% 'runTitle', 'listPath', 'listFile', ...
% 'dD2H0_dT0', 'seD2HPres', 'seD2H0Pres', ...
% 'sampleName', 'sampleLon', 'sampleLat', 'sampleH', ...
% 'sampleAge', 'sampleD2H', 'seD2HSample', ...
% 'd2H0Pres', 'd2HPresLocal', 'd2HPresCatch', ...
% 'iWetPresLocal', 'iWetPresCatch', ...
% 'd2H0Ref', 'd2HRefLocal', 'd2HRefCatch', ...
% 'PhiLiftLocal', 'PhiLiftCatch', ...
% 'T0Pres', 'TPres', 'T0Ref', 'TRef', 'PhiClim', 'sampleComment', ...
% 'ageRecord', 'T0Record', 'd2H0Record', ...
% 'T0RecordSmooth', 'd2H0RecordSmooth', '-v7.3');
    load([matPath, '/', matFilename], ...
    'matfilesPath_Climate', 'matFileBenthicForamClimate', 'matFileMEBM', ...
    'runTitle', 'listPath', 'listFile', ...
    'seD2HPres', 'seD2H0Pres', ...
    'sampleAge', 'sampleD2H', 'seD2HSample', ...
    'd2H0Pres', 'd2HPresLocal', 'd2HPresCatch', ...
    'd2H0Ref', 'd2HRefLocal', 'd2HRefCatch', ...
    'PhiLiftLocal', 'PhiLiftCatch', ...
    'T0Pres', 'T0Ref', 'PhiClim', ...
    'ageRecord', 'T0Record', 'd2H0Record', ...
    'T0RecordSmooth', 'd2H0RecordSmooth', 'spanAge');
    
%... Select option to use local versus catchment data
while true
    optionLC = input('Option: Local or catchment data (L, C): ', 's');
    optionLC = upper(optionLC(1));
    if strcmp('L', optionLC) || strcmp('C', optionLC), break, end
    fprintf('Incorrect selection. Try again.\n')
end
switch optionLC
    case 'L'
        d2HRef = d2HRefLocal;
        d2HPres = d2HPresLocal;
        PhiLift = PhiLiftLocal;
    case 'C'
        d2HRef = d2HRefCatch;
        d2HPres = d2HPresCatch;
        PhiLift = PhiLiftCatch;
end
clear d2HRefLocal d2HLocalPres PhiLiftLocal
clear d2HRefCatch d2HCatchPres PhiLiftCatch

%% Compute
%... Calculate SE for PhiLiftLocal and PiLiftTotal
% There are three error components. 
% The error for sample d2H, which is set to the SE for the sample.
% The errors associated with the reference d2H and base d2H at the
% location and age of the sample, where "reference" refers to the case 
% where the topography is set to the same as present, and "base" refers to
% the case where there is no topography (and no orographic precpitation). 
% The second and third error components are estimated using SE estimates
% from OPI for modern d2H and d2H0 at the sample location, and the 
% standard deviation and covariance for the d2H and d2H0 time series at 
% the sample location, which is designed to account for the low precision
% of the sample age relative to the variation with age in the d2H and d2H0
% time series. 
% Note that the SE for PhiLift is calculated in two ways. 
% The "total" estimate includes all of the errors indicated above.
% The "partial" estimate ignores the error associated high-frequency 
% variation in the time series.

%... Estimate time series for PhiClim and d2H at centroid
% PhiClim is empirically related to T0 for the samples using,
%    PhiClim = 1 + kClim*ln(T0Record/T0Pres),
% where T0Ref and T0Pres are in Kelvin, and kClim is a empirically 
% estimated dimensionless coefficient.
logRatioT0 = log(T0Ref ./ T0Pres);
% Best-fit value for kClim
kClim = sum((PhiClim - 1) .* logRatioT0) ./ sum(logRatioT0.^2);
% Use this empirical relation to estimate time series for PhiClim
PhiClimRecord = 1 + kClim*log(T0Record/T0Record(1));
PhiClimRecordSmooth = 1 + kClim*log(T0RecordSmooth/T0RecordSmooth(1));
% Calculate time series for d2H
d2HRecord = PhiClimRecord .* (d2HPres(1) - d2H0Pres(1)) + d2H0Record;
d2HRecordSmooth = ...
    PhiClimRecordSmooth .* (d2HPres(1) - d2H0Pres(1)) + d2H0RecordSmooth;

% Select record elements that correspond to the age range for the samples.
iSelect = ageRecord<=max(sampleAge) & ageRecord>=min(sampleAge);
nSelect = sum(iSelect);
% Calculate nSpan for specified spanAge
nSpan = ceil(nSelect*spanAge/(max(sampleAge) - min(sampleAge))); 
% For reporting, calculate high-frequency SD for d2H0 and d2 records.
sdD2H0Record = std(d2H0Record(iSelect) - d2H0RecordSmooth(iSelect));
sdD2HRecord = std(d2HRecord(iSelect) - d2HRecordSmooth(iSelect));

%... Construct variance matrix 
% Method: Tellinghuise, 2001, Statistical error propagation
% Note that V(1,1) refers to the variance associated with sampled2H.
% This value is added below since it varies with each sample.
% Calculate the "partial" variance, which ignores the high-frequency 
% variance estimated from the time series.
VPartial = zeros(3,3);
VPartial(2,2) = seD2HPres^2;
VPartial(3,3) = seD2H0Pres^2;
% Calculate the "total" variance, which includes all sources of variance,
% including variance estimated from time series.
VTotal = zeros(3,3);
% Calculate 2x2 covariance matrix for high-frequency variation 
% of d2H and d2H0. Note that the first factor adjusts for the 
% correct degrees of freedom.
VTotal(2:3,2:3) = (nSelect - 1)/(nSpan - 1) ...
             *cov(d2HRecord(iSelect) - d2HRecordSmooth(iSelect), ...
                  d2H0Record(iSelect) - d2H0RecordSmooth(iSelect));
% Add variance for OPI estimates for modern d2H and d2H0 values.
VTotal(2,2) = VTotal(2,2) + seD2HPres^2;
VTotal(3,3) = VTotal(3,3) + seD2H0Pres^2;

% Calculate SE(PhiLift) for each sample
nSample = length(sampleD2H);
sePhiLiftPartial = nan(nSample,1);
sePhiLiftTotal = nan(nSample,1);
for k = 1:nSample
    % Calculate derivatives for PhiLift
    % PhiLift = (sampleD2H - sampleD2H0) / (d2HRef - sampleD2H0)
    % g(1) = dPhiLift_dsampleD2H, g(2) = dPhiLift_dD2H, and
    % g(3) = dPhiLift_dRefD2H0;
    g = zeros(3,1);
    g(1) = 1 / ((log(1 + d2HRef(k)) - log(1 + d2H0Ref(k)))*(1 + sampleD2H(k))); 
    g(2) = -(log(1 + sampleD2H(k)) - log(1 + d2H0Ref(k))) ...
          / ((log(1 + d2HRef(k)) - log(1 + d2H0Ref(k)))^2*(1 + d2HRef(k)));
    g(3) = (log(1 + sampleD2H(k)) - log(1 + d2HRef(k))) ...
          / ((log(1 + d2HRef(k)) - log(1 + d2H0Ref(k)))^2*(1 + d2H0Ref(k)));     
    VPartial(1,1) = seD2HSample(k)^2;
    sePhiLiftPartial(k) = sqrt(g'*VPartial*g);
    VTotal(1,1) = seD2HSample(k)^2;
    sePhiLiftTotal(k) = sqrt(g'*VTotal*g);
end

%% Smooth sample results for plotting
% Estimate average interval between sample ages.
ageIntervalAverageSample = (max(sampleAge) - min(sampleAge))/ nSample;
% Set nSpan for loess smoothing to a length in the series equal to spanAge.
nSpan = ceil(spanAge/ageIntervalAverageSample);
% Smooth results
sampleD2HSmooth = smooth(sampleD2H, nSpan, 'loess');
d2HRefSmooth = smooth(d2HRef, nSpan, 'loess');
PhiLiftSampleSmooth = smooth(PhiLift, nSpan, 'loess');
% Use unsmoothed modern values for age = 0 Ma (last value in series)
sampleD2HSmooth(end) = d2HPres(1);
d2HRefSmooth(end) = d2HPres(1);
PhiLiftSampleSmooth(end) = 1;

%% Report results
logFilename=[matPath, '/', mfilename, '_Log.txt'];
if isfile(logFilename), delete (logFilename); end
diary(logFilename);
fprintf (['Program: ', mfilename, '\n'])
fprintf('Start time: %s\n', startTime)
fprintf('\n------- Source files used for opiPredictCalc --------\n')
fprintf('Path for climate matfiles:\n%s\n', matfilesPath_Climate)
fprintf('Mat filename for BenthicForamClimate data:\n%s\n', ...
    matFileBenthicForamClimate);
fprintf('Mat filename for MEBM data:\n%s\n', matFileMEBM)
fprintf('\n--------- opiPredictCalc matfile used here ----------\n')
fprintf('Path:\n%s\n', matPath)
fprintf('Filename:\n%s\n', matFilename)
fprintf('List file path:\n%s\n', listPath)
fprintf('List file name:\n%s\n', listFile)

fprintf('\n-------------- Background Information ---------------\n')
if strcmp(optionLC, 'L'), str = 'local'; else, str = 'catchment'; end 
fprintf('Option for precipitation data: %s\n', str);
fprintf('Average age interval between samples (Ma): %g\n', ageIntervalAverageSample)
fprintf('Average age interval between record values (Ma): %g\n', ...
    abs(ageRecord(2) - ageRecord(1)))
fprintf('Age span for lowess smoothing (Ma): %g\n', spanAge);
fprintf('\nAverage standard error for sample d2H (per mil): %g\n', ...
    sqrt(mean(seD2HSample.^2))*1e3);
fprintf('Standard error for OPI estimate of modern d2H (per mil): %g\n', ...
    seD2HPres*1e3)
fprintf('Standard error for OPI estimate of d2H0 (per mil): %g\n', seD2H0Pres*1e3)
fprintf('Standard deviation for high-frequency variation in d2H0 (per mil): %g\n', ...
    sdD2H0Record*1e3)
fprintf('Standard deviation for high-frequency variation in d2H (per mil): %g\n', ...
    sdD2HRecord*1e3);
fprintf('\nPhiClimate ≈ 1 + kClim*ln(T0Ref/T0Pres), kClim (dimensionless): %g\n', kClim) 
fprintf('\nAverage, min, and max for partial standard error for PhiLift: %g, %g to %g\n', ...
    rms(sePhiLiftPartial), min(sePhiLiftPartial), max(sePhiLiftPartial))
fprintf('Average, min, and max for total standard error for PhiLift: %g, %g to %g\n\n', ...
    rms(sePhiLiftTotal), min(sePhiLiftTotal), max(sePhiLiftTotal))

% Close diary for the last time
diary off

%% Plot
%... Create title string
titleStr = {runTitle; matFileBenthicForamClimate};

figure(1)
ax1 = subplot(2,1,1);
hold on
plot(ageRecord, d2H0Record*1e3, '-', 'LineWidth', 2, 'Color', [0.6 0.6 0.6])
plot(ageRecord, d2H0RecordSmooth*1e3, '-b', 'LineWidth', 2)
plot(sampleAge, d2HRefSmooth*1e3, '-b', 'LineWidth', 1);
plot(sampleAge, d2HRef*1e3, '.b', 'LineWidth', 2);
plot(sampleAge, sampleD2HSmooth*1e3, '-k', 'LineWidth', 3)
errorbar(sampleAge, sampleD2H*1e3, seD2HSample*1e3, '.r', 'MarkerSize', 15, 'LineWidth', 1)
plot(0, d2HPres(1)*1e3, 'or')
%... Format plot
box on
grid on
hA = gca;
hA.XLim = [0, 60];
hA.FontSize = 14;
hA.LineWidth = 1;
hA.XDir = 'reverse';
hA.XTickLabel = {};
% hA.YLim = [-140, 10];
ylabel(['Precipitation \delta^{2}H (', char(8240), ')']);
title('Fig. 1. Observed and Predicted Isotopic Fractionation');

ax2 = subplot(2,1,2);
hold on
plot(ageRecord, T0Record - TC2K, '-', 'LineWidth', 2, 'Color', [0.6 0.6 0.6])
plot(ageRecord, T0RecordSmooth - TC2K, '-b', 'LineWidth', 2)
%... Format plot
box on
grid on
hA = gca;
hA.XLim = [0, 60];
hA.FontSize = 14;
hA.LineWidth = 1;
hA.XDir = 'reverse';
% hA.YLim = [15, 30];
xlabel('Age (Ma)')
ylabel({'Sea-Level'; 'Temperature (°C)'});

% Modify positions to so that vertical extend of the top plot is 
% twice that for the bottom plot.
yBuff = 0.1;
height1 = 0.75*(1 - 2.5*yBuff);
height2 = 0.25*(1 - 2.5*yBuff);
ax1.Position(2) = 1.5*yBuff + height2;
ax1.Position(4) = height1;
ax2.Position(2) = yBuff;
ax2.Position(4) = height2;
% Print to pdf
printFigure(matPath)

figure(2)
hold on
plot(sampleAge, PhiLiftSampleSmooth, '-k', 'LineWidth', 3)
errorbar(sampleAge, PhiLift, sePhiLiftTotal, '.r', 'MarkerSize', 15, 'LineWidth', 1)
plot(0, 1, 'or') 
hL = refline(0, 1);
hL.LineStyle = '--';
hL.LineWidth = 2;
hL.Color = 'k';
uistack(hL, 'down');
%... Format plot
box on
grid on
hA = gca;
hA.FontSize = 14;
hA.LineWidth = 1;
hA.XDir = 'reverse';
hA.YLim(1) = 0;
xlabel('Age (Ma)')
ylabel('\Phi_{lift}', 'FontSize', 20);
title('Fig. 2. Predicted Evolution of \Phi_{lift}');
% Print to pdf
printFigure(matPath)

end
