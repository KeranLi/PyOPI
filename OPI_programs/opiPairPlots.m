function opiPairPlots
%... Create pair plots for candidate solutions generated in OPI search
% for best-fit solution.
% The input file should have the following structure:
% (Lines that start with a % character are ignored.)
% Line 1: Run title string
% Line 2: Number of samples in observed data used for fit
% Line 3: Parameter labels, separated by the "|" character
% Line 4: Power-of-10 exponents for scaling parameter values for plots
% Line 5: Lower constraints for parameter search
% Line 6: Upper constraints for parameter search
% Subsequent lines: Candidate solution for each row, with nSolutions rows.
%      The first element of the row gives the solution number.
%      The second element gives the reduced chi-square for the solution.
%      The remaining elements give the parameters for each solution.
% After the last solution:
% Line: SOLUTION
% Line: Reduced X2 for best-fit solution
% Line: Degrees of freedom for best-fit solution
% Line: Best-fit solution
% A parameter that has a fixed value for all solutions will be reported
% in the log file, but will be otherwise ignored for the pair plots.
% A fixed value means that the lower and upper constraints for that
% parameter are equal (no variation is allowed).
%
% v. 1.1: Before remaining elements are the nParameters parameter values
%         for that solution. % Lines with leading % symbol are ignored.
% Function opiFitCRS_Solutions.m writes most of this information,
% except for the plotting, solution values are multiplied by 10^exponent
%         (which is opposite the normal convention).
% v. 1.2: Solution values are assumed to have already been multipled by
%         by 10^exponent.
% v. 1.3: The definition of exponent has been reversed so that solution
%         values are multiplied by 10^-exponent, and 10^exponent is
%         is the scaling factor shown in the axis label.
% v. 1.4: Accounts for parameters that are constrained to a fixed value.
%         Revised definition for epsilon = stdev(F_searchSet).
% Dependencies: cmapscale.m, fnPersistentPath.m, getSolutions.m, 
%   printFigure.m

% Mark Brandon, Yale University, 2016-2022

%% Initialize system
close all
clc
dbstop if error

%% Constants
%... Start time for calculation
startTime = datetime;

%% Select solutions file
solutionsPath = fnPersistentPath;
[solutionsFile, solutionsPath] = ...
    uigetfile([solutionsPath, '/*opiFit_Solutions*.*']);
if solutionsFile==0, error('No solutions file selected'), end
fnPersistentPath(solutionsPath);

%% First phase of reporting results
%... Start diary file
logFilename=[solutionsPath, '/', mfilename, '_Log.txt'];
if isfile(logFilename), delete (logFilename); end
diary(logFilename);
%... General information
fprintf ('Program: %s\n', mfilename)
fprintf('Start time: %s\n', startTime)
fprintf('Path for solutions file:\n%s\n', solutionsPath)
fprintf('Filename for solutions file:\n%s\n', solutionsFile)
% fprintf(['\n>> Read solutions file, and report lines that contain nulls.\n', ...
%         '   (Note that the data are recovered by stripping out the nulls.)\n']);

%% Parse solutions file
[solutionsTitle, nSamples, parameterLabelsAll, exponentsAll, ...
    lbAll, ubAll, mSet, epsilon0, ...
    solutionsAll, chiR2Best, nuBest, betaBest] = ...
    getSolutions(solutionsPath, solutionsFile);
% fprintf('>> End of report for null lines\n\n');

%... Check lengths for nParameters, parameter labels, and power-of-ten factors 
nParameters = length(parameterLabelsAll);
if nParameters~=(size(solutionsAll, 2) - 2) || nParameters~=length(exponentsAll)
    error('Mismatch between number of parameters, parameter labels, and power-of-ten factors')
end

%... Parse solutions, and remove solutions with chiR2==nan
iNan = isnan(solutionsAll(:,1));
solutionsAll = solutionsAll(~iNan, :);
chiR2All = solutionsAll(:,1);
solutionsAll = solutionsAll(:,3:end);
nSolutionsAll = length(chiR2All);

%... Calculate epsilon as a function of search iteration
epsilon = zeros(nSolutionsAll-(mSet-1),1);
f = chiR2All;
fSet = f(1:mSet);
epsilon(1) = std(fSet);
for iSearch = mSet+1:nSolutionsAll
    % Increment fSet with the next solution, and sort
    fSet = sort([fSet; f(iSearch)]);
    % Update fSet with mSet best solutions
    fSet = fSet(1:mSet);
    % Calculate associated epsilon
    epsilon(iSearch-(mSet-1)) = std(fSet);
end

%... Apply power-of-10 factoring for parameter values
% For this version, each value, x, is converted via x*10^-exponents.
solutionsAll = ...
    solutionsAll.*repmat((10.^-exponentsAll)', nSolutionsAll, 1);
lbAll = lbAll.*10.^-exponentsAll';
ubAll = ubAll.*10.^-exponentsAll';

%... Check to find free parameters
iFree = (ubAll - lbAll)~=0;

%... Trim down to free parameters
parameterLabelsFree = parameterLabelsAll(iFree');
lbFree = lbAll(iFree);
ubFree = ubAll(iFree);
nParametersFree = sum(iFree);

%... Calculate chiR2 offset for 95 percent confidence regions using the
% chi2 contour method of Press et al. 2007, Numerical Recipes (p. 812-816). 
% The 95 percent confidence limits for point and paired estimates
% has chi2 offset from the best-fit chi2 value equal to 3.84 and
% 5.99, respectively (from chi2inv(0.95, 1), and chi2inv(0.95, 2)).
% The solutions that lie within the confidence region are defined by
% chiR2 <= chiR2Limit and chiR2 > chiR2Best.
chiR2UnivariateLimit = chiR2Best + 3.84/nuBest;
chiR2BivariateLimit = chiR2Best + 5.99/nuBest;
iUnivariate95CL = (chiR2All<=chiR2UnivariateLimit) & (chiR2All>=chiR2Best);
iBivariate95CL = (chiR2All<=chiR2BivariateLimit) & (chiR2All>=chiR2Best);
 
%% Report results
fprintf('Solutions title: \n%s\n\n', char(solutionsTitle));
%... Tables showing search constraints, best-fit solution, and confidence limits
% Table 1. Parameter constraints
fprintf('Table 1. The parameter constraints used in the search.\n')
fprintf('%-s\n', '                               Power-of-10')
fprintf('%-s\n', 'Axis Labels                      Factors  (  Search Constraints  )')
for k=1:nParameters
    fprintf('%-2d: %-31s %-+4d  (%-10g, %-10g)   \n', ...
        k, parameterLabelsAll{k}, exponentsAll(k), ...
        lbAll(k).*10.^exponentsAll(k), ubAll(k).*10.^exponentsAll(k));
end
fprintf('\n\n');

% Table 2. Best-fit solution and confidence limits
fprintf('Table 2. The best-fit solution and associated univariate 95-percent confidence limits.\n')
fprintf('%-s\n', ...
    'Axis Labels                     Estimate        ( 95% Confidence Limit )')
for k=1:nParameters
    fprintf('%-2d: %-26s ', k, parameterLabelsAll{k});
    if iFree(k)
        %... Report best fit and 95% confidence limit for kth parameter
        betaUnivariateLw95CL = min(solutionsAll(iUnivariate95CL,k)).*10.^exponentsAll(k);
        betaUnivariateHi95CL = max(solutionsAll(iUnivariate95CL,k)).*10.^exponentsAll(k);
        fprintf(' %-15g (%-15g, %-15g)\n', ...
            betaBest(k), betaUnivariateLw95CL, betaUnivariateHi95CL);
    else
        %... Report fixed value for kth parameter
        fprintf(' Fixed Parameter\n');
    end
end
fprintf('\n\n');

%... Background details
fprintf('------- Background Details about the Search  --------\n')
fprintf('Number of observations for fit: %d\n', nSamples);
fprintf('Number of parameters: %d\n', nParameters);
fprintf('Number of free parameters: %d\n', nParametersFree);
fprintf('Reduced chi-square for best-fit solution: %g\n', chiR2Best);
fprintf('Degrees of freedom for best-fit solution: %d\n', nuBest);
fprintf('Size of solution set: %d\n', mSet);
fprintf('Specified epsilon for stopping criterion, epsilon: %g\n', epsilon0);
fprintf('Total number of solutions (ignoring nan solutions): %d\n', nSolutionsAll)
fprintf('Number of solutions with chiR2==nan: %d\n', sum(iNan));
fprintf('Reduced chi-square for bivariate 95%% confidence limit: %g\n', chiR2BivariateLimit);
fprintf('Number of solutions within bivariate 95%% confidence limits: %d\n', sum(iBivariate95CL));

%... Close the diary
diary off

%% Plot figures
%... Fig01: Plot progress of reduced chi-square
Fig01

%... Fig02: Plot progress of epsilon
Fig02

%... Fig03: Plot, from worst to best, those solutions within the bivariate 
% 95% confidence region (solutions with chiR2 < chiR2Best are ignored). 
chiR2 = chiR2All(iBivariate95CL);
solutions = solutionsAll(iBivariate95CL, :);
[chiR2, iSort] = sort(chiR2, 'descend');
solutions = solutions(iSort, iFree);
Fig03_04(3)

%... Fig04: Plot solutions starting with worst and ending with best
% for the range chiR2Best <= chiR2 < 6*chiR2Best.
chiR2 = chiR2All(chiR2All>=chiR2Best & chiR2All<6*chiR2Best);
solutions = solutionsAll(chiR2All>=chiR2Best, :);
[chiR2, iSort] = sort(chiR2, 'descend');
solutions = solutions(iSort, iFree);
Fig03_04(4)

%% Fig 01, Plot progress of reduced chi-square
    function Fig01
        figure(1)
        plot(log10(chiR2All), 'b-', 'LineWidth', 2);
        xlabel('Iteration during Search', 'FontSize', 14);
        ylabel('Log10 Reduced \chi^2', 'FontSize', 14);
        hA = gca;
        hA.FontSize = 14;
        hA.LineWidth = 2;
        hA.XLim(1) = 0;
        hA.YLim(1) = 0;
        %.... Write title
        str = {'Fig. 1. Minimum reduced chi-square as function of the search iteration'; ...
            solutionsTitle};
        hT = title(str,'interpreter','none','FontSize',10);
        hT.Units = 'normalized';
        hT.Position(2) = hT.Position(2) + 0.02;        
        %... Save figure in pdf format
        printFigure(solutionsPath);
    end

%% Fig 02, Plot progress of epsilon
    function Fig02
        figure(2)
        hold on
        plot([mSet, nSolutionsAll], log10(epsilon0*[1 1]), 'r-');
        plot(mSet:nSolutionsAll, log10(epsilon), 'b-', 'LineWidth', 2);
        xlabel('Iteration during Search', 'FontSize', 14);
        ylabel('Log_{10} Epsilon', 'FontSize', 14);
        hA = gca;
        hA.FontSize = 14;
        hA.LineWidth = 2;
        hA.XLim(1) = 0;
        hA.YLim(1) = log10(epsilon0)-1;
        %.... Write title
        str ={'Fig. 2. Termination variable, epsilon, as a function of the search iteration'; ...
            solutionsTitle};
        hT = title(str,'interpreter','none','FontSize',10);
        hT.Units = 'normalized';
        hT.Position(2) = hT.Position(2) + 0.02;        
        %... Save figure in pdf format
        printFigure(solutionsPath);
    end

%% Figs 03 and 04: Pair plot
    function Fig03_04(figNum)
        %... Scale font size for axis labels and tic labels in pair plots,
        % relative to 10 point font when plotting 9 parameters.
        fontSizeAxes = 10*(9/(nParameters - 1));
        %... Scale symbols for plotting location of solutions, 
        % relative to 10 point marker size when plotting 9 parameters.
        markerSizeSolutions = 10*(9/(nParameters - 1));

        figure(figNum)
        bigAx = newplot;
        hFig = ancestor(bigAx, 'figure');
        set(bigAx, 'Visible', 'off', 'color', 'none');
        % Calculate position and size of plots
        pos = get(bigAx, 'Position');
        width = pos(3)/(nParametersFree-1);
        height = pos(4)/(nParametersFree-1);
        space = 0.1; % 10 percent space between axes
        pos(1:2) = pos(1:2) + space*[width height];
        xlim = zeros([nParametersFree, nParametersFree-1, 2]);
        ylim = zeros([nParametersFree, nParametersFree-1, 2]);
        bigAxHV = get(bigAx, 'HandleVisibility');
        bigAxParent = get(bigAx, 'Parent');
        %... Set master colorbar
        hCB = colorbar(bigAx, 'north');
        cMap0 = flipud(parula(254));
        colormap(hCB, cMap0);
        hCB.Position(1) = hCB.Position(1) + hCB.Position(3)/3;
        hCB.Position(3) = hCB.Position(3)/2;
        switch figNum
            case 3
                xlabel(hCB, 'Reduced \chi^2 (95% CL)');
            case 4
                xlabel(hCB, 'Reduced \chi^2 (Solutions < 6×min[\chi_r^2])');
        end
        %... Rescale cMap1 for plots and construct ticks and labels
        % for colorbar. hCB.Ticks are rescaled for the [0 1] limits.
        nTicks = 6;
        nRound = 2;
        [cMap1, hCB.Ticks, hCB.TickLabels] = ...
            cmapscale(chiR2, cMap0, 1, [], nTicks, nRound);
        hCB.FontSize = 12;
        hCB.Ticks = (hCB.Ticks - hCB.Ticks(1))/(hCB.Ticks(end) - hCB.Ticks(1));

        %... Draw plots
        % Preallocate graphics-object array
        ax = gobjects(nParametersFree, nParametersFree-1);
        for i = 2:nParametersFree
            for j = 1:i-1
                %... Plot lower triangle, where i>j.
                % i and j are row and column positions in full plot matrix
                %... Initialize axis for plot
                axPos = [pos(1)+(j-1)*width, pos(2)+(nParametersFree-i)*height, ...
                    width*(1-space), height*(1-space)];
                ax(i,j) = axes('Position', axPos, 'HandleVisibility', bigAxHV, ...
                    'parent', bigAxParent);
                %... Set background for each plot
                ax(i,j).Color = [0.75 0.75 0.75];
                ax(i,j).LineWidth = 1.0;
                ax(i,j).TickLength = [0.03,0.05];
                ax(i,j).TickDir = 'out';
                ax(i,j).FontSize = fontSizeAxes;
                ax(i,j).Box = 'on';
                ax(i,j).Layer = 'top';
                ax(i,j).Visible = 'on';
                %... Find data limits and increase to create 20 percent margin
                xMinAll = lbFree(j);
                xMaxAll = ubFree(j);
                xMin = min(solutions(:,j));
                xMax = max(solutions(:,j));
                xBest =  solutions(end,j);
                dX = 1.20*max([xBest - xMin, xMax - xBest]);
                yMinAll = lbFree(i);
                yMaxAll = ubFree(i);
                yMin = min(solutions(:,i));
                yMax = max(solutions(:,i));
                yBest =  solutions(end,i);
                dY = 1.20*max([yBest - yMin, yMax - yBest]);
                %... Set axis limits
                xlim(i,j,1) = max([xBest - dX, xMinAll]);
                xlim(i,j,2) = min([xBest + dX, xMaxAll]);
                ylim(i,j,1) = max([yBest - dY, yMinAll]);
                ylim(i,j,2) = min([yBest + dY, yMaxAll]);
                ax(i,j).XLim = xlim(i,j,:);
                ax(i,j).YLim = ylim(i,j,:);
                ax(i,j).XGrid = 'off';
                ax(i,j).YGrid = 'off';
                hold(ax(i,j), 'on');
                
                %... Label plots for testing purposes
                %str=sprintf('%d, %d',i,j);
                %text(xMin+(xMax-xMin)/2,yMin+(yMax-yMin)/2,str, ...
                %     'HorizontalAlignment','center','fontsize',10);
                
                %... Set color map
                colormap(ax(i,j), cMap1);
                %... Plot location with crossing lines for best-fit solution
                plot(ax(i,j), [solutions(end,j), solutions(end,j)], ...
                    [ylim(i,j,1), ylim(i,j,2)], '-', ...
                    'color', 'r', 'linewidth',1);
                plot(ax(i,j), [xlim(i,j,1), xlim(i,j,2)], ...
                    [solutions(end,i), solutions(end,i)], '-', ...
                    'color', 'r', 'linewidth',1);
                
                %... Plot locations for candidate solutions, which are ordered
                % so that the best solutions are plotted last.
                % plot(solutions(~iBest,j), solutions(~iBest,i), '.b');
                scatter(solutions(:,j), solutions(:,i), ...
                    25, chiR2, 'filled', 'SizeData', markerSizeSolutions);
                
                %... Plot red circle for best-fit solution
                plot(ax(i,j),solutions(end,j), solutions(end,i), ...
                    'or', 'MarkerSize', markerSizeSolutions);
                
                %... Draw lines for right and bottom axis lines to account
                % for error in Matlab
                plot(ax(i,j), [xlim(i,j,1),xlim(i,j,1)], ...
                    [ylim(i,j,1), ylim(i,j,2)], ...
                    '-k', 'linewidth',1);
                plot(ax(i,j), [xlim(i,j,1),xlim(i,j,2)], ...
                    [ylim(i,j,1), ylim(i,j,1)], ...
                    '-k', 'linewidth',1);
            end
        end
        
        %... Add axis labels to the right and bottom of the plot matrix, and
        % turn off ticks inside the plot matrix
        for i = 2:nParametersFree
            for j = 1:i-1
                if i==nParametersFree
                    xlabel(ax(i,j), parameterLabelsFree{j});
                    ax(i,j).FontSize = fontSizeAxes;
                else
                    ax(i,j).XTickLabel = '';
                end
                if j==1
                    ylabel(ax(i,j), parameterLabelsFree{i});
                    ax(i,j).FontSize = fontSizeAxes;
                else
                    ax(i,j).YTickLabel = '';
                end
            end
        end
        
        %... Link X axes in each column
        for j = 1:nParametersFree-1
            linkaxes(ax(j+1:nParametersFree,j), 'x');
        end
        
        %... Make bigAx the CurrentAxes
        set(hFig,'CurrentAx',bigAx)
        
        %... Set bigAx so Title and X/YLabel visibility are on and strings are empty
        set([get(bigAx,'Title'); get(bigAx,'XLabel'); get(bigAx,'YLabel')], ...
            'String', '', 'Visible', 'on')
        
        %.... Write title
        switch figNum
            case 3
                str = 'Figure 3. Confidence limits (95%) for best-fit solution.';
            case 4
                str = 'Figure 4. Maps showing reduced \chi^2 aroung best-fit solution.';
        end
        str = {str; solutionsTitle};
        title(bigAx, str,'interpreter','none','FontSize',10);
        bigAx.Position(2) = bigAx.Position(2) + 0.01;
        %... Add figure label
        
        %... Save figure in pdf format
        printFigure(solutionsPath);
    end
end
