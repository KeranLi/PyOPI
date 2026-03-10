function [betaMin, fMin] = fminCRS3(fun, betaLB, betaUB, ...
    mu, epsilon0, isParallel, fOut, solutions)
%fminCRS3 Controlled random search to minimize an objectve function.
%   The objective function is defined by [f, nu] = fun(beta),
%   where beta is a vector of parameters, and f and nu are scalars,
%   representing the value and degrees of freedom for the objective 
%   function.
%
%   betaMin = fminCRS3(f, betaLB, betaUB, mu, epsilon0)
%   finds betaMin that minimizes f, subject to constraints,
%   betaLB <= beta <= betaUB, where betaLB, betaUB, beta, and betaMIN 
%   are vectors of the same size. fun is a user-defined function to be 
%   minimized with respect to beta. f must be solely a function of
%   beta, but an anonymous function can be used to account for additional
%   variables required for the calculation, as illustrated below. 
%   The CRS3 search is controlled by 2 parameters.
%   The first, mu, is a factor that determines the size of the search set,
%   mSet = ceil(mu*(nFree - 1)), where nFree is the number of free 
%   parameters. 
%   The second, epsilon0, defines the stopping criterion, 
%   epsilon0 > stdev(f(1:mSet)) < epsilon0, where f(1:mSet) is the mSet 
%   of candidate solutions with the smallest f values, 
%   and stdev is the standard deviation function. 
%   Recommended values: mu = 10, epsilon0 = 1e-4.

%   betaMin = fminCRS3(..., isParallel), where isParallel is a
%   logical variable, where false (default) and true indicate
%   serial and parallel computation, respectively.
%
%   betaMin = fminCRS3(..., isParallel, fOUT), where
%   fOUT is a handle that points to a user-defined function to record
%   results during the search. If fOUT is empty or missing, then no
%   intermediate results are recorded.
%
%   betaMin = fminCRS3(..., isParallel, fOUT, solutions),
%   where solutions is an array with a set of "restart" solutions,
%   which allow one to restart an aborted search from a previous
%   solutions file. Each row corresponds to a trial solution, starting
%   with the reduced chi-square value and nu (degrees of freedom), and
%   followed by the parameters beta. If solutions is empty or missing,
%   then the search starts with a new set of initial solutions.
%
%   [betaMin, nu] = fminCRS3(...) returns nu, the degrees of freedom
%   associated with the minimum solution, which can be used to 
%   calculate uncertainties for the estimate of betaMin. 
%
%   Example, with illustration of how to pass extra parameters:
%     function crsExample
%         %... Set lower and upper bounds
%         betaLB = [0, 0, 0];
%         betaUB = [10, 10, 10];
%         %... Define anonymous function, which allows passing
%         % of additional fixed parameters.
%         argFixed = [5, 4, 6];
%         chiR2 = @(beta) myFun(beta, argFixed);
%         %... Call fminCRS3 function
%         beta = fminCRS3(chiR2, betaLB, betaUB, mu0, mu, epsilon0);
%     end
%     function [chiR2, nu] = myFun(beta, argFixed)
%         % User function to be minimized for beta
%         chiR2 = beta(1)*exp(argFixed) + beta(2);
%         nu = 100;
%     end
%
%   Algorithm: Modified after the original CRS3 version of Price (1987).
%   The modifications, which are from Brachetti et al. (1997) and
%   Jiao et al. (2006), are as follows:
%   1) Weighted calculation of centroid and reflection points (Brachetti).
%   2) Three-point quadratic solution (Jiao), which exploits the quadratic
%      algorithm of Brachetti but with fewer points.
%   3) Independent sizes for the initial and main search sets. 
%   3) The weighting parameter omega is set to 1000.
%
%   Note on omega: Brachetti's algorithm uses the chiR2 variable to weight 
%   estimates for new points (weighted centroid, and weighted reflect). 
%   The influence of the weighting is gradually increased as the 
%   epsilon decreases. The variable omega (set below in the code) is a 
%   non-negative constant that determines the amount of weighting at the 
%   start of the search. Brachetti suggested setting omega = 1000, 
%   which means that there is almost no initial weighting, which should
%   result in a slightly more explorative search at the start. Jiao
%   found that omega = 1.1 gave a faster convergence time, persumably 
%   because the search is more exploitive of the available search
%   results. Setting omega = 0 cause the weighting to remain the 
%   same throughout the search. My preference is omega = 1000.
%
%   References:
%   Brachetti, P., De Felice Ciccoli, M., Di Pillo, G., and
%      Lucidi, S., 1997, A new version of the Price's algorithm for
%      global optimization: Journal of Global Optimization, v. 10,
%      p. 165-184.
%   Jiao, Y.-C., Dang, C., Leung, Y., Hao, Y., 2006. A
%      modification to the new version of the Price's Algorithm for
%      continuous global optimization problems. J Global Optim,
%      v. 36, p. 609-626.
%   Price, W.L., 1987. Global optimization algorithms for a CAD
%      workstation. Journal of Optimization Theory and Applications,
%      v. 55(1), p.133-146.
%
%   June 17, 2019: Set up separate mu0 and mu factors for initial and
%   main search sets.
%   Sept 15, 2019: I removed quasi-random search, which works poorly with 
%   nParameter > ~5.
%   July 24, 2022: Return to using mu for both initial and main search
%   sets, and set up a new stopping criteron, where the search is halted
%   when stdev(fBest) < epsilon0, where fBest is the mSet of candidate
%   solutions with the smallest f values. 

%   Mark Brandon, Yale University, 2005, 2015-2021

%% Initialize variables
%... Set the CRS3 variable, omega, which scales the weighting
% during the search. Recommended value: 1000
omega = 1000;

%% Seed random-number generator using the current time
rng('shuffle');

%% Parse input arguments
%... Check input arguments
narginchk(5,8)

%... Check search parameters
if mu<1, error('mu, as used in fminCRS3, is not correctly set'), end
if epsilon0<=0, error('epsilon, as used in fminCRS3, is not correctly set'), end

%... Convert to row vectors and check bounds
betaLB = betaLB(:)';
betaUB = betaUB(:)';
nParameters = length(betaLB);
if length(betaLB) ~= length(betaUB)
    error('Error: vectors for upper and lower bounds must be same size');
end
if any(betaLB > betaUB)
    error('Error: upper bounds must be greater than lower bounds');
end

%... Determine n, the number of free parameters
nFree = sum((betaUB - betaLB)~=0);

%... Define size of solution set (see Price, 1987, p. 136) 
mSet = ceil(mu*(nFree + 1));

%... Set parallel mode flag 
% Determine number of available workers
nWorkers = feature('NumCores');
if ~islogical(isParallel)
    error('isParallel argument must be set to logical value')
end
if nargin < 6 || isempty(isParallel) || nWorkers==1
    isParallel = false; 
end

%... Process argument for fOut function
if nargin < 7
    %... No fOut function
    fOut = [];
else
    %... Validate fOut function
    if ~isa(fOut, 'function_handle')
        error('Last argument is not a properly formed function handle')
    end
end

%... Process solutions argument
if nargin < 8
    %... No solutions argument
    solutions = [];
    %... Validate solutions argument
    if ~isnumeric(solutions)
        error('Solutions from restart file are not numeric.')
    end
end

%... Send note to fOut function about search parameters
if ~isempty(fOut)
    str = sprintf(...
        ['%% Number of solutions in restart file: %d\n', ...
        '%% Size of search set:\n%d\n', ...
        '%% Specified epsilon for stopping criterion: \n%g\n'], ...
        size(solutions, 1), mSet, epsilon0);
    fOut('note', str);
end

% Send restart solutions to output function, if present
% Must do before initating parallel mode to avoid IO conflict
if ~isempty(solutions)
    if ~isempty(fOut)
        for i = 1:size(solutions, 1), fOut(solutions(i,:)); end
    end
end

% Setup serial or parallel mode, as selected
if isParallel
    % Retain one worker for Matlab client when total workers > 4
    if nWorkers>4, nWorkers = nWorkers-1; end
    %... Create parallel pool (delete pool first to initialize it)
    delete(gcp('nocreate'))
    p = parpool(nWorkers);
    nWorkers = p.NumWorkers;
    %... Initialize jobs array with future objects
    % Using trivial function (@disp) to initialize. fetchOutputs ensures
    % that all jobs have an initial state equal to "ready and read".
    jobs(1:nWorkers) = parfeval(p, @disp, 0, 0);
    fetchOutputs(jobs);
else
    %... Use serial mode
    nWorkers = 1;
end

%% Summarize starting condition of search
fprintf('Modified controlled random search, fminCRS3\n')
fprintf('Number of parameters = %g\n', nParameters);
fprintf('Number of free parameters = %d\n', nFree);
fprintf('Factor for size of search set, mu = %g\n', mu);
fprintf('Size of search set = %g\n', mSet);
fprintf('Specified epsilon for stopping criterion = %g\n', epsilon0);
fprintf('Workers for parallel calculation = %d\n\n', nWorkers);

%% Start search
%... Assemble intial search set. Parameters are stored in betaSet
% as n column vectors, corresponding to mSet potential solutions.
% The function values, which are to be minizimed, are stored in fSet.
% If restart solutions are provided (input argument: solutions),
% then they are added first to this search set.
%... Process restart solutions, if present
if ~isempty(solutions)
	% Remove solutions where f is equal to nan
	solutions = solutions(~isnan(solutions(:,1)),:);
	% Load initial solutions into search set
	m = size(solutions, 1);
	fSet(1:m,:) = solutions(:,1);
	betaSet(1:m,:) = solutions(:,3:end);
else
	m = 0;
	fSet = zeros(mSet,1);
	betaSet = zeros(mSet,nParameters);
end
%... Complete rest of initial solution set
if m<mSet
    if isParallel
        %... Process solutions in parallel mode
        while true
            % Start solutions on available workers
            while any([jobs.Read])
                i = find([jobs.Read], 1);                
                beta = betaLB + (betaUB - betaLB).*rand(1, nParameters);
                jobs(i) = parfeval(p, fun, 3, beta);
            end
            % Fetch a completed solution
            [i, f, nu] = fetchNext(jobs);
            beta = cell2mat(jobs(i).InputArguments);
            % Send trial solution to output function, if present
            if ~isempty(fOut), fOut([f, nu, beta]); end
            % Add to set (skip solutions where f==nan)
            if ~isnan(f)
                m = m + 1;
                betaSet(m,:) = beta;
                fSet(m) = f;
            end
            %... Check for exit
            if m==mSet
                % Cancel and reset jobs to "ready and read" status (no
                cancel(jobs);
                jobs(1:end) = parfeval(p, @disp, 0, 0);
                fetchOutputs(jobs);
                break
            end
        end
    else
        %... Process solutions in serial mode
        while m<mSet
            beta = betaLB + (betaUB - betaLB).*rand(1, nParameters);
            [f, nu] = feval(fun, beta);
            %... Send trial solution to output function
            if ~isempty(fOut), fOut([f, nu, beta]); end
            % Add to set (skip solutions where f==nan)
            if ~isnan(f)
                m = m + 1;
                betaSet(m,:) = beta;
                fSet(m) = f;
            end
        end
    end
end
%... Set up main search set
% Save best and worst solutions from the first 1:mSet solutions
f0Min = min(fSet(1:mSet));
f0Max = max(fSet(1:mSet));
% For all available solutions, save the best mSet solutions, and 
% also the best three and the worst of that set.
[~, iSort] = sort(fSet, 'ascend');
betaSet = betaSet(iSort(1:mSet),:);
fSet = fSet(iSort(1:mSet));
iMin = (1:3)';
fMin = fSet(iMin);
iMax = mSet;
fMax = fSet(iMax);
%... Start main search
isNewMin = false;
isNewJob = false;
while true
    if ~isParallel || any([jobs.Read])
        %... Generate new point in parameter space
        if isNewMin==true
            %... Try to estimate a better parameter point using an
            % quadratic approximation. Used when there is new solution
            % among the best three.
            beta = 0.5*( ...
                (betaSet(iMin(2), :).^2 - betaSet(iMin(3), :).^2)*fMin(1) ...
                + (betaSet(iMin(3), :).^2 - betaSet(iMin(1), :).^2)*fMin(2) ...
                + (betaSet(iMin(1), :).^2 - betaSet(iMin(2), :).^2)*fMin(3))...
                ./((betaSet(iMin(2), :) - betaSet(iMin(3), :))*fMin(1) ...
                + (betaSet(iMin(3), :) - betaSet(iMin(1), :))*fMin(2) ...
                + (betaSet(iMin(1), :) - betaSet(iMin(2), :))*fMin(3));
            %... Check if quadratic estimate is within specified bounds.
            % If true, set isNew==true, which signals a solution at this
            % point. If false, set isNew==false, which calls for the
            % reflection method below to find a solution within bounds.
            isNewMin = false;
            if all(beta >= betaLB) && all(beta <= betaUB), isNewMin = true; end
        end
        if isNewMin==false
            %... Use reflection method to estimate new parameter point.
            % Select at random N+1 trial solutions from the search set
            % Find a weighted centroid for 2:N+1 subset and then do a weighted
            % reflection of the first point around the weighted centroid to get
            % a new trial solution.
            while true
                iSubset = randi(mSet, nParameters+1, 1);
                betaSubset = betaSet(iSubset,:);
                fSubset = fSet(iSubset);
                %... Calculate weighted centroid
                phi = omega*(fMax - fMin(1))^2/(f0Max-f0Min);
                nu = 1./(fSubset(2:end) - fMin(1) + phi);
                w = nu./sum(nu);
                betaCenter = w'*betaSubset(2:end,:);
                fCenter = w'*fSubset(2:end);
                %... Calculate weighted reflection point
                if fCenter <= fSubset(1)
                    alpha = 1 - (fSubset(1) - fCenter)/(fMax - fMin(1) + phi);
                    beta = betaCenter - alpha*(betaSubset(1,:) - betaCenter);
                else
                    alpha = 1 + (fSubset(1) - fCenter)/(fMax - fMin(1) + phi);
                    beta = betaSubset(1,:) - alpha*(betaCenter - betaSubset(1,:));
                end
                %... Check if new point is within specified bounds
                if all(beta >= betaLB) && all(beta <= betaUB), break; end
            end
        end
        %... Submit parameter point for solution
        if isParallel
            i = find([jobs.Read], 1);
            jobs(i) = parfeval(p, fun, 2, beta);
        else
            [f, nu] = feval(fun, beta);
            isNewJob = true;
        end
    end
    if isParallel && any(strcmp({jobs.State},'finished') & ~[jobs.Read])
        %... Fetch completed job from parallel pool
        [i, f, nu] = fetchNext(jobs);
        beta = cell2mat(jobs(i).InputArguments);
        isNewJob = true;
    end
    if isNewJob == true
        %... Process results from newly completed job
        isNewJob = false;
        %... Send trial solution to output function, if present
        if ~isempty(fOut), fOut([f, nu, beta]); end
        % Check to see if there is new minimum solution
        isNewMin = f < fMin(3);
        %... Update solution set
        if f < fMax
            %... Use the trial solution as a replacement for
            % worst solution in the current search set.
            betaSet(iMax,:) = beta;
            fSet(iMax) = f;
            %... Sort revised search set
            [~, iSort] = sort(fSet, 'ascend');
            iMax = iSort(end);
            fMax = fSet(iMax);
            iMin = iSort(1:3);
            fMin = fSet(iMin);
            %... Check stopping criterion, which is defined
            % in terms of the standard deviation of f for the 
            % current search set. 
            if std(fSet) < epsilon0
                %... Stop and exit function with solution.
                betaMin = betaSet(iMin(1),:);
                fMin = fSet(iMin(1));
                %... Close output function
                if ~isempty(fOut), fOut('close'); end
                %... Delete parallel pool
                if isParallel, delete(gcp); end
                return
            end
        end
    end
end
end