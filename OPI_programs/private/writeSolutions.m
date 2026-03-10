function writeSolutions(data, varargin)
% FMINCRSOUT Reports candidate solutions as generated in
% real time by fminCRS.
%
% Mark Brandon, Yale University, 2016-2019

%% Process call
persistent isStart nIterations nParameters fid runPath logFilename
switch true
    case strcmp(data, 'initialize')
        %... Initialize solution file
        isStart = true;
        nIterations = 0;
        runPath =  varargin{1};
        runTitle = varargin{2};
        nSamples = varargin{3};
        parameterLabels = varargin{4};
        exponents = varargin{5};
        lb = varargin{6};
        ub = varargin{7};
        nParameters = length(exponents);
        logFilename = [runPath, '/', 'opiFit_Solutions.txt'];
        if isfile(logFilename)
            movefile(logFilename, [runPath, '/', 'opiFit_Solutions.bk!']);
        end
        % 'W' option invokes buffered output, with 4k (4096 byte) buffer.
        % This option is needed to fix output errors when using the HPC,
        % but it means that output is written only every ~50 lines.
        % fid = fopen(logFilename, 'W', 'native', 'UTF-8');
        % Using 'w' option, which means output after each call
        fid = fopen(logFilename, 'w', 'native', 'UTF-8');
        fprintf(fid, '%% Solution file for optimization using opiFit\n');
        fprintf(fid, '%% Start time: %s\n', char(datetime));
        fprintf(fid, '%% Original run path:\n%% %s\n', runPath);
        fprintf(fid, '%% Run title:\n%s\n', runTitle);
        fprintf(fid, '%% Number of observations:\n%d\n', nSamples);
        fprintf(fid, '%% Parameter labels:\n');
        fprintf(fid, '%s\n', parameterLabels);
        fprintf(fid, '%% Exponent for power of 10 factoring for plotted variables labeled above:\n');
        fprintf(fid, '%d', exponents(1));
        fprintf(fid, '\t%d', exponents(2:end));
        fprintf(fid, '\n');
        fprintf(fid, '%% Lower and upper constraints for parameter search:\n');
        fprintf(fid, '%g', lb(1));
        fprintf(fid, '\t%g', lb(2:end));
        fprintf(fid, '\n');
        fprintf(fid, '%g', ub(1));
        fprintf(fid, '\t%g', ub(2:end));
        fprintf(fid, '\n');
    case strcmp(data, 'note')
        %... Write a string to the solutions file
        str =  varargin{1};
        fprintf(fid, '%s', str);        
    case strcmp(data, 'close')
        %... Close file
        if isStart
            fprintf(fid, '%%Finish time: %s\n', char(datetime));
            fclose(fid);
            % copyfile(logFilename, logFilenameShadow)
            isStart = false;
        end
    otherwise
        %... Write solution to file
        if isStart
            %... Write column labels for solutions
            if nIterations==0
                fprintf(fid, '%%    n\t        chiR2\t    nu');
                for i = 1:nParameters
                    fprintf(fid, '\t         beta%02d', i); 
                end
                fprintf(fid, '\n');
            end
            %... Write nth solution
            nIterations = nIterations + 1;
            str = [sprintf('%6d', nIterations), ...
                sprintf('\t%#15.6g', data(1)), ...
                sprintf('\t%6d', data(2)), ...
                sprintf('\t%#15.6g', data(3:end)), '\n'];
            fprintf(fid, str);
        end
end
end
