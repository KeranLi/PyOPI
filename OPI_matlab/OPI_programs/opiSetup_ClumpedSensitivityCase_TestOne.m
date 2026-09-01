function opiSetup_ClumpedSensitivityCase_TestOne()
% opiSetup_ClumpedSensitivityCase_TestOne builds one minimal test case for
% validating the self-contained case-generation workflow.

opiSetup_ClumpedSensitivityCases("T0_1_290K");

fprintf(['Created test sensitivity case:\n' ...
    'sensitivity_parameter_local_clumped/T0_1_290K\n']);
fprintf('Check that the folder contains a run file, best-run file, topo, samples, divide, proxy_clumped, and README.\n');
end
