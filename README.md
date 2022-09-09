# OPI, Orographic Precipitation and Isotopes

## Matlab Programs for the Analysis of Orographic Precipitation and Isotopes, with Implications for the Study of Paleotopography

### Overview
The Orographic Precipitation and Isotopes (OPI) programs provide computational tools for the analysis of precipitation and isotope fractionation associated with steady atmospheric flow over an arbitrary three-dimensional topography.

Author: Mark Brandon mark.brandon@yale.edu

### Requirements
Matlab (currently running with Matlab 2022a).

Regional-scale calculations may be limited by available memory. Personal experience indicates that about 10 to 20 GB are needed for a regional-scale problem. The opiFit program can be explicitly set to run in with multiple CPUs (parallel mode). Some Matlab functions will automatically make use of multiple CPUs, when available.

### Running the OPI programs
Unzip the OPI package.

The top level of the “OPI programs” directory contains the main functions for the programs.

Supporting functions for the OPI programs are in the “private” subdirectory. The main input files are best stored in a “data” directory for the study area. These include a digital topography file (Matlab mat format), and a sample file with stable isotope data (Excel format). A run directory should be set up to hold the input and output for a specific run. This directory must contain a run file (text format), which defines variables used by the run, and the path and file names for the data files. Each run file includes comments to indicate the information needed. A run file is set up by editing an existing run file, such as the one included with the example for this release.

To execute a program, load the main function into Matlab, and select the run option in the Matlab menu. The OPI program will then prompt the user to load one or two input files, which should be located in the run directory. The program will then run to completion, and produce a log file. Most programs also produce figures, and some will produce matfiles containing numerical results. All output is saved in the run directory, including a pdf for each figure.

Note that there are two sets of OPI programs, as indicated by the suffix to the program name. “OneWind” indicates programs that handle solutions (nine parameters) for single steady wind field. “TwoWinds” indicates programs that handle solutions (19 parameters) for a mixture of two unique steady wind fields.

There are two pdf documents in this release, which provide further details about the concepts, algorithms, and data used for the OPI programs.

The programs are commonly used in a sequence:

1) opiFit_OneWind, opiFit_TwoWinds: These programs use a direct search to find a least-squares estimate of a set of OPI parameters that provides the best fit of the sample data to the OPI model. For input, opiFit requires a run file (text format), a topography file (mat format), and a sample file (Excel format). For output, the program produces a log file (text format), and list of candidate solutions (opiFit_Solutions.txt, text format). The best-fit solution is indicated in the log file and is also appended to the run file. Run times for opiFit can range from 200 to 5,200 CPU hours, depending on the size of the region, and whether the solution is for one or two wind directions.

2) opiPairPlots: This program provides a summary of the output from opi_Fit. For input, the program uses the “opiFit_Solutions.txt” file. The user is prompted to select this file at the start of the program. The output is a log file (text format) and four figures. The log file provides information about the best-fit solution and estimated confidence limits. Two of the figures show a set of “pair plots”, which illustrate the distribution of the misfit (reduced chi-square) for each pair of estimated parameters.

3) opiCalc_OneWind, opiCalc_TwoWinds: These programs calculate results for the solution located at the end of the run file. For input, opiCalc requires a run file, a topography file, and a sample file. The user is prompted to select the run file at the start of the program. opiCalc_OneWind provides an option to calculate synthetic results (such as the example included in this release). This is done by setting the sample file name in the run file to “no”, and then modifying the OPI solution at the end of the run file, to conform to the desired example. The output from opiCalc is saved as a log file (text format), and a mat file with calculated results (opiCalc_OneWind_Results.mat, opiCalc_TwoWinds_Results.mat).

4) opiPlots_OneWind, opiPlots_TwoWinds: These programs generate plots to illustrate the OPI results. For input, opiPlots uses the mat file produced by opiCalc. The user is prompted for this file at the start of the program. For output, the program produces a log file, and a pdf file for each figure.

5) opiMaps_OneWind, opiMaps_TwoWinds: These programs create maps to illustrate the OPI results. For input, opiMaps requires the mat file produced by opiCalc, and the topography file associated with that solution. The user is prompted to select the opiCalc mat file at the start of the program. For output, the program produces a log file, and a pdf file for each figure.

6) opiPredictCalc, opiPredictPlot: These programs calculate the prediction of precipitation isotopes for times in the geologic past. opiPredictCalc uses only a one-wind solution, but the user is able to select that solution from a two-winds result. For input, the opiPredictCalc requires an opiCalc mat file, and a list of locations, ages, and isotope measurements (Excel format). The program also uses three mat files that contain Cenozoic climate data: “Cramer2011BenthicForamClimate.mat”, “Miller2020BenthicForamClimate.mat”, “MEBM_vary_OLR.mat” (see "private" directory). The opiPredictCalc program includes a summary of how these data are used in the calculation. Note that opiPredictCalc runs using a single CPU, and typically requires about 1 CPU hour to finish.


### Example
This release includes a synthetic example of the precipitation and precipitation isotopes caused by steady flow of moist air over a Gaussian mountain range. The release also includes the input files for this example, which can be used to see how the programs work. The file “OPI simulation of orographic precipitation and isotopes for a synthetic Gaussian topography.pdf” provides a summary for this example, along with the log files and figures produced by the OPI programs. The program “opiSyntheticTopography” shows how to construct topography mat files for a synthetic example.

Note that, for this synthetic example, some figure numbers are missing from the figure sequence from each program. The reason is that there are no observed data for this example, so those figures that show only observed data are not included in the summary.

Brandon et al. (2022b, 2022c) provide two real examples of the use of the OPI programs to analyze orographic precipitation and fractionation of water isotopes in the vicinity of the Patagonian Andes and the South-Central Andes. The Patagonia Andes case is used in Chang et al. (2022) to discuss the influence of evaporation on modern precipitation isotopes from that area. The South-Central Andes case is used in Fennell et al. (2022) as a basis for the interpreting precipitation isotope measurements from Cenozoic samples of hydrated volcanic glass. The research objective in Fennell et al. is the interpretation of the Cenozoic topographic evolution of area of Argentina east of the main Andes. The example in Brandon et al. (2022c) shows how opiPredict was used for this objective.

### References
Brandon, M.T., 2022a, Matlab Programs for the Analysis of Orographic Precipitation and Isotopes, with Implications for the Study of Paleotopography. Zenodo open-access repository.
.
Brandon, M.T., Chang, Q., and Hren, M.T., 2022b, Analysis of orographic precipitation and isotopes in the vicinity of the Patagonian Andes (latitude 54.8 to 40.1 S), Zenodo open-access repository.

Brandon, M.T., Fennell, L.M., and Hren, M.T., 2022c, Analysis of orographic precipitation and isotopes in the vicinity of the South-Central Andes (latitude 37.6 to 32.4 S), Zenodo open-access repository.

Chang, Q., Brandon, M.T., and Hren, M.T., 2022, Topography, Evaporation and Precipitation Isotopes Across the Patagonian Andes (in review).

Fennell, L.M., Brandon, M.T., and Hren, M.T., 2022, Cenozoic topographic evolution of the Southern Central Andes foreland region as revealed by hydrogen stable isotopes in hydrated volcanic glass (in review).
