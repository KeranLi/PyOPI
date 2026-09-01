function out = opiAnalyze_AridityGeomorphicPatterns(resultRoot)
% Compare aridity in normalized geomorphic coordinates, not gridwise space.
if nargin<1
    resultRoot='/Users/keranli/Desktop/Coding/OPI_matlab/scenarios/aridity_modern_common_baseline_north_source_state2';
end
names={'modern_Tibet','Q4_V35_G4','Q4_V25_G4'};
R=cell(1,3);
for k=1:3
    S=load(fullfile(resultRoot,names{k},'opiAridity_Results.mat'),'result');
    R{k}=S.result;
end
modernAI=R{1}.aridityAbsolute(:,:,1);
modernMedian=median(modernAI(isfinite(modernAI)));

edges=linspace(0,1,21); centers=(edges(1:end-1)+edges(2:end))/2;
profile=table(); elevTable=table(); zoneTable=table();
plateauProfile=table(); plateauBounds=table();
elevEdges=[0 1000 2000 3000 4000 5000 inf];
for k=1:3
    z=R{k}.aridityAbsolute(:,:,1)/modernMedian;
    latNorm=(R{k}.lat-min(R{k}.lat))/(max(R{k}.lat)-min(R{k}.lat));
    latGrid=repmat(latNorm(:),1,numel(R{k}.lon));
    for j=1:numel(centers)
        m=latGrid>=edges(j)&latGrid<edges(j+1)&isfinite(z);
        profile=[profile; table(string(names{k}),centers(j),sum(m,'all'), ...
            median(z(m),'omitnan'),mean(z(m),'omitnan'), ...
            'VariableNames',{'case','normalizedNorthing','n','medianRelativeP_PET','meanRelativeP_PET'})]; %#ok<AGROW>
    end
    for j=1:numel(elevEdges)-1
        m=R{k}.hGridM>=elevEdges(j)&R{k}.hGridM<elevEdges(j+1)&isfinite(z);
        elevTable=[elevTable; table(string(names{k}),elevEdges(j),elevEdges(j+1),sum(m,'all'), ...
            median(z(m),'omitnan'),mean(z(m)<1,'omitnan'), ...
            'VariableNames',{'case','elevationMinM','elevationMaxM','n','medianRelativeP_PET','fractionBelowModernMedian'})]; %#ok<AGROW>
    end
    zones={R{k}.hGridM<2000,R{k}.hGridM>=2000&R{k}.hGridM<3500,R{k}.hGridM>=3500};
    zoneNames={'lowland_lt2000m','slope_2000_3500m','plateau_ge3500m'};
    for j=1:3
        m=zones{j}&isfinite(z);
        zoneTable=[zoneTable; table(string(names{k}),string(zoneNames{j}),sum(m,'all'), ...
            median(z(m),'omitnan'),mean(z(m)<1,'omitnan'), ...
            'VariableNames',{'case','zone','n','medianRelativeP_PET','fractionBelowModernMedian'})]; %#ok<AGROW>
    end

    % Align each case by its own plateau edges using zonal-median elevation.
    rowElevation=median(R{k}.hGridM,2,'omitnan');
    plateauRows=find(rowElevation>=3000);
    if numel(plateauRows)>=2
        southLat=R{k}.lat(plateauRows(1)); northLat=R{k}.lat(plateauRows(end));
        geomCoord=(R{k}.lat-southLat)/(northLat-southLat);
        geomGrid=repmat(geomCoord(:),1,numel(R{k}.lon));
        gEdges=linspace(-0.5,1.5,21); gCenters=(gEdges(1:end-1)+gEdges(2:end))/2;
        plateauBounds=[plateauBounds; table(string(names{k}),southLat,northLat, ...
            'VariableNames',{'case','southPlateauEdgeLat','northPlateauEdgeLat'})]; %#ok<AGROW>
        for j=1:numel(gCenters)
            m=geomGrid>=gEdges(j)&geomGrid<gEdges(j+1)&isfinite(z);
            plateauProfile=[plateauProfile; table(string(names{k}),gCenters(j),sum(m,'all'), ...
                median(z(m),'omitnan'),mean(z(m),'omitnan'), ...
                'VariableNames',{'case','plateauRelativeCoordinate','n','medianRelativeP_PET','meanRelativeP_PET'})]; %#ok<AGROW>
        end
    end
end
writetable(profile,fullfile(resultRoot,'geomorphic_northing_profiles.csv'));
writetable(elevTable,fullfile(resultRoot,'geomorphic_elevation_bands.csv'));
writetable(zoneTable,fullfile(resultRoot,'geomorphic_zone_summary.csv'));
writetable(plateauProfile,fullfile(resultRoot,'geomorphic_plateau_aligned_profiles.csv'));
writetable(plateauBounds,fullfile(resultRoot,'geomorphic_plateau_boundaries.csv'));

f=figure('Visible','off','Color','w'); hold on;
for k=1:3
    m=profile.case==names{k}; plot(profile.normalizedNorthing(m),profile.medianRelativeP_PET(m),'-o','LineWidth',1.5,'DisplayName',names{k});
end
yline(1,'--'); xlabel('Normalized south-to-north position'); ylabel('Relative P/PET (modern median = 1)'); legend('Location','best'); grid on;
exportgraphics(f,fullfile(resultRoot,'Fig_Geomorphic_Northing_Profile.png'),'Resolution',180); close(f);
f=figure('Visible','off','Color','w'); hold on;
for k=1:3
    m=plateauProfile.case==names{k}&plateauProfile.n>0;
    plot(plateauProfile.plateauRelativeCoordinate(m),plateauProfile.medianRelativeP_PET(m),'-o','LineWidth',1.5,'DisplayName',names{k});
end
xline(0,':','South plateau edge','HandleVisibility','off');
xline(1,':','North plateau edge','HandleVisibility','off');
yline(1,'--','Modern median','HandleVisibility','off');
xlabel('Plateau-relative coordinate'); ylabel('Relative P/PET'); legend('Location','best'); grid on;
exportgraphics(f,fullfile(resultRoot,'Fig_Geomorphic_PlateauAligned_Profile.png'),'Resolution',180); close(f);
out=struct('profile',profile,'plateauProfile',plateauProfile,'plateauBounds',plateauBounds,'elevationBands',elevTable,'zones',zoneTable);
fprintf('Wrote geomorphic-pattern analysis to %s\n',resultRoot);
end
