clear;
%Load urdf file
Robot=importrobot('IRB 1300-1150 URDF.urdf');

%Generate workspace and configurations
%[wksp,cfgs] = generateRobotWorkspace(Robot,{},MaxNumSamples=5000);
[wksp,cfgs] = generateRobotWorkspace(Robot,{});
%wsAlpha = alphaShape(wksp(:,1),wksp(:,2),wksp(:,3));

%Generate manipulability index
%IndexInvCond = manipulabilityIndex(Robot,cfgs,IndexType ="inverse-condition");
IndexInvCond = manipulabilityIndex(Robot,cfgs,IndexType="yoshikawa");

%Generate plots
%randoConfig=randomConfiguration(Robot);
%show(Robot,randoConfig,Visuals="on");
%show(Robot,homeConfiguration(Robot),Visuals="on");
show(Robot,Visuals="on");
hold on

%sdf = wksp(IndexInvCond>0.1,:);
showWorkspaceAnalysis(wksp,IndexInvCond,Voxelize=false);
%a = plot(wsAlpha,FaceAlpha=0.45,EdgeColor="none");
%randomConfiguration(Robot)

hold off
title("Manipulability Index - Yoshikawa - IRB1300-1150");

axis auto


%Output=[wksp, IndexInvCond];
%writematrix(Output,'IRB_1300-1150_Robot_MI.csv');