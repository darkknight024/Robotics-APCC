clear;
%Robot=importrobot('irb4600-60_2.05.urdf');
%Robot=importrobot('irb1300-7_1.4.urdf');
Robot=importrobot('urdf/IRB-1300 1400 URDF.urdf');

%Generate workspace and configurations
[wksp,cfgs] = generateRobotWorkspace(Robot,{},MaxNumSamples=5000);
%Generate manipulability index
%IndexInvCond = manipulabilityIndex(Robot,cfgs,IndexType ="inverse-condition");
IndexInvCond = manipulabilityIndex(Robot,cfgs,IndexType="yoshikawa");
%plot


show(Robot);
hold on
%sdf = wksp(IndexInvCond>0.1,:);
showWorkspaceAnalysis(wksp,IndexInvCond,Voxelize=false);
hold off
title("Manipulability Index");

axis auto


%Output=[wksp, IndexInvCond];
%writematrix(Output,'IRB_1300-1150_Robot_MI.csv');r