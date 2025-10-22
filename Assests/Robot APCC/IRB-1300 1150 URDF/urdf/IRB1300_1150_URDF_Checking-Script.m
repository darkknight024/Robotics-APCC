clear;
%Load urdf file
Robot=importrobot('IRB 1300-1150 URDF.urdf');
Robot.Gravity = [0 0 -9.81];

%Define Table
Table= collisionBox(.822,1.137,.15);
T = trvec2tform([.2475 0 -.092]);
Table.Pose = T;

config = homeConfiguration(Robot)

% config(1).JointPosition = 90*pi/180;
% show(Robot,config);
% title("Axis 1 - 90degs");

% config(2).JointPosition = 155*pi/180;
% show(Robot,config);
% title("Axis 2 - +155degs");
% 
% config(2).JointPosition = -95*pi/180;
% show(Robot,config);
% title("Axis 2 - -95degs");
% %show(Robot,Visuals="on");

% config(3).JointPosition = 65*pi/180;
% show(Robot,config);
% title("Axis 3 - +65");

% config(3).JointPosition = -210*pi/180;
% show(Robot,config);
% title("Axis 3 - -210degs");

% config(4).JointPosition = 90*pi/180;
% show(Robot,config);
% title("Axis 3 - 90degs");

% config(5).JointPosition = 90*pi/180;
% show(Robot,config);
% title("Axis 5 - 90degs");

config(6).JointPosition = 90*pi/180;
show(Robot,config);
title("Axis 6 - 90degs");


hold on
show(Table);

hold off
axis auto

