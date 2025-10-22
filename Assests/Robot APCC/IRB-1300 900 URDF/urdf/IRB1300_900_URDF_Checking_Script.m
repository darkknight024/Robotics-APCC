clear;
%Load urdf file
Robot=importrobot('IRB-1300 900 URDF.urdf');
Robot.Gravity = [0 0 -9.81];

%Define Table
Table= collisionBox(.822,1.137,.15);
T = trvec2tform([.2475 0 -.092]);
Table.Pose = T;

config = homeConfiguration(Robot)

% config(1).JointPosition = 90*pi/180;
% show(Robot,config);
% title("IRB-1300 900 Joint 1 +90degs");

% config(2).JointPosition = 130*pi/180;
% show(Robot,config);
% title("IRB-1300 900 Joint 2 +130degs");

% config(2).JointPosition = -100*pi/180;
% show(Robot,config);
% title("IRB-1300 900 Joint 2 -100degs");


% config(3).JointPosition = 65*pi/180;
% show(Robot,config);
% title("IRB-1300 900 Joint 3 +65degs");

% config(3).JointPosition = -210*pi/180;
% show(Robot,config);
% title("IRB-1300 900 Joint 3 -210degs");

% config(4).JointPosition = 90*pi/180;
% show(Robot,config);
% title("IRB-1300 900 Joint 4 +90degs");

% config(5).JointPosition = 90*pi/180;
% show(Robot,config);
% title("IRB-1300 900 Joint 5 +90degs");

config(6).JointPosition = 90*pi/180;
show(Robot,config);
title("IRB-1300 900 Joint 6 +90degs");


hold on
show(Table);
view(-30,20);
hold off
axis auto

