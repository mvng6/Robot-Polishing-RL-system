clear;
close all;
clc;

%% C++ 기록 데이터
data = readmatrix("Data_Speedl_F_35_P_80.00_I_130.00_D_0.50.csv");

t = (data(:,1) - data(1,1)) * 0.001;

RL_u = data(:,end-2);
RL_send_flag = data(:,end-1);
RL_ep_flag = data(:,end);

%%
figure;
subplot(3,1,1);
plot(t,RL_u);
xlim([0 t(end)]);

subplot(3,1,2);
plot(t,RL_send_flag);
xlim([0 t(end)]);

subplot(3,1,3);
plot(t,RL_ep_flag);
xlim([0 t(end)]);