% implementation of RNN 
clc
clear
close all
%% training dataset generation
% for i = 1:largest_number+1
%     binary{i} = dec2bin(i-1, 8);
%     int2binary{i} = binary{i};
% end
load D:/research/Class/train_west_Final_4section_rnn.mat
load D:/research/Class/Regression/FinalExperiment/IntervalTime/thirdring/NoJump/10min/net/3ker_6_22h_west_0
addpath(genpath('D:/research/DeepLearnToolbox-master'));

%% input variables
sy.alpha = 0.01;
% input_dim = 108;
% hidden_dim = 16;%?
% output_dim = 1;

%% initialize neural network weights
net =cnnff(cnn,train_x_new1);
sy.W = net.ffW;
sy.b = net.ffb;
sy.U = 1;
sy.H = 1;
sy.dW = zeros(size(sy.W));
sy.db = zeros(size(sy.b));
sy.dU = zeros(size(sy.U));
sy.dH = zeros(size(sy.H));
sy.numpoches = 10;  %%迭代次数

sy.ErrorMin = 1000;
train_x = net.fv;
train_y = train_y_new;
%%重新组织train_y只保留工作日的代码
bk = [];
bk1 =[];
ak = [];
ak1 =[];
net1 = cnnff(cnn,test_x_new1);
u = [2,3,9,10,16,17];
for i =0:20*31-1
    if(mod(i+1,7)~=2 && mod(i+1,7)~=3 )
        bk =[bk train_y_new(i*192+1:(i+1)*192)];
        ak = [ak net.fv(:,i*192+1:(i+1)*192)];
        if(i<=7*31-1)
           bk1 =[bk1 test_y_new(i*192+1:(i+1)*192)];
           ak1 =[ak1 net1.fv(:,i*192+1:(i+1)*192)];
        end
    end
end;
net.fv = ak;
net1.fv =ak1;
train_y_new = bk;
test_y_new =bk1;

%% train logic

for i = 1:192:192
    error =0;
    error2 = 0;
    sy.left =i;
    sy.right=i+191;
    MAPEall =0;
    
for input = 0:30
[sy1,error1,MAPE1,MAPE2,SD1,SD2]=Cnn_rnn_train(sy,net.fv(:,input*192*14+1:(input+1)*192*14),train_y_new(input*192*14+1:(input+1)*192*14));
sy.H = sy1.H;
error3= Cnn_rnn_test(sy,net1.fv(:,input*192*5+1:(input+1)*192*5),test_y_new(input*192*5+1:(input+1)*192*5));
error = error1+error;
error2 = error3+error2;
MAPEall = MAPEall +(MAPE1-MAPE2);
disp(SD1);
disp(SD2);
end
disp(MAPEall);
%disp(error/31);
%disp(error2/31)
fid1 = fopen('D:/research/result/10min/2rd_Crn_picture.txt','a');
fprintf(fid1,'%f %f \n',error,error2);
% disp(error1/20);
% disp(error2/20);
% disp(error3/num);
aaa=0;
end