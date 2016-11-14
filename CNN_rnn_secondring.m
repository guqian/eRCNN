% implementation of RNN 
clc
clear
close all
%% training dataset generation
% for i = 1:largest_number+1
%     binary{i} = dec2bin(i-1, 8);
%     int2binary{i} = binary{i};
% end
%load D:/research/Class/train_west_Final_4section_rnn.mat
%load D:/research/Class/Regression/FinalExperiment/IntervalTime/thirdring/NoJump/10min/net/3ker_6_22h_west_0
addpath(genpath('D:/research/DeepLearnToolbox-master'));

%% input variables
left = 18;
right = 66;
TrainIndex =[];
TestIndex = [];
for i = 1:5760
        if rem(i,288) >left*4 && rem(i,288)<= right*4 %&& rem(i,288)~=0
            TrainIndex = [TrainIndex i];
            if i <=2016 
            TestIndex = [TestIndex i];
            end
        end
end
for time = 10:100:30
for jump = 6:6
fid1 = fopen(['D:/research/result/All_3rd_Crn_' num2str(jump) '.txt'],'a');
for input2 = 0:79
load (['D:/research/Mypicture/secondring/model_fan' num2str(jump) ' ' num2str(input2)])
load (['D:/research/Mypicture/secondring/train_fan-1 ' num2str(input2)])
sy.alpha = 0.01;
% input_dim = 108;
% hidden_dim = 16;%?
% output_dim = 1;
train_y_new = importdata(['D:/research/trainFinal/Cnn2rdtest_10_new_fan/1train_y' num2str(input2) '.txt']);
test_y_new=importdata(['D:/research/trainFinal/Cnn2rdtest_10_new_fan/1test_y' num2str(input2) '.txt']);
input = 0;
%% initialize neural network weights
net =cnnff(cnn1,train_x_new(:,:,TrainIndex));
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
% bk = [];
% bk1 =[];
% ak = [];
% ak1 =[];
net1 = cnnff(cnn1,test_x_new(:,:,TestIndex));
% u = [2,3,9,10,16,17];
% for i =0:20*31-1
%     if(mod(i+1,7)~=2 && mod(i+1,7)~=3 )
%         bk =[bk train_y_new(i*192+1:(i+1)*192)];
%         ak = [ak net.fv(:,i*192+1:(i+1)*192)];
%         if(i<=7*31-1)
%            bk1 =[bk1 test_y_new(i*192+1:(i+1)*192)];
%            ak1 =[ak1 net1.fv(:,i*192+1:(i+1)*192)];
%         end
%     end
% end;
% net.fv = ak;
% net1.fv =ak1;
% train_y_new = bk;
% test_y_new =bk1;

%% train logic
i=1;
error =0;
error2 = 0;
sy.left =i;
sy.right=i+191;
MAPEall =0;
if jump ==-1
    jump1 =0;
else
    jump1 =jump;
end
sy = Cnn_rnn_setup(sy,net.ffW,net.ffb,4,4,0.1);
[sy1,error1,MAPE1,MAPE2,SD1,SD2]=Cnn_rnn_train(sy,net.fv,train_y_new(TrainIndex+jump),jump1,192);
sy.H = sy1.H;
[error1,error2,MAPE1,MAPE2,RMSE1,RMSE2]= Cnn_rnn_test(sy,net1.fv,test_y_new(TestIndex+jump),jump1);
%error = error1+error;
%error2 = error3+error2;
%MAPEall = MAPEall +(MAPE1-MAPE2);
%disp(SD1);
%disp(SD2);
%disp(MAPEall);
%disp(error/31);
%disp(error2/31)

disp(error1)
disp(error2)
fprintf(fid1,'%d %f %f %f %f %f %f \n',input2,error1,error2,MAPE1,MAPE2,RMSE1,RMSE2);
% disp(error1/20);
% disp(error2/20);
% disp(error3/num);
aaa=0;
end
end
end