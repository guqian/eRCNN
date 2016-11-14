function CNNfeature_new_zheng(test_x,cnn)
%сисз
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
%load D:/research/Class/CnnNorth_50_Final.mat
load D:/research/Class/train_north_Final_4section.mat
load D:/research/Class/Regression/FinalExperiment/IntervalTime/thirdring/NoJump/10min/net/3ker_6_22h_new_north
%load D:/research/Class/Regression/FinalExperiment/IntervalTime/thirdring/NoJump/10min/net/3ker_6_22h_new_north
addpath(genpath('D:/research/DeepLearnToolbox-master')); 
left =18;
right = 66;
% fea.layers = {
%     struct('type', 'i') %input layer
%     struct('type', 'c', 'outputmaps', 18, 'kernelsize', 5) %convolution layer
%     struct('type', 's', 'scale', 2) %sub sampling layer
% %     struct('type', 'c', 'outputmaps', 18, 'kernelsize', 5) %convolution layer
% %     struct('type', 's', 'scale', 2) %subsampling layer
% };
opts.alpha = 1;
opts.batchsize = 24;
opts.numepochs = 50;
outputnum = cnn.layers{2}.outputmaps;
% fea.layers{2}.k = cell(1,18);
% for l = 1:outputnum
%     fea.layers{2}.k{l} = cell(12,2);
%     for i = 1:12
%         for j = 1:2
%             z = zeros(14,4);
%             z(i:i+2,j:j+2) = z(i:i+2,j:j+2) + rot180(cnn.layers{2}.k{1}{l});         
%             fea.layers{2}.k{l}{i,j} = z;
%         end
%     end
% end
% 
% z = ones(14,4,1);
% 
% fea.layers{1}.a{1} = z;
% 
% fea.layers{2}.p = cell(1,18);
% outputnum = cnn.layers{2}.outputmaps;
%bis = zeros(14,4,24);
% %input = 28;
testsize = 2016;
% 
fid = fopen('D:/research/result/thirdring/Sensitivy_zheng.txt','w');
for left =18:3:63

for i = 1:122;
     Influence(i) =0;
end
right = left +3;

for input = 0:121
%test_x1 = importdata(['D:/research/trainnew/test/test_x_Edecline' num2str(input) '.txt']);
%if input == 0
%    test_x=test_x1(:,indexbusytotal2);
%else
%    test_x=[test_x,test_x1(:,indexbusytotal2)];
%end
%end
if(input ==0)
    num =31;
    off = 0;
    fea = Setup(cnn,outputnum);
    bis = zeros(14,4,24);
end
if(input ==31)
    load D:/research/Class/train_east_Final_4section.mat
    load D:/research/Class/Regression/FinalExperiment/IntervalTime/thirdring/NoJump/10min/net/3ker_6_22h_new_east
    %load D:/research/Class/train_east_Final_4section.mat
    %load D:/research/Class/Regression/FinalExperiment/IntervalTime/thirdring/NoJump/10min/net/3ker_6_22h_new_east
    num =30;
    off = 31;
    fea = Setup(cnn,outputnum);
    bis = zeros(14,4,24);
end
if(input == 61)
   load D:/research/Class/train_south_Final_4section.mat
   load D:/research/Class/Regression/FinalExperiment/IntervalTime/thirdring/NoJump/10min/net/3ker_6_22h_new_south
   %load D:/research/Class/train_south_Final_4section.mat
   %load D:/research/Class/Regression/FinalExperiment/IntervalTime/thirdring/NoJump/10min/net/3ker_6_22h_new_south
   num = 30;
   off =61;
   fea = Setup(cnn,outputnum);
   bis = zeros(14,4,24);
end
if(input == 91)
    load D:/research/Class/train_west_Final_4section.mat
    load D:/research/Class/Regression/FinalExperiment/IntervalTime/thirdring/NoJump/10min/net/3ker_6_22h_new_west
    num = 31;
    off  = 91;
    fea = Setup(cnn,outputnum);
    bis = zeros(14,4,24);
end
input2 = input-off;

test_x_new = test_x(1:28,1+16*input2*testsize:testsize*16*(input2+1));
test_x_new = reshape(test_x_new,28,16,testsize);
kernel = 3;
TestIndex = [];
 for i = 1 : 2016
        test_x_new1(:,:,i) = test_x_new((17-kernel)/2:(37+kernel)/2,12-kernel:12,i);
        if rem(i,288) >left*4 && rem(i,288)<= right*4 
            TestIndex = [TestIndex i];
        end
end
test_x_new = test_x_new1(:,:,TestIndex);
m = size(test_x_new, 3);
kk = randperm(m);

for dis = 1:1
   
    fea.layers{1}.a{1} = test_x_new(:, :, kk((dis - 1) * opts.batchsize + 1 : dis * opts.batchsize));
    cnn = cnnff(cnn,fea.layers{1}.a{1});
    outputnum = cnn.layers{2}.outputmaps;
    disp(dis);
for l = 1:outputnum
    fea.layers{2}.p{l} = cell(24,12);
  
       for i = 1:12
          for j = 1:2
              out = repmat(cnn.layers{2}.a{l}(i,j,:),14,4);
              z = (reshape(repmat(fea.layers{2}.k{l}{i,j},1,24),14,4,24)).*(out.*(1-out));

              fea.layers{2}.p{l}{i,j} = z;
          end
       end
end

fea.layers{3}.p = cell(1,18);
for l =1:18
    fea.layers{3}.p{l} = cell(4,1);
    for i =1:6
        for j =1:1
            z = fea.layers{2}.p{l}{(i-1)*2+1,(j-1)*2+1}.*0.25+fea.layers{2}.p{l}{(i-1)*2+1,(j-1)*2+2}.*0.25+fea.layers{2}.p{l}{(i-1)*2+2,(j-1)*2+1}.*0.25+fea.layers{2}.p{l}{(i-1)*2+2,(j-1)*2+2}.*0.25;
            fea.layers{3}.p{l}{i,j} = z;
        end
    end
end
i = 0;
z = zeros(14,4,24);
for l = 1:18
    for m = 1:6
       for j = 1:1
           i = i+1;
           z = z + fea.layers{3}.p{l}{m,j}.*cnn.ffW(1,i);
       end
    end
end
o = reshape(cnn.o(1,1:24),1,1,24);
o = repmat(o,14,4);

 fea.out = z.*(o.*(1-o));

 bis = bis + fea.out;
 
end
 hhh = sum(bis,3);
 j =0;
 for i =  input-7:input+6
     j=j+1;
     if(i <0)
        k = 123 +i;
     else if i >121
            k = i - 121;
         else
          k = i+1;
         end
     end
     Influence(k) = Influence(k) + mean(abs(hhh(j,:)));
 end
% xlswrite(['D:/research/Excel/' num2str(input) 'sum.xlsx'],hhh);

% hhh = sum(o,3);
% xlswrite('D:/research/123456.xlsx',hhh,1,'A1');
end
   for i = 1:122
       fprintf(fid,'%f ',Influence(i));
   end
   fprintf(fid,'\n');
end
end
    function X = rot180(X)
        X = flipdim(flipdim(X, 1), 2);
    end
    function fea = Setup(cnn,outputnum)
       fea.layers = {
            struct('type', 'i') %input layer
            struct('type', 'c', 'outputmaps', 18, 'kernelsize', 5) %convolution layer
            struct('type', 's', 'scale', 2) %sub sampling layer
        %     struct('type', 'c', 'outputmaps', 18, 'kernelsize', 5) %convolution layer
        %     struct('type', 's', 'scale', 2) %subsampling layer
        };
        fea.layers{2}.k = cell(1,18);
        for l = 1:outputnum
            fea.layers{2}.k{l} = cell(12,2);
            for i = 1:12
                for j = 1:2
                    z = zeros(14,4);
                    z(i:i+2,j:j+2) = z(i:i+2,j:j+2) + rot180(cnn.layers{2}.k{1}{l});         
                    fea.layers{2}.k{l}{i,j} = z;
                end
            end
        end

        z = ones(14,4,1);

        fea.layers{1}.a{1} = z;

        fea.layers{2}.p = cell(1,18);

    end

