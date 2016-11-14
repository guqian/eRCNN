function [error1,error2,bestMAPE1,bestMAPE2,bestRMSE1,bestRMSE2] = Cnn_ernn_test(sy,test_x,test_y,jump)
addpath(genpath('D:/research/DeepLearnToolbox-master'));
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
ErrorMin = 1000;
error1 =0;
error2 =0;
MAPE1 =0;
MAPE2 =0;
SD1 =0;
SD2 =0;
fid =fopen('D:/research/result/40minFenBu_time_new.txt','a');
fid2 =fopen('D:/research/result/FenBu_time2_begin6:00new.txt','a');
for j = 0:(size(test_y,2)/192-1)  
    layer_1_errors =zeros(1,sy.left-1)';
    layer_1_values = zeros(1,sy.left-1)';
    layer_1_values = [layer_1_values;1];
    layer_2_values = zeros(1,sy.left-1)';
    layer_2_deltas =zeros(1,sy.left-1)';
    %layer_1_values = [layer_1_values; zeros(1, hidden_dim)];
    flag =zeros(1,sy.right);
    % 开始对一个序列进行处理，搞清楚一个东西，一个LSTM单元的输出其实就是隐含层
    for position = sy.left:sy.right
%         X = [a(binary_dim - position)-'0' b(binary_dim - position)-'0'];   % X 是 input
%         y = [c(binary_dim - position)-'0']';                               % Y 是label，用来计算最后误差
        %disp(192*j+position)
        X=  test_x(:,192*j+position);
        Y = test_y(192*j+position);
        if(position >sy.left)
            preY = test_y(192*j+position-1);
        end
        % 这里是RNN，因此隐含层比较简单
        % X ------------------------> input
        % sunapse_0 ----------------> U_i
        % layer_1_values(end, :) ---> previous hidden layer （S(t-1)）
        % synapse_h ----------------> W_i
        % layer_1 ------------------> new hidden layer (S(t))
        layer_1 = sigmoid(sy.W*X + sy.b);
        
        % layer_1 ------------------> hidden layer (S(t))
        % layer_2 ------------------> 最终的输出结果，其维度应该与 label (Y) 的维度是一致的
        % 这里的 sigmoid 其实就是一个变换，将 hidden layer (size: 1 x 16) 变换为 1 x 1
        % 有写时候，如果输入与输出不匹配的话，使可以使用 softmax 进行变化的
        % output layer (new binary representation)
        %layer_2 = sigmoid(layer_1*synapse_1)-0.5;
%         if(layer_1 <0.05)
%              layer_1 =0;
%         end
        if(position >sy.left+jump)
            %f(abs(error(end))>0.2 ||preY ==0)
                layer_2  = layer_1*sy.U +sy.H*layer_1_errors(end-jump);
            %else
                %layer_2 = layer_1*synapse_1;
            %end;
        else
           layer_2 = layer_1;
        end;
                %尝试先只计算layer_1（一层）
        if(layer_2 < 0.05)
            layer_2 = 0;
            flag(position) =1;
        else if (layer_2 >=0.9)
            layer_2 = 0.9;
            flag(position) =1;
            end
        end
        layer_2_values = [layer_2_values;layer_2];
        layer_1_errors = [layer_1_errors;Y-layer_1];
        
        % 计算误差，根据误差进行反向传播
        % layer_2_error ------------> 此次（第 position+1 次的误差）
        % l 是真实结果
        % layer_2 是输出结果
        % layer_2_deltas 输出层的变化结果，使用了反向传播，见那个求导（输出层的输入是 layer_2，那就对输入求导即可，然后乘以误差就可以得到输出的diff）
        % did we miss?... if so, by how much?
        layer_2_error = Y - layer_2;
        if(Y<=0.833)
            MAPE2 = MAPE2 + abs(layer_2_error)/(1-Y);
            MAPE1 = MAPE1+ abs(Y-layer_1)/(1-Y);
            SD1 = SD1 + (Y-layer_1)^2;
            SD2 = SD2 + abs(layer_2_error)^2;
        end
        %layer_2_deltas = [layer_2_deltas; layer_2_error*sigmoid_output_to_derivative(layer_2)];%先算个简单的 不更新layer_2.
        layer_2_deltas = [layer_2_deltas;layer_2_error];
        % 总体的误差（误差有正有负，用绝对值）
        error2 = error2 + abs(layer_2_error);
        % decode estimate so we can print it out
        % 就是记录此位置的输出，用于显示结果
        %d(binary_dim - position) = round(layer_2(1));
        
        % 记录下此次的隐含层 (S(t))
        % store hidden layer so we can use it in the next timestep
        layer_1_values = [layer_1_values; layer_1];
    end
    
    % 计算隐含层的diff，用于求参数的变化，并用来更新参数，还是每一个timestep来进行计算
    %future_layer_1_delta = zeros(1, hidden_dim);
    
    % 开始进行反向传播，计算 hidden_layer 的diff，以及参数的 diff

     for position = sy.right:-1:sy.left+1+jump
        if(flag(position-sy.left+1)==0)
        sy.dH = sy.dH+layer_1_errors(position-1-jump)*layer_2_deltas(position);
        end;
     end
       sy.H = sy.H + sy.dH*sy.alpha;
       sy.dH = sy.dH * 0;
       
       %synapse_b = synapse_b + synapse_b_update*alpha;
       %synapse_b_update = synapse_b_update * 0;
       
       
%     synapse_0 = synapse_0 + synapse_0_update * alpha;
%     synapse_1 = synapse_1 + synapse_1_update * alpha;
%     synapse_h = synapse_h + synapse_h_update * alpha;
%     
%synapse_0_update = synapse_0_update * 0;
%     synapse_1_update = synapse_1_update * 0;
%     synapse_h_update = synapse_h_update * 0;

%     
%     if(mod(j,1000) == 0)
%         err = sprintf('Error:%s\n', num2str(overallError)); fprintf(err);
%         d = bin2dec(num2str(d));
%         pred = sprintf('Pred:%s\n',dec2bin(d,8)); fprintf(pred);
%         Tru = sprintf('True:%s\n', num2str(c)); fprintf(Tru);
%         out = 0;
%         size(c)
%         sep = sprintf('-------------\n'); fprintf(sep);
%     end
        error1 = error1+ mean(abs(layer_1_errors(sy.left:sy.right)));
        %error2 = error2 +overallError/(sy.right-sy.left+1); 
      for i = 1:192
          if j ~= [2,3]
            fprintf(fid,'%f %f\n',layer_2_deltas(i),layer_1_errors(i)); 
          end
            fprintf(fid2,'%f %f\n',layer_2_deltas(i),layer_1_errors(i)); 
      end
end
    if (error2/(7*(sy.right-sy.left+1)) < ErrorMin)
        ErrorMin = error2/(7*(sy.right-sy.left+1));
        bestMAPE1 = MAPE1/(7*(sy.right-sy.left+1));
        bestMAPE2 = MAPE2/(7*(sy.right-sy.left+1));
        bestRMSE1 = sqrt(SD1/(7*(sy.right-sy.left+1)));
        bestRMSE2 = sqrt(SD2/(7*(sy.right-sy.left+1)));
        bestH = sy.H;
    end
fid1 = fopen('D:/research/result/10min/2rd_Crn_new_test_1.txt','a');
%fprintf(fid1,'%f %f %f %f %f %f \n',error1/7,ErrorMin);
%disp(error1/20);
%disp(ErrorMin);
error1 = error1/7;
error2 = ErrorMin;
%disp(bestH);
end
