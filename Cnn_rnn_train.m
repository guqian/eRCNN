function [sy,MAE,bestMAPE1,bestMAPE,bestRMSE1,bestRMSE2] = Cnn_rnn_train( sy,train_x,train_y,jump,sequence_length)
%addpath(genpath('D:/research/DeepLearnToolbox-master'));
% sequence_length����ÿ�����еĳ���
%MAEMin = 1000;  %��������������matlab�еĴ������á�
%E_l =4;
%K = 4;    %���õ���Ԫ���������ｵ���˸��Ӷȣ�����R��E��KֵҲ�����ò�ͬ
alpha=0.01;
[m,n] =size(train_x);
for q = 1:sy.numpoches %��MAE��MAPE��RMSE�ֱ���0
    MAE =0;
    %sy.W_E = zeros(E_l,1);
for j = 0:(size(train_y,2)/sequence_length-1)
    %��Щ����Ӧ��д��һ��setup��
    layer.errors =zeros(1,sy.right);   %��Ԫerror���о�����0�������õĴ�СΪ4
    layer_E.output = zeros(sy.K,sy.right);  %��Ԫֵ������0
    layer_R.output = zeros(sy.K,sy.right);  %ÿ����Ԫ������ͬ�������ʼֵ
    %layer_1.values = [layer_1.values;1];
    %layer_2.values = zeros(1,sy.left-1)';
    
    %layer_2.deltas =zeros(1,sy.left-1)';  %�ڶ�������
    flag =zeros(1,sy.right);
    for position = sy.left:sy.right       %��Ҫ��������䷶Χ
        X=  train_x(:,sequence_length*j+position);  %��������
        Y = train_y(sequence_length*j+position);    %ʵ�ʽ��
        E =[];
        for i = position-sy.E_l:position-1             %����e����
            if(i <=0)
                E = [E;0];
            else
                E = [E;layer.errors(i)];
            end;
        end;
        %if(position >sy.left)
         %   preY = train_y(sequence_length*j+position-1);
        %end
        layer_R.output(:,position) = sigmoid(sy.W_R*X + sy.b_R)'; %X��cnnȫ���ӵ������Ҳ�ɹ����ÿ������˵Ľ������һ����Ԫ��
        layer_E.output(:,position) = sigmoid(sy.W_E*E + sy.b_E)'; 
        layer.output = sy.W_OR*layer_R.output(:,position) + sy.W_OE*layer_E.output(:,position);
        if(layer.output <= 0)
            layer.output = 0;
            flag(position) =1;   %dead��Ԫ
        else if (layer.output >=1)
            layer.output = 1;
            flag(position) =1;
            end
        end        %���Ʒ�Χ��������0-1
        layer.errors(position) = Y - layer.output;  %����error���ֵ
        %layer_2_values = [layer_2_values;layer_2];
        %layer_1_MAEs = [layer_1_MAEs;Y-layer_1];
        %layer_2_delta = Y - layer_2;
        %layer_2_deltas = [layer_2_deltas; layer_2_MAE*sigmoid_output_to_derivative(layer_2)];%������򵥵� ������layer_2.
        %layer_2_deltas = [layer_2_deltas;layer_2_MAE];
        % ���������������и����þ���ֵ��
        
        %MAE = MAE + abs(layer_2_MAE);
        %if(Y<=0.833)
            %MAPE = MAPE + abs(layer_2_MAE)/(1-Y);
            %MAPE1 = MAPE1+ abs(Y-layer_1)/(1-Y);
            %RMSE1 = RMSE1 + (Y-layer_1)^2;
            %RMSE2 = RMSE2 + abs(layer_2_MAE)^2;
        %end;
        
        
        % ��¼�´˴ε������� (S(t))
        % store hidden layer so we can use it in the next timestep
        %layer_1_values = [layer_1_values; layer_1];
    end
    
    % �����������diff������������ı仯�����������²���������ÿһ��timestep�����м���
    %future_layer_1_delta = zeros(1, hidden_dim);
    
    % ��ʼ���з��򴫲������� hidden_layer ��diff���Լ������� diff
     sy.d_E = zeros(sy.K,sy.right); 
     sy.d_O = zeros(1,sy.right);
     sy.d_R = zeros(sy.K,sy.right);
     delta.W_OR=0;
     delta.W_OE=0;
     delta.W_R=0;
     delta.W_E=0;
     delta.b_O =0;
     delta.b_E =0;
     delta.b_R = 0;
     for position = sy.right:-1:sy.left
        d_E = [];
        for i = position+1:position + sy.E_l;    
            if(i > sy.right)
                d_E = [d_E,zeros(1,sy.K)'];
            else
                d_E = [d_E,sy.d_E(:,i)];
            end
        end
        if(flag(position)==0)
            sy.d_O(position) = layer.errors(position)+ sum(sum(rot90(sy.W_E,2).*d_E)');                  %ע��˴�Ҫ��ת,Ȼ�����ڻ�
        else                                                              %sy.dH = sy.dH+layer_1_MAEs(position-1-jump)*layer_2_deltas(position)
            sy.d_O(position) = sum((sum(rot90(sy.W_E,2).*d_E))');
        end
        for i = 1:sy.K
            sy.d_E(i,position)=sy.d_O(position)*sy.W_OE(i)*layer_E.output(i,position)*(1-layer_E.output(i,position));
            sy.d_R(i,position)=sy.d_O(position)*sy.W_OR(i)*layer_R.output(i,position)*(1-layer_R.output(i,position));
        end 
        %�ݶ��½������и��£�����ÿ�����СΪ192������������ȡ�鷽�����иĽ���
        delta.W_OR = delta.W_OR + sy.d_O(position)*layer_R.output(:,position);
        delta.W_OE = delta.W_OE + sy.d_O(position)*layer_E.output(:,position);
        delta.W_R = delta.W_R + sy.d_R(position)*train_x(:,sequence_length*j+position);
        E = [];
        for i = position-sy.E_l:position-1             %����e����
            if(i <=0)
                E = [E,0];
            else
                E = [E,layer.errors(i)];
            end;
        end;
        delta.W_E = delta.W_E + sy.d_E(position)*E;
        delta.b_O = delta.b_O + sy.d_O(position);
        delta.b_R = delta.b_R + sy.d_R(position);
        delta.b_E = delta.b_E + sy.d_E(position);
     end 
end
end
fid1 = fopen('D:/research/result/10min/2rd_Crn_new1.txt','a');
fprintf(fid1,'%f %f\n',MAE1/14,MAEMin);
%disp(MAE1/20);
%disp(MAEMin);
MAE = (MAE1/14 - MAEMin);
%disp(bestH);
end

