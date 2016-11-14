% implementation of RNN 
clc
clear
close all
%%  input strction
load D:/research/Class/train_ANN_Final.mat;
% for input = 0:121
%     %D:/research/train/Cnn3rdtest_20/train_y
%     %train_x1 = importdata(['D:/research/trainnew/test/train_x_Edecline' num2str(input) '.txt']);
%     %train_y1 = importdata(['J:/Cnn3rdtest_20_third/train_y' num2str(input)%'.txt']);
%     %train_x1 = importdata(['D:/research/train/OneStreet/new_train_x_SVM' num2str(input) '.txt']);
%     %test_x1 = importdata(['D:/research/train/OneStreet/new_test_x_SVM' num2str(input) '.txt']);
%     train_y1 = importdata(['D:/research/trainFinal/Cnn3rdtest_10_new_4section/1train_y' num2str(input) '.txt']);
%     %test_x1 = importdata(['D:/research/trainnew/test/test_x_Edecline' num2str(input) '.txt']);
%     %test_y1 = importdata(['J:/Cnn3rdtest_20_third/test_y' num2str(input) '.txt']);
%     test_y1 = importdata(['D:/research/trainFinal/Cnn3rdtest_10_new_4section/1test_y' num2str(input) '.txt']);
%     %indexbusy1 = importdata(['J:/Static/indexbusy_train' num2str(input) '.txt']);
%     %indexbusy2 = importdata(['J:/Static/indexbusy_test' num2str(input) '.txt']);
%     disp(input);
%     if input ==0
%         %train_x = train_x1;
%         %test_x =test_x1;
%         %train_x = reshape(train_x1,28,28,1440);
%         %test_x = reshape(test_x1,28,28,504);
%          %train_x = train_x1(:,indexbusytotal1);
%          %test_x = test_x1(:,indexbusytotal2);
%          train_y =train_y1;%(:,indexbusy1/4);
%          test_y =test_y1;%(:,indexbusy2/4);
%          fid1 =load(['D:/research/trainFinal/Speed_new/speed_y_20_zheng_' num2str(input) '.txt'],'r');
%     else
%         %train_x = [train_x train_x1];
%         %test_x = [test_x test_x1];
%         %train_x1 = train_x1(:,indexbusytotal1);
%         %test_x1 = test_x1(:,indexbusytotal2);
%         %train_y1 = train_y1(:,indexbusy1);
%         %test_y1 = test_y1(:,indexbusy2);
%         %train_x = [train_x,train_x1(:,indexbusytotal1)];
%         %test_x = [test_x,test_x1(:,indexbusytotal2)];
%         train_y = [train_y,train_y1];
%         test_y = [test_y,test_y1];        
%         fid1 =[fid1 load(['D:/research/trainFinal/Speed_new/speed_y_20_zheng_' num2str(input) '.txt'],'r')];
%     end
% end
% save ('D:/research/Class/train_ANN_Final.mat','train_x','test_x','train_y','test_y');
fid =fopen('D:/research/result/10min/rnn_0.txt','w');
train_x =reshape(train_x,4,5760*122);
test_x = reshape(test_x,4,2016*122);
left = 18;
right = 66;
% for i =1:5760
%         if rem(i,2) == 1
%                 indexbusy1 = [indexbusy1 i];
%             if(i<=2016)
%                 indexbusy2 = [indexbusy2 i];
%             end;
%         end;
% end;

%% training dataset generation
binary_dim = 8;

largest_number = 2^binary_dim-1;
binary = cell(largest_number,1);
int2binary = cell(largest_number,1);
for i = 1:largest_number+1
    binary{i} = dec2bin(i-1, 8);
    int2binary{i} = binary{i};
end

%% input variables
alpha = 0.1;
input_dim = 4;
hidden_dim = 16;
output_dim = 1;

%% initialize neural network weights

%% train logic
for input = 0:30
    synapse_0 = 2*rand(input_dim,hidden_dim) - 1;
    synapse_1 = 2*rand(hidden_dim,output_dim) - 1;
    synapse_h = 2*rand(hidden_dim,hidden_dim) - 1;

    synapse_0_update = zeros(size(synapse_0));
    synapse_1_update = zeros(size(synapse_1));
    synapse_h_update = zeros(size(synapse_h));

    for j = 1:100
    % generate a simple addition problem (a + b = c)
%     a_int = randi(round(largest_number/2)); % int version
%     a = int2binary{a_int+1}; % binary encoding
%     
%     b_int = randi(floor(largest_number/2)); % int version
%     b = int2binary{b_int+1}; % binary encoding
%     
%     % true answer
%     c_int = a_int + b_int;
%     c = int2binary{c_int+1};
    
    % where we'll store our best guess (binary encoded)
%     d = zeros(size(c));
%     
%     if length(d)<8
%         pause;
%     end
    
    overallError = 0;
    
    layer_2_deltas = [];
    layer_1_values = [];
    layer_1_values = [layer_1_values; zeros(1, hidden_dim)];
    trainsize = 5760;
    testsize = 2016;
    TrainIndex = [];
    TestIndex = [];
    for i = 1:trainsize
            if rem(i,72*4) >left*4 && rem(i,72*4)<= right*4
                TrainIndex = [TrainIndex i];
            end
            if i <=testsize && rem(i,72*4) >left*4 && rem(i,72*4)<= right*4 
                TestIndex = [TestIndex i];
            end
    end
    TrainIndex = TrainIndex + input*trainsize;
    TestIndex = TestIndex + input*testsize;
    train_x_new = train_x(:,TrainIndex);
    test_x_new = test_x(:,TestIndex);
    train_y_new = train_y(TrainIndex);
    test_y_new = test_y(TestIndex);
    % ��ʼ��һ�����н��д��������һ��������һ��LSTM��Ԫ�������ʵ����������
    for position = 1:size(TrainIndex,2)/20
%         X = [a(binary_dim - position)-'0' b(binary_dim - position)-'0'];   % X �� input
%         y = [c(binary_dim - position)-'0']';                               % Y ��label����������������
          X = train_x_new(:,position)';
          y = train_y_new(position);
        % ������RNN�����������Ƚϼ�
        % X ------------------------> input
        % sunapse_0 ----------------> U_i
        % layer_1_values(end, :) ---> previous hidden layer ��S(t-1)��
        % synapse_h ----------------> W_i
        % layer_1 ------------------> new hidden layer (S(t))
        layer_1 = sigmoid(X*synapse_0 + layer_1_values(end, :)*synapse_h);
        
        % layer_1 ------------------> hidden layer (S(t))
        % layer_2 ------------------> ���յ�����������ά��Ӧ���� label (Y) ��ά����һ�µ�
        % ����� sigmoid ��ʵ����һ���任���� hidden layer (size: 1 x 16) �任Ϊ 1 x 1
        % ��дʱ����������������ƥ��Ļ���ʹ����ʹ�� softmax ���б仯��
        % output layer (new binary representation)
        layer_2 = sigmoid(layer_1*synapse_1);
        
        % ���������������з��򴫲�
        % layer_2_error ------------> �˴Σ��� position+1 �ε���
        % l ����ʵ���
        % layer_2 ��������
        % layer_2_deltas �����ı仯�����ʹ���˷��򴫲������Ǹ��󵼣������������� layer_2���ǾͶ������󵼼��ɣ�Ȼ��������Ϳ��Եõ������diff��
        % did we miss?... if so, by how much?
        layer_2_error = y - layer_2;
        layer_2_deltas = [layer_2_deltas; layer_2_error*sigmoid_output_to_derivative(layer_2)];
        
        % ���������������и����þ���ֵ��
        overallError = overallError + abs(layer_2_error(1));
        
        % decode estimate so we can print it out
        % ���Ǽ�¼��λ�õ������������ʾ���
        %d(binary_dim - position) = round(layer_2(1));
        
        % ��¼�´˴ε������� (S(t))
        % store hidden layer so we can use it in the next timestep
        layer_1_values = [layer_1_values; layer_1];
    end
    
    % �����������diff������������ı仯�����������²���������ÿһ��timestep�����м���
    future_layer_1_delta = zeros(1, hidden_dim);
    
    % ��ʼ���з��򴫲������� hidden_layer ��diff���Լ������� diff
    for position = 0:size(TrainIndex)/20-1
        % ��Ϊ��ͨ������õ������㣬������ﻹ����Ҫ�õ������
        % a -> (operation) -> y, x_diff = derivative(x) * y_diff
        % ע����������ʼ��ǰ��
        X = train_x_new(:,size(TrainIndex)/20-1-position)';
        % layer_1 -----------------> ��ʾ������ hidden_layer (S(t))
        % prev_layer_1 ------------> (S(t-1))
        layer_1 = layer_1_values(size(TrainIndex)/20-1-position, :);
        prev_layer_1 = layer_1_values(size(TrainIndex)/20-1-position-1, :);
        
        % layer_2_delta -----------> �����������diff
        % hidden_layer_diff,��������������������diff�Լ���һ���������diff
        % error at output layer
        layer_2_delta = layer_2_deltas(size(TrainIndex)/20-1-position, :);
        % ����ط��� hidden_layer �����������棬��Ϊ hidden_layer -> next timestep, hidden_layer -> output��
        % ����䷴�򴫲�Ҳ��������
        % error at hidden layer
        layer_1_delta = (future_layer_1_delta*(synapse_h') + layer_2_delta*(synapse_1')) ...
                        .* sigmoid_output_to_derivative(layer_1);
        
        % let's update all our weights so we can try again
        synapse_1_update = synapse_1_update + (layer_1')*(layer_2_delta);
        synapse_h_update = synapse_h_update + (prev_layer_1')*(layer_1_delta);
        synapse_0_update = synapse_0_update + (X')*(layer_1_delta);
        
        future_layer_1_delta = layer_1_delta;
    end
    
    synapse_0 = synapse_0 + synapse_0_update * alpha;
    synapse_1 = synapse_1 + synapse_1_update * alpha;
    synapse_h = synapse_h + synapse_h_update * alpha;
    
    synapse_0_update = synapse_0_update * 0;
    synapse_1_update = synapse_1_update * 0;
    synapse_h_update = synapse_h_update * 0;
    
%     if(mod(j,1000) == 0)
%         err = sprintf('Error:%s\n', num2str(overallError)); fprintf(err);
%         d = bin2dec(num2str(d));
%         pred = sprintf('Pred:%s\n',dec2bin(d,8)); fprintf(pred);
%         Tru = sprintf('True:%s\n', num2str(c)); fprintf(Tru);
%         out = 0;
%         size(c)
%         sep = sprintf('-------------\n'); fprintf(sep);
%     end
    end
    fprintf(fid,'%d %f\n',input,overallError/192);
end
% %% training dataset generation
% binary_dim = 8;
% 
% largest_number = 2^binary_dim-1;
% binary = cell(largest_number,1);
% int2binary = cell(largest_number,1);
% for i = 1:largest_number+1
%     binary{i} = dec2bin(i-1, 8);
%     int2binary{i} = binary{i};
% end
% 
% %% input variables
% alpha = 0.1;
% input_dim = 2;
% hidden_dim = 16;
% output_dim = 1;
% 
% %% initialize neural network weights
% synapse_0 = 2*rand(input_dim,hidden_dim) - 1;
% synapse_1 = 2*rand(hidden_dim,output_dim) - 1;
% synapse_h = 2*rand(hidden_dim,hidden_dim) - 1;
% 
% synapse_0_update = zeros(size(synapse_0));
% synapse_1_update = zeros(size(synapse_1));
% synapse_h_update = zeros(size(synapse_h));
% 
% %% train logic
% for j = 0:19999
%     % generate a simple addition problem (a + b = c)
%     a_int = randi(round(largest_number/2)); % int version
%     a = int2binary{a_int+1}; % binary encoding
%     
%     b_int = randi(floor(largest_number/2)); % int version
%     b = int2binary{b_int+1}; % binary encoding
%     
%     % true answer
%     c_int = a_int + b_int;
%     c = int2binary{c_int+1};
%     
%     % where we'll store our best guess (binary encoded)
%     d = zeros(size(c));
%     
%     if length(d)<8
%         pause;
%     end
%     
%     overallError = 0;
%     
%     layer_2_deltas = [];
%     layer_1_values = [];
%     layer_1_values = [layer_1_values; zeros(1, hidden_dim)];
%     
%     % ��ʼ��һ�����н��д��������һ��������һ��LSTM��Ԫ�������ʵ����������
%     for position = 0:binary_dim-1
%         X = [a(binary_dim - position)-'0' b(binary_dim - position)-'0'];   % X �� input
%         y = [c(binary_dim - position)-'0']';                               % Y ��label����������������
%         
%         % ������RNN�����������Ƚϼ�
%         % X ------------------------> input
%         % sunapse_0 ----------------> U_i
%         % layer_1_values(end, :) ---> previous hidden layer ��S(t-1)��
%         % synapse_h ----------------> W_i
%         % layer_1 ------------------> new hidden layer (S(t))
%         layer_1 = sigmoid(X*synapse_0 + layer_1_values(end, :)*synapse_h);
%         
%         % layer_1 ------------------> hidden layer (S(t))
%         % layer_2 ------------------> ���յ�����������ά��Ӧ���� label (Y) ��ά����һ�µ�
%         % ����� sigmoid ��ʵ����һ���任���� hidden layer (size: 1 x 16) �任Ϊ 1 x 1
%         % ��дʱ����������������ƥ��Ļ���ʹ����ʹ�� softmax ���б仯��
%         % output layer (new binary representation)
%         layer_2 = sigmoid(layer_1*synapse_1);
%         
%         % ���������������з��򴫲�
%         % layer_2_error ------------> �˴Σ��� position+1 �ε���
%         % l ����ʵ���
%         % layer_2 ��������
%         % layer_2_deltas �����ı仯�����ʹ���˷��򴫲������Ǹ��󵼣������������� layer_2���ǾͶ������󵼼��ɣ�Ȼ��������Ϳ��Եõ������diff��
%         % did we miss?... if so, by how much?
%         layer_2_error = y - layer_2;
%         layer_2_deltas = [layer_2_deltas; layer_2_error*sigmoid_output_to_derivative(layer_2)];
%         
%         % ���������������и����þ���ֵ��
%         overallError = overallError + abs(layer_2_error(1));
%         
%         % decode estimate so we can print it out
%         % ���Ǽ�¼��λ�õ������������ʾ���
%         d(binary_dim - position) = round(layer_2(1));
%         
%         % ��¼�´˴ε������� (S(t))
%         % store hidden layer so we can use it in the next timestep
%         layer_1_values = [layer_1_values; layer_1];
%     end
%     
%     % �����������diff������������ı仯�����������²���������ÿһ��timestep�����м���
%     future_layer_1_delta = zeros(1, hidden_dim);
%     
%     % ��ʼ���з��򴫲������� hidden_layer ��diff���Լ������� diff
%     for position = 0:binary_dim-1
%         % ��Ϊ��ͨ������õ������㣬������ﻹ����Ҫ�õ������
%         % a -> (operation) -> y, x_diff = derivative(x) * y_diff
%         % ע����������ʼ��ǰ��
%         X = [a(position+1)-'0' b(position+1)-'0'];
%         % layer_1 -----------------> ��ʾ������ hidden_layer (S(t))
%         % prev_layer_1 ------------> (S(t-1))
%         layer_1 = layer_1_values(end-position, :);
%         prev_layer_1 = layer_1_values(end-position-1, :);
%         
%         % layer_2_delta -----------> �����������diff
%         % hidden_layer_diff,��������������������diff�Լ���һ���������diff
%         % error at output layer
%         layer_2_delta = layer_2_deltas(end-position, :);
%         % ����ط��� hidden_layer �����������棬��Ϊ hidden_layer -> next timestep, hidden_layer -> output��
%         % ����䷴�򴫲�Ҳ��������
%         % error at hidden layer
%         layer_1_delta = (future_layer_1_delta*(synapse_h') + layer_2_delta*(synapse_1')) ...
%                         .* sigmoid_output_to_derivative(layer_1);
%         
%         % let's update all our weights so we can try again
%         synapse_1_update = synapse_1_update + (layer_1')*(layer_2_delta);
%         synapse_h_update = synapse_h_update + (prev_layer_1')*(layer_1_delta);
%         synapse_0_update = synapse_0_update + (X')*(layer_1_delta);
%         
%         future_layer_1_delta = layer_1_delta;
%     end
%     
%     synapse_0 = synapse_0 + synapse_0_update * alpha;
%     synapse_1 = synapse_1 + synapse_1_update * alpha;
%     synapse_h = synapse_h + synapse_h_update * alpha;
%     
%     synapse_0_update = synapse_0_update * 0;
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
%     
% end