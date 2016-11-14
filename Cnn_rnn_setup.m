function [sy] = Cnn_rnn_setup( sy,W_R,b_R,E_l,K,alpha)
sy.alpha = alpha;
sy.E_l = E_l;
sy.K = K;
sy.W_R = repmat(W_R,sy.K,1);  %��ʼֵ���ɼ���΢С�Ŷ�
sy.b_R = repmat(b_R,sy.K,1); 
sy.W_E  = sy.alpha*rand(sy.K,sy.E_l);
sy.b_E = sy.alpha*rand(1,sy.K);
sy.W_OR = sy.alpha*rand(1,sy.K);
sy.W_OE = sy.alpha*rand(1,sy.K);
sy.left =1;
sy.right=192;
sy.numpoches = 1;