%�˺��������ã�����һ��vector��·�ε��ٶ�״̬�仯���У����ͼ������õ��أ�������Ŀ�Ԥ����
%N:��Ҫ�Ĵ�N
%entropy:����������Ϣ��
function y=CalculatePredicability(N,entropy)
syms phymax;
ourFunction=-phymax*log2(phymax)-(1-phymax)*log2(1-phymax)+(1-phymax)*log2(N-1);
results=solve(ourFunction==entropy);

%�ų������еĸ�����
y=nan;   %Ĭ��û�н�
for i=1:size(results)
    if isreal(results(i))
        y=results(i);
        break;
    end
end
