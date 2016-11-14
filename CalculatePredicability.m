%此函数的作用，输入一个vector（路段的速度状态变化序列），和计算所得的熵，输出它的可预测性
%N:需要的大N
%entropy:计算所得信息熵
function y=CalculatePredicability(N,entropy)
syms phymax;
ourFunction=-phymax*log2(phymax)-(1-phymax)*log2(1-phymax)+(1-phymax)*log2(N-1);
results=solve(ourFunction==entropy);

%排除掉所有的复数解
y=nan;   %默认没有解
for i=1:size(results)
    if isreal(results(i))
        y=results(i);
        break;
    end
end
