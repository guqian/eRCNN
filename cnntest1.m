function [er,er2] = cnntest1(net,x,y)
   net = cnnff(net,x);
   er = mean(abs(net.o-y));
   %er2 = sqrt(mean(abs(net.o -y).^2));
   er2 = mean((abs(net.o-y)-repmat(mean(abs(net.o-y)),size(y),1)).^2);
   disp([num2str(er) ' ' num2str(er2)]);
   return;
   right =0;
   for i = 1:size(y,2)
       if (net.o(i) < 0.8 && y(i) < 0.8)  || (net.o(i) >= 0.8 && y(i) >= 0.8)
           right = right +1
       end
   end
   disp(right/size(y,2));
end
