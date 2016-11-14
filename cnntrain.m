function [netBest,mode] = cnntrain(net, x, y,testx,testy, input,opts,fid1)
    m = size(x, 3);
    numbatches = m / opts.batchsize;
    if rem(numbatches, 1) ~= 0
        error('numbatches not integer');
    end
    net.rL = [];
    ermin.id =1;
    ermin.advance=1;
    flag = 0;
    ermin.value = cnntest1(net,x,y);
    netBest =net;
    mode = 1;
    for i = 1 : opts.numepochs
        disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)]);
        tic;
        kk = randperm(m);
        %kk = 1:960;
        for l = 1 : numbatches
            batch_x = x(:, :, kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
            batch_y = y(:,    kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
            net = cnnff(net, batch_x);
            net = cnnbp(net, batch_y);
            net = cnnapplygrads(net, opts);
            if isempty(net.rL)
                %net.L =gather(net.L);
                net.rL(1) = net.L;
            end
            %net.rL = gpuArray(net.rL);
            net.rL(end + 1) = 0.99 * net.rL(end) + 0.01 * net.L;
        end
        toc;
        %[er1,er2,er3] = cnntest(net, x,y ,input,15,15,fid1);
        er1 = cnntest1(net,x,y);
        
        %if er1 < er2 
            %if er1 < er3
             %   if er1 <ermin.value
              %      ermin.value =er1;
               %     ermin.id = i;
                %    netBest =net;
                 %   mode =1;
                %end;
            %else if er3 <ermin.value  
             %       ermin.value =er3;
              %      ermin.id = i;
               %     netBest =net;
                %    mode =3;
                %end;
            %end;
        %else if er2 < er3
         %      if er2 <ermin.value
          %          ermin.value =er2;
           %         ermin.id = i;
            %        netBest =net;
             %       mode =2;
              % end;
            %else if er3 <ermin.value
             %       ermin.value =er3;
              %      ermin.id = i;
               %     netBest =net;
                %    mode =3;
                %end;
           % end;
        %end;
        if er1 == ermin.advance
            flag = flag+1;
        else 
            ermin.advance=er1;
            flag = 0;
        end;
        if flag >=100
            break;
        end;
        if er1 <ermin.value
            ermin.value =er1;
            ermin.id = i;
            netBest =net;
            mode =1;
        end;    
        %cnntest(net,testx,testy,input,15,15,fid1);
        cnntest1(net,testx,testy);
    end
    
end
