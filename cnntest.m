function [er,er2, er3,er4,er5,er6,numbusy,numunbusy] = cnntest(net, x, y,i,xishu,mode,fid1)
    %  feedforward

    net = cnnff(net, x);
    %if mode ~=0
        %partweight = importdata(['J:/jarrateNew/' num2str(i) 'jarrate30.txt']);
    %end;
    %partweightset = repmat(partweight,1,xishu);
    %if mode ==3
        %doubtable = importdata(['/home/guqian/Static/num_test' num2str(i) '.txt']);
    %else
        %doubtable = importdata(['/home/guqian/Static/num_train' num2str(i) '.txt']);
    %end;
   
    %fid =fopen(['D:/research/trainnew/result/Two' num2str(i) '.txt'],'a');
    [~, h] = max(net.o);
    [~, a] = max(y);
    if mode == 20

        fid3 =fopen(['D:/research/trainnew/ResultOld/10New3rd_20_01_' num2str(i) '.txt'],'w');
        fid2 = fopen(['D:/research/trainnew/ResultOld/10New3rd_20_float_' num2str(i) '.txt'],'w');
        for q = 1:length(h)
            fprintf(fid3,'%d ',h(q)-1);
            fprintf(fid2,'%d ',net.o(1,q));
        end
        fclose(fid2);
        fclose(fid3);
        %eval(['save D:/research/trainnew/result/4rd_20_01_' num2str(i) '.txt h -ASCII']) 
    end

        %doubtable = [1:144 289:504];
    
    %doubtable = 1:504;
    %h =h(doubtable);
    %a = a(doubtable);
    bad = find(h ~= a);
    indexbusy = find(a == 1);
    badbusy = find(h(indexbusy) ~= a(indexbusy));
    %for i =20:20
        %if i >=0

        %else
            %fid2 =importdata(['J:/3rdtest28-28da1/speed_y' num2str(i) '.txt'],'r');
            %fid1 = [fid1 fid2];
        %end;
    %end;
    if mode == 5 || mode== 6
        er =numel(bad);
        er2 =numel(badbusy);
        er4 =0;
        er5= 0;
        abnormal1 =0;
        abnormal2 = 0;
        abnormal3 = 0;
        for i = 1:size(a,2)
            if fid1(i) == 0 || fid1(i) == 30
                abnormal1 = abnormal1 + 1;
                if h(i) ==1 
                    abnormal2 = abnormal2 + 1;
                end
            end
        end
        for i = 1:numel(bad)
            if fid1(bad(i)) == 0 || fid1(bad(i)) == 30 || fid1(bad(i)) ==60;
                er = er -1;
                if fid1(bad(i)) == 0 || fid1(bad(i)) == 30
                    er2 =er2 -1;
                end;
            else if fid1(bad(i)) <=35 && fid1(bad(i))>=25 
                er4 =er4 +1;
                else if fid1(bad(i)) <=65 && fid1(bad(i))>=55
                er5 =er5+1;
                    end;
                 end;
            end;
        end;
        er = er/(size(a,2)-abnormal1);
        er2 =er2/(numel(indexbusy)-abnormal2);
        er4 =er4 /size(a,2);
        er5 =er5 /size(a,2);
    else
        er =numel(bad)/ size(a, 2);
        er2 =numel(badbusy)/numel(indexbusy);
        er4 =0;
        er5 =0;
    end;
    indexunbusy = find(a == 2);
    badunbusy = find(h(indexunbusy) ~= a(indexunbusy));
    er3 = numel(badunbusy)/numel(indexunbusy);
    indexunbusy2 =find (a == 3);
    badunbusy2 = find(h(indexunbusy2) ~= a(indexunbusy2));
    er4 = numel(badunbusy2)/numel(indexunbusy2);
    if mode == 15
        index4 = find(a == 4);
        bad4 = find(h(index4) ~= a(index4));
        er5 = numel(bad4)/numel(index4);
        index5 = find(a == 5);
        bad5 = find(h(index5) ~= a(index5));
        er6 = numel(bad5)/numel(index5);
    disp([num2str(er) ' ' num2str(numel(indexbusy)) ':' num2str(er2) ' ' num2str(numel(indexunbusy)) ':' num2str(er3) ' ' num2str(numel(indexunbusy2)) ':' num2str(er4) ' ' num2str(numel(index4)) ':' num2str(er5) ' ' num2str(numel(index5)) ':' num2str(er6)] );
    index6 = find(h == 1);
    bad6 = find(h(index6) ~= a(index6));
    er7 = numel(bad6)/numel(index6);
    index7 = find(h == 2);
    bad7 = find(h(index7) ~= a(index7));
    er8 = numel(bad7)/numel(index7);
    index8 = find(h == 3);
    bad8 = find(h(index8) ~= a(index8));
    er9 = numel(bad8)/numel(index8);
    index9 = find(h == 4);
    bad9 = find(h(index9) ~= a(index9));
    er10 = numel(bad9)/numel(index9);
    index10 = find(h == 5);
    bad10 = find(h(index10) ~= a(index10));
    er11 = numel(bad10)/numel(index10);
    disp([num2str(er7) ' ' num2str(er8) ' ' num2str(er9) ' ' num2str(er10) ' ' num2str(er11)] )
    else
         disp([num2str(er) ' ' num2str(numel(indexbusy)) ':' num2str(er2) ' ' num2str(numel(indexunbusy)) ':' num2str(er3) ' ' num2str(numel(indexunbusy2)) ':' num2str(er4) ' ' ] )
    end
    if mode == 2
        partweightnew = repmat(partweight,1,7);
        [~, h] = max(0.7*net.o+0.3*partweightnew);
        [~, a] = max(y);
        bad = find(h ~= a);
        indexbusy = find(a == 1);
        badbusy = find(h(indexbusy) ~= a(indexbusy));
        er2 = numel(badbusy)/numel(indexbusy);
        indexunbusy = find(a == 2);
        badunbusy = find(h(indexunbusy) ~= a(indexunbusy));
        er3 = numel(badunbusy)/numel(indexunbusy);
        er = numel(bad) / size(y, 2);
        disp([num2str(er) ' ' num2str(numel(indexbusy)) ':' num2str(er2) ' ' num2str(numel(indexunbusy)) ':' num2str(er3)]);
    end;
    if mode == 1
        fid =fopen(['/home/guqian/jarrateNew/' num2str(i) 'jarrate30.txt'],'w');
        badtotal = 0;
        for k = 1:72
            %H{k} = h(:,k:72:k+19*72);
            A = a(:,k:72:k+19*72);
            output = net.o(:,k:72:k+19*72);
            %b = partweight(k)
            %partweightnew{k} = repmat(partweight(k),1,72)
            badbest =20;
            j =0;
            for i = 0:100
                b (1,1) = i*0.01;
                b (2,1) = 1-i*0.01;
                partweightnew = repmat(b,1,20);
                [~,H]=max(0.7*output+0.3*partweightnew);
                bad = find(H ~= A);
                indexbusy = find(A == 1);
                badbusy = find(H(indexbusy) ~= A(indexbusy));
                er1 = numel(badbusy)/numel(indexbusy);
                if numel(bad) <= badbest
                    badbest =numel(bad);
                    j = i;
                end;
                if numel(bad) == 0
                    break;
                end;
            end;
            badtotal = badtotal + badbest;
            partweight(1,k)=j*0.01;
            partweight(2,k)=1 -j*0.01;
        end;
        for k = 1 : 72
            fprintf(fid,'%f ',partweight(1,k));
        end;
        fprintf(fid,'\n');
        for k =1 : 72
            fprintf(fid,'%f ',partweight(2,k));
        end;
        fprintf(fid,'\n');
    %fclose(fid1);
    %fclose(fid2);
    %fclose(fid3);
    end;
    numbusy = numel(indexbusy);
    numunbusy = numel(indexunbusy);

    %[~, h] = max(partweight);
    %[~, a] = max(y);
    %bad = find(h ~= a);
    %er2 = numel(bad) / size(y, 2);
    %disp(er2);
    %[~, h] = max(0.7*net.o+0.3*partweight);
    %[~, a] = max(y);
    %bad = find(h ~= a);
    %er3 = numel(bad) / size(y, 2);
    %disp(er3);
    %bad1 = find((h==1) ~
    %3= (a ==1));
    %if(~isempty(a == 1))
    %    er1 = numel(bad1) /max(size(a(a==1),2),size(h(h==1),2));
    %else
    %    er1 = 0
    %end;
    %bad2 = find((h==2) ~= (a ==2));
    %if(~isempty(a == 2))
    %    er2 = numel(bad2) /max(size(a(a==2),2),size(h(h==2),2));
    %else
    %    er2 = 0;
    %end;
    %bad3 = find((h==3) ~= (a ==3));
    %if(~isempty(a == 3))
    %    er3 = numel(bad3) /max(size(a(a==3),2),size(h(h==3),2));
    %else
    %    er3 = 0;
    %end;
    %[~, h] = max(partweight);
    %bad = find(h ~= a);
    %er2 = numel(bad) / size(y, 2);
    %disp(er2)
    %[~, h] = max(0.3*partweight+0.7*net.o);
    %bad = find(h ~= a);
    %er3 = numel(bad) / size(y, 2);
    %disp(er3)
    %fclose(fid1)

end
