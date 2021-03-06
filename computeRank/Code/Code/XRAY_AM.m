clear all;
close all;

%XRAY_AM.m

load /home/local/ANT/pjagdeep/Dropbox/NMF/Code/Reuters7285_new_preprocessed.mat;
%load ../Code/Reuters10_orig_3.mat;
%load ../Code/Reuters_preprocessed_new.mat
% load /home/local/ANT/pjagdeep/Dropbox/NMF/Code/TDT2_preprocessed_new.mat;
%load TDT2_preprocessed_12345.mat;
%load TDT-2_6.mat;
% load TDT2_0_10.mat;
%load /home/local/ANT/pjagdeep/Dropbox/NMF/Code/Twentyng_fea_preprocessed_new.mat;
%load /home/local/ANT/pjagdeep/Dropbox/NMF/Code/yale32_A_4096.mat;
% fea = A';
% clear A;
% load ../Code/Reuters_preprocessed_new.mat
% load tdt2_top30_online.mat;
% clearvars -except X Y;
% fea = X;
% gnd = Y;

k=10;
lowrank = 0;
%fea(:,sum(fea,1)<1)=[]; % Removing orphan words (not present in any document)

%idxf= find(sum(fea,2)<1);
%fea(idxf,:) = [];
%gnd(idxf) =[];
[n,d] = size(fea);

doc_length = sum(fea,2);

fea=tfidf_l1(fea,1);

fea = spdiags(doc_length,0,n,n) * fea;
A=fea';
clear fea;
% w0=1/k; binary=0; eps1 =

outpath_main = '/home/local/ANT/pjagdeep/Desktop/NMF/results_ER_SPA_AM_RTR_test';
mkdir(outpath_main);
no_of_iter = 1;
binary =0;
topic=zeros(no_of_iter,n);
Accuracy = zeros(no_of_iter,1);
nmi_val = zeros(no_of_iter,1);
Clustering_time = zeros(no_of_iter,1);


ndocs = size(A,2);
for i=0:no_of_iter-1
    outpath_iter = strcat(outpath_main,sprintf('/iter_%d',i));
    mkdir(outpath_iter);
    temp_start = tic;
    if i==0
%         temp_idx=reshape(FastConicalHull(A',k), 1,k);
%           temp_idx=reshape(spa(A',0.1,k), 1,k);
        %temp_idx=reshape(ER_spa_svds_v2(A',0.1,k), 1,k);
        temp_idx=reshape(pw_spa(A',0.1,k), 1,k);
%         temp_idx=reshape(ER_pw_spa_svds_v2(A',0.1,k), 1,k);
% temp_idx=reshape(ER_xray_svds_v2(A',0.1,k), 1,k);
%         [~,temp_idx, ~] = fastconichull2(A', A', k, 'maxres');
% 	   [~,temp_idx,obj] = fastconichull2(A', A', k, 'max');

if lowrank==1
    [Ua,Da,Va]= svds(A,k);
    A_k = Ua*Da*Va';
        C = A_k(temp_idx,:);
else
        C = A(temp_idx,:);
end
        B = nnlsHALSupdt(A',C'); B=B';
        [~,max_idx] = max(C,[],1);
        filename1 = strcat(outpath_iter,'/C_UTSVD.mat');
        save(filename1,'C','max_idx');
        
                C_tmp = C;
        max1 = zeros(ndocs,1);
        max2 = zeros(ndocs,1);
        for l1 = 1:ndocs
            max1(l1) = C_tmp(max_idx(l1),l1);
            C_tmp(max_idx(l1),l1) = 0;
        end
        [~,max_idx2] = max(C_tmp,[],1);
           for l2 = 1:ndocs 
            
            max2(l2) = C_tmp(max_idx2(l2),l2);
           end
           sm = sum(C,1);
           sm = sm';
           max1 = max1./sm;
           max2 = max2./sm;
           
           maxd = max1 - max2;
           %size(maxd)
           %size(sm)
           fprintf('max of max1 is %f\n',max(maxd))
           fprintf('min of max1 is %f\n',min(maxd))
           linspace(0,1,11)
           h = hist(maxd,10)
           
           
           res_a(1) = 100 * sum((max1 >0.4) .* (max2 < 0.2))/ndocs;
           res_a(2) = 100 * sum((max1 >0.4) .* (max2 < 0.3))/ndocs;
           res_a(3) = 100 * sum((max1 >0.5) .* (max2 < 0.2))/ndocs;
           res_a(4) = 100 * sum((max1 >0.5) .* (max2 < 0.3))/ndocs;
           res_a(5) = 100 * sum((max1 >0.95))/ndocs;
           
           fprintf('\n%f%% docs satisy max > 0.4 and max2 < 0.2 \n',res_a(1));
           fprintf('\n%f%% docs satisy max > 0.4 and max2 < 0.3 \n',res_a(2));
           fprintf('\n%f%% docs satisy max > 0.5 and max2 < 0.2 \n',res_a(3));
           fprintf('\n%f%% docs satisy max > 0.5 and max2 < 0.3 \n',res_a(4));
           fprintf('\n%f%% docs satisy max > 0.95 \n',100 * sum((max1 >0.95))/ndocs);
           
           fprintf(' 0.4 \n%f%% docs satisy max > 0.4 and max2 < 0.2 \n',100 * sum((max1 >0.4) .* (max2 < 0.2))/ndocs);
           100 * h(6:10)'/ndocs
           res_a'
           
           
        
    else
        
        C = nnlsHALSupdt(A,B);
        B = nnlsHALSupdt(A',C'); B=B';
        [~,max_idx] = max(C,[],1);
        filename1 = strcat(outpath_iter,'/C_UTSVD.mat');
        save(filename1,'C','max_idx');
        
    end
    
    topic(i+1,:) = help_empty_topics(max_idx,k,n);
    filename1 = strcat(outpath_iter,'/topic.mat');
    save(filename1,'topic');
    Accuracy(i+1) = find_accuracy_munkres(topic(i+1,:),gnd,k);
%     nmi_val(i+1) = nmi3(gnd', topic(i+1,:));
   nmi_val(i+1) = normalized_mutual_information(topic(i+1,:),gnd');
    Clustering_time(i+1) = toc(temp_start);
    fprintf('Clustering done in %f secs with nmi = %f, Accuracy = %d \n',Clustering_time(i+1),nmi_val(i+1),Accuracy(i+1));
    
    filename1 = strcat(outpath_iter,'/Result.mat');
    save(filename1,'nmi_val','Accuracy','Clustering_time');
    
end


