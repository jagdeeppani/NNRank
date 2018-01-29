%UTSVD_AM.m
%load ../Code/Reuters7285_new_preprocessed.mat;
% load ../Code/Reuters10_orig_3.mat;
%load ../Code/Reuters_preprocessed_new.mat
load ../Code/TDT2_preprocessed_new.mat;
% load TDT2_preprocessed_12345.mat;
% load TDT-2_6.mat;
% load TDT2_0_10.mat;
%load ../Code/Twentyng_fea_preprocessed_new.mat;
%load ../Code/yale32_A_4096.mat;
%fea = A';
% clear A;
% load ../Code/Reuters_preprocessed_new.mat
% load tdt2_top30_online.mat;
% clearvars -except X Y;
% fea = X;
% gnd = Y;


k=30;
fea(:,sum(fea,1)<1)=[]; % Removing orphan words (not present in any document)

idxf= find(sum(fea,2)<1);
fea(idxf,:) = [];
gnd(idxf) =[];
[n,d] = size(fea);

doc_length = sum(fea,2);
%
%fea=tfidf(fea,1);
%
%fea = spdiags(doc_length,0,n,n) * fea;


A=fea';
clear fea;
% w0=1/k; binary=0;

outpath_main = '/home/jagdeep/Desktop/NMF/Experiments/ICML16/Clustering/Kmeans';
no_of_iter = 1;
binary =0;
topic=zeros(no_of_iter,n);
Accuracy = zeros(no_of_iter,1);
nmi_val = zeros(no_of_iter,1);
Clustering_time = zeros(no_of_iter,1);
reps = 5;


for i=0:no_of_iter-1
    outpath_iter = strcat(outpath_main,sprintf('/iter_%d',i));
    mkdir(outpath_iter);
    temp_start = tic;
    fprintf('Performing k-means on columns of B_k\n')
    % This is k-means %%
    tic;
    q_best = Inf;
    cluster_id = [];
    for r = 1:reps
        [~,ini_center] = kmeanspp_ini(A,k);
        [~, c_id, ~,q2,~] = kmeans_fast(A,ini_center,2,reps > 1);
        if q2 < q_best
            cluster_id = c_id;
            q_best = q2;
        end
    end
    disp(q_best)
    %fprintf('Quality:%.4f\n',q_best);
    toc;
    max_idx = cluster_id;
    filename1 = strcat(outpath_iter,'/cluster_id.mat');
    save(filename1,'cluster_id');
    
    topic(i+1,:) = help_empty_topics(max_idx,k,n);
    filename1 = strcat(outpath_iter,'/topic.mat');
    save(filename1,'topic');
    Accuracy(i+1) = find_accuracy_munkres(topic(i+1,:),gnd,k);
    nmi_val(i+1) = nmi3(gnd', topic(i+1,:));
    Clustering_time(i+1) = toc(temp_start);
    fprintf('Clustering done in %f secs with nmi = %f, Accuracy = %d \n',Clustering_time(i+1),nmi_val(i+1),Accuracy(i+1));
    
    filename1 = strcat(outpath_iter,'/Result.mat');
    save(filename1,'nmi_val','Accuracy','Clustering_time');
    
end


