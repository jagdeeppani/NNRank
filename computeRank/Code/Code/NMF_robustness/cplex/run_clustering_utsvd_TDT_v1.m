clear all;
close all;
load TDT2.mat;
%clear trainIdx testIdx;
%addpath /home/jagdeep/Desktop/NMF/Experiments/glpkmex-master;
% Reducing the size of data similar to Mizutani etal.


% Preprocessing the data
% tic;
% stop_words=importdata('stopwords.txt');
% data1=importdata('TDT2_Terms.txt');
% R_terms=data1.textdata;
% clear data1;
% fea = preprocess_dataset(fea,R_terms,stop_words);
% disp('preprocessing done');
% toc;
%loading the preprocessed data
clear fea;
load TDT2_fea_processed.mat;


%fea = fea(1:8258,:); % 7285 x 18933
%gnd = gnd(1:8258);      % 7285 x 1
%fea = fea;
%gnd = gnd;
%fea1=fea;
%gnd1=gnd;
%ind = trainIdx<7286;
%trainIdx = trainIdx(ind);
% testIdx = testIdx(testIdx<7286);

%alt_min=0:1:20;
max_alt_min=6;

k=30;
fea(:,find(sum(fea,1)<6))=[]; % Removing orphan words (not present in any document)

noise_level=0.3;

% funs = {
%     @spa, ...
%     @fast_hull,...
%     @utsvd,...
%     };
funs={@ER_spa_svds};
% Descriptive names for the algorithms
% funs_str = {
%     'SPA',...
%     'XRAY',...
%     'UTSVD',...
%     };
funs_str = {'ER-SPA'};

num_algs = length(funs_str);

A=fea';
clear fea;
[m,n] = size(A);

% Accuracy=zeros(num_algs,max_alt_min+1);
% Accuracy_sd = zeros(num_algs,max_alt_min+1);
%
% nmi_val = zeros(num_algs,max_alt_min+1);
% nmi_val_sd = zeros(num_algs,max_alt_min+1);

alg_time=zeros(num_algs,1);
%failed_count=zeros(length(alt_min),1);

anchor_indices_all=zeros(num_algs,k);
err_flag=0;

for alg_idx = 1:num_algs
    if strcmp(funs_str{alg_idx},'UTSVD')
        disp('UTSVD started');
        outpath = '/home/jagdeep/Desktop/NMF/Experiments/tsvd-code-birch-v3/tsvd_cplex/output_clustering/tdt2_erspa/tsvd_output';
        mkdir(outpath);
        utime=tic;
        [M_utsvd,cluster_id2] = TSVD_nowrite(A,outpath,k);
        disp('M found');
        alg_time(alg_idx)=toc(utime);
        fprintf('%s completed in %f secs\n',funs_str{alg_idx},alg_time(alg_idx));
        
    else
        fprintf('%s started',funs_str{alg_idx})
        fun = funs{alg_idx};
        fun_time=tic;
        temp_idx1=reshape(fun(A', noise_level,k), 1,k);
        
        if (temp_idx1==0)
            disp('error happened');
            err_flag=1;
            break;
        end
        anchor_indices_all(alg_idx,:) = temp_idx1;
        alg_time(alg_idx)=toc(fun_time);
        fprintf('%s completed in %f secs\n',funs_str{alg_idx},alg_time(alg_idx));
    end
    
end

if (err_flag==1)
        disp('error happened in %s\n',funs_str{alg_idx})
        return;
end

outpath11 = '/home/jagdeep/Desktop/NMF/Experiments/tsvd-code-birch-v3/tsvd_cplex/output_clustering/tdt2_erspa/output';
mkdir(outpath11);
filename0 = strcat(outpath11,'/anchor_indices_all_tdt2.mat');
save(filename0,'anchor_indices_all');


Accuracy=zeros(num_algs,max_alt_min);
nmi_val=zeros(num_algs,max_alt_min);
optimization_time=zeros(num_algs,max_alt_min);

residual_L2 = zeros(num_algs,max_alt_min);
residual_L1 = zeros(num_algs,max_alt_min);
residual_s = zeros(num_algs,max_alt_min);



for alg_idx=1:num_algs
    
    for alt_min_count=0:max_alt_min
        topic=zeros(alg_idx,n);
        
        temp_start=tic;
        if  strcmp(funs_str{alg_idx},'UTSVD') && alt_min_count==0;
            M=M_utsvd;
            optimization_time(alg_idx,alt_min_count+1) = toc(temp_start);
            nmi_val(alg_idx,1) = nmi(gnd,cluster_id2);
            continue;
            
        elseif strcmp(funs_str{alg_idx},'UTSVD') && alt_min_count==1;
            W= nnlsHALSupdt( A , M );
            
        elseif strcmp(funs_str{alg_idx},'UTSVD')
            M = nnlsHALSupdt(A',W'); M=M';
            W = nnlsHALSupdt(A,M);
        elseif alt_min_count==0;
            W=A(anchor_indices_all(alg_idx,:),:);
        else
            M = nnlsHALSupdt(A',W'); M=M';
            W = nnlsHALSupdt(A,M);
            
        end
        
        [~,max_idx] = max(W);
        
        topic(alg_idx,:) = max_idx;
        optimization_time(alg_idx,alt_min_count+1) = toc(temp_start);
        fprintf('Topic found in %f secs\n',optimization_time(alg_idx,alt_min_count+1));
        
        if alt_min_count>0
            
            residual_L2(alg_idx,alt_min_count+1) = 1 - (norm(A-M*W,'fro')/norm(A,'fro'));
            residual_L1(alg_idx,alt_min_count+1) = 1 - (norm(A-M*W,1)/norm(A,1));
            residual_s(alg_idx,alt_min_count+1) = 1- (sumabs(A-M*W)/sumabs(A));
        end
        % Accuracy evaluation started
        
        Accuracy(alg_idx,alt_min_count+1) = find_accuracy(topic(alg_idx,:),gnd,k);
        
        % Accuracy evaluation done
        
        % This portion takes care of orphan topics i.e it ensures that each topic has atleast one document
        % for alg_idx=1:num_algs+1
        for adjust_topic=1:k
            topic_count2(adjust_topic)=sum(topic(alg_idx,:)==adjust_topic);
        end
        orphan_topics=find(topic_count2==0);
        [~,max_top]=max(topic_count2);
        
        doc_id2=1;
        for all_count=1:length(orphan_topics)
            found1=0;
            while(~found1)
                if topic(alg_idx,doc_id2)==max_top
                    found1=1;
                    topic(alg_idx,doc_id2)=orphan_topics(all_count);
                end
                
                doc_id2=doc_id2+1;
            end
        end
        
        % end
        
        % calculating the nmi
        nmi_val(alg_idx,alt_min_count+1) = nmi(gnd', topic(alg_idx,:));
        
    end
end

%outpath11 = '/home/jagdeep/Desktop/NMF/Experiments/tsvd-code-birch-v2/tsvd_cplex/output/clustering/alt_min_iter_files-temp1';
%mkdir(outpath11);
filename1 = strcat(outpath11,'/nmi_val.mat');
save(filename1,'nmi_val');
filename2 = strcat(outpath11,'/Accuracy_val.mat');
save(filename2,'Accuracy');
filename3 = strcat(outpath11,'/Residual_norm_L2.mat');
save(filename3,'residual_L2');
filename4 = strcat(outpath11,'/Residual_norm_L1.mat');
save(filename4,'residual_L1');
% filename5 = strcat(outpath11,'/Total_time.mat');
% save(filename5,'Total_time');
filename5 = strcat(outpath11,'/alg_time.mat');
save(filename5,'alg_time');
filename5 = strcat(outpath11,'/optimization_time.mat');
save(filename5,'optimization_time');

filename4 = strcat(outpath11,'/Residual_norm_s.mat');
save(filename4,'residual_s');




