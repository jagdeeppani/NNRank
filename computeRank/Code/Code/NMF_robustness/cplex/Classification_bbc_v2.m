function Classification_bbc_v2
clear all;
close all;
load bbc.mat;
clear trainIdx testIdx;
addpath /home/jagdeep/Desktop/NMF/Experiments/LIBLINEAR/liblinear-1.96/matlab;

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
% clear fea;
% load TwentyNG_fea_processed_v2.mat;


% fea = fea(1:8258,:); % 7285 x 18933
% gnd = gnd(1:8258);      % 7285 x 1
load train_test_indices_bbc.mat
% Train data and Test data
trainf = fea(trainIdx,:);
trainl = gnd(trainIdx,:);
testf = fea(testIdx,:);
testl = gnd(testIdx,:);

no_of_features=[100];

%alt_min=0:1:20;
max_alt_min=10;

k=100;
%fea(:,find(sum(fea,1)<1))=[]; % Removing orphan words (not present in any document)

noise_level=0.3;

funs = {@utsvd,@spa, @fast_hull,@heur_spa,@random_anchor};
% funs={@spa,@utsvd};
% Descriptive names for the algorithms
funs_str = {'UTSVD','SPA','XRAY','Heur_SPA','random'};
%  funs_str = {'SPA','UTSVD'};

num_algs = length(funs_str);

A=fea';
clear fea;
[m,n] = size(A);
outpath_data = '/home/jagdeep/Desktop/NMF/Experiments/tsvd-code-birch-v3/tsvd_cplex/output_classification/bbc_v2/data';
mkdir(outpath_data);
filename1 = strcat(outpath_data,'/features_A.mat');
save(filename1,'A');
filename1 = strcat(outpath_data,'/ground_truth.mat');
save(filename1,'gnd');

alg_time=zeros(num_algs,1);

anchor_indices_all=zeros(num_algs,k);
err_flag=0;


outpath_result = '/home/jagdeep/Desktop/NMF/Experiments/tsvd-code-birch-v3/tsvd_cplex/output_classification/bbc_v2/result';
mkdir(outpath_result);

svm_accuracy=zeros(num_algs+1,max_alt_min+1);

optimization_time=zeros(num_algs+1,max_alt_min+1);

residual_L2 = zeros(num_algs,max_alt_min+1);
residual_L1 = zeros(num_algs,max_alt_min+1);
residual_s = zeros(num_algs,max_alt_min+1);

time_orig_features=tic;
svm_accuracy(num_algs+1,:) = liblinear_svm_solver(trainf,trainl,testf,testl);
optimization_time(num_algs+1,:) = toc(time_orig_features);

for alg_idx = 1:num_algs
    if strcmp(funs_str{alg_idx},'UTSVD')
        disp('UTSVD started');
        outpath_tsvd = '/home/jagdeep/Desktop/NMF/Experiments/tsvd-code-birch-v3/tsvd_cplex/output_classification/bbc_v2/tsvd_output';
        mkdir(outpath_tsvd);
        utime=tic;
        [M_utsvd,cluster_id] = TSVD(A,outpath_tsvd,k);
        disp('M found');
        alg_time(alg_idx)=toc(utime);
        fprintf('%s completed in %f secs\n',funs_str{alg_idx},alg_time(alg_idx));
    
      
    
    elseif strcmp(funs_str{alg_idx},'random')
         temp_idx1=randperm(m);
         temp_idx1 = temp_idx1(1:k);
         anchor_indices_all(alg_idx,:) = temp_idx1;
                
    else   fprintf('%s started\n',funs_str{alg_idx})
        fun = funs{alg_idx};
        fun_time=tic;
        temp_idx1=reshape(fun(A', noise_level,k), 1,k);
        
        if (temp_idx1==0)
            disp('error happened\n');
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
%
% outpath_data = '/home/jagdeep/Desktop/NMF/Experiments/tsvd-code-birch-v3/tsvd_cplex/output_classification/tdt2_erspa/output';
% mkdir(outpath11);
filename0 = strcat(outpath_data,'/anchor_indices_all_bbc_v2.mat');
save(filename0,'anchor_indices_all');




for alg_idx=1:num_algs
    
    for alt_min_count=0:max_alt_min
        %topic=zeros(alg_idx,n);
        fprintf('Algorithm: %s, Iteration number: %d\n',funs_str{alg_idx},alt_min_count);
        outpath_data_iter=strcat(outpath_data,'/iter',num2str(alt_min_count));
        mkdir(outpath_data_iter);
        
        temp_start=tic;
        if  strcmp(funs_str{alg_idx},'UTSVD') && alt_min_count==0;
            M=M_utsvd;
            
            
            W = zeros(k,n);
            for i=1:n
                W(cluster_id(i),i) = 1;
            end
            
            train_anchor_f = (W(:,trainIdx))';
            test_anchor_f = (W(:,testIdx))';
            svm_accuracy(alg_idx,alt_min_count+1)=liblinear_svm_solver(train_anchor_f,trainl,test_anchor_f,testl);
            
            optimization_time(alg_idx,alt_min_count+1) = toc(temp_start);
            filename1 = strcat(outpath_data_iter,'/M_UTSVD.mat');
            save(filename1,'M');
            filename1 = strcat(outpath_data_iter,'/W_UTSVD.mat');
            save(filename1,'W');
            
            continue;
            
        elseif strcmp(funs_str{alg_idx},'UTSVD') && alt_min_count==1;
            W= nnlsHALSupdt( A , M );
            filename1 = strcat(outpath_data_iter,'/W_UTSVD.mat');
            save(filename1,'W');
            filename1 = strcat(outpath_data_iter,'/M_UTSVD.mat');
            save(filename1,'M');
            
        elseif strcmp(funs_str{alg_idx},'UTSVD')
            M = nnlsHALSupdt(A',W'); M=M';
            W = nnlsHALSupdt(A,M);
            filename1 = strcat(outpath_data_iter,'/M_UTSVD.mat');
            save(filename1,'M');
            filename1 = strcat(outpath_data_iter,'/W_UTSVD.mat');
            save(filename1,'W');
            
        elseif alt_min_count==0;
            W=A(anchor_indices_all(alg_idx,:),:);
            filename1 = strcat(outpath_data_iter,'/W_',funs_str{alg_idx},'.mat');
            save(filename1,'W');
            
        else
            M = nnlsHALSupdt(A',W'); M=M';
            W = nnlsHALSupdt(A,M);
            filename1 = strcat(outpath_data_iter,'/M_',funs_str{alg_idx},'.mat');
            save(filename1,'M');
            filename1 = strcat(outpath_data_iter,'/W_',funs_str{alg_idx},'.mat');
            save(filename1,'W');
        end
        
        train_anchor_f = (W(:,trainIdx))';
        test_anchor_f = (W(:,testIdx))';
        svm_accuracy(alg_idx,alt_min_count+1)=liblinear_svm_solver(train_anchor_f,trainl,test_anchor_f,testl);
        
        optimization_time(alg_idx,alt_min_count+1) = toc(temp_start);
        fprintf('SVM accuracy for algorithm %s with %d iterations: %f and time taken: %f secs\n',funs_str{alg_idx},alt_min_count,svm_accuracy(alg_idx,alt_min_count+1),optimization_time(alg_idx,alt_min_count+1));
%         filename1 = strcat(outpath_data_iter,'/topic_',funs_str{alg_idx},'.mat');
%         save(filename1,'topic');
        if alt_min_count>0
            
            residual_L2(alg_idx,alt_min_count+1) = 1 - (norm(A-M*W,'fro')/norm(A,'fro'));
            residual_L1(alg_idx,alt_min_count+1) = 1 - (norm(A-M*W,1)/norm(A,1));
            residual_s(alg_idx,alt_min_count+1) = 1- (sumabs(A-M*W)/sumabs(A));
        end
       
    end
end
temp_alg_time = repmat(alg_time,1,max_alt_min+1);
temp_alg_time(num_algs+1,:) = zeros(1,max_alt_min+1); 
Total_time = cumsum(optimization_time,2)+ temp_alg_time;
%outpath11 = '/home/jagdeep/Desktop/NMF/Experiments/tsvd-code-birch-v2/tsvd_cplex/output/clustering/alt_min_iter_files-temp1';
%mkdir(outpath11);

filename2 = strcat(outpath_result,'/svm_accuracy.mat');
save(filename2,'svm_accuracy');
filename3 = strcat(outpath_result,'/Residual_norm_L2.mat');
save(filename3,'residual_L2');
filename4 = strcat(outpath_result,'/Residual_norm_L1.mat');
save(filename4,'residual_L1');
filename4 = strcat(outpath_result,'/Residual_norm_s.mat');
save(filename4,'residual_s');
filename5 = strcat(outpath_result,'/Total_time.mat');
save(filename5,'Total_time');
filename5 = strcat(outpath_result,'/alg_time.mat');
save(filename5,'alg_time');
filename5 = strcat(outpath_result,'/optimization_time.mat');
save(filename5,'optimization_time');

filename4 = strcat(outpath_result,'/Residual_norm_s.mat');
save(filename4,'residual_s');
end


