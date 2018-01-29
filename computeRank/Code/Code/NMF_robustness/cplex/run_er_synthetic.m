clear all;
close all;
% run mizutani et al. synthetic experiment

m=5000;
n=250;
k= 10;
% sigma_arr = 0.4:0.1:0.5;
sigma_arr = [0.2];


funs = {
    @spa, ...
    @fast_hull,...
    @ER_spa_svds,...
    @ER_xray_svds
    };

% Descriptive names for the algorithms
funs_str = {'SPA', 'XRAY','ER-SPA','ER_XRAY','UTSVD','true-K'};

num_algs=length(funs);
no_of_iter=50;
iter_algo=zeros(num_algs,no_of_iter);
residual_norm_orig_temp=zeros(num_algs,no_of_iter);
outpath_data = '/home/jagdeep/Desktop/NMF/Experiments/tsvd-code-birch-v3/tsvd_cplex/output_er_synthetic/data';
mkdir(outpath_data);
outpath_result = '/home/jagdeep/Desktop/NMF/Experiments/tsvd-code-birch-v3/tsvd_cplex/output_er_synthetic/result';
mkdir(outpath_result);
outpath_tsvd = '/home/jagdeep/Desktop/NMF/Experiments/tsvd-code-birch-v3/tsvd_cplex/output_er_synthetic/tsvd';
mkdir(outpath_tsvd);


for sigma_idx=1:length(sigma_arr)
    sigma = sigma_arr(sigma_idx);
    
    
    for iter_count = 1:no_of_iter
        [A,true_index] = synthetic_data_mizutani(m,n,k,sigma); % A is a word x document matrix
        
        for alg_idx = 1:num_algs
            if strcmp(funs_str{alg_idx},'UTSVD')
                utime=tic;
                [M,~] = TSVD(A,outpath_tsvd,k);
                disp('M found');
                alg_time(alg_idx,iter_count)=toc(utime);
                fprintf('%s completed in %f secs\n',funs_str{alg_idx},alg_time(alg_idx));
                W = nnlsHALSupdt(A,M);
                
                
            else
                fun = funs{alg_idx};
                alg_time_temp=tic;
                anchor_index = fun(A',sigma, k);
                % anchor_index=reshape(anchor_index,1,k);
                W = A(anchor_index,:);
                M = nnlsHALSupdt(A',W'); %  solving for 1 norm error and Rev(Inf,1) error are same. % solve lp or nnls, which is better
                M=M';
                alg_time(alg_idx,iter_count)=toc(alg_time_temp);
            end
            
            residual = A-M*W;
            residual_fro_norm_temp(alg_idx,iter_count) = norm(residual,'fro');
            temp = 1- ( norm(residual,1) / norm(A,1) );
            iter_algo(alg_idx,iter_count)=temp;
            res_s_norm_temp(alg_idx,iter_count) = sumabs(residual);
            res_s_normalized_temp(alg_idx,iter_count) = 1-(sumabs(residual)/sumabs(A));
            recovery_rate_temp(alg_idx,iter_count) = length(intersect(true_index,anchor_index))/k;
            
        end
    end
    
    residual_norm_orig(:,sigma_idx) = mean(residual_norm_orig_temp,2);
    residual_norm(:,sigma_idx) = mean(iter_algo,2);
    res_s_norm_final(:,sigma_idx) = mean(res_s_norm_temp,2);
    res_s_normalized_final(:,sigma_idx) = mean(res_s_normalized_temp,2);
    recovery_rate_final(:,sigma_idx) = mean(recovery_rate_temp,2);
    
    %final_fractiles1(:,sigma_idx)=mean(fractiles1,2);
    %final_fractiles2(:,sigma_idx)=mean(fractiles2,2);
    % clear iter_algo residual_norm_orig_temp;
    disp(recovery_rate_final);
    
    
    
    
end









