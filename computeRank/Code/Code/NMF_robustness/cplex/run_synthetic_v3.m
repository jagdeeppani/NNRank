clear all;
close all;
n = 10000;
d = 5000;
k = 50;
cw_arr=[25];

no_of_iter = 4;
etta = 0.1;   % Here we assume, sum of the topic word proportion of the catch words is 0.1
alpha_arr = [0.01 0.02 0.03];  % Each column of W is sampled from dirichlet with parameter 0.01
beta_arr = 0:0.05:0.4;
% beta_arr = 0.1;
normalize = 2;    % This says how the noise matrix will be bounded
residual_norm_param=1;
% 
funs = {
    @heur_spa,...
    @PostPrecSPA,...
    @PrecSPA,...
    @spa, ...
    @fast_hull,...
    @utsvd,...
    };
% funs={@utsvd};
funs_str = {'Heur-SPA','SPA','XRAY','UTSVD'};
% funs_str = {'UTSVD'};%,'XRAY','UTSVD'};
num_algs = length(funs);

res_norm_L1_final = zeros(num_algs,length(beta_arr));
res_norm_L1_normalized_final = zeros(num_algs,length(beta_arr));
res_norm_L2_final = zeros(num_algs,length(beta_arr));
res_norm_L2_normalized_final = zeros(num_algs,length(beta_arr));

res_norm_orig_L1_final = zeros(num_algs,length(beta_arr));
res_norm_orig_L1_normalized_final = zeros(num_algs,length(beta_arr));
res_norm_orig_L2_final = zeros(num_algs,length(beta_arr));
res_norm_orig_L2_normalized_final = zeros(num_algs,length(beta_arr));

res_norm_M_L1_final = zeros(num_algs,length(beta_arr));
res_norm_M_L2_final = zeros(num_algs,length(beta_arr));

res_norm_W_L1_final = zeros(num_algs,length(beta_arr));
res_norm_W_L2_final = zeros(num_algs,length(beta_arr));

time_final = zeros(num_algs,length(beta_arr));

for cw_count =1:length(cw_arr)
    cw = cw_arr(cw_count);
    for alpha_count=1:length(alpha_arr)
        alpha = alpha_arr(alpha_count);
        
        outpath2 = sprintf('/home/jagdeep/Desktop/NMF/Experiments/tsvd-code-birch-v2/tsvd_cplex/tsvd_synthetic25_birch/res_norm1/alpha-%f-cw-%d',alpha,cw);
        mkdir(outpath2);
        for beta_count=1:length(beta_arr)
            beta=beta_arr(beta_count);
            
            res_norm_L1_temp=zeros(num_algs,no_of_iter);
            res_norm_L1_normalized_temp = zeros(num_algs,no_of_iter);
            res_norm_L2_temp = zeros(num_algs,no_of_iter);
            res_norm_L2_normalized_temp = zeros(num_algs,no_of_iter);
            
            res_norm_orig_L1_temp= zeros(num_algs,no_of_iter);
            res_norm_orig_L1_normalized_temp = zeros(num_algs,no_of_iter);
            res_norm_orig_L2_temp = zeros(num_algs,no_of_iter);
            res_norm_orig_L2_normalized_temp = zeros(num_algs,no_of_iter);
            
            res_norm_M_L1_temp = zeros(num_algs,no_of_iter);
            res_norm_M_L2_temp = zeros(num_algs,no_of_iter);
            
            res_norm_W_L1_temp = zeros(num_algs,no_of_iter);
            res_norm_W_L2_temp = zeros(num_algs,no_of_iter);
            time_temp = zeros(num_algs,no_of_iter);
            
            
            current_iter=1;
            
            while current_iter<=no_of_iter
                for iter_count=current_iter:no_of_iter
                    [A,A_orig,M_orig,W_orig,permute_vect] = generate_synthetic(d,n,k,cw,etta,alpha,beta,normalize); % A = (M_orig * W_orig + N)(perm_vect,:); 
%             
%                     filename = strcat(outpath_data,'/A_orig.mat');
%             save(filename,'A_orig');
            filename = strcat(outpath_data,'/M_orig.mat');
            save(filename,'M_orig');
            filename = strcat(outpath_data,'/W_orig.mat');
            save(filename,'W_orig');
            filename = strcat(outpath_data,'/A.mat');
            save(filename,'A');
            filename = strcat(outpath_data,'/permute_vect.mat');
            save(filename,'permute_vect');
            
                    err_flag=0;
                    fprintf('Current beta: %f  , iter: %f \n',beta,iter_count);
                    for alg_idx=1:num_algs
                        
                        if  strcmp(funs_str{alg_idx},'UTSVD')
                            outpath = sprintf('/home/jagdeep/Desktop/NMF/Experiments/tsvd-code-birch-v2/tsvd_cplex/tsvd_synthetic25_birch/tsvd_output1/Par-%d-%f-%f-%d',cw,alpha,beta,iter_count);
                            mkdir(outpath);
                            tusvd=tic;
                            [M,~] = TSVD(A,outpath,k);
                            disp('M found');
                            W = nnlsHALSupdt(A,M); % solve lp or nnls, which is better
                            disp('W found');
                            time_temp(alg_idx,iter_count)=toc(tusvd);
                        else
                            fun=funs{alg_idx};
                            alg_time=tic;
                            anchor_indices = fun(A',beta, k);
                            
                            if length(anchor_indices)<k
                                disp('cant find k topics');
                                err_flag=1;
                                current_iter=iter_count;
                                toc(alg_time);
                                break;
                                
                            end
                            
                            W = A(anchor_indices,:);
                            M = nnlsHALSupdt(A',W');   M=M';  % solving for 1 norm error and Rev(Inf,1) error are same. % solve lp or nnls, which is better
                            time_temp(alg_idx,iter_count)=toc(alg_time);
                            
                        end
                        
                        residual= A-M*W;
                        res_norm_L1_temp(alg_idx,iter_count) = norm(residual,1);
                        res_norm_L1_normalized_temp(alg_idx,iter_count) = norm(residual,1) / norm(A,1);
                        res_norm_L2_temp(alg_idx,iter_count) = norm(residual,'fro');
                        res_norm_L2_normalized_temp(alg_idx,iter_count) = norm(residual,'fro') / norm(A,'fro');
                        
                        residual= A_orig-M*W;
                        res_norm_orig_L1_temp(alg_idx,iter_count) = norm(residual,1);
                        res_norm_orig_L1_normalized_temp(alg_idx,iter_count) = norm(residual,1) / norm(A_orig,1);
                        res_norm_orig_L2_temp(alg_idx,iter_count) = norm(residual,'fro');
                        res_norm_orig_L2_normalized_temp(alg_idx,iter_count) = norm(residual,'fro') / norm(A_orig,'fro');
                        
                        % cost_mat = -(M_orig' * M); % cost_mat(i,j) says the cost of assigning ith column of M_orig to j th column of M
                        cost_mat=zeros(k,k);
                        for c_idx1=1:k
                            for c_idx2=1:k
                                cost_mat(c_idx1,c_idx2)=norm(M_orig(:,c_idx1)-M(:,c_idx2),1);
                            end
                        end
                        [Assign,cost] = munkres(cost_mat);
                        M_new = M(:,Assign);
                        
                        residual= M_new-M_orig;
                        res_norm_M_L1_temp(alg_idx,iter_count) = norm(residual,1);
                        res_norm_M_L2_temp(alg_idx,iter_count) = norm(residual,'fro');
                        
                        W_new = W(Assign,:);
                        
                        residual= (W_new-W_orig);
                        res_norm_W_L1_temp(alg_idx,iter_count) = norm(residual,1);
                        res_norm_W_L2_temp(alg_idx,iter_count) = norm(residual,'fro');
                        
                        
                        
                        fprintf('%10s : L1 residual norm    : %5.1f : beta: %f\n',funs_str{alg_idx},res_norm_L1_temp(alg_idx,iter_count),beta);
                        fprintf('%10s : L1 residual norm(N) : %5.1f : beta: %f \n',funs_str{alg_idx},res_norm_L1_normalized_temp(alg_idx,iter_count),beta);
                        fprintf('%10s : L2 residual norm    : %5.1f : beta: %f \n',funs_str{alg_idx},res_norm_L2_temp(alg_idx,iter_count),beta);
                        fprintf('%10s : L2 residual norm(N) : %5.1f : beta: %f \n',funs_str{alg_idx},res_norm_L2_normalized_temp(alg_idx,iter_count),beta);
                        
                    end
                    if err_flag==1
                        break;
                    end
                end
                if iter_count==no_of_iter
                    current_iter=no_of_iter+1;
                end
            end
            
            res_norm_L1_final(:,beta_count) = mean(res_norm_L1_temp,2);
            res_norm_L1_normalized_final(:,beta_count) = mean(res_norm_L1_normalized_temp,2);
            
            
            res_norm_L2_final(:,beta_count) = mean(res_norm_L2_temp,2);
            res_norm_L2_normalized_final(:,beta_count) = mean(res_norm_L2_normalized_temp,2);
            
            
            res_norm_orig_L1_final(:,beta_count) = mean(res_norm_orig_L1_temp,2);
            res_norm_orig_L1_normalized_final(:,beta_count) = mean(res_norm_orig_L1_normalized_temp,2);
            res_norm_orig_L2_final(:,beta_count) = mean(res_norm_orig_L2_temp,2);
            res_norm_orig_L2_normalized_final(:,beta_count) = mean(res_norm_orig_L2_normalized_temp,2);
            
            
            res_norm_M_L1_final(:,beta_count) = mean(res_norm_M_L1_temp,2);
            res_norm_M_L2_final(:,beta_count) = mean(res_norm_M_L2_temp,2);
            
            
            res_norm_W_L1_final(:,beta_count) = mean(res_norm_W_L1_temp,2);
            res_norm_W_L2_final(:,beta_count) = mean(res_norm_W_L2_temp,2);
            
            time_final(:,beta_count) = mean(time_temp,2);
           
%             outpath2 = sprintf('/home/jagdeep/Desktop/NMF/Experiments/tsvd-code-birch-v2/tsvd_cplex/tsvd_synthetic25_birch/res_norm1/alpha-%f-cw-%d',alpha,cw);
%         mkdir(outpath2);
        filename = strcat(outpath2,'/res_norm_L1_final.mat');
        save(filename,'res_norm_L1_final');
        
        filename = strcat(outpath2,'/res_norm_L1_normalized_final.mat');
        save(filename,'res_norm_L1_normalized_final');
        
        filename = strcat(outpath2,'/res_norm_L2_final.mat');
        save(filename,'res_norm_L2_final');
        
        filename = strcat(outpath2,'/res_norm_L2_normalized_final.mat');
        save(filename,'res_norm_L2_normalized_final');
        
        filename = strcat(outpath2,'/res_norm_orig_L1_final.mat');
        save(filename,'res_norm_orig_L1_final');
        
        filename = strcat(outpath2,'/res_norm_orig_L1_normalized_final.mat');
        save(filename,'res_norm_orig_L1_normalized_final');
        
        filename = strcat(outpath2,'/res_norm_orig_L2_final.mat');
        save(filename,'res_norm_orig_L2_final');
        
        filename = strcat(outpath2,'/res_norm_orig_L2_normalized_final.mat');
        save(filename,'res_norm_orig_L2_normalized_final');
        
        filename = strcat(outpath2,'/res_norm_M_L1_final.mat');
        save(filename,'res_norm_M_L1_final');
        
        filename = strcat(outpath2,'/res_norm_M_L2_final.mat');
        save(filename,'res_norm_M_L2_final');
        
        filename = strcat(outpath2,'/res_norm_W_L1_final.mat');
        save(filename,'res_norm_W_L1_final');
        
        filename = strcat(outpath2,'/res_norm_W_L2_final.mat');
        save(filename,'res_norm_W_L2_final');
        
        filename = strcat(outpath2,'/running_time.mat');
        save(filename,'time_final');
            
            
            
        end
        
    end
end







