clear all;
close all;
n = 100;
d = 100;
k = 10;
cw_arr=[3];

no_of_iter = 10;
etta = 0.1;   % Here we assume, sum of the topic word proportion of the catch words is 0.1
alpha_arr = [0.05];  % Each column of W is sampled from dirichlet with parameter 0.01
%beta_arr = 0.65:0.05:0.7;
beta_arr = [0.5,1,2];
normalize = 2;    % This says how the noise matrix will be bounded
residual_norm_param=1;
%
funs = {@LPsepNMF_oone, @heur_spa,@spa,@fast_hull,@ER_spa_svds_v2,@ER_xray_svds_v2,@TSVD,@UTSVD};
funs_str = {'LP-rho1','Heur-SPA','SPA','XRAY','ER-SPA','ER-XRAY','TSVD','UTSVD'};
num_algs = length(funs_str);
addpath '/home/local/ANT/pjagdeep/Downloads/glpkmex-master'
noise_level = 2;
for cw_count =1:length(cw_arr)
    cw = cw_arr(cw_count);
    for alpha_count=1:length(alpha_arr)
        alpha = alpha_arr(alpha_count);
        
        outpath_res_norm = sprintf('/home/local/ANT/pjagdeep/NMF_Experiments/synth_dominant_gaussian_small/res_norm/alpha-%f-cw-%d',alpha,cw);
        mkdir(outpath_res_norm);
        
        outpath_res_norm_db = sprintf('/home/local/ANT/pjagdeep/Dropbox/NMF/synth_dominant_gaussian_small/res_norm/alpha-%f-cw-%d',alpha,cw);
        mkdir(outpath_res_norm_db);
        
        
        res_norm_L1_final = zeros(num_algs,length(beta_arr));
        res_norm_L1_normalized_final = zeros(num_algs,length(beta_arr));
        res_norm_L2_final = zeros(num_algs,length(beta_arr));
        res_norm_L2_normalized_final = zeros(num_algs,length(beta_arr));
        res_norm_s_final = zeros(num_algs,length(beta_arr));
        res_norm_s_normalized_final = zeros(num_algs,length(beta_arr));
        res_norm_spct_normalized_final = zeros(num_algs,length(beta_arr));
        
        res_norm_orig_L1_final = zeros(num_algs,length(beta_arr));
        res_norm_orig_L1_normalized_final = zeros(num_algs,length(beta_arr));
        res_norm_orig_L2_final = zeros(num_algs,length(beta_arr));
        res_norm_orig_L2_normalized_final = zeros(num_algs,length(beta_arr));
        res_norm_orig_s_final = zeros(num_algs,length(beta_arr));
        res_norm_orig_s_normalized_final = zeros(num_algs,length(beta_arr));
        res_norm_orig_spct_normalized_final = zeros(num_algs,length(beta_arr));
        
        res_norm_M_L1_final = zeros(num_algs,length(beta_arr));
        res_norm_M_L2_final = zeros(num_algs,length(beta_arr));
        res_norm_M_s_final = zeros(num_algs,length(beta_arr));
        res_norm_M_spct_final = zeros(num_algs,length(beta_arr));
        
        res_norm_W_L1_final = zeros(num_algs,length(beta_arr));
        res_norm_W_L2_final = zeros(num_algs,length(beta_arr));
        res_norm_W_s_final = zeros(num_algs,length(beta_arr));
        res_norm_W_spct_final = zeros(num_algs,length(beta_arr));
        
        time_final = zeros(num_algs,length(beta_arr));
        
        for beta_count=1:length(beta_arr)
            beta=beta_arr(beta_count);
            
            res_norm_L1_temp=zeros(num_algs,no_of_iter);
            res_norm_L1_normalized_temp = zeros(num_algs,no_of_iter);
            res_norm_L2_temp = zeros(num_algs,no_of_iter);
            res_norm_L2_normalized_temp = zeros(num_algs,no_of_iter);
            res_norm_s_temp = zeros(num_algs,no_of_iter);
            res_norm_s_normalized_temp = zeros(num_algs,no_of_iter);
            res_norm_spct_normalized_temp = zeros(num_algs,no_of_iter);
            
            res_norm_orig_L1_temp= zeros(num_algs,no_of_iter);
            res_norm_orig_L1_normalized_temp = zeros(num_algs,no_of_iter);
            res_norm_orig_L2_temp = zeros(num_algs,no_of_iter);
            res_norm_orig_L2_normalized_temp = zeros(num_algs,no_of_iter);
            res_norm_orig_s_temp = zeros(num_algs,no_of_iter);
            res_norm_orig_s_normalized_temp = zeros(num_algs,no_of_iter);
            res_norm_orig_spct_normalized_temp = zeros(num_algs,no_of_iter);
            
            res_norm_M_L1_temp = zeros(num_algs,no_of_iter);
            res_norm_M_L2_temp = zeros(num_algs,no_of_iter);
            res_norm_M_s_temp = zeros(num_algs,no_of_iter);
            res_norm_M_spct_temp = zeros(num_algs,no_of_iter);
            
            res_norm_W_L1_temp = zeros(num_algs,no_of_iter);
            res_norm_W_L2_temp = zeros(num_algs,no_of_iter);
            res_norm_W_s_temp = zeros(num_algs,no_of_iter);
            res_norm_W_spct_temp = zeros(num_algs,no_of_iter);
            
            time_temp = zeros(num_algs,no_of_iter);
            
            
            current_iter=1;
            
            while current_iter<=no_of_iter
                for iter_count=current_iter:no_of_iter
                    %                     current_path = pwd;
                    %                     inpath_data = sprintf('/media/BM/Jagdeep/NMF/Results_new/synthetic_catchwords/data/cabi-%d-%f-%f-%d',cw,alpha,beta,iter_count);
                    
                    %                     cd(inpath_data);
                    %                     load A.mat;
                    %                     load M_orig.mat;
                    %                     load W_orig.mat;
                    %                     load permute_vect.mat;
                    
                    
                    %                     cd(current_path);
                    %                     A_orig = M_orig * W_orig;
                    [A,A_orig,M_orig,W_orig,permute_vect] = generate_synthetic(d,n,k,cw,etta,alpha,beta,normalize);  % We set A = (M_orig * W_orig + N)(perm_vect,:)
                    %
                    %                     filename = strcat(outpath_data,'/A_orig.mat');
                    %             save(filename,'A_orig');
                    outpath_data = sprintf('/home/local/ANT/pjagdeep/NMF_Experiments/synth_dominant_gaussian_small/data/cabi-%d-%f-%f-%d',cw,alpha,beta,iter_count);
                    mkdir(outpath_data);
                    %
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
                            outpath = sprintf('/home/local/ANT/pjagdeep/NMF_Experiments/synth_dominant_gaussian_small/utsvd_output1/Par-%d-%f-%f-%d',cw,alpha,beta,iter_count);
                            mkdir(outpath);
                            tusvd=tic;
                            [M,~] = TSVD_new_threshold(A,outpath,k,0);
                            disp('M found');
                            W = nnlsHALSupdt(A,M); % solve lp or nnls, which is better
                            disp('W found');
                            time_temp(alg_idx,iter_count)=toc(tusvd);
                            
                            filename = strcat(outpath_data,'/M_UTSVD.mat');
                            save(filename,'M');
                            filename = strcat(outpath_data,'/W_UTSVD.mat');
                            save(filename,'W');
                            
                        elseif  strcmp(funs_str{alg_idx},'TSVD')
                            outpath = sprintf('/home/local/ANT/pjagdeep/NMF_Experiments/synth_dominant_gaussian_small/tsvd_output1/Par-%d-%f-%f-%d',cw,alpha,beta,iter_count);
                            mkdir(outpath);
                            tusvd=tic;
                            [M,~] = TSVD(A,outpath,k);
                            disp('M found');
                            W = nnlsHALSupdt(A,M); % solve lp or nnls, which is better
                            disp('W found');
                            time_temp(alg_idx,iter_count)=toc(tusvd);
                            
                            filename = strcat(outpath_data,'/M_TSVD.mat');
                            save(filename,'M');
                            filename = strcat(outpath_data,'/W_TSVD.mat');
                            save(filename,'W');
                        else
                            fun=funs{alg_idx};
                            alg_time=tic;
                            anchor_indices = fun(A',noise_level, k);
                            
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
                            
                            strM = strcat('/M_',funs_str{alg_idx},'.mat');
                            strW = strcat('/W_',funs_str{alg_idx},'.mat');
                            
                            filename = strcat(outpath_data,strM);
                            save(filename,'M');
                            filename = strcat(outpath_data,strW);
                            save(filename,'W');
                            
                            
                        end
                        prediction = M*W;
                        residual= A-prediction;
                        res_norm_L1_temp(alg_idx,iter_count) = 1 - norm(residual,1);
                        res_norm_L1_normalized_temp(alg_idx,iter_count) = 1- norm(residual,1) / norm(A,1);
                        res_norm_L2_temp(alg_idx,iter_count) = 1 - norm(residual,'fro');
                        res_norm_L2_normalized_temp(alg_idx,iter_count) =1- norm(residual,'fro') / norm(A,'fro');
                        res_norm_s_temp(alg_idx,iter_count) = 1-sumabs(residual);
                        res_norm_s_normalized_temp(alg_idx,iter_count) = 1-(sumabs(residual)/sumabs(A));
                        res_norm_spct_normalized_temp(alg_idx,iter_count) = 1-(norm(residual)/norm(A));
                        
                        residual= A_orig-prediction;
                        res_norm_orig_L1_temp(alg_idx,iter_count) = norm(residual,1);
                        res_norm_orig_L1_normalized_temp(alg_idx,iter_count) = norm(residual,1) / norm(A_orig,1);
                        res_norm_orig_L2_temp(alg_idx,iter_count) = norm(residual,'fro');
                        res_norm_orig_L2_normalized_temp(alg_idx,iter_count) = norm(residual,'fro') / norm(A_orig,'fro');
                        res_norm_orig_s_temp(alg_idx,iter_count) = 1-sumabs(residual);
                        res_norm_orig_s_normalized_temp(alg_idx,iter_count) = 1-(sumabs(residual)/sumabs(A_orig));
                        res_norm_orig_spct_normalized_temp(alg_idx,iter_count) = 1-(norm(residual)/norm(A_orig));
                        
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
                        res_norm_M_L1_temp(alg_idx,iter_count) = 1 - norm(residual,1);
                        res_norm_M_L2_temp(alg_idx,iter_count) = 1 - norm(residual,'fro');
                        res_norm_M_s_temp(alg_idx,iter_count) = 1 - sumabs(residual);
                        %res_norm_M_spct_temp(alg_idx,iter_count) = 1 - norm(residual);
                        
                        W_new = W(Assign,:);
                        
                        residual= (W_new-W_orig);
                        res_norm_W_L1_temp(alg_idx,iter_count) = norm(residual,1);
                        res_norm_W_L2_temp(alg_idx,iter_count) = norm(residual,'fro');
                        res_norm_W_s_temp(alg_idx,iter_count) = 1 - sumabs(residual);
                        %res_norm_W_spct_temp(alg_idx,iter_count) = 1 - norm(residual);
                        
                        
                        fprintf('%10s : Ls residual norm    : %5.1f, beta: %f, cw= %d, alpha=%f \n',funs_str{alg_idx},res_norm_s_normalized_temp(alg_idx,iter_count),beta,cw,alpha);
                        fprintf('%10s : Ls residual norm(orig) : %5.1f, beta: %f, cw= %d, alpha=%f  \n',funs_str{alg_idx},res_norm_orig_s_normalized_temp(alg_idx,iter_count),beta,cw,alpha);
                        fprintf('%10s : spct residual norm    : %5.1f, beta: %f, cw= %d, alpha=%f  \n',funs_str{alg_idx},res_norm_spct_normalized_temp(alg_idx,iter_count),beta,cw,alpha);
                        fprintf('%10s : spct residual norm(orig) : %5.1f, beta: %f, cw= %d, alpha=%f  \n',funs_str{alg_idx},res_norm_orig_spct_normalized_temp(alg_idx,iter_count),beta,cw,alpha);
                        
                    end
                    if err_flag==1
                        break;
                    end
                    
                    %fileID = fopen('/home/jagdeep/Dropbox/NMF/Code_ICML16/status_catch.txt','w');
                    %status1 = [beta;iter_count];
                    %fprintf(fileID,'%4f %2f\n',status1);
                    %fclose(fileID);
                    
                end
                if iter_count==no_of_iter
                    current_iter=no_of_iter+1;
                end
                
                
            end
            
            res_norm_L1_final(:,beta_count) = mean(res_norm_L1_temp,2);
            res_norm_L1_normalized_final(:,beta_count) = mean(res_norm_L1_normalized_temp,2);
            
            
            res_norm_L2_final(:,beta_count) = mean(res_norm_L2_temp,2);
            res_norm_L2_normalized_final(:,beta_count) = mean(res_norm_L2_normalized_temp,2);
            
            res_norm_s_final(:,beta_count) = mean(res_norm_s_temp,2);
            res_norm_s_normalized_final(:,beta_count) = mean(res_norm_s_normalized_temp,2);
            res_norm_spct_normalized_final(:,beta_count) = mean(res_norm_spct_normalized_temp,2);
            
            res_norm_orig_L1_final(:,beta_count) = mean(res_norm_orig_L1_temp,2);
            res_norm_orig_L1_normalized_final(:,beta_count) = mean(res_norm_orig_L1_normalized_temp,2);
            res_norm_orig_L2_final(:,beta_count) = mean(res_norm_orig_L2_temp,2);
            res_norm_orig_L2_normalized_final(:,beta_count) = mean(res_norm_orig_L2_normalized_temp,2);
            res_norm_orig_s_final(:,beta_count) = mean(res_norm_orig_s_temp,2);
            res_norm_orig_s_normalized_final(:,beta_count) = mean(res_norm_orig_s_normalized_temp,2);
            res_norm_orig_spct_normalized_final(:,beta_count) = mean(res_norm_orig_spct_normalized_temp,2);
            
            
            res_norm_M_L1_final(:,beta_count) = mean(res_norm_M_L1_temp,2);
            res_norm_M_L2_final(:,beta_count) = mean(res_norm_M_L2_temp,2);
            res_norm_M_s_final(:,beta_count) = mean(res_norm_M_s_temp,2);
            
            res_norm_W_L1_final(:,beta_count) = mean(res_norm_W_L1_temp,2);
            res_norm_W_L2_final(:,beta_count) = mean(res_norm_W_L2_temp,2);
            res_norm_W_s_final(:,beta_count) = mean(res_norm_W_s_temp,2);
            
            time_final(:,beta_count) = mean(time_temp,2);
            
            % outpath2 = sprintf('/home/jagdeep/Desktop/NMF/Experiments/tsvd-code-birch-v2/tsvd_cplex/tsvd_synthetic25_birch/res_norm1/alpha-%f-cw-%d',alpha,cw);
            % mkdir(outpath2);
            filename = strcat(outpath_res_norm,'/res_norm_L1_final.mat');
            save(filename,'res_norm_L1_final');
            
            filename = strcat(outpath_res_norm,'/res_norm_L1_normalized_final.mat');
            save(filename,'res_norm_L1_normalized_final');
            
            filename = strcat(outpath_res_norm,'/res_norm_L2_final.mat');
            save(filename,'res_norm_L2_final');
            
            filename = strcat(outpath_res_norm,'/res_norm_L2_normalized_final.mat');
            save(filename,'res_norm_L2_normalized_final');
            
            filename = strcat(outpath_res_norm,'/res_norm_orig_L1_final.mat');
            save(filename,'res_norm_orig_L1_final');
            
            filename = strcat(outpath_res_norm,'/res_norm_orig_L1_normalized_final.mat');
            save(filename,'res_norm_orig_L1_normalized_final');
            
            filename = strcat(outpath_res_norm,'/res_norm_orig_L2_final.mat');
            save(filename,'res_norm_orig_L2_final');
            
            filename = strcat(outpath_res_norm,'/res_norm_orig_L2_normalized_final.mat');
            save(filename,'res_norm_orig_L2_normalized_final');
            
            filename = strcat(outpath_res_norm,'/res_norm_M_L1_final.mat');
            save(filename,'res_norm_M_L1_final');
            
            filename = strcat(outpath_res_norm,'/res_norm_M_L2_final.mat');
            save(filename,'res_norm_M_L2_final');
            
            filename = strcat(outpath_res_norm,'/res_norm_W_L1_final.mat');
            save(filename,'res_norm_W_L1_final');
            
            filename = strcat(outpath_res_norm,'/res_norm_W_L2_final.mat');
            save(filename,'res_norm_W_L2_final');
            
            filename = strcat(outpath_res_norm,'/res_norm_W_s_final.mat');
            save(filename,'res_norm_W_s_final');
            
            filename = strcat(outpath_res_norm,'/res_norm_M_s_final.mat');
            save(filename,'res_norm_M_s_final');
            
            
            filename = strcat(outpath_res_norm,'/res_norm_s_final.mat');
            save(filename,'res_norm_s_final');
            
            filename = strcat(outpath_res_norm,'/res_norm_s_normalized_final.mat');
            save(filename,'res_norm_s_normalized_final');
            
            filename = strcat(outpath_res_norm,'/res_norm_orig_s_normalized_final.mat');
            save(filename,'res_norm_orig_s_normalized_final');
            
            filename = strcat(outpath_res_norm,'/res_norm_orig_s_final.mat');
            save(filename,'res_norm_orig_s_final');
            
            filename = strcat(outpath_res_norm,'/res_norm_spct_normalized_final.mat');
            save(filename,'res_norm_spct_normalized_final');
            
            filename = strcat(outpath_res_norm,'/res_norm_orig_spct_normalized_final.mat');
            save(filename,'res_norm_orig_spct_normalized_final');
            
            
            
            filename = strcat(outpath_res_norm_db,'/res_norm_s_normalized_final.mat');
            save(filename,'res_norm_s_normalized_final');
            
            filename = strcat(outpath_res_norm_db,'/res_norm_orig_s_normalized_final.mat');
            save(filename,'res_norm_orig_s_normalized_final');
                        
            filename = strcat(outpath_res_norm_db,'/res_norm_spct_normalized_final.mat');
            save(filename,'res_norm_spct_normalized_final');
            
            filename = strcat(outpath_res_norm_db,'/res_norm_orig_spct_normalized_final.mat');
            save(filename,'res_norm_orig_spct_normalized_final');
           
            
            
            
            filename = strcat(outpath_res_norm,'/running_time.mat');
            save(filename,'time_final');
            
            
        end
        
    end
end







