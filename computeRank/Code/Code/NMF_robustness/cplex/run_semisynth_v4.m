% A=M*W

%function run_gillislp_vs_ntsvd
clear all;
close all;


M_orig= dlmread('M');
M_orig=M_orig';
W_orig= dlmread('nyt5kr-50topics-theta-0.01-20000_300');
W_orig=W_orig';
[m,k]=size(M_orig);
n=size(W_orig,2);

load A_orig_20k.mat;
% [m,n]=size(A_orig); % m is the vocab size  % n is the number of documents   % number of topics

no_of_iter = 5;
beta_arr=[0.2];

funs = {
    @heur_spa,...
    @spa, ...
    @fast_hull,...
    @utsvd,...
    };
% funs = {@ER_spa_svds};
funs_str = {'Heur-SPA','SPA','XRAY','UTSVD'};
% funs_str = {'SPA','true_K'};

num_algs = length(funs_str);    % UTSVD and True k are not counted in num_algs

res_norm_L1_A_final = zeros(num_algs,length(beta_arr));
res_norm_L2_A_final = zeros(num_algs,length(beta_arr));
res_t_norm_L1_A_final = zeros(num_algs,length(beta_arr));

res_norm_L1_Aorig_final = zeros(num_algs,length(beta_arr));
res_norm_L2_Aorig_final = zeros(num_algs,length(beta_arr));
res_t_norm_L1_Aorig_final = zeros(num_algs,length(beta_arr));

res_norm_L1_M_final = zeros(num_algs,length(beta_arr));
res_norm_L2_M_final = zeros(num_algs,length(beta_arr));
res_t_norm_L1_M_final = zeros(num_algs,length(beta_arr));

res_norm_L1_W_final = zeros(num_algs,length(beta_arr));
res_norm_L2_W_final = zeros(num_algs,length(beta_arr));
res_t_norm_L1_W_final = zeros(num_algs,length(beta_arr));

time_final = zeros(num_algs,length(beta_arr));


normalize=2; % Normalize is column wise l1 or l2. if l1, ||N||_1<=eps1
density = 1.0;   % Proportion of nonzero entries in the noise matrix (but at least one nonzero per column).

for beta_idx = 1:length(beta_arr)
    beta = beta_arr(beta_idx);
    
    
    current_iter=1;
    loop_count=1;
    
    
    res_norm_L1_A_temp = zeros(num_algs,no_of_iter);   %To be updated
    res_norm_L2_A_temp = zeros(num_algs,no_of_iter);
    res_t_norm_L1_A_temp = zeros(num_algs,no_of_iter);
    
    res_norm_L1_Aorig_temp = zeros(num_algs,no_of_iter);
    res_norm_L2_Aorig_temp = zeros(num_algs,no_of_iter);
    res_t_norm_L1_Aorig_temp = zeros(num_algs,no_of_iter);
    
    res_norm_L1_M_temp = zeros(num_algs,no_of_iter);
    res_norm_L2_M_temp = zeros(num_algs,no_of_iter);
    res_t_norm_L1_M_temp = zeros(num_algs,no_of_iter);
    
    res_norm_L1_W_temp = zeros(num_algs,no_of_iter);
    res_norm_L2_W_temp = zeros(num_algs,no_of_iter);
    res_t_norm_L1_W_temp = zeros(num_algs,no_of_iter);
    
    time_temp = zeros(num_algs,no_of_iter);
    
    while current_iter<=no_of_iter && loop_count<20
        err_flag=0;
        for iter_count = current_iter:no_of_iter
            
            if (beta>0)
                N=normrnd(0,1,m,n);
                % Normalizing the Noise matrix
                
                if normalize == 1
                    cn_a = sum(abs(A_orig));
                    %cn_n = sum(abs(N));
                    N = N * (beta/sqrt(m)) * diag(cn_a);
                    
                elseif normalize == 2
                    cn_a = sqrt(sum(A_orig.^2));
                    %cn_n = sqrt(sum(N.^2));
                    N = N * (beta/sqrt(m)) * diag(cn_a);
                else
                    error('Invalid normalization type');
                end
            else
                N=zeros(m,n);
            end
            
            
            A=A_orig+N;
            clear N;
            
            fprintf('Data generated with  - beta   : %d\n',beta);
            %fprintf('   - Density of the noise: %d%%\n',100*density);
            disp('************************************************************************************');
            
            for alg_idx = 1:num_algs
                if  strcmp(funs_str{alg_idx},'UTSVD')
                    outpath = sprintf('/home/jagdeep/Desktop/NMF/Experiments/tsvd-code-birch-v2/tsvd_cplex/semisynthetic_20k/tsvd_out/beta_iter%d_%d',beta,iter_count);
                    mkdir(outpath);
                    tusvd=tic;
                    [M,~] = TSVD(A,outpath,k);  % Birch is set inside TSVD
                    disp('M found');
                    W = nnlsHALSupdt(A,M); % solve lp or nnls, which is better
                    disp('W found');
                    time_temp(alg_idx,iter_count)=toc(tusvd);
               else
                    
                    fun = funs{alg_idx};
                    alg_time= tic;
                    anchor_indices = fun(A', beta, k);
                    if anchor_indices==0
                        current_iter=iter_count;
                        err_flag=1;
                        toc
                        break;
                    end
                    W = A(anchor_indices,:);
                    M = nnlsHALSupdt(A',W'); % solving for 1 norm error and Rev(Inf,1) error are same. % solve lp or nnls, which is better
                    M = M';
                    time_temp(alg_idx,iter_count)=toc(alg_time);
                end
                
                residual=A-(M*W);
                res_norm_L1_A_temp(alg_idx,iter_count) = 1 - ( norm(residual,1) / norm(A,1) );   %To be updated
                res_norm_L2_A_temp(alg_idx,iter_count) = 1 - ( norm(residual,'fro') / norm(A,'fro') );
                res_t_norm_L1_A_temp(alg_idx,iter_count) = 1 - ( norm(residual',1) / norm(A',1) );   %To be updated
                
                residual = A_orig-(M*W);
                res_norm_L1_Aorig_temp(alg_idx,iter_count) = 1 - ( norm(residual,1) / norm(A_orig,1) );   %To be updated
                res_norm_L2_Aorig_temp(alg_idx,iter_count) = 1 - ( norm(residual,'fro') / norm(A_orig,'fro') );
                res_t_norm_L1_Aorig_temp(alg_idx,iter_count) = 1 - ( norm(residual',1) / norm(A_orig',1) );
                
                residual = M_orig-M;
                res_norm_L1_M_temp(alg_idx,iter_count) = 1 - ( norm(residual,1) / norm(M_orig,1) );   %To be updated
                res_norm_L2_M_temp(alg_idx,iter_count) = 1 - ( norm(residual,'fro') / norm(M_orig,'fro') );
                res_t_norm_L1_M_temp(alg_idx,iter_count) = 1 - ( norm(residual',1) / norm(M_orig',1) );
                
                residual = W_orig-W;
                res_norm_L1_W_temp(alg_idx,iter_count) = 1 - ( norm(residual,1) / norm(W_orig,1) );   %To be updated
                res_norm_L2_W_temp(alg_idx,iter_count) = 1 - ( norm(residual,'fro') / norm(W_orig,'fro') );
                res_t_norm_L1_W_temp(alg_idx,iter_count) = 1 - ( norm(residual',1) / norm(W_orig',1) );
                
                
                fprintf('%s : L2 residual norm : (%5.1f) with noise %f and iter %d\n', funs_str{alg_idx},res_norm_L2_A_temp(alg_idx,iter_count) ,beta,iter_count);
                %disp(sort(K_test));
            end
            
            if err_flag==1
                disp('error happened');
                break;
            end
            
            if iter_count==no_of_iter       % All iterations are successful
                current_iter=no_of_iter+1;
            end
            
        end
        loop_count=loop_count+1;
    end
    %     if loop_count>=20
    %         fprintf('Loop count exceeds 20 at noise level: %f',noise_arr(noise_idx));
    %         final_residual_norm(:,noise_idx) = NaN;
    %         final_residual_norm2(:,noise_idx) = NaN;
    %         time_final(:,noise_idx) = NaN;
    %         continue;
    %
    %     end
    
    res_norm_L1_A_final(:,beta_idx) = mean(res_norm_L1_A_temp,2);   %To be updated
    res_norm_L2_A_final(:,beta_idx) = mean(res_norm_L2_A_temp,2);
    res_t_norm_L1_A_final(:,beta_idx) = mean(res_t_norm_L1_A_temp,2);
    
    res_norm_L1_Aorig_final(:,beta_idx) = mean(res_norm_L1_Aorig_temp,2);
    res_norm_L2_Aorig_final(:,beta_idx) = mean(res_norm_L2_Aorig_temp,2);
    res_t_norm_L1_Aorig_final(:,beta_idx) = mean(res_t_norm_L1_Aorig_temp,2);
    
    res_norm_L1_M_final(:,beta_idx) = mean(res_norm_L1_M_temp,2);
    res_norm_L2_M_final(:,beta_idx) = mean(res_norm_L2_M_temp,2);
    res_t_norm_L1_M_final(:,beta_idx) = mean(res_t_norm_L1_M_temp,2);
    
    res_norm_L1_W_final(:,beta_idx) = mean(res_norm_L1_W_temp,2);
    res_norm_L2_W_final(:,beta_idx) = mean(res_norm_L2_W_temp,2);
    res_t_norm_L1_W_final(:,beta_idx) = mean(res_t_norm_L1_W_temp,2);
    
    time_final(:,beta_idx) = mean(time_temp,2);
    
    
    disp(res_norm_L2_A_final);
    outpath11 = sprintf('/home/jagdeep/Desktop/NMF/Experiments/tsvd-code-birch-v2/tsvd_cplex/semisynthetic_20k/residual_norm/noise%d',beta);
    mkdir(outpath11);
    filename = strcat(outpath11,'/res_norm_L1_A_final.mat');
    save(filename,'res_norm_L1_A_final');
    
    filename = strcat(outpath11,'/res_norm_L2_A_final.mat');
    save(filename,'res_norm_L2_A_final');
    
    filename = strcat(outpath11,'/res_t_norm_L1_A_final.mat');
    save(filename,'res_t_norm_L1_A_final');
    
    filename = strcat(outpath11,'/res_norm_L1_Aorig_final.mat');
    save(filename,'res_norm_L1_Aorig_final');
    
    filename = strcat(outpath11,'/res_norm_L2_Aorig_final.mat');
    save(filename,'res_norm_L2_Aorig_final');
    
    filename = strcat(outpath11,'/res_t_norm_L1_Aorig_final.mat');
    save(filename,'res_t_norm_L1_Aorig_final');
    
    filename = strcat(outpath11,'/res_norm_L1_M_final.mat');
    save(filename,'res_norm_L1_M_final');
    
    filename = strcat(outpath11,'/res_norm_L2_M_final.mat');
    save(filename,'res_norm_L2_M_final');
    
    filename = strcat(outpath11,'/res_t_norm_L1_M_final.mat');
    save(filename,'res_t_norm_L1_M_final');
    
    filename = strcat(outpath11,'/res_norm_L1_W_final.mat');
    save(filename,'res_norm_L1_W_final');
    
    filename = strcat(outpath11,'/res_norm_L2_W_final.mat');
    save(filename,'res_norm_L2_W_final');
    
    filename = strcat(outpath11,'/res_t_norm_L1_W_final.mat');
    save(filename,'res_t_norm_L1_W_final');
    
    
    
    filename = strcat(outpath11,'/time_final.mat');
    save(filename,'time_final');
end

disp('************************************************************************************');
disp(res_norm_L2_A_final);


%plot
x = beta_arr;
y = final_residual_norm';
plot(x,y(:,1),'g-+',x,y(:,2),'b-o',x,y(:,3),'c-s',x,y(:,4),'r-o',x,y(:,5),'g--',x,y(:,6),'k-+',x,y(:,7),'r-d',x,y(:,8),'m-<');
grid on;
hold on;
legend(funs_str{1},funs_str{2},funs_str{3},funs_str{4},funs_str{5},funs_str{6},funs_str{7},funs_str{8});
title('L2 Residual norm for $A$ (middlepoint,dense) ');
xlabel('Noise level (beta)');
ylabel('L1 Residual norm');

