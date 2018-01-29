% A=M*W

%function run_gillislp_vs_ntsvd
clear all;
close all;


% The problem size of the synthetic data set
m = 100;   % m is the vocab size
n = 50;    % n is the number of documents
k = 10;      % number of topics

noise_arr = linspace(0.01,1,10);
% noise_arr = noise_arr(35:40);   % form 35 to 40
%noise_arr = [0.3];
no_of_iter = 10;

funs = {
    @prec_spa,...
    @post_prec_spa,...
    @hottopixx, ...
    @spa, ...
    @fast_hull,...
    @LPsepNMF_oone, ...
    };
% funs = {@ER_spa_svds};
funs_str = {'PrecSPA','PostPrecSPA','HottTopixx','SPA', 'XRAY','LP-rho1','true_K','UTSVD'};
% funs_str = {'SPA','true_K'};

num_algs = length(funs_str);    % UTSVD and True k are not counted in num_algs
res_norm_L1_temp=zeros(num_algs,no_of_iter);
time_temp=zeros(length(funs)+2,no_of_iter);
%final_residual_norm = zeros(num_algs+2,length(noise_arr));
final_time = zeros(num_algs+2,length(noise_arr));

for noise_idx = 1:length(noise_arr)
    noise_level = noise_arr(noise_idx);
    type = 2;        % 1=middlepoint (requires n >= r+r(r-1)/2), 2=Dirichlet.
    scaling = 1.0;   % Different scaling of the columns of Mtilde.
    normalize = 1.0; % Different normalization of the Noise.
    density = 0.25;   % Proportion of nonzero entries in the noise matrix (but at least one nonzero per column).
    
    current_iter=1;
    loop_count=1;
    while current_iter<=no_of_iter && loop_count<20
        err_flag=0;
        for iter_count = current_iter:no_of_iter
            
            [A_orig, A, W_orig, M_orig, ~, anchor_indices_true] = synthetic_data_gillis_lp(n, m, k,noise_level, type, scaling, density, normalize);    A=A'; A_orig=A_orig'; W_orig=W_orig'; M_orig=M_orig';         % A is a words x docs matrix
            
            disp(' Properties of the generated noisy separable matrix: ');
            if type == 1
                disp('   - Type                : Middle Points');
            else
                disp('   - Type                : Dirichlet');
            end
            fprintf('   - Noise level         : %d%%\n',100*noise_level);
            disp('************************************************************************************');
            
            for alg_idx = 1:num_algs
                if  strcmp(funs_str{alg_idx},'UTSVD')
                    outpath = sprintf('/home/jagdeep/Desktop/NMF/Experiments/tsvd-code-birch-v2/tsvd_cplex/output_final_gillislp_temp/noise_iter%d_%d',noise_level,iter_count);
                    mkdir(outpath);
                    tusvd=tic;
                    [M,~] = TSVD(A,outpath,k);
                    disp('M found');
                    W = solvelp(A,M); % solve lp or nnls, which is better
                    disp('W found');
                    time_temp(alg_idx,iter_count)=toc(tusvd);
                elseif strcmp(funs_str{alg_idx},'true_K')
                    t_truek=tic;
                    W = A(anchor_indices_true,:);
                    M = solvelp(A',W'); % solving for 1 norm error and Rev(Inf,1) error are same. % solve lp or nnls, which is better
                    M = M';
                    time_temp(num_algs+1,iter_count)=toc(t_truek);
                    
                else
                    
                    fun = funs{alg_idx};
                    alg_time= tic;
                    anchor_indices = fun(A', noise_level, k);
                    if anchor_indices==0
                        current_iter=iter_count;
                        err_flag=1;
                        toc
                        break;
                    end
                    W = A(anchor_indices,:);
                    M = solvelp(A',W'); % solving for 1 norm error and Rev(Inf,1) error are same. % solve lp or nnls, which is better
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
                
                
                fprintf('%s : L1 residual norm : (%5.1f) with noise %d\n', funs_str{alg_idx}, temp,noise_level);
                %disp(sort(K_test));
            end
            
            if err_flag==1
                break;
                disp('error happened');
                aa=123;
                test3=9;
            end
            
            if iter_count==no_of_iter       % All iterations are successful
                current_iter=no_of_iter+1;
            end
            
        end
        loop_count=loop_count+1;
    end
    if loop_count>=20
        fprintf('Loop count exceeds 20 at noise level: %f',noise_arr(noise_idx));
        final_residual_norm(:,noise_idx) = NaN;
        final_residual_norm2(:,noise_idx) = NaN;
        final_time(:,noise_idx) = NaN;
        continue;
        
    end
    
    
    res_norm_L1_temp(find(res_norm_L1_temp==0))=NaN;
    res_norm_temp2(find(res_norm_temp2==0))=NaN;
    
    
    final_residual_norm(:,noise_idx) = nanmean(res_norm_L1_temp,2);
    final_residual_norm2(:,noise_idx) = nanmean(res_norm_temp2,2);
    final_time(:,noise_idx) = mean(time_temp,2);
    
    index=[3,4,5,1,2,6,7,8];
    final_residual_norm=final_residual_norm(index,:);
    final_residual_norm2=final_residual_norm2(index,:);
    final_time=final_time(index,:);
    
    disp(final_residual_norm);
    outpath11 = sprintf('/home/jagdeep/Desktop/NMF/Experiments/tsvd-code-birch-v2/tsvd_cplex/residual_norm_final_gillislp_temp/noise%d',noise_level);
    mkdir(outpath11);
    filename = strcat(outpath11,'/residualnorm1.mat');
    save(filename,'final_residual_norm');
    filename2 = strcat(outpath11,'/residualnorm2.mat');
    save(filename2,'final_residual_norm2');
    filename1 = strcat(outpath11,'/time.mat');
    save(filename1,'final_time');
end

disp('************************************************************************************');
disp(final_residual_norm);


%plot
x = noise_arr;
%plot(x,l1_residual_norm(:,1),'g','LineWidth',0.8,x,l1_residual_norm(:,2),'b--o','LineWidth',0.8,x,l1_residual_norm(:,3),'c-*','LineWidth',0.8,x,l1_residual_norm(:,4),'r:d','LineWidth',0.8,x,l1_residual_norm(:,5),'m-.x',
% 'LineWidth',0.8,x,l1_residual_norm(:,6),'y-s','LineWidth',0.8,x,l1_residual_norm(:,7),'k-+','LineWidth',0.8);

%     plot(x,l1_residual_norm(:,1),'g',x,l1_residual_norm(:,2),'b--o',x,l1_residual_norm(:,3),'c-*',x,l1_residual_norm(:,4),'r:d',x,l1_residual_norm(:,5),'m-.x',x,l1_residual_norm(:,6),'y-s',x,l1_residual_norm(:,7),'k-+');
plot(x,final_residual_norm(:,1),'g',x,final_residual_norm(:,2),'b--o',x,final_residual_norm(:,3),'c-*',x,final_residual_norm(:,4),'r:d',x,final_residual_norm(:,5), ...
    'm-.x',x,final_residual_norm(:,6),'y-.>',x,final_residual_norm(:,7),'b-.<',x,final_residual_norm(:,8),'m-.+');
plot(x,final_residual_norm)
grid on;
hold on;
legend(funs_str{1},funs_str{2},funs_str{3},funs_str{4},funs_str{5},funs_str{6},funs_str{7},'un-normalized tsvd');
title('L1 Residual on gillislp synthetic data(middle point / dense noise)');
xlabel('Noise level');
ylabel('L1 Residual norm (Gillis etal.)');

%end % of minibench
