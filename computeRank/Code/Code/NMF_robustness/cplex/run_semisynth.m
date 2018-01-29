% Run on semi-synthetic data
%function run_semisynth
clear all;
close all;

M_orig= dlmread('M');
M_orig=M_orig';
W_orig= dlmread('nyt5kr-50topics-theta-0.01-20000_300');
W_orig=W_orig';
%W_orig=W_orig(:,1:6000);
[m,k]=size(M_orig);
n=size(W_orig,2);



% Generating Noise matrix
% A=M*W +N


sigma_arr=[0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0];
%sigma_arr=[0 0.01];
normalize=2; % Normalize is column wise l1 or l2. if l1, ||N||_1<=eps1
residual_norm_param=1;

no_of_iter = 5;
density = 0.9;  % Proportion of nonzero entries in the noise matrix (but at least one nonzero per column). 
funs = {
        %@hottopixx, ...
        @spa, ...
        @fast_hull,...
        %@ER_nmf
        %@LPsepNMF_oone, ...
        %@LPsepNMF_otwo, ...
        };

    % Descriptive names for the algorithms
    funs_str = {'SPA', 'XRAY'};
     %disp('************************************************************************************'); 
    %disp('          Comparing near-separable NMF algorithms on a synthetic data set'); 
    %disp('************************************************************************************'); 
    
num_algs = length(funs);
A_orig=M_orig*W_orig;
%noise_arr = [0.2];
for sigma_idx = 1:length(sigma_arr)
    % Parameters for generating the noisy separable matrix Mtilde
    % ---> see synthetic_data.m for more details 
    
    
    sigma = sigma_arr(sigma_idx);
    eps1=sigma*sqrt(n/m);
    if (sigma>0)
        N=normrnd(0,sigma,m,n);
        

        % Normalizing the Noise matrix

        if normalize == 1
            cn_a = sum(abs(A_orig));
            cn_n = sum(abs(N));
            N = N * eps1 * diag(cn_a./cn_n);

        elseif normalize == 2
            cn_a = sqrt(sum(A_orig.^2));
            cn_n = sqrt(sum(N.^2));
            N = N * eps1 * diag(cn_a./cn_n);
        else 
            error('Invalid normalization type');
        end
    else
        N=zeros(m,n);
    end
    
    
    A=A_orig+N;    
    
    fprintf('   - Sigma         : %d\n',sigma); 
    %fprintf('   - Density of the noise: %d%%\n',100*density); 
    disp('************************************************************************************'); 
    iter_algo=zeros(num_algs+1,no_of_iter);

    for iter_count=1:no_of_iter
        
    
        for alg_idx = 1:num_algs
            fun = funs{alg_idx};
            tic
            K_test = fun(A',eps1, k); 
            toc
            
            W = nnlsHALSupdt(A',(A(K_test,:))'); % A'(:,K_test) = (A(K_test,:))'    solving for 1 norm error and Rev(Inf,1) error are same. % solve lp or nnls, which is better

            if residual_norm_param==1;
                residual=(A'-(A(K_test,:))'*W)';
                residual_norm_orig_temp(alg_idx,iter_count) = norm(residual,1);
                %temp = 1- ( norm(A'-(A(K_test,:))'*W,1) / norm(A,1) );
                temp = 1- ( norm(residual,1) / norm(A,1) );
                iter_algo(alg_idx,iter_count)=temp;
                
                % fractile calculation of residual 
                rsum= sum(abs(residual));
                rsum=sort(rsum);
                fractile_param=[0.7 0.9];
                fractile_values=zeros(1,length(fractile_param));
                for i=1:length(fractile_param)
                    f_index=ceil(n*fractile_param(i));
                    fractile_values(i)= rsum(f_index+1);
                end
                fractiles1(alg_idx,iter_count)=fractile_values(1);
                fractiles2(alg_idx,iter_count)=fractile_values(2);
                
                
                
            elseif residual_norm_param==2
                temp = 1- ( norm(A'-(A(K_test,:))'*W,'fro') / norm(A,'fro') );
                iter_algo(alg_idx,iter_count)=temp;
            else
                error('invalid residual norm');
            end
            
            fprintf('%10s : L1 residual norm : (%5.1f): \n', funs_str{alg_idx}, temp);
            %disp(sort(K_test));
        end
    
      
    outpath = sprintf('/home/jagdeep/Desktop/NMF/Experiments/tsvd-code-birch-v2/tsvd_cplex/output4/%d_%d',sigma,iter_count);
    mkdir(outpath);
    
    %A_positive(A'<0)=0;
    [M,~] = TSVD(A,outpath,k);
    disp('M found');
    W = nnlsHALSupdt(A,M); % solve lp or nnls, which is better
    disp('W found');
    if residual_norm_param==1;
                residual=A-M*W;
                residual_norm_orig_temp(num_algs+1,iter_count) = norm(residual,1);
                temp = 1- ( norm(residual,1) / norm(A,1) );
                iter_algo(num_algs+1,iter_count)=temp;
                
                % Fractile calculation started
                rsum= sum(abs(residual));
                rsum=sort(rsum);
                fractile_param=[0.7 0.9];
                fractile_values=zeros(1,length(fractile_param));
                for i=1:length(fractile_param)
                    f_index=ceil(n*fractile_param(i));
                    fractile_values(i)= rsum(f_index+1);
                end
                fractiles1(num_algs+1,iter_count)=fractile_values(1);
                fractiles2(num_algs+1,iter_count)=fractile_values(2);
                
                
            elseif residual_norm_param==2
                temp = 1- ( norm(A-M*W,'fro') / norm(A,'fro') );
                iter_algo(num_algs+1,iter_count)=temp;
            else
                error('invalid residual norm');
    end
    
    end
    
    residual_norm_orig(:,sigma_idx) = mean(residual_norm_orig_temp,2);
    residual_norm(:,sigma_idx) = mean(iter_algo,2);
    final_fractiles1(:,sigma_idx)=mean(fractiles1,2);
    final_fractiles2(:,sigma_idx)=mean(fractiles2,2);
    clear fractiles1 fractiles2 iter_algo residual_norm_orig_temp;
    %l1_residual_norm(noise_idx,alg_idx) = temp;
    %l1_residual_norm(noise_idx,num_algs+1) = temp;
        
   disp(residual_norm);
   outpath11 = sprintf('/home/jagdeep/Desktop/NMF/Experiments/tsvd-code-birch-v2/tsvd_cplex/residualnorm_semisynth1/%d',sigma);
   mkdir(outpath11);
   filename = strcat(outpath11,'/residualnorm.mat');
   save(filename,'residual_norm');
   filename2 = strcat(outpath11,'/f_fractiles1.mat');
   save(filename2,'final_fractiles1');
   filename3 = strcat(outpath11,'/f_fractiles2.mat');
   save(filename3,'final_fractiles2');
   filename4 = strcat(outpath11,'/residual_norm_orig.mat');
   save(filename4,'residual_norm_orig');
   
end

disp('************************************************************************************'); 
disp(residual_norm);


%plot
x =sigma_arr;
%plot(x,l1_residual_norm(:,1),'g','LineWidth',0.8,x,l1_residual_norm(:,2),'b--o','LineWidth',0.8,x,l1_residual_norm(:,3),'c-*','LineWidth',0.8,x,l1_residual_norm(:,4),'r:d','LineWidth',0.8,x,l1_residual_norm(:,5),'m-.x',
% 'LineWidth',0.8,x,l1_residual_norm(:,6),'y-s','LineWidth',0.8,x,l1_residual_norm(:,7),'k-+','LineWidth',0.8);

    %plot(x,l1_residual_norm(:,1),'g',x,l1_residual_norm(:,2),'b--o',x,l1_residual_norm(:,3),'c-*',x,l1_residual_norm(:,4),'r:d',x,l1_residual_norm(:,5),'m-.x',x,l1_residual_norm(:,6),'y-s',x,l1_residual_norm(:,7),'k-+');
    plot(x,residual_norm(1,:),'g-*',x,residual_norm(2,:),'b--o',x,residual_norm(3,:),'k-+');
    grid on;
    hold on;
    legend(funs_str{1},funs_str{2},'un-normalized tsvd');
    title('Experiment on semisynthetic data with dense noise');
    xlabel('Sigma');
    ylabel('Residual norm');

%end % of minibench
