% A=M*W

%function run_gillislp_vs_ntsvd
clear all;
close all;


% The problem size of the synthetic data set
n = 100;   % n is the number of documents
d = 100;    % d is the vocab size
k = 10;      % number of topics

%noise_arr = linspace(0.01,1,40);
%noise_arr = noise_arr(5:40);   % form 35 to 40
noise_arr = [10,20,40,60,80,100];
no_of_iter = 10;

cw = 3;
etta = 0.1;   % Here we assume, sum of the topic word proportion of the catch words is 0.1
alpha = 0.05;  % Each column of W is sampled from dirichlet with parameter 0.01


normalize = 2;    % This says how the noise matrix will be bounded
addpath '/home/local/ANT/pjagdeep/Downloads/glpkmex-master'
%funs = {
    %@prec_spa,...
    %@post_prec_spa,...
    %@hottopixx, ...
    %@spa, ...
    %@fast_hull,...
    %@LPsepNMF_oone, ...
    %@true_k,...
    %@utsvd,...
    %};
funs = {@LPsepNMF_oone, @heur_spa,@spa,@fast_hull,@ER_spa_svds_v2,@ER_xray_svds_v2,@TSVD,@UTSVD};
funs_str = {'LP-rho1','Heur-SPA','SPA','XRAY','ER-SPA','ER-XRAY','TSVD','UTSVD'};


%funs = {@utsvd};
%funs_str = {'PrecSPA','PostPrecSPA','HottTopixx','SPA', 'XRAY','LP-rho1','true_K','UTSVD'};
%funs_str = {'UTSVD'};



num_algs = length(funs_str);    % UTSVD and True k are not counted in num_algs

res_norm_L1_A_final = zeros(num_algs,length(noise_arr));
res_norm_L2_A_final = zeros(num_algs,length(noise_arr));
res_t_norm_L1_A_final = zeros(num_algs,length(noise_arr));
res_norm_s_A_final = zeros(num_algs,length(noise_arr));
res_norm_spct_A_final = zeros(num_algs,length(noise_arr));

res_norm_L1_Aorig_final = zeros(num_algs,length(noise_arr));
res_norm_L2_Aorig_final = zeros(num_algs,length(noise_arr));
res_t_norm_L1_Aorig_final = zeros(num_algs,length(noise_arr));
res_norm_s_Aorig_final = zeros(num_algs,length(noise_arr));
res_norm_spct_Aorig_final = zeros(num_algs,length(noise_arr));

res_norm_L1_M_final = zeros(num_algs,length(noise_arr));
res_norm_L2_M_final = zeros(num_algs,length(noise_arr));
res_t_norm_L1_M_final = zeros(num_algs,length(noise_arr));
res_norm_s_M_final = zeros(num_algs,length(noise_arr));
res_norm_spct_M_final = zeros(num_algs,length(noise_arr));

res_norm_L1_W_final = zeros(num_algs,length(noise_arr));
res_norm_L2_W_final = zeros(num_algs,length(noise_arr));
res_t_norm_L1_W_final = zeros(num_algs,length(noise_arr));
res_norm_s_W_final = zeros(num_algs,length(noise_arr));
res_norm_spct_W_final = zeros(num_algs,length(noise_arr));

time_final = zeros(num_algs,length(noise_arr));
% negative_idx=zeros(1,length(noise_arr));

n_level= 2;
for noise_idx = 1:length(noise_arr)
    noise_level = noise_arr(noise_idx);
    
    type = 2;        % 1=middlepoint (requires n >= r+r(r-1)/2), 2=Dirichlet.
    scaling = 1.0;   % Different scaling of the columns of Mtilde.
    normalize = 1.0; % Different normalization of the Noise.
    density = 1;   % Proportion of nonzero entries in the noise matrix (but at least one nonzero per column).
    
    current_iter=1;
    loop_count=1;
    outpath_residual_norm= sprintf('/home/local/ANT/pjagdeep/NMF_Experiments/synth_dominant_multin_small/residual_norm/noise%d',noise_arr);
    mkdir(outpath_residual_norm);
    
    outpath_res_norm_db= sprintf('/home/local/ANT/pjagdeep/Dropbox/NMF/synth_dominant_multin_small/residual_norm_m10/noise%d',noise_arr);
    mkdir(outpath_res_norm_db);
    
    res_norm_L1_A_temp = zeros(num_algs,no_of_iter);   %To be updated
    res_norm_L2_A_temp = zeros(num_algs,no_of_iter);
    res_t_norm_L1_A_temp = zeros(num_algs,no_of_iter);
    res_norm_s_A_temp = zeros(num_algs,no_of_iter);
    
    res_norm_L1_Aorig_temp = zeros(num_algs,no_of_iter);
    res_norm_L2_Aorig_temp = zeros(num_algs,no_of_iter);
    res_t_norm_L1_Aorig_temp = zeros(num_algs,no_of_iter);
    res_norm_s_Aorig_temp = zeros(num_algs,no_of_iter);
    
    res_norm_L1_M_temp = zeros(num_algs,no_of_iter);
    res_norm_L2_M_temp = zeros(num_algs,no_of_iter);
    res_t_norm_L1_M_temp = zeros(num_algs,no_of_iter);
    res_norm_s_M_temp = zeros(num_algs,no_of_iter);
    
    res_norm_L1_W_temp = zeros(num_algs,no_of_iter);
    res_norm_L2_W_temp = zeros(num_algs,no_of_iter);
    res_t_norm_L1_W_temp = zeros(num_algs,no_of_iter);
    res_norm_s_W_temp = zeros(num_algs,no_of_iter);
    
    time_temp = zeros(num_algs,no_of_iter);
    
    while current_iter<=no_of_iter && loop_count<20
        err_flag=0;
        for iter_count = current_iter:no_of_iter
            
            %[A_orig, A, M_orig, W_orig, ~, anchor_indices_true] = synthetic_data_gillis_lp(m, n, k,noise_level, type, scaling, density, normalize);
            [A,A_orig,B_orig,C_orig,permute_vect] = generate_dominant_multinomial(d,n,k,cw,etta,alpha,noise_level,normalize); 
            
             %A=A'; A_orig=A_orig'; W_orig=W_orig'; M_orig=M_orig';         % A is a words x docs matrix
%             if min(min(A))<0
%                 A(A<0)=0;
%                 negative_idx(noise_idx)=1;
%                 fprintf('-ve elements found in A with setup, type: %d, density: %f, noise: %f, iter: %d \n',type,density,noise_level,iter_count);
%             end
            
          
            outpath_data = sprintf('/home/local/ANT/pjagdeep/NMF_Experiments/synth_dominant_multin_small/data/noise_iter%d_%d',noise_arr,iter_count);
            mkdir(outpath_data);
            
            filename = strcat(outpath_data,'/A_orig.mat');
            save(filename,'A_orig');
            filename = strcat(outpath_data,'/B_orig.mat');
            save(filename,'B_orig');
            filename = strcat(outpath_data,'/C_orig.mat');
            save(filename,'C_orig');
            filename = strcat(outpath_data,'/A.mat');
            save(filename,'A');
            %filename = strcat(outpath_data,'/anchor_indices_true.mat');
            %save(filename,'anchor_indices_true');
            
            
            fprintf('   - Noise m        : %d\n',noise_level);
            disp('************************************************************************************');
            A(A<0) = 0;
            for alg_idx = 1:num_algs
                if  strcmp(funs_str{alg_idx},'UTSVD')
                    outpath_utsvd = sprintf('/home/local/ANT/pjagdeep/NMF_Experiments/synth_dominant_multin_small/utsvd_out/noise_iter%d_%d',noise_arr,iter_count);
                    mkdir(outpath_utsvd);
                    tusvd=tic;
                    [B,~] = TSVD_new_threshold(A,outpath_utsvd,k,0);  % Birch is set inside TSVD
                 
                    disp('B found');
                    C = nnlsHALSupdt(A,B); % solve lp or nnls, which is better
                    
                    disp('C found');
                    time_temp(alg_idx,iter_count)=toc(tusvd);
                    
                    filename = strcat(outpath_data,'/B_UTSVD.mat');
                    save(filename,'B');
                    filename = strcat(outpath_data,'/C_UTSVD.mat');
                    save(filename,'C');
                
                    
                    
                elseif strcmp(funs_str{alg_idx},'TSVD')
                            outpath_tsvd = sprintf('/home/local/ANT/pjagdeep/NMF_Experiments/synth_dominant_multin_small/tsvd_out/noise_iter%d_%d',noise_arr,iter_count);
                            mkdir(outpath_tsvd);
                            tusvd=tic;
                            [B,~] = TSVD(A,outpath_tsvd,k);
                            disp('M found');
                            C = nnlsHALSupdt(A,B); % solve lp or nnls, which is better
                            disp('W found');
                            time_temp(alg_idx,iter_count)=toc(tusvd);
                            
                            filename = strcat(outpath_data,'/B_TSVD.mat');
                            save(filename,'B');
                            filename = strcat(outpath_data,'/C_TSVD.mat');
                            save(filename,'C');
                    
                elseif strcmp(funs_str{alg_idx},'true_K')
                    t_truek=tic;
                    B = A(:,anchor_indices_true);
                    C = nnlsHALSupdt(A,B); % solving for 1 norm error and Rev(Inf,1) error are same. % solve lp or nnls, which is better
               
                    time_temp(alg_idx,iter_count)=toc(t_truek);
                    filename = strcat(outpath_data,'/M_true_K.mat');
                    save(filename,'M');
                    filename = strcat(outpath_data,'/W_true_K.mat');
                    save(filename,'W');
                    
                                        
                else
                    
                    fun = funs{alg_idx};
                    alg_time= tic;
                    anchor_indices = fun(A', 2, k);
                    if anchor_indices==0
                        current_iter=iter_count;
                        err_flag=1;
                        toc
                        break;
                    end
                    C = A(anchor_indices,:);
                    B = nnlsHALSupdt(A',C'); % solving for 1 norm error and Rev(Inf,1) error are same. % solve lp or nnls, which is better
                    B = B';
                    
                    time_temp(alg_idx,iter_count)=toc(alg_time);
                    
                    strB = strcat('/B_',funs_str{alg_idx},'.mat');
                    strC = strcat('/C_',funs_str{alg_idx},'.mat');
                    
                    filename = strcat(outpath_data,strB);
                    save(filename,'B');
                    filename = strcat(outpath_data,strC);
                    save(filename,'C');
                end
                size(B)
                size(C)
                
                prediction = B*C;
                residual=A-prediction;
                res_norm_L1_A_temp(alg_idx,iter_count) = 1 - ( norm(residual,1) / norm(A,1) );   %To be updated
                res_norm_L2_A_temp(alg_idx,iter_count) = 1 - ( norm(residual,'fro') / norm(A,'fro') );
                res_t_norm_L1_A_temp(alg_idx,iter_count) = 1 - ( norm(residual',1) / norm(A',1) );   %To be updated
                res_norm_s_A_temp(alg_idx,iter_count) = 1- (sumabs(residual)/sumabs(A));
                res_norm_spct_A_temp(alg_idx,iter_count) = 1- (norm(residual)/norm(A));
                
                residual = A_orig-prediction;
                res_norm_L1_Aorig_temp(alg_idx,iter_count) = 1 - ( norm(residual,1) / norm(A_orig,1) );   %To be updated
                res_norm_L2_Aorig_temp(alg_idx,iter_count) = 1 - ( norm(residual,'fro') / norm(A_orig,'fro') );
                res_t_norm_L1_Aorig_temp(alg_idx,iter_count) = 1 - ( norm(residual',1) / norm(A_orig',1) );
                res_norm_s_Aorig_temp(alg_idx,iter_count) = 1- (sumabs(residual)/sumabs(A_orig));
                res_norm_spct_Aorig_temp(alg_idx,iter_count) = 1- (norm(residual)/norm(A_orig));
                
size(B)
size(B_orig)
                residual = C_orig-C;
                res_norm_L1_M_temp(alg_idx,iter_count) = 1 - ( norm(residual,1) / norm(B_orig,1) );   %To be updated
                res_norm_L2_M_temp(alg_idx,iter_count) = 1 - ( norm(residual,'fro') / norm(B_orig,'fro') );
                res_t_norm_L1_M_temp(alg_idx,iter_count) = 1 - ( norm(residual',1) / norm(B_orig',1) );
                res_norm_s_M_temp(alg_idx,iter_count) = 1- (sumabs(residual)/sumabs(B_orig));              
                res_norm_spct_M_temp(alg_idx,iter_count) = 1- (norm(residual)/norm(B_orig));              
                
                
                cost_mat=zeros(k,k);
                for c_idx1=1:k
                    for c_idx2=1:k
                        cost_mat(c_idx1,c_idx2)=norm(B_orig(:,c_idx1)- B(:,c_idx2),1);
                    end
                end
                [Assign,cost] = munkres(cost_mat);
                B_new = B(:,Assign);
                
                residual = B_orig-B;
                res_norm_L1_W_temp(alg_idx,iter_count) = 1 - ( norm(residual,1) / norm(C_orig,1) );   %To be updated
                res_norm_L2_W_temp(alg_idx,iter_count) = 1 - ( norm(residual,'fro') / norm(C_orig,'fro') );
                res_t_norm_L1_W_temp(alg_idx,iter_count) = 1 - ( norm(residual',1) / norm(C_orig',1) );
                res_norm_s_W_temp(alg_idx,iter_count) = 1- (sumabs(residual)/sumabs(C_orig));
                res_norm_spct_W_temp(alg_idx,iter_count) = 1- (norm(residual)/norm(C_orig));
                
                fprintf('%s : L2 residual norm : (%5.3f) with noise %f\n and iter %d\n', funs_str{alg_idx},res_norm_s_A_temp(alg_idx,iter_count) ,noise_arr,iter_count);
                fprintf('%s : L2 residual norm : (%5.3f) with noise %f\n and iter %d\n', funs_str{alg_idx},res_norm_s_Aorig_temp(alg_idx,iter_count) ,noise_arr,iter_count);
                fprintf('%s : L2 residual norm : (%5.3f) with noise %f\n and iter %d\n', funs_str{alg_idx},res_norm_s_W_temp(alg_idx,iter_count) ,noise_arr,iter_count);
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
    
    res_norm_L1_A_final(:,noise_idx) = mean(res_norm_L1_A_temp,2);   %To be updated
    res_norm_L2_A_final(:,noise_idx) = mean(res_norm_L2_A_temp,2);
    res_t_norm_L1_A_final(:,noise_idx) = mean(res_t_norm_L1_A_temp,2);
    res_norm_s_A_final(:,noise_idx) =mean(res_norm_s_A_temp,2);
    res_norm_spct_A_final(:,noise_idx) =mean(res_norm_spct_A_temp,2);
    
    res_norm_L1_Aorig_final(:,noise_idx) = mean(res_norm_L1_Aorig_temp,2);
    res_norm_L2_Aorig_final(:,noise_idx) = mean(res_norm_L2_Aorig_temp,2);
    res_t_norm_L1_Aorig_final(:,noise_idx) = mean(res_t_norm_L1_Aorig_temp,2);
    res_norm_s_Aorig_final(:,noise_idx) =mean(res_norm_s_Aorig_temp,2);
    res_norm_spct_Aorig_final(:,noise_idx) =mean(res_norm_spct_Aorig_temp,2);
    
    res_norm_L1_M_final(:,noise_idx) = mean(res_norm_L1_M_temp,2);
    res_norm_L2_M_final(:,noise_idx) = mean(res_norm_L2_M_temp,2);
    res_t_norm_L1_M_final(:,noise_idx) = mean(res_t_norm_L1_M_temp,2);
    res_norm_s_M_final(:,noise_idx) =mean(res_norm_s_M_temp,2);
    res_norm_spct_M_final(:,noise_idx) =mean(res_norm_spct_M_temp,2);
    
    res_norm_L1_W_final(:,noise_idx) = mean(res_norm_L1_W_temp,2);
    res_norm_L2_W_final(:,noise_idx) = mean(res_norm_L2_W_temp,2);
    res_t_norm_L1_W_final(:,noise_idx) = mean(res_t_norm_L1_W_temp,2);
    res_norm_s_W_final(:,noise_idx) =mean(res_norm_s_W_temp,2);
    res_norm_spct_W_final(:,noise_idx) =mean(res_norm_spct_W_temp,2);
    
    
    time_final(:,noise_idx) = mean(time_temp,2);
    
    
    disp(res_norm_s_A_final);

    filename = strcat(outpath_residual_norm,'/res_norm_L1_A_final.mat');
    save(filename,'res_norm_L1_A_final');
    
    filename = strcat(outpath_residual_norm,'/res_norm_L2_A_final.mat');
    save(filename,'res_norm_L2_A_final');
    
    filename = strcat(outpath_residual_norm,'/res_t_norm_L1_A_final.mat');
    save(filename,'res_t_norm_L1_A_final');
    
    filename = strcat(outpath_residual_norm,'/res_norm_L1_Aorig_final.mat');
    save(filename,'res_norm_L1_Aorig_final');
    
    filename = strcat(outpath_residual_norm,'/res_norm_L2_Aorig_final.mat');
    save(filename,'res_norm_L2_Aorig_final');
    
    filename = strcat(outpath_residual_norm,'/res_t_norm_L1_Aorig_final.mat');
    save(filename,'res_t_norm_L1_Aorig_final');
    
    filename = strcat(outpath_residual_norm,'/res_norm_L1_M_final.mat');
    save(filename,'res_norm_L1_M_final');
    
    filename = strcat(outpath_residual_norm,'/res_norm_L2_M_final.mat');
    save(filename,'res_norm_L2_M_final');
    
    filename = strcat(outpath_residual_norm,'/res_t_norm_L1_M_final.mat');
    save(filename,'res_t_norm_L1_M_final');
    
    filename = strcat(outpath_residual_norm,'/res_norm_L1_W_final.mat');
    save(filename,'res_norm_L1_W_final');
    
    filename = strcat(outpath_residual_norm,'/res_norm_L2_W_final.mat');
    save(filename,'res_norm_L2_W_final');
    
    filename = strcat(outpath_residual_norm,'/res_t_norm_L1_W_final.mat');
    save(filename,'res_t_norm_L1_W_final');
    
    filename = strcat(outpath_residual_norm,'/res_norm_s_A_final.mat');
    save(filename,'res_norm_s_A_final');
    
    filename = strcat(outpath_residual_norm,'/res_norm_s_Aorig_final.mat');
    save(filename,'res_norm_s_Aorig_final');
    
    filename = strcat(outpath_residual_norm,'/res_norm_s_M_final.mat');
    save(filename,'res_norm_s_M_final');
    
    filename = strcat(outpath_residual_norm,'/res_norm_s_W_final.mat');
    save(filename,'res_norm_s_W_final');
    
    filename = strcat(outpath_residual_norm,'/res_norm_spct_A_final.mat');
    save(filename,'res_norm_spct_A_final');
    
    filename = strcat(outpath_residual_norm,'/res_norm_spct_Aorig_final.mat');
    save(filename,'res_norm_spct_Aorig_final');
    
    filename = strcat(outpath_residual_norm,'/res_norm_spct_M_final.mat');
    save(filename,'res_norm_spct_M_final');
    
    filename = strcat(outpath_residual_norm,'/res_norm_spct_W_final.mat');
    save(filename,'res_norm_spct_W_final');
    
    
    filename = strcat(outpath_res_norm_db,'/res_norm_s_Aorig_final.mat');
    save(filename,'res_norm_s_Aorig_final');
            
    filename = strcat(outpath_res_norm_db,'/res_norm_s_W_final.mat');
    save(filename,'res_norm_s_W_final');

    filename = strcat(outpath_res_norm_db,'/res_norm_s_A_final.mat');
    save(filename,'res_norm_s_A_final');
    
    
    
    filename = strcat(outpath_residual_norm,'/time_final.mat');
    save(filename,'time_final');
end

disp('************************************************************************************');
disp(res_norm_s_A_final);
disp('************************************************************************************');
disp(res_norm_s_Aorig_final);



%plot
%x = noise_arr;
%y = res_norm_s_A_final;
%plot(x,y(:,1),'g-+',x,y(:,2),'b-o',x,y(:,3),'c-s',x,y(:,4),'r-o',x,y(:,5),'g--',x,y(:,6),'k-+',x,y(:,7),'r-d',x,y(:,8),'m-<');
%grid on;
% hold on;
% legend(funs_str{1},funs_str{2},funs_str{3},funs_str{4},funs_str{5},funs_str{6},funs_str{7},funs_str{8});
% title('L2 Residual norm for $A$ (middlepoint,dense) ');
% xlabel('Noise level (beta)');
% ylabel('L1 Residual norm');

