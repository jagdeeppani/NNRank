load Reuters7285_new_preprocessed.mat;
%load Twentyng_fea_preprocessed_new.mat;
%load yale32_A_4096.mat;
A=fea';
clear fea;
[d,n] = size(A);
fprintf('Sparsity(in %%): %d\n',(nnz(A)*100)/(d*n));

beta=0.1;
k_arr = [10,15,20,25,30,40,50,60,80,100,120,130,140,150,160];
res_norm_l2 = zeros(1,length(k_arr));
is_spa=0;
is_utsvd=1;
is_alternativeMin=0;
frobA = norm(A,'fro');

for k_count=1:length(k_arr)
    k=k_arr(k_count);
    if is_spa==1
        fprintf('Computing NMF by SPA\n');
        anchor_indices = spa(A', beta, k);
        if anchor_indices==0
            fprintf('No anchor indices\n');
        end
        C = A(anchor_indices,:);
        B = nnlsHALSupdt(A',C'); % solving for 1 norm error and Rev(Inf,1) error are same. % solve lp or nnls, which is better
        B = B';
    end
    if is_utsvd==1
        fprintf('Computing NMF by UTSVD\n');
        [B,cluster_id2] = TSVD_new_threshold(A,'/home/local/ANT/pjagdeep/Dropbox/largeScaleNMF/computeRank/tsvdOutput',k,0);
        C = nnlsHALSupdt(A,B); % solving for 1 norm error and Rev(Inf,1) error are same. % solve lp or nnls, which is better
    end
        
    if is_alternativeMin==1
        [B,C] = nnmf(A,k);
    end
    residual=A-(B*C);
    residual_err = norm(residual,'fro') / frobA;
    fprintf('Residual error at k=%d is : %f\n',k,residual_err);
    res_norm_l2(k_count) = residual_err;
end

csvwrite('Residuals_reuters_tsvdnmf.csv',[k_arr; res_norm_l2]);




