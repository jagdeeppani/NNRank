funs_str_U = {'U-Heur-SPA','U-SPA','U-ER-SPA','UTSVD','UTSVD-AM','U-LR-SPA','U-LR-ER-SPA','LR-AM-UTSVD'};

funs_str_N = {'N-Heur-SPA','N-SPA','N-ER-SPA','NTSVD','NTSVD-AM','N-LR-SPA','N-LR-ER-SPA','LR-AM-NTSVD'};
k_arr=[20];
for k_count=1:length(k_arr)
    k=k_arr(k_count);
    fprintf('\n Dataset: Reuters, k = %d\n',k);
    fprintf('Algorithm\t Running time\t Accuracy\t NMI\n');
    
    for alg_idx=1:length(funs_str_N)
        fprintf('%12s\t %f\t %f\t %f\n',funs_str_U{alg_idx},alg_time_final_U(alg_idx,k_count),Accuracy_final_U(alg_idx,k_count),nmi_val_final_U(alg_idx,k_count));
    end
    fprintf('\n************************************\n')
end


for k_count=1:length(k_arr)
    k=k_arr(k_count);
    fprintf('\n Dataset: Reuters, k = %d\n',k);
    fprintf('Algorithm\t Running time\t Accuracy\t NMI\n');
    
    for alg_idx=1:length(funs_str_N)
        fprintf('%12s\t %f\t %f\t %f\n',funs_str_N{alg_idx},alg_time_final_N(alg_idx,k_count),Accuracy_final_N(alg_idx,k_count),nmi_val_final_N(alg_idx,k_count));
    end
    fprintf('\n************************************\n')
end