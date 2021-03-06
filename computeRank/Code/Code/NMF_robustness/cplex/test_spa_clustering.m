load Reuters21578.mat;
clear trainIdx testIdx;
% Reducing the size of data similar to Mizutani etal.
fea = fea(1:8258,:); % 7285 x 18933
gnd = gnd(1:8258);      % 7285 x 1

fea(:,sum(fea,1)<1)=[]; % Removing orphan words (not present in any document)
fea_orig = fea;
gnd_orig = gnd;

orig_k=48;
k_arr=[6 8 10 12];
no_of_repetitions=5;
noise_level=0.3;

% funs = {@heur_spa,@spa,@fast_hull,@ER_spa_svds, @ER_xray_svds,@utsvd};
% funs_str = {'Heur-SPA','SPA','XRAY','ER-SPA','ER-XRAY','UTSVD'};
funs = {@fast_hull,@spa_doc_norm,@utsvd,@ntsvd};
funs_str = {'XRAY','SPA-doc-norm','UTSVD','NTSVD','UTSVD_W','NTSVD_W'};

num_algs = length(funs_str);

alg_time_final = zeros(num_algs+2,length(k_arr));
nmi_val_final = zeros(num_algs+2,length(k_arr));
nmi_val_final2 = zeros(num_algs+2,length(k_arr));
Accuracy_final = zeros(num_algs+2,length(k_arr));


for k_count=1:length(k_arr)
    k = k_arr(k_count);
    
    alg_time=zeros(num_algs+2,no_of_repetitions);
    nmi_val=zeros(num_algs+2,no_of_repetitions);
    nmi_val2=zeros(num_algs+2,no_of_repetitions);
    Accuracy = zeros(num_algs+2,no_of_repetitions);
    
    
    for repeat_count=1:no_of_repetitions
        fprintf('number of classes = %d, repetitions = %d\n',k,repeat_count);
        
        class_indices = randperm(orig_k);
        class_indices = class_indices(1:k);
        
        indices=[];
        for ci_count=1:k
            indices = union(indices,find(gnd_orig==class_indices(ci_count)) );
        end
        fea = fea_orig(indices,:);
        gnd = gnd_orig(indices);
        
        A=fea';
        clear fea;
        [m,n] = size(A);
        
        label=gnd;
        mapped_label = label;
        unique_label = unique(label); % returns sorted unique gnd
        
        for ug_count=1:k
            mapped_label(label == (unique_label(ug_count)) ) = ug_count;
        end
        gnd=mapped_label;
        
        
        
        topic=zeros(num_algs+2,n);
        outpath_data = sprintf('/home/jagdeep/Desktop/NMF/Experiments/tsvd-code-birch-v3/tsvd_cplex/output_clustering_temp112/rtr_v3/k_%d/rpt_%d/data',k,repeat_count);
        mkdir(outpath_data);
        
        filename1 = strcat(outpath_data,'/features_A.mat');
        save(filename1,'A');
        filename1 = strcat(outpath_data,'/ground_truth.mat');
        save(filename1,'label');
        
        
        A_un=A;
        A = A*spdiags(1./sum(A,1)',0,n,n);
        anchor_indices_all=zeros(num_algs-1,k);
        err_flag=0;
        
        
        outpath_result = sprintf('/home/jagdeep/Desktop/NMF/Experiments/tsvd-code-birch-v3/tsvd_cplex/output_clustering_temp112/rtr_v3/k_%d/rpt_%d/result',k,repeat_count);
        mkdir(outpath_result);
        
        for alg_idx = 1:num_algs
            if  strcmp(funs_str{alg_idx},'UTSVD')
                disp('UTSVD started\n');
                outpath_tsvd = sprintf('/home/jagdeep/Desktop/NMF/Experiments/tsvd-code-birch-v3/tsvd_cplex/output_clustering_temp112/rtr_v3/k_%d/rpt_%d/tsvd',k,repeat_count);
                mkdir(outpath_tsvd);
                utime=tic;
                [M_utsvd,cluster_id2] = TSVD(A_un,outpath_tsvd,k);
                disp('M found\n');
                alg_time(alg_idx,repeat_count)=toc(utime);
                fprintf('NMF: %s completed in %f secs\n',funs_str{alg_idx},alg_time(alg_idx,repeat_count));
                
                filename1 = strcat(outpath_data,'/M_UTSVD.mat');
                save(filename1,'M_utsvd');
                filename1 = strcat(outpath_data,'/cluster_id2.mat');
                save(filename1,'cluster_id2');
                
                M=M_utsvd;
                cluster_id2 = help_empty_topics(cluster_id2,k,n);
                topic(alg_idx,:) = cluster_id2;
                
                fprintf('Topic found for %s \n',funs_str{alg_idx});
                filename1 = strcat(outpath_data,'/topic.mat');
                save(filename1,'topic');
                
                Accuracy(alg_idx,repeat_count) = find_accuracy(topic(alg_idx,:),gnd,k);
                nmi_val(alg_idx,repeat_count) = nmi(gnd,topic(alg_idx,:));
                nmi_val2(alg_idx,repeat_count) = nmi2(gnd,topic(alg_idx,:));
                
                fprintf('Clustering completed by %s\n',funs_str{alg_idx});
                
            elseif  strcmp(funs_str{alg_idx},'NTSVD')
                disp('NTSVD started\n');
                outpath_tsvd = sprintf('/home/jagdeep/Desktop/NMF/Experiments/tsvd-code-birch-v3/tsvd_cplex/output_clustering_temp112/rtr_v3/k_%d/rpt_%d/tsvd',k,repeat_count);
                mkdir(outpath_tsvd);
                utime=tic;
                [M_utsvd,cluster_id2] = TSVD(A,outpath_tsvd,k);
                disp('M found\n');
                alg_time(alg_idx,repeat_count)=toc(utime);
                fprintf('NMF: %s completed in %f secs\n',funs_str{alg_idx},alg_time(alg_idx,repeat_count));
                
                filename1 = strcat(outpath_data,'/M_UTSVD.mat');
                save(filename1,'M_utsvd');
                filename1 = strcat(outpath_data,'/cluster_id2.mat');
                save(filename1,'cluster_id2');
                
                M=M_utsvd;
                cluster_id2 = help_empty_topics(cluster_id2,k,n);
                topic(alg_idx,:) = cluster_id2;
                
                fprintf('Topic found for %s \n',funs_str{alg_idx});
                filename1 = strcat(outpath_data,'/topic.mat');
                save(filename1,'topic');
                
                Accuracy(alg_idx,repeat_count) = find_accuracy(topic(alg_idx,:),gnd,k);
                nmi_val(alg_idx,repeat_count) = nmi(gnd,topic(alg_idx,:));
                nmi_val2(alg_idx,repeat_count) = nmi2(gnd,topic(alg_idx,:));
                
                fprintf('Clustering completed by %s\n',funs_str{alg_idx});
                
            elseif  strcmp(funs_str{alg_idx},'UTSVD_W')
                disp('UTSVD_W started\n');
                outpath_tsvd = sprintf('/home/jagdeep/Desktop/NMF/Experiments/tsvd-code-birch-v3/tsvd_cplex/output_clustering_er/rtr_v3/k_%d/rpt_%d/tsvd',k,repeat_count);
                mkdir(outpath_tsvd);
                utime=tic;
                [M_utsvd,~] = TSVD(A_un,outpath_tsvd,k);
                disp('M found\n');
                W_utsvd = nnlsHALSupdt(A_un,M_utsvd);
                
                alg_time(alg_idx,repeat_count)=toc(utime);
                fprintf('NMF: %s completed in %f secs\n',funs_str{alg_idx},alg_time(alg_idx,repeat_count));
                
                [~,max_idx] = max(W_utsvd);
                max_idx = help_empty_topics(max_idx,k,n);
                topic(alg_idx,:) = max_idx;
                fprintf('Topic found for %s \n',funs_str{alg_idx});
                
                Accuracy(alg_idx,repeat_count) = find_accuracy(topic(alg_idx,:),gnd,k);
                nmi_val(alg_idx,repeat_count) = nmi(gnd,topic(alg_idx,:));
                
                filename1 = strcat(outpath_data,'/M_utsvd_w.mat');
                save(filename1,'M_utsvd');
                filename1 = strcat(outpath_data,'/cluster_id2_utsvd_w.mat');
                save(filename1,'cluster_id2');
                filename1 = strcat(outpath_data,'/W_utsvd_w.mat');
                save(filename1,'W_utsvd');
                filename1 = strcat(outpath_data,'/topic.mat');
                save(filename1,'topic');
                
                fprintf('Clustering completed by %s\n',funs_str{alg_idx});
                
            elseif  strcmp(funs_str{alg_idx},'NTSVD_W')
                disp('NTSVD_W started\n');
                outpath_tsvd = sprintf('/home/jagdeep/Desktop/NMF/Experiments/tsvd-code-birch-v3/tsvd_cplex/output_clustering_er/rtr_v3/k_%d/rpt_%d/tsvd',k,repeat_count);
                mkdir(outpath_tsvd);
                utime=tic;
                [M_utsvd,~] = TSVD(A,outpath_tsvd,k);
                disp('M found\n');
                W_utsvd = nnlsHALSupdt(A,M_utsvd);
                
                alg_time(alg_idx,repeat_count)=toc(utime);
                fprintf('NMF: %s completed in %f secs\n',funs_str{alg_idx},alg_time(alg_idx,repeat_count));
                
                [~,max_idx] = max(W_utsvd);
                max_idx = help_empty_topics(max_idx,k,n);
                topic(alg_idx,:) = max_idx;
                fprintf('Topic found for %s \n',funs_str{alg_idx});
                
                Accuracy(alg_idx,repeat_count) = find_accuracy(topic(alg_idx,:),gnd,k);
                nmi_val(alg_idx,repeat_count) = nmi(gnd,topic(alg_idx,:));
                
                filename1 = strcat(outpath_data,'/M_ntsvd_w.mat');
                save(filename1,'M_utsvd');
                filename1 = strcat(outpath_data,'/cluster_id2_ntsvd_w.mat');
                save(filename1,'cluster_id2');
                filename1 = strcat(outpath_data,'/W_ntsvd_w.mat');
                save(filename1,'W_utsvd');
                filename1 = strcat(outpath_data,'/topic.mat');
                save(filename1,'topic');
                
                fprintf('Clustering completed by %s\n',funs_str{alg_idx});
                
                
            else
                fprintf('%s started\n',funs_str{alg_idx})
                fun = funs{alg_idx};
                fun_time=tic;
                temp_idx1=reshape(fun(A', noise_level,k), 1,k);
                
                if (temp_idx1==0)
                    disp('error happened\n');
                    err_flag=1;
                    break;
                end
                anchor_indices_all(alg_idx,:) = temp_idx1;
                alg_time(alg_idx,repeat_count)=toc(fun_time);
                fprintf('%s completed in %f secs\n',funs_str{alg_idx},alg_time(alg_idx,repeat_count));
                filename0 = strcat(outpath_data,'/anchor_indices_all.mat');
                save(filename0,'anchor_indices_all');
                
                W = A(anchor_indices_all(alg_idx,:),:);
                [~,max_idx] = max(W);
                max_idx = help_empty_topics(max_idx,k,n);
                topic(alg_idx,:) = max_idx;
                fprintf('Topic found for %s \n',funs_str{alg_idx});
                filename1 = strcat(outpath_data,'/topic.mat');
                save(filename1,'topic');
                
                Accuracy(alg_idx,repeat_count) = find_accuracy(topic(alg_idx,:),gnd,k);
                nmi_val(alg_idx,repeat_count) = nmi(gnd,topic(alg_idx,:));
                nmi_val2(alg_idx,repeat_count) = nmi2(gnd,topic(alg_idx,:));
                
            end
            
        end
        
        
        if (err_flag==1)
            disp('error happened in %s\n',funs_str{alg_idx})
            return;
        end
        
        %         [U,D,V]= svds(A,k);
        %         B=U*D*V'; % B is the best rank k approximation of A
        %
        %         for alg_idx=1:num_algs
        %
        %             if  strcmp(funs_str{alg_idx},'SPA')
        %                 W = B(anchor_indices_all(alg_idx,:),:);
        %                 [~,max_idx] = max(W);
        %                 max_idx = help_empty_topics(max_idx,k,n);
        %                 topic(num_algs+1,:) = max_idx;
        %                 fprintf('Topic found for Lowrank-SPA \n');
        %                 filename1 = strcat(outpath_data,'/topic.mat');
        %                 save(filename1,'topic');
        %
        %                 Accuracy(num_algs+1,repeat_count) = find_accuracy(topic(num_algs+1,:),gnd,k);
        %                 nmi_val(num_algs+1,repeat_count) = nmi(gnd,topic(num_algs+1,:));
        %                 alg_time(num_algs+1,repeat_count) = alg_time(alg_idx,repeat_count);
        %
        %             elseif strcmp(funs_str{alg_idx},'ER-SPA')
        %                 W = B(anchor_indices_all(alg_idx,:),:);
        %                 [~,max_idx] = max(W);
        %                 max_idx = help_empty_topics(max_idx,k,n);
        %                 topic(num_algs+2,:) = max_idx;
        %                 fprintf('Topic found for Lowrank-ER-SPA \n');
        %                 filename1 = strcat(outpath_data,'/topic.mat');
        %                 save(filename1,'topic');
        %
        %                 Accuracy(num_algs+2,repeat_count) = find_accuracy(topic(num_algs+2,:),gnd,k);
        %                 nmi_val(num_algs+2,repeat_count) = nmi(gnd,topic(num_algs+2,:));
        %                 alg_time(num_algs+2,repeat_count) = alg_time(alg_idx,repeat_count);
        %             end
        %
        %         end
        
        
    end
    alg_time_final(:,k_count) = mean(alg_time,2);
    nmi_val_final(:,k_count) = mean(nmi_val,2);
    nmi_val_final2(:,k_count) = mean(nmi_val2,2);
    Accuracy_final(:,k_count) = mean(Accuracy,2);
    
    filename1 = strcat(outpath_result,'/nmi_val_final.mat');
    save(filename1,'nmi_val_final');
    filename1 = strcat(outpath_result,'/nmi_val_final2.mat');
    save(filename1,'nmi_val_final2');
    filename2 = strcat(outpath_result,'/Accuracy_final.mat');
    save(filename2,'Accuracy_final');
    filename5 = strcat(outpath_result,'/alg_time_final.mat');
    save(filename5,'alg_time_final');
    
    
end
funs_str = {'XRAY','SPA-doc-norm','UTSVD','NTSVD','UTSVD_W','NTSVD_W'};
outpath_home = '/home/jagdeep/Desktop/NMF/Experiments/tsvd-code-birch-v3/tsvd_cplex/output_clustering_temp112/rtr_v3';
filename5 = strcat(outpath_home,'/funs_str.mat');
save(filename5,'funs_str');


for k_count=1:length(k_arr)
    k=k_arr(k_count);
    fprintf('\n Dataset: Reuters, k = %d\n',k);
    fprintf('Algorithm\t Running time\t Accuracy\t NMI\t NMI2\n');
    
    for alg_idx=1:length(funs_str)
        fprintf('%s\t %f\t %f\t %f\t %f\n',funs_str{alg_idx},alg_time_final(alg_idx,k_count),Accuracy_final(alg_idx,k_count),nmi_val_final(alg_idx,k_count), nmi_val_final2(alg_idx,k_count));
    end
    fprintf('\n************************************\n')
end


