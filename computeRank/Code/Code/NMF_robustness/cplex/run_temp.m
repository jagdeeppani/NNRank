k = k_arr(k_count);
    for repeat_count=1:no_of_repetitions
        fprintf('number of classes = %d, repetitions = %d\n',k,repeat_count);
        
        class_indices = randperm(orig_k);
        class_indices = class_indices(1:k);
        
        indices=[];
        for ci_count=1:k
            indices = union(indices,find(gnd_orig==class_indices(ci_count)) );
        end
        fea = fea_orig(indices,:);
        gnd = gnd_orig(indices,:);
        
        noise_level=0.3;
        
        funs = {@heur_spa,@spa,@fast_hull,@ER_spa_svds, @ER_xray_svds,@utsvd};
        funs_str = {'Heur-SPA','SPA','XRAY','ER-SPA','ER-XRAY','UTSVD'};
        
        
        num_algs = length(funs_str);
        
        A=fea';
        clear fea;
        [m,n] = size(A);
        %         outpath_data = '/home/jagdeep/Desktop/NMF/Experiments/tsvd-code-birch-v3/tsvd_cplex/output_clustering/rtr_v3/data';
        outpath_data = sprintf('/home/jagdeep/Desktop/NMF/Experiments/tsvd-code-birch-v3/tsvd_cplex/output_clustering/rtr_v3/k_%d/rpt_%d/data',k,repeat_count);
        mkdir(outpath_data);
        
        
        filename1 = strcat(outpath_data,'/features_A.mat');
        save(filename1,'A');
        filename1 = strcat(outpath_data,'/ground_truth.mat');
        save(filename1,'gnd');
        
        alg_time=zeros(num_algs,1);
        
        anchor_indices_all=zeros(num_algs,k);
        err_flag=0;
        
        
        outpath_result = sprintf('/home/jagdeep/Desktop/NMF/Experiments/tsvd-code-birch-v3/tsvd_cplex/output_clustering/rtr_v3/k_%d/rpt_%d/result',k,repeat_count);
        mkdir(outpath_result);
        
        for alg_idx = 1:num_algs
            if strcmp(funs_str{alg_idx},'UTSVD')
                disp('UTSVD started');
                outpath_tsvd = sprintf('/home/jagdeep/Desktop/NMF/Experiments/tsvd-code-birch-v3/tsvd_cplex/output_clustering/rtr_v3/k_%d/rpt_%d/tsvd',k,repeat_count);
                mkdir(outpath_tsvd);
                utime=tic;
                [M_utsvd,cluster_id2] = TSVD(A,outpath_tsvd,k);
                disp('M found');
                alg_time(alg_idx)=toc(utime);
                fprintf('%s completed in %f secs\n',funs_str{alg_idx},alg_time(alg_idx));
                
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
                alg_time(alg_idx)=toc(fun_time);
                fprintf('%s completed in %f secs\n',funs_str{alg_idx},alg_time(alg_idx));
            end
            
        end
        
        if (err_flag==1)
            disp('error happened in %s\n',funs_str{alg_idx})
            return;
        end
        %
        % outpath_data = '/home/jagdeep/Desktop/NMF/Experiments/tsvd-code-birch-v3/tsvd_cplex/output_clustering/tdt2_erspa/output';
        % mkdir(outpath11);
        filename0 = strcat(outpath_data,'/anchor_indices_all_tdt2.mat');
        save(filename0,'anchor_indices_all');
        
        
        Accuracy=zeros(num_algs,max_alt_min);
        nmi_val=zeros(num_algs,max_alt_min);
        optimization_time=zeros(num_algs,max_alt_min);
        
        residual_L2 = zeros(num_algs,max_alt_min);
        residual_L1 = zeros(num_algs,max_alt_min);
        residual_s = zeros(num_algs,max_alt_min);
        
        
        
        for alg_idx=1:num_algs
            
            for alt_min_count=0:max_alt_min
                topic=zeros(alg_idx,n);
                fprintf('Algorithm: %s, Iteration number: %d\n',funs_str{alg_idx},alt_min_count);
                outpath_data_iter=strcat(outpath_data,'/iter',num2str(alt_min_count));
                mkdir(outpath_data_iter);
                temp_start=tic;
                if  strcmp(funs_str{alg_idx},'UTSVD') && alt_min_count==0;
                    M=M_utsvd;
                    optimization_time(alg_idx,alt_min_count+1) = toc(temp_start);
                    nmi_val(alg_idx,1) = nmi(gnd,cluster_id2);
                    
                    filename1 = strcat(outpath_data_iter,'/M_UTSVD.mat');
                    save(filename1,'M');
                    continue;
                    
                elseif strcmp(funs_str{alg_idx},'UTSVD') && alt_min_count==1;
                    W= nnlsHALSupdt( A , M );
                    filename1 = strcat(outpath_data_iter,'/W_UTSVD.mat');
                    save(filename1,'W');
                    
                elseif strcmp(funs_str{alg_idx},'UTSVD')
                    M = nnlsHALSupdt(A',W'); M=M';
                    W = nnlsHALSupdt(A,M);
                    filename1 = strcat(outpath_data_iter,'/M_UTSVD.mat');
                    save(filename1,'M');
                    filename1 = strcat(outpath_data_iter,'/W_UTSVD.mat');
                    save(filename1,'W');
                    
                elseif alt_min_count==0;
                    W=A(anchor_indices_all(alg_idx,:),:);
                    filename1 = strcat(outpath_data_iter,'/W_',funs_str{alg_idx},'.mat');
                    save(filename1,'W');
                    
                else
                    M = nnlsHALSupdt(A',W'); M=M';
                    W = nnlsHALSupdt(A,M);
                    filename1 = strcat(outpath_data_iter,'/M_',funs_str{alg_idx},'.mat');
                    save(filename1,'M');
                    filename1 = strcat(outpath_data_iter,'/W_',funs_str{alg_idx},'.mat');
                    save(filename1,'W');
                end
                
                [~,max_idx] = max(W);
                
                topic(alg_idx,:) = max_idx;
                optimization_time(alg_idx,alt_min_count+1) = toc(temp_start);
                fprintf('Topic found in %f secs\n',optimization_time(alg_idx,alt_min_count+1));
                filename1 = strcat(outpath_data_iter,'/topic_',funs_str{alg_idx},'.mat');
                save(filename1,'topic');
                if alt_min_count>0
                    
                    residual_L2(alg_idx,alt_min_count+1) = 1 - (norm(A-M*W,'fro')/norm(A,'fro'));
                    residual_L1(alg_idx,alt_min_count+1) = 1 - (norm(A-M*W,1)/norm(A,1));
                    residual_s(alg_idx,alt_min_count+1) = 1- (sumabs(A-M*W)/sumabs(A));
                end
                
                % This portion takes care of orphan topics i.e it ensures that each topic has atleast one document
                % for alg_idx=1:num_algs+1
                for adjust_topic=1:k
                    topic_count2(adjust_topic)=sum(topic(alg_idx,:)==adjust_topic);
                end
                orphan_topics=find(topic_count2==0);
                [~,max_top]=max(topic_count2);
                
                doc_id2=1;
                for all_count=1:length(orphan_topics)
                    found1=0;
                    while(~found1)
                        if topic(alg_idx,doc_id2)==max_top
                            found1=1;
                            topic(alg_idx,doc_id2)=orphan_topics(all_count);
                        end
                        
                        doc_id2=doc_id2+1;
                    end
                end
                
                % end
                
                
                % Accuracy evaluation started
                
                Accuracy(alg_idx,alt_min_count+1) = find_accuracy(topic(alg_idx,:),gnd,k);
                
                % Accuracy evaluation done
                
                
                
                % calculating the nmi
                nmi_val(alg_idx,alt_min_count+1) = nmi(gnd', topic(alg_idx,:));
                
            end
        end
        
        Total_time=cumsum(optimization_time,2)+repmat(alg_time,1,max_alt_min+1);
        
        filename1 = strcat(outpath_result,'/nmi_val.mat');
        save(filename1,'nmi_val');
        filename2 = strcat(outpath_result,'/Accuracy_val.mat');
        save(filename2,'Accuracy');
        filename3 = strcat(outpath_result,'/Residual_norm_L2.mat');
        save(filename3,'residual_L2');
        filename4 = strcat(outpath_result,'/Residual_norm_L1.mat');
        save(filename4,'residual_L1');
        filename4 = strcat(outpath_result,'/Residual_norm_s.mat');
        save(filename4,'residual_s');
        filename5 = strcat(outpath_result,'/Total_time.mat');
        save(filename5,'Total_time');
        filename5 = strcat(outpath_result,'/alg_time.mat');
        save(filename5,'alg_time');
        filename5 = strcat(outpath_result,'/optimization_time.mat');
        save(filename5,'optimization_time');
        
        filename4 = strcat(outpath_result,'/Residual_norm_s.mat');
        save(filename4,'residual_s');
        
        [val,ind] = max(nmi_val,[],2);
        max_nmi_val(:,repeat_count) = val;
        max_nmi_index(:,repeat_count) = ind;
        
        time_taken=zeros(size(Total_time,1),1);
        for alg_idx=1:size(Total_time,1)
            time_taken(alg_idx) = Total_time(alg_idx,max_nmi_index(alg_idx,repeat_count));
        end
        Total_clustering_time(:,repeat_count) = time_taken;
        
    end
    
    outpath_final_result = sprintf('/home/jagdeep/Desktop/NMF/Experiments/tsvd-code-birch-v3/tsvd_cplex/output_clustering/rtr_v3/k_%d',k);
    %     mkdir(outpath_final_result);
    
    filename = strcat(outpath_final_result,'/max_nmi_val.mat');
    save(filename,'max_nmi_val');
    filename = strcat(outpath_final_result,'/max_nmi_index.mat');
    save(filename,'max_nmi_index');
    filename = strcat(outpath_final_result,'/Total_clustering_time.mat');
    save(filename,'Total_clustering_time');
    
    avg_nmi_val = mean(max_nmi_val,2);
    avg_time_taken = mean(Total_clustering_time,2);
    
    fprintf('\n Dataset: Reuters, k = %d\n',k);
    fprintf('Algorithm\t Running time\t NMI\n');
    
    for alg_idx=1:num_algs
        fprintf('%s\t %f\t %f\n',funs_str{alg_idx},avg_time_taken(alg_idx),avg_nmi_val(alg_idx));
    end