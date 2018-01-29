clear all;
close all;
load bbc.mat;
clear trainIdx testIdx;
%addpath /home/jagdeep/Desktop/NMF/Experiments/glpkmex-master;
% Reducing the size of data similar to Mizutani etal.
% Assumes data points are sorted according to labels



% Preprocessing the data
% stop_words=importdata('stopwords.txt');
% data1=importdata('Reuter_terms.txt');
% R_terms=data1.textdata;
% clear data1;
% fea1=preprocess_dataset_reuters(fea,R_terms,stop_words);

%loading the preprocessed data
%clear fea;
%load Reuters_fea_processed.mat;


%fea = fea(1:8258,:); % 7285 x 18933
%gnd = gnd(1:8258);      % 7285 x 1
%fea = fea;
%gnd = gnd;
%fea1=fea;
%gnd1=gnd;
%ind = trainIdx<7286;    
%trainIdx = trainIdx(ind);
% testIdx = testIdx(testIdx<7286);

% alt_min=0:1:20;
%alt_min=0:1:20;
alt_min=0:1:11;
k=5;
fea(:,find(sum(fea,1)<1))=[]; % Removing orphan words (not present in any document) 
    
noise_level=0.3;
    
funs = {
    %@utsvd_nmf, ...
    %@hottopixx, ...
    %@spa, ...
    %@fast_hull,...
    %@ER_spa_svds,...
%     @ER_xray_svds,...
    %@LPsepNMF_oone, ...
    %@LPsepNMF_otwo, ...
    };

    % Descriptive names for the algorithms
%     funs_str = {
        %'U-tsvd', ...
    %'Hottopixx', ...
%     'SPA',...
%     'XRAY',...
%     'ER-SPA',...
%     'ER-XRAY',...
%     'UTSVD',...
    %'LP-rho1', ...
    %'LP-rho2', ...

%     };

funs={@heur_spa};
funs_str = {'Heur-SPA'};


num_algs = length(funs);

A=fea';
[m,n] = size(A);



Accuracy=zeros(num_algs+1,length(alt_min));
Accuracy_sd = zeros(num_algs+1,length(alt_min));
    
nmi_val = zeros(num_algs+1,length(alt_min));
nmi_val_sd = zeros(num_algs+1,length(alt_min));

final_K_time=zeros(num_algs+1,length(alt_min));
failed_count=zeros(length(alt_min),1);

anchor_indices_all=zeros(num_algs,k);
err_flag=0;

for alg_idx = 1:num_algs
    fun = funs{alg_idx};
    tic
    temp_idx1=reshape(fun(fea, noise_level,k), 1,k);
    
    if (temp_idx1==0)
        err_flag=1;
        break;
    end
    
    anchor_indices_all(alg_idx,:) = temp_idx1;
    fprintf('%s completed in %f secs',funs_str{alg_idx},toc);
end
if (err_flag==1)
    disp('error happened in %s',funs_str{alg_idx})
    return;
end
outpath11 = '/home/jagdeep/Documents/NMF/tsvd_cplex/output_clustering_heur/bbc';
mkdir(outpath11);
filename0 = strcat(outpath11,'/anchor_indices_all_bbc.mat');
save(filename0,'anchor_indices_all');

%load anchor_indices_all_bbc.mat;


% Running TSVD
 %outpath = sprintf('/home/jagdeep/Desktop/NMF/Experiments/tsvd-code-birch-v2/tsvd_cplex/output/clustering/alt_min_iter');
    %outpath = '/home/jagdeep/Documents/NMF/tsvd_cplex/output_clustering/bbc_birch';
    %mkdir(outpath);

%     tic
%     [M_utsvd,cluster_id2] = TSVD(A,outpath11,k);
%     disp('M found');
    %W = solvelp(A,M); % solve lp or nnls, which is better
        % used nnls, as solvelp is needing more memory


temp_Accuracy=zeros(num_algs,length(alt_min));
    temp_nmi_val=zeros(num_algs,length(alt_min));
    K_time=zeros(num_algs,length(alt_min));

    
for alg_idx=1:num_algs+1
    if  strcmp(funs_str{alg_idx},'UTSVD')
        M = M_utsvd;
        W= nnlsHALSupdt( A , M );
        %nmi(gnd,cluster_id2);
                
    else
       anchor_indices = anchor_indices_all(alg_idx,:);
        W = A(anchor_indices,:);
        M=nnlsHALSupdt(A',W');
        M=M';
    end

    
	for alt_min_count=1:length(alt_min)
        [~,max_idx] = max(W);
        tic      
        topic(alg_idx,:) = max_idx;
        K_time(alg_idx,alt_min_count)=toc;
        fprintf('Topic found in %f secs\n',K_time(alg_idx,alt_min_count));
        residual_L2(alg_idx,alt_min_count) = norm(A-M*W,'fro');
        residual_L1(alg_idx,alt_min_count) = norm(A-M*W,1);  
                
%         [newtopic] = bestMap(gnd',topic(alg_idx,:));
%         sumAc=0;
%         for topic_number=1:k
%             docs_top=find(newtopic==topic_number);
%             docs_gnd=find(newtopic==topic_number);
%             sumAc = sumAc + length(intersect(docs_top,docs_gnd));
%         
%         end
%         temp_Accuracy(alg_idx,alt_min_count) = sumAc/n;

 % Accuracy evaluation started
        % Assignment of topics to clusters
            % Formation of topic class matrix
            topic_class = zeros(k,k);
                for doc_idx=1:n
                    temp1=topic(alg_idx,doc_idx);
                    temp2=gnd(doc_idx);
                    topic_class(temp1,temp2) = topic_class(temp1,temp2)+1;
                end
            
          % solving the assignment problem

          Aeq=zeros(2*k,k^2);
          for i=1:k
            temp1=zeros(k,k);
            temp2=zeros(k,k);
            temp1(i,:)=ones(k,1)';
            temp2(:,i)=ones(k,1);
            Aeq(i,:)=reshape(temp1,k^2,1);
            Aeq(k+i,:)=reshape(temp2,k^2,1);
          end

              intersect_mat=squeeze(topic_class);
              f = reshape(intersect_mat,k^2,1);
              beq=ones(2*k,1);
              LB=zeros(k^2,1);
              UB=Inf*ones(k^2,1);

              solution = linprog(-1*f,[],[],Aeq,beq,LB,UB);
              temp_Accuracy(alg_idx,alt_min_count) = (sum(sum(reshape(solution,k,k).*intersect_mat)))/n;
          
          % Accuracy evaluation done
         
         
          
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
          
          
          % calculating the nmi
          temp_nmi_val(alg_idx,alt_min_count) = nmi(gnd', topic(alg_idx,:));
    if  alt_min_count~=length(alt_min)  
        if  strcmp(funs_str{alg_idx},'UTSVD')
            M= nnlsHALSupdt( A' , W' );
            M=M';
            W = nnlsHALSupdt( A ,M );
        else
            W = nnlsHALSupdt(A,M);
            M = nnlsHALSupdt(A',W');
            M = M';
        end
    end  
                    
end

end
fprintf('nmi val is %f',nmi(gnd,cluster_id2));
%outpath11 = '/home/jagdeep/Desktop/NMF/Experiments/tsvd-code-birch-v2/tsvd_cplex/output/clustering/alt_min_iter_files-temp1';
%mkdir(outpath11);
filename1 = strcat(outpath11,'/nmi_val.mat');
save(filename1,'temp_nmi_val');
filename2 = strcat(outpath11,'/Accuracy_val.mat');
save(filename2,'temp_Accuracy');
filename3 = strcat(outpath11,'/Residual_norm_L2.mat');
save(filename3,'residual_L2');
filename4 = strcat(outpath11,'/Residual_norm_L1.mat');
save(filename4,'residual_L1');
filename5 = strcat(outpath11,'/K_time.mat');
save(filename5,'K_time');

% x = 2*alt_min(1:11);
% plot(x,temp_nmi_val(:,1:11));
% grid on;
% xlabel='Iterations';
% ylabel='Normalized Mutual Information(NMI)';
% legend(funs_str{1},funs_str{2},funs_str{3},funs_str{4},funs_str{5});




































  
    