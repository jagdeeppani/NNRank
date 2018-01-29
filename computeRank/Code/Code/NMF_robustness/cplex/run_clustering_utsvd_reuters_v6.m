clear all;
close all;
load Reuters21578.mat;
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


fea = fea(1:7285,:); % 7285 x 18933
gnd = gnd(1:7285);      % 7285 x 1
%fea = fea;
%gnd = gnd;
%fea1=fea;
%gnd1=gnd;
%ind = trainIdx<7286;    
%trainIdx = trainIdx(ind);
% testIdx = testIdx(testIdx<7286);

alt_min=0:1:20;
k=10;
fea(:,find(sum(fea,1)<1))=[]; % Removing orphan words (not present in any document) 
    
noise_level=0.3;
    
funs = {
    %@utsvd_nmf, ...
    %@hottopixx, ...
    @spa, ...
    @fast_hull,...
    @ER_spa,...
    @ER_xray,...
    %@LPsepNMF_oone, ...
    %@LPsepNMF_otwo, ...
    };

    % Descriptive names for the algorithms
    funs_str = {
        %'U-tsvd', ...
    %'Hottopixx', ...
    'SPA',...
    'XRAY',...
    'ER-SPA',...
    'ER-XRAY',...
    %'LP-rho1', ...
    %'LP-rho2', ...

    };

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

% Running TSVD
 %outpath = sprintf('/home/jagdeep/Desktop/NMF/Experiments/tsvd-code-birch-v2/tsvd_cplex/output/clustering/alt_min_iter');
    outpath = '/home/jagdeep/Desktop/NMF/Experiments/tsvd-code-birch-v2/tsvd_cplex/output/clustering/alt_min_iter';
    mkdir(outpath);

    tic
    [M_utsvd,~] = TSVD(A,outpath,k);
    disp('M found');
    %W = solvelp(A,M); % solve lp or nnls, which is better
        % used nnls, as solvelp is needing more memory


temp_Accuracy=zeros(num_algs+1,length(alt_min));
    temp_nmi_val=zeros(num_algs+1,length(alt_min));
    K_time=zeros(num_algs+1,length(alt_min));

    
for alg_idx=1:num_algs
	anchor_indices = anchor_indices_all(alg_idx,:);
	W = A(anchor_indices,:);
	M=nnlsHALSupdt(A',W');
	M=M';
	

	for alt_min_count=1:length(alt_min)
		[~,max_idx] = max(W);
              
                topic(alg_idx,:) = max_idx;
                K_time(alg_idx,alt_min_count)=toc;
                fprintf('Topic found in %f secs\n',K_time(alg_idx,alt_min_count));
                residual_L2(alg_idx,alt_min_count) = norm(A-M*W,'fro');
                residual_L1(alg_idx,alt_min_count) = norm(A-M*W,1);  

			topic_class = zeros(num_algs+1,k,k);
            for alg_idx=1:size(topic,1);

                for doc_idx=1:n
                    temp1=topic(alg_idx,doc_idx);
                    temp2=gnd(doc_idx);
                    topic_class(alg_idx,temp1,temp2) = topic_class(alg_idx,temp1,temp2)+1;
                end
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

          for alg_idx=1:num_algs+1
              intersect_mat=squeeze(topic_class(alg_idx,:,:));
              f = reshape(intersect_mat,k^2,1);
              beq=ones(2*k,1);
              LB=zeros(k^2,1);
              UB=Inf*ones(k^2,1);

              solution = linprog(-1*f,[],[],Aeq,beq,LB,UB);
              temp_Accuracy(alg_idx,alt_min_count) = (sum(sum(reshape(solution,k,k).*intersect_mat)))/n;
          end
          % Accuracy evaluation done
          
          % This portion takes care of orphan topics i.e it ensures that each topic has atleast one document 
          for alg_idx=1:num_algs+1
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
                           
          end
          
          
          
          
          % calculating the nmi
          for alg_idx=1:num_algs+1
              temp_nmi_val(alg_idx,alt_min_count) = nmi(gnd', topic(alg_idx,:));
          end
           


		W = nnlsHALSupdt(A,M);
         M = nnlsHALSupdt(A',W');
         M = M';
                    
     end














for alt_min_count = 1:length(alt_min)
    alt_min_current=alt_min(alt_min_count);

    %failed_idx=[];
    %for iter_count=1:length(alt_min)
        % Choosing randomly k classes and the corresponding datapoint and
        % labels
        
%         unique_gnd=unique(gnd);
%         permute1=randperm(length(unique_gnd));
%         label=unique_gnd(permute1(1:k));
%         position=0;
%         for i=1:k
%             temp = find(gnd==label(i));
%             data_idx(position+1:position+length(temp))=temp;
%             position=position+length(temp);
%         end
% 
%         data_idx=sort(data_idx);
%         gnd = gnd(data_idx);
%         fea = fea(data_idx,:);
%         unique_gnd=unique(label);
%         clear data_idx;

        
        %size(fea)
         % m is the vocabulary size, n: No of documents
            topic = zeros(num_algs+1,n);
            %normalized_fea = fea*spdiags(1./sum(fea,1)',0,n,n);
            for alg_idx = 1:num_algs
                
                tic
                anchor_indices = anchor_indices_all(alg_idx,:);
            
                W = A(anchor_indices,:);
                M=nnlsHALSupdt(A',W');
                M=M';
                
                for temp_count=1:alt_min_current
                    W = nnlsHALSupdt(A,M);
                    M = nnlsHALSupdt(A',W');
                    M = M';
                    
                end
                
                [~,max_idx] = max(W);
                %size(W)
                topic(alg_idx,:) = max_idx;
                K_time(alg_idx,alt_min_count)=toc;
                fprintf('Topic found in %f secs\n',K_time(alg_idx,alt_min_count));
                residual_L2(alg_idx,alt_min_count) = norm(A-M*W,'fro');
                residual_L1(alg_idx,alt_min_count) = norm(A-M*W,1);

                %W = solvelp(A,A(:,anchor_indices)); % solving for 1 norm error and Rev(Inf,1) error are same. % solve lp or nnls, which is better
                %anchor_indices = reshape(anchor_indices, 1, length(anchor_indices));  
                %pct = 100 * measureIndex(K_true, anchor_indices);
                %fprintf('Noise level : %f',noise_level);
                %fprintf('%10s : L1 residual norm : (%5.1f): ', funs_str{alg_idx}, temp);
                %disp(sort(anchor_indices));
            end
%            if err_flag==1
%                K_time(:,alt_min_count)=0;
%                failed_count(alt_min)=failed_count(alt_min)+1;
%                continue;
%            end

        %UTSVD started
        M=M_utsvd;
        W = nnlsHALSupdt(A,M);
        for temp_count=1:alt_min_current
            M = nnlsHALSupdt(A',W');
            M = M';
            W = nnlsHALSupdt(A,M);
        end
        disp('W found');
        

        [~,max_idx] = max(W); % C_tsvd is a vector (1xn) containing the cluster number of each document
        topic(num_algs+1,:)=max_idx;
        K_time(num_algs+1,alt_min_count)=toc;
        fprintf('Topic found in %f secs\n',K_time(num_algs+1,alt_min_count));
        residual_L2(num_algs+1,alt_min_count) = norm(A-M*W,'fro');
        residual_L1(num_algs+1,alt_min_count) = norm(A-M*W,1);

        % Accuracy evaluation started
        % Assignment of topics to clusters
            % Formation of topic class matrix
            topic_class = zeros(num_algs+1,k,k);
            for alg_idx=1:size(topic,1);

                for doc_idx=1:n
                    temp1=topic(alg_idx,doc_idx);
                    temp2=gnd(doc_idx);
                    topic_class(alg_idx,temp1,temp2) = topic_class(alg_idx,temp1,temp2)+1;
                end
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

          for alg_idx=1:num_algs+1
              intersect_mat=squeeze(topic_class(alg_idx,:,:));
              f = reshape(intersect_mat,k^2,1);
              beq=ones(2*k,1);
              LB=zeros(k^2,1);
              UB=Inf*ones(k^2,1);

              solution = linprog(-1*f,[],[],Aeq,beq,LB,UB);
              temp_Accuracy(alg_idx,alt_min_count) = (sum(sum(reshape(solution,k,k).*intersect_mat)))/n;
          end
          % Accuracy evaluation done
          
          % This portion takes care of orphan topics i.e it ensures that each topic has atleast one document 
          for alg_idx=1:num_algs+1
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
                           
          end
          
          
          
          
          % calculating the nmi
          for alg_idx=1:num_algs+1
              temp_nmi_val(alg_idx,alt_min_count) = nmi(gnd', topic(alg_idx,:));
          end
    
          
    %end
    
    %temp_Accuracy(:,find((sum(temp_Accuracy))==0))=[];
    %temp_nmi_val(:,find((sum(temp_nmi_val))==0))=[];
    %Accuracy(:,alt_min_count)=mean(temp_Accuracy,2);
    %Accuracy_sd(:,alt_min_count) = std(temp_Accuracy,0,2); 
    
    %nmi_val(:,alt_min_count)=mean(temp_nmi_val,2);
    %nmi_val_sd(:,alt_min_count)= std(temp_nmi_val,0,2);
    
    %final_K_time(:,find((sum(final_K_time))==0))=[];
    %final_K_time(:,alt_min_count)=mean(K_time,2);
end

outpath11 = '/home/jagdeep/Desktop/NMF/Experiments/tsvd-code-birch-v2/tsvd_cplex/output/clustering/alt_min_iter_files';
mkdir(outpath11);
filename1 = strcat(outpath11,'/nmi_val.mat');
save(filename1,'temp_nmi_val');
filename2 = strcat(outpath11,'/Accuracy_val.mat');
save(filename2,'temp_Accuracy');
filename3 = strcat(outpath11,'/Residual_norm_L2.mat');
save(filename3,'residual_L2');
filename = strcat(outpath11,'/Residual_norm_L1.mat');
save(filename,'residual_L1');
filename = strcat(outpath11,'/K_time.mat');
save(filename,'K_time');

   












    
