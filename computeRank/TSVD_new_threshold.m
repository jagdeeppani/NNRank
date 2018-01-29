function [M_hat,cluster_id2] = TSVD_new_threshold(A,outpath,K,binary,varargin)
%************ Thresolded SVD based K-Means for Topic Recovery *************
% 
% [M_hat, dominantTopic] = TSVD(inpath, outpath, K)
%
% Inputs:
% inpath : Path to data matrix A (output from Amatrix function, see ReadMe)
% outpath : Output folder (final topic matrix will be written here)
% K : Number of topics
%
% Output:
% M_hat : Final estimated topic matrix
% dominantTopic : Cluster (dominant topic) assignment of documents
%
% Optional Inputs (Algorithm Parameters, need to be passed in order):
% (For details refer to paper, steps here refer to TSVD algo section 3.3)
% w_0 : double, lower bound of pobability of dominant topic, see step 1.a
%       (default 1/k)
% \epsilon_1 : double, \epsilon parameter for second thresolding in 
%              step 1(a) (default 1/6)
% \epsilon_2 : double, \epsilon_0 for computing g(.,.) in step 5(a)
%              (default 1/6)
% \gamma : double, for finding the set J(.) in step 5.b (default 1.1)
% \epsilon_3 : double, \epsilon_0 for finding top documents, step 6 
%              (default 1.0)
% reps : int, number of repitions for the first k-means step 4.a, more 
%        repititions are better though computationally expensive 
%        (default depends on data size: 1,3,5)
% 
% 
% Example:
% inpath = 'demo_nips/nips-proc.txt';
% outpath = 'out/';
% K = 50;
% [M, labels] = TSVD(inpath, outpath, K);
%

%
%-------------------------------------------------------------------------
% Author: Trapit Bansal
% Last Modified: August 6, 2014
%-------------------------------------------------------------------------
% 



if ~isempty(varargin)   % Algo parameters
    [w0,eps1,eps2,rho,eps3,reps,eps5,alpha5,beta5,tolerance] = deal(varargin{:});
else
%         w0 = 1.0/K;
%         eps1 = 1/6;
%         eps2 = 1/3;
%         rho = 1.1;
%         eps3 = 1.0;
%         reps = 3;  % will be changed later if data is too big or too small

        w0 = 1.0/K;
        eps1 = 1/6;
        eps2 = 1/3;
        rho = 1.05;
        eps3 = 1.1;
        reps = 3;
        
        eps5 = 0.04;
        alpha5 = 0.9;
        beta5 = 0.02;
        tolerance = 15;

%         eps5 = 0.01;
%         alpha5 = 0.75;
%         beta5 = 0.02;
%         tolerance = 5;


end

%A = Amatrix(infile);  %Reads the data and returns the A matrix
[w,d] = size(A);
fprintf('\nNo of documents read:%d\n',d);

if isempty(varargin)
    if d <= 5000, reps = 5;
    elseif d >= 50000, reps = 1;
    else reps = 10  ;
    end
end
% fprintf('reps is %d\n',reps);
if w*d > 1e8
    save(strcat(outpath, '/A.mat'),'A');  % Save A for later use, if its large
end

start_time = tic;

% A = A*spdiags(1./sum(A,1)',0,d,d); %normalized A

fprintf('Thresholding on A\n')
tic;
A = A'; % Row slicing is inefficient for sparse matrix
% [B, thres] = threshold_sparse_v3(A, w0, eps1, w, d);

fprintf('binary is %d\n',binary)
if binary
    [B, thres] = threshold_sparse_v3_binary(A, eps5, alpha5, beta5,tolerance,outpath);
else
[B, thres] = threshold_sparse_v3_0(A, eps5, alpha5, beta5,tolerance,outpath);
end


% [B, thres] = threshold_docs_try(A, 3/4);

toc;
fprintf('Thresholding on A done..\n')
% RReurn;

% new remove rows with thres 1
% B = B(sum(B,2)~=0,:);

if w*d > 1e8, clear A; % clear to save space
else A = A'; end

% Uncomment the follwing line to write the thresholds (\zeta)
% tic
dlmwrite(strcat(outpath,'/Thresholds'),full(thres),'delimiter','\n');
% clear thres
% fprintf('Time take to write threshold : %f secs',toc);
retained_docs = find(sum(B,1)~=0);   %Columns which are not completely zero
B = B(:,retained_docs);
fprintf('size of B is %d\n',size(B));

%% Computing the K-rank approximation for the matrix B
fprintf('Computing SVD Projection\n')
tic;
if w*d <= 5e7 && w > d
    [~,S,V] = svds(B,K);
    B_k = sparse(S)*V';
    clear S V;
else
    % computing BB^T then finding top eigenvectors!
    BBt = B*B';
%     BBt = spSymProd(B);
    [U,~] = eigs(BBt,[],K,'lm',struct('issym',1));
    clear BBt;
    B_k = U'*B;
    clear U;
end
toc;
% filename1 = strcat(outpath,'/B_k.mat');   % For debugging
%         save(filename1,'B_k');


% 
%% K-means on projected matrix
% fprintf('Performing k-means on columns of B_k\n')
%% This is k-means %%
% tic;
% q_best = Inf;
% cluster_id = [];
% for r = 1:reps
%     [~,ini_center] = kmeanspp_ini(B_k,K);
%     [~, c_id, ~,q2,~] = kmeans_fast(B_k,ini_center,2,reps > 1);
%     if q2 < q_best
%         cluster_id = c_id; 
%         q_best = q2;
%     end
% end
% fprintf('Quality:%.4f\n',q_best);
% toc;
% clear B_k ini_center c_id;
%%

%% THIS IS BIRCH
fprintf('Performing BIRCH on columns of B_k\n')
tic;
dlmwrite('../BIRCH/B_k',B_k', 'delimiter',' ','precision','%.6f');
[C, ~,~] = BIRCH(K,K,50,2);

% filename1 = strcat(outpath,'/C.mat');   % For debugging
%         save(filename1,'C');

fprintf('Number of clusters:%d\n',size(C,1));
C = C';
[~,cluster_id] = max(bsxfun(@minus,2*real(C'*B_k),dot(C,C,1).'));
clear C B_k
toc;

filename1 = strcat(outpath,'/cluster_id.mat');   % For debugging
        save(filename1,'cluster_id');
        
% filename1 = strcat(outpath,'/retained_docs.mat');   % For debugging
%         save(filename1,'retained_docs');


P1 = zeros(K,w);  % Finding centers in original space
for k=1:K
    cols = find(cluster_id==k);
    P1(k,:) = sum(B(:,cols),2)./length(cols);
end

% Uncomment following line to write clusting info
% doc_ids = zeros(d,1) - 1;
% doc_ids(retained_docs) = cluster_id; %removed docs are -1
% dlmwrite(strcat(outpath,'/P1'),full(P1),'delimiter',' ','precision','%.6f');
% dlmwrite(strcat(outpath,'/doc_cluster_id'),full(doc_ids),'\n');
% dlmwrite(strcat(outpath,'/clusterID'),doc_ids,'\n');
% clear doc_ids

% filename1 = strcat(outpath,'/P1.mat');   % For debugging
%         save(filename1,'P1');
%         filename1 = strcat(outpath,'/B.mat');   % For debugging
%         save(filename1,'B');



%% Lloyds on B with start from B_k
fprintf('Performing Lloyds on B with centers from B_k clustering\n')
tic;
[~, cluster_id2, ~,q2,~] = kmeans_fast(B,P1,2,0);
clear B cluster_id
% fprintf('Quality:%.4f\n',q2);
toc;

%code added by Jagdeep (This portion takes care of orphan topics i.e it ensures that each topic has atleast one document)
%start
     
%         filename1 = strcat(outpath,'/cluster_id2.mat');   % For debugging
%         save(filename1,'cluster_id2');
% 
%         for adjust_topic=1:K
%             topic_count2(adjust_topic)=sum(cluster_id2==adjust_topic);
%         end
%         orphan_topics=find(topic_count2==0);
%         [~,max_top]=max(topic_count2);
%         
%       doc_id2=1;
%         for all_count=1:length(orphan_topics)
%             
%             found1=0;
%             while(~found1)
%                 if cluster_id2(doc_id2)==max_top
%                     found1=1;
%                     cluster_id2(doc_id2)=orphan_topics(all_count);
%                 end
%                 
%                 doc_id2=doc_id2+1;
%             end
%         end

  cluster_id2 = help_empty_topics(cluster_id2,K,d);   
%end



if w*d > 1e8, load(strcat(outpath, '/A.mat')); end
%A = A*spdiags(1./sum(A,1)',0,d,d); %normalized A
A1_rowsum = full(sum(A,2));

P2 = zeros(w,K);    % this will be topic matrix without using cathwords
for k=1:K
    cols = retained_docs(cluster_id2==k);
    P2(:,k) = sum(A(:,cols),2)./length(cols);
end

% Uncomment following two lines to write clustering info
% tic
dlmwrite(strcat(outpath,'/P2'),full(P2'),'delimiter',' ','precision','%.6f');
dlmwrite(strcat(outpath,'/clusterID2'),cluster_id2,'\n');
% 
% fprintf('Time take to write P2 and clusterID2 : %f secs',toc);
%%
% fprintf('Performing Lloyds on B with centers from B_k clustering\n')
% 
% numIters = 20;
% % numSamples = (d/2000)*k; % number of docs sampled per iteration
% numSamples = 30*k; % number of docs sampled per iteration
% probs_i = thres/sum(thres);
% 
% if w*d > 1e8, load(strcat(outpath, '/A.mat')); end
% A = A*spdiags(1./sum(A,1)',0,d,d); %normalized A
% A1_rowsum = full(sum(A,2));
% 
% P2 = zeros(w,K);    % this will be topic matrix from first clustering
% for k=1:K
%     cols = retained_docs(cluster_id==k);
%     P2(:,k) = sum(A(:,cols),2)./length(cols);
% end
% dlmwrite(strcat(outpath,'/P1_A'),full(P2'),'delimiter',' ','precision','%.6f');
% clear P2
% 
% centers = P1;
% tic;
% A = A';
% for iter = 1:numIters
%     samples_i = randsample(w,numSamples,true,probs_i);
%     samples_j = zeros(numSamples,1);
%     remaining_j = 1:d;
%     for r=1:numSamples
%         i = samples_i(r);
%         u = rand;
%         j_id = find(u*sum(A(remaining_j,i)) <= cumsum(A(remaining_j,i)),1);
%         j = remaining_j(j_id);
%         samples_j(r) = j;
%         remaining_j(j_id) = [];
%     end
%     [centers, cluster_id2, ~,q2,~] = kmeans_fast2(B(:,samples_j),centers,1,1,1); %take one step of Lloyds
%     fprintf('Iteration %d, %.3f\n',iter,q2);
% end
% A = A';
% toc;
% dlmwrite(strcat(outpath,'/centers'),full(centers'),'delimiter',' ','precision','%.6f');
% [centers, cluster_id2, ~,q2,~] = kmeans_fast2(B,centers,1,1,1); %take one step of Lloyds
% fprintf('Iteration Final, %.3f\n',q2);
% dlmwrite(strcat(outpath,'/centers_new'),full(centers'),'delimiter',' ','precision','%.6f');
% P2 = zeros(w,K);    % this will be topic matrix without using cathwords
% for k=1:K
%     cols = retained_docs(cluster_id2==k);
%     P2(:,k) = sum(A(:,cols),2)./length(cols);
% end
% dlmwrite(strcat(outpath,'/P2'),full(P2'),'delimiter',' ','precision','%.6f');
% dlmwrite(strcat(outpath,'/clusterID2'),cluster_id2,'\n');

% fprintf('Finding Catchwords\n');
% [M_hat,catchword] = find_catchword(A,cluster_id2,retained_docs,eps2,eps3,w0,rho,P2);
% dlmwrite(strcat(outpath,'/M_hat_new'),full(M_hat'),'delimiter',' ','precision','%.6f');
% 
% 
% tic;
% [centers, cluster_id2, ~,q2,~] = kmeans_fast2(B,P1,2,-1,1);
% % clear B cluster_id
% fprintf('Quality:%.4f\n',q2);
% toc;
% dlmwrite(strcat(outpath,'/centers_old'),full(centers'),'delimiter',' ','precision','%.6f');
% 
% P2 = zeros(w,K);    % this will be topic matrix without using cathwords
% for k=1:K
%     cols = retained_docs(cluster_id2==k);
%     P2(:,k) = sum(A(:,cols),2)./length(cols);
% end
% 
% % Uncomment following two lines to write clustering info
% dlmwrite(strcat(outpath,'/P2'),full(P2'),'delimiter',' ','precision','%.6f');
% dlmwrite(strcat(outpath,'/clusterID2'),cluster_id2,'\n');
% 

%% Find Catchwords
fprintf('Finding Catchwords\n');
tic;
fractiles = zeros(w,K); % This will store the values g(i,l)

for l=1:K
    if (sum(cluster_id2==l)==0)
        fprintf('There is a topic for which no documents are present');
    end
    T = A(:,retained_docs(cluster_id2==l))'; %columns of T are words
    
    
    % sorting on columns is faster for sparse matrix
    T = sort(T,1,'descend'); % sort cols in descending
%     fractiles(:,l) = T(min(floor(eps2*w0*d/2),size(T,1)),:);   % was written by Trapit
    fractiles(:,l) = T(min(max(1,floor(eps2*w0*d/2)),size(T,1)),:); % edited by Jagdeep
end
clear T;

catchword = false(w,K);
for l =1:K
    for i=1:w
        catchword(i,l) = false;
        fractile_1 = fractiles(i,l);
        isanchor = false;
        for l2 = 1:K
            if (l2==l), continue; end
            fractile_2 = fractiles(i,l2);
            isanchor = (fractile_1 > rho*fractile_2);
            if ~isanchor
                break
            end
        end
        if isanchor
            catchword(i,l)  = true;
        end
    end
end

catchy_topics = find(sum(catchword,1)~=0);

catchless = setdiff(1:K,catchy_topics);
for l=1:K
    % check that frequency over catchwords is not very small
    if (~ismember(l,catchless) && sum(A1_rowsum(catchword(:,l))) <= 0.001*d/(2*K))
        catchless = horzcat(catchless,l);
    end
end

if ~isempty(catchless)
    fprintf('Catchless topics: ');
    fprintf('%d ',catchless); fprintf('\n');
end

% Uncomment the following line to write the catchords indicator matrix
% tic;
dlmwrite(strcat(outpath,'/catchwords'),full(catchword),'delimiter',' ');
% fprintf('Time take to write catchwords : %f secs',toc);
M_hat = zeros(w,K);
for l=1:K
    if ismember(l,catchless)
        M_hat(:,l) = P2(:,l);
        continue;
    end
    n = max(floor(eps3*w0*d/2),1); % new - 08/14
    [~,inds1]=sort(sum(A(catchword(:,l),:),1),'descend');
    alpha1 = inds1(1:n);
    M_hat(:,l) = sum(A(:,alpha1),2)*1.0/n;
end

toc;

end_time = toc(start_time);

fprintf('Writing topic matrix\n');
% tic;
dlmwrite(strcat(outpath,'/M_hat'),full(M_hat'),'delimiter',' ','precision','%.6f');
% fprintf('Time take to write threshold : %f secs',toc);

fprintf('\nAll Done, algorithm took %.2f seconds\n', end_time);
end
