clear all;
close all;

load Reuters21578.mat;
addpath /home/jagdeep/Desktop/NMF/Experiments/LIBLINEAR/liblinear-1.96/matlab;

% Taking only first 10 classes
fea = fea(1:8258,:); % 7285 x 18933
gnd = gnd(1:8258);      % 7285 x 1
%ind = trainIdx<7286;    
%trainIdx = trainIdx(ind);
%testIdx = testIdx(testIdx<7286);

fea(:,find(sum(fea,1)<1))=[]; % Removing orphan words (not present in any document) 

% five percent docs are in training set and rest are test set
% unique_gnd=unique(gnd);
% trainIdx_subset=[];
% testIdx_subset=[];
% for i=1:length(unique_gnd)
%     Idx_subset= find(gnd==unique_gnd(i));
%     p=randperm(length(Idx_subset));
%     Idx_subset = Idx_subset(p);
%     trainIdx_subset = union( trainIdx_subset, Idx_subset(1:ceil(0.05*length(Idx_subset))));
%     testIdx_subset = union(testIdx_subset,setdiff(Idx_subset,trainIdx_subset) );
% end
% trainIdx = trainIdx_subset; % describes which docs are in train set
% testIdx = testIdx_subset;   % % describes which docs are in test set
% 
load train_test_indices.mat
% Train data and Test data
trainf = fea(trainIdx,:);
trainl = gnd(trainIdx,:);
testf = fea(testIdx,:);
testl = gnd(testIdx,:);





%no_of_features=[200 250 300 350 400 450 500];
no_of_features = [200];

funs = {
    %@spa,...
    %@fast_hull,...
    %@ER_spa_svds,...
    %@ER_xray_svds,...
    };

    % Descriptive names for the algorithms
%     funs_str = {
%     'SPA',...
%     'XRAY',...
%     %'ER-SPA',...
%     %'ER-XRAY',...
%     'UTSVD',...
%     };

    funs_str = {
    %'SPA',...
    %'XRAY',...
    'UTSVD',...
    };

num_algs=length(funs);

c_arr=[0.1,0.5,1,10,16,32];


for c_count=1:length(c_arr)
    
svmoptions = sprintf( '-c %s -e 0.001 -v 4 -q',num2str(c_arr(c_count)) );    
temp_acc_val(c_count) = train(trainl, trainf, svmoptions);     % 5 cross validation , tolerance for termination criterion
%[~, accuracy_full_feature, ~] = predict(testl, testf, model, '-b 0');
end
[acc_maxval,maxind_c]=max(temp_acc_val);
%temp_acc_val(c_count)
% svmoptions = strcat( '-c ',num2str(c_arr(maxind_c)),' -e 0.001');    
svmoptions = sprintf( '-c %s -e 0.001 -q',num2str(c_arr(maxind_c)) ); 
model = train(trainl, trainf, svmoptions); 
[~, accuracy_full_feature, ~] = predict(testl, testf, model, '-b 0');

fprintf('svm on full features is done');
disp(accuracy_full_feature(1));
clear temp_acc_val model;
%clear model;

%model = svmtrain(trainl,trainf,'-s 0 -t 0 -c 0.1 -e 0.001 -wi 1');
%[~, accuracy, prob_estimates] = svmpredict(testl, testf, model,'-b 0');

%acc(1:length(no_of_features),1)=accuracy(1);
noise_level=0.3;
A=fea';
max_alt_min_iter = 10;
Time_taken = zeros(num_algs+1,length(no_of_features));
Residual_norm = zeros(num_algs+1,max_alt_min_iter);
for feature_count=1:length(no_of_features)
    k=no_of_features(feature_count);
    anchor_indices=zeros(num_algs,k);
    accuracy=zeros(num_algs+2,max_alt_min_iter);		% UTSVD and All feature methods are out of this set, so 2 is added
    
    outpath11 = sprintf('/home/jagdeep/Desktop/NMF/Experiments/tsvd-code-birch-v2/tsvd_cplex/output_classification/reuters/feature_%d',no_of_features(feature_count));
    mkdir(outpath11);
    
    for alg_idx=1:num_algs+1
        tic;
        if strcmp(funs_str{alg_idx},'UTSVD')
            %outpath = '/home/jagdeep/Desktop/NMF/Experiments/tsvd-code-birch-v2/tsvd_cplex/output_classification/reuters';
            %mkdir(outpath);
            [M,~] = TSVD(A,outpath11,k);
            disp('M found');
            W=nnlsHALSupdt(A,M);
            Residual_norm(alg_idx,1)= norm(A- M*W,'fro');
            
        else
            fun=funs{alg_idx};
            anchor_indices(alg_idx,:) = reshape(fun(fea,noise_level,k),1,k); 
            fprintf('%s anchor_indices found\n',funs_str{alg_idx});
            W = A(anchor_indices(alg_idx,:),:);
        end
        
            train_anchor_f = (W(:,trainIdx))';
            test_anchor_f = (W(:,testIdx))';
    
            for c_count=1:length(c_arr)
                svmoptions = sprintf( '-c %s -e 0.001 -v 4 -q',num2str(c_arr(c_count)) );    
                temp_acc_val(c_count) = train(trainl, sparse(train_anchor_f), svmoptions);     % 5 cross validation , tolerance for termination criterion
            end
            
            [acc_maxval,maxind_c]=max(temp_acc_val);
            %temp_acc_val(c_count)
            % svmoptions = strcat( '-c ',num2str(c_arr(maxind_c)),' -e 0.001');    
            svmoptions = sprintf( '-c %s -e 0.001 -q',num2str(c_arr(maxind_c)) ); 
            model = train(trainl, sparse(train_anchor_f), svmoptions); 
            [~, temp_accuracy, ~] = predict(testl, sparse(test_anchor_f), model, '-b 0');
            accuracy(alg_idx,1)=temp_accuracy(1);
            fprintf('SVM with c= %d completed for %s with  %d alt_min iterations\n',c_arr(maxind_c),funs_str{alg_idx},0);
	        clear model;   
        
        
        
        
        for alt_iter=1:max_alt_min_iter
            M=nnlsHALSupdt(A',W');
            M=M';
            W=nnlsHALSupdt(A,M);
            disp('residual calculation started');
            tic
            Residual_norm(alg_idx,alt_iter+1)=norm(A-M*W,'fro');
            disp('residual calculation done');
            toc
            train_anchor_f = (W(:,trainIdx))';
            test_anchor_f = (W(:,testIdx))';
    
            for c_count=1:length(c_arr)

                svmoptions = sprintf( '-c %s -e 0.001 -v 4 -q',num2str(c_arr(c_count)) );    
                temp_acc_val(c_count) = train(trainl, sparse(train_anchor_f), svmoptions);     % 5 cross validation , tolerance for termination criterion
            end
            
            [acc_maxval,maxind_c]=max(temp_acc_val);
            %temp_acc_val(c_count)
            % svmoptions = strcat( '-c ',num2str(c_arr(maxind_c)),' -e 0.001');    
            svmoptions = sprintf( '-c %s -e 0.001 -q',num2str(c_arr(maxind_c)) ); 
            model = train(trainl, sparse(train_anchor_f), svmoptions); 
            [~, temp_accuracy, ~] = predict(testl, sparse(test_anchor_f), model, '-b 0');
            accuracy(alg_idx,alt_iter+1)=temp_accuracy(1);
            fprintf('SVM with c= %d completed for %s with  %d alt_min iterations\n',c_arr(maxind_c),funs_str{alg_idx},alt_iter);
	        clear model;   

        end
        Time_taken(alg_idx,feature_count)=toc;
        
    end
    accuracy(num_algs+2,:)=accuracy_full_feature(1);
    
    
    fname1 = strcat(outpath11,'/svm_accuracy.mat');
    save(fname1,'accuracy');
    fname2 = strcat(outpath11,'/anchor_indices.mat');
    save(fname2,'anchor_indices');
    fname3 = strcat(outpath11,'/Time_taken.mat');
    save(fname3,'Time_taken');
    
end
    fname4 = strcat(outpath11,'/train_test_indices.mat');
    save (fname4,'trainIdx','testIdx');


%     plot(no_of_features,acc);
%     grid on;
%     title('Classification experiment on Reuters')
%     xlabel('No of features ')
%     ylabel('Accuracy')
%     legend('Full features','XRAY','SPA')







