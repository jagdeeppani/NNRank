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

trainIdx_mat=zeros(50,435);
testIdx_mat=zeros(50,7823);
c_arr=[0.1,0.5,1,10,16,32,64,128,256,512];
accuracy_mat = zeros(50,length(c_arr));


for iter_s=1:50

unique_gnd=unique(gnd);
trainIdx_subset=[];
testIdx_subset=[];
for i=1:length(unique_gnd)
    Idx_subset= find(gnd==unique_gnd(i));
    p=randperm(length(Idx_subset));
    Idx_subset = Idx_subset(p);
    trainIdx_subset = union( trainIdx_subset, Idx_subset(1:ceil(0.05*length(Idx_subset))));
    testIdx_subset = union(testIdx_subset,setdiff(Idx_subset,trainIdx_subset) );
end
trainIdx = trainIdx_subset; % describes which docs are in train set
testIdx = testIdx_subset;   % % describes which docs are in test set


% Train data and Test data
trainf = fea(trainIdx,:);
trainl = gnd(trainIdx,:);
testf = fea(testIdx,:);
testl = gnd(testIdx,:);



for c_count=1:length(c_arr)
    
svmoptions = sprintf( '-c %s -e 0.001 -v 4 -q',num2str(c_arr(c_count)) );    
accuracy_mat(iter_s,c_count) = train(trainl, trainf, svmoptions);     % 5 cross validation , tolerance for termination criterion
%[~, accuracy_full_feature, ~] = predict(testl, testf, model, '-b 0');


end
%[acc_maxval,maxind_c]=max(temp_acc_val);
%temp_acc_val(c_count)
% svmoptions = strcat( '-c ',num2str(c_arr(maxind_c)),' -e 0.001');    
% svmoptions = sprintf( '-c %s -e 0.001 -q',num2str(c_arr(maxind_c)) ); 
% model = train(trainl, trainf, svmoptions); 
% [~, accuracy_full_feature, ~] = predict(testl, testf, model, '-b 0');

fprintf('svm on full features is done\n');

trainIdx_mat(iter_s,:)=trainIdx';
testIdx_mat(iter_s,:)=testIdx';
%clear temp_acc_val model;


end

