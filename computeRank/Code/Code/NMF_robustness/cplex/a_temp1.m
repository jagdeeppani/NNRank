% five percent docs are in training set and rest are test set
% load bbc.mat;
% clear trainIdx testIdx;
%fea(:,find(sum(fea,1)<5))=[];
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

trainf = fea(trainIdx,:);
trainl = gnd(trainIdx,:);
testf = fea(testIdx,:);
testl = gnd(testIdx,:);

svm_accuracy = liblinear_svm_solver(trainf,trainl,testf,testl)
