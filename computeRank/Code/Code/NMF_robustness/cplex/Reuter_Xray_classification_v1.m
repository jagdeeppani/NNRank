clear all;
close all;

load Reuters21578.mat;

% Taking only first 10 classes
fea = fea(1:7285,:); % 7285 x 18933
gnd = gnd(1:7285);      % 7285 x 1
ind = trainIdx<7286;    
trainIdx = trainIdx(ind);
testIdx = testIdx(testIdx<7286);

% Train data and Test data
trainf = fea(trainIdx,:);
trainl = gnd(trainIdx,:);
testf = fea(testIdx,:);
testl = gnd(testIdx,:);


%no_of_features=[10 20 50 100 150 200 250 300 350 400 450 500];
no_of_features = [100];

funs = {
    @spa, ...
    @fast_hull,...
    @ER_spa_svds,...
    @ER_xray_svds,...
    };

    % Descriptive names for the algorithms
    funs_str = {
    'SPA',...
    'XRAY',...
    'ER-SPA',...
    'ER-XRAY',...
    'UTSVD',...
    };


num_algs=length(funs);
accuracy=zeros(num_algs+2,length(no_of_features));		% UTSVD and All feature methods are out of this set, so 2 is added

model = train(trainl, trainf, '-v 5 -e 0.001');     % 5 cross validation , tolerance for termination criterion
[~, accuracy_full_feature, ~] = predict(testl, testf, model, '-b 0');
accuracy_full_feature
%clear model;

%model = svmtrain(trainl,trainf,'-s 0 -t 0 -c 0.1 -e 0.001 -wi 1');
%[~, accuracy, prob_estimates] = svmpredict(testl, testf, model,'-b 0');

%acc(1:length(no_of_features),1)=accuracy(1);
noise_level=0.3;
    
for feature_count=1:length(no_of_features)
    k=no_of_features(count);
    
    for alg_idx=1:num_algs
	fun=funs(alg_idx);
    	anchor_index(alg_idx,:) = fun(fea,noise_level,k); 
    	fprintf('%s anchor_index found\n',funs_str{alg_idx});

    	train_anchor_f = trainf(:, anchor_index(alg_idx,:));
    	test_anchor_f = testf(:,anchor_index(alg_idx,:));
    
    	model = train (trainl, train_anchor_f, '-v 5 -e 0.001');
    	[~, temp_accuracy, ~] = predict(testl, testf_xray, model, '-b 0');
	accuracy(alg_idx,feature_count)=temp_accuracy(1);
	    	

	%model = svmtrain(trainl,trainf_xray,'-s 0 -t 0 -c 0.1 -e 0.001 -wi 1');
    	%[~, accuracy_xray, prob_estimates_xray] = svmpredict(testl, testf_xray, model,'-b 0');

	outpath = '/home/jagdeep/Desktop/NMF/Experiments/tsvd-code-birch-v2/tsvd_cplex/output_clustering/TwentyNG';
    	mkdir(outpath);
   	tic
    	[M,~] = TSVD(A,outpath,k);
    	disp('M found');    	

	model = train (trainl, trainf_spa, '-v 5 -e 0.001');
    	[~, accuracy_spa, ~] = predict(testl, testf_spa, model, '-b 0');
    
    %model = svmtrain(trainl,trainf_spa,'-s 0 -t 0 -c 0.1 -e 0.001 -wi 1');
    %[~, accuracy_spa, prob_estimates_spa] = svmpredict(testl, testf_spa, model,'-b 0');


    acc(count,2:3)=[accuracy_xray(1) accuracy_spa(1)];
    disp(r);
end

    plot(no_of_features,acc);
    grid on;
    title('Classification experiment on Reuters')
    xlabel('No of features ')
    ylabel('Accuracy')
    legend('Full features','XRAY','SPA')
