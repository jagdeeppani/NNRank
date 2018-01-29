function accuracy=liblinear_svm_solver(trainf,trainl,testf,testl)

c_arr=[0.1,0.5,1,10,16,32];

temp_acc_val=zeros(1,length(c_arr));
for c_count=1:length(c_arr)
    
    svmoptions = sprintf( '-c %s -e 0.001 -v 4 -q',num2str(c_arr(c_count)) );
    temp_acc_val(c_count) = train(trainl, sparse(trainf), svmoptions);     % 5 cross validation , tolerance for termination criterion
    %[~, accuracy_full_feature, ~] = predict(testl, testf, model, '-b 0');
end
[~,maxind_c]=max(temp_acc_val);
%temp_acc_val(c_count)
% svmoptions = strcat( '-c ',num2str(c_arr(maxind_c)),' -e 0.001');
svmoptions = sprintf( '-c %s -e 0.001 -q',num2str(c_arr(maxind_c)) );
model = train(trainl, sparse(trainf), svmoptions);
[~, accuracy_full_feature, ~] = predict(testl, sparse(testf), model, '-b 0');

fprintf('svm on full features is done');
disp(accuracy_full_feature(1));
accuracy=accuracy_full_feature(1);

end

