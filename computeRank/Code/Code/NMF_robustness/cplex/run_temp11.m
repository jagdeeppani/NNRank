alt_iter=6;
iter_time=tic;
            M=nnlsHALSupdt(A',W');
            M=M';
            W=nnlsHALSupdt(A,M);
            disp('residual calculation started');
            
            Residual_norm_L2(alg_idx,alt_iter+1)=norm(A-M*W,'fro');
            disp('residual calculation done');
            
            svm_time=tic;
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
            Time_taken(alg_idx,alt_iter+1)=toc(iter_time)+Time_taken(alg_idx,alt_iter)-temp_time;
            temp_time=toc(svm_time);