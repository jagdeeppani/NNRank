k=300;
singularVals = zeros(k,5);

%Reuters with 10 classes
load Reuters7285_new_preprocessed.mat;
A=fea';
clear fea;
[d,n] = size(A);
fprintf('Sparsity(in %%): %d\n',(nnz(A)*100)/(d*n));

fprintf('Computing Singular values of Reuters10\n');
tic;
%k=200;           %Number of topics or latent factors
singularVals(:,1) = svds(A,k);
toc;
fprintf('sprank of Reuters10 : %d\n',sprank(A));
%csvwrite('s.csv',s);




%Reuters with 48 classes
load Reuters_preprocessed_new.mat
A=fea';
clear fea;
[d,n] = size(A);
fprintf('Sparsity(in %%): %d\n',(nnz(A)*100)/(d*n));

fprintf('Computing Singular values of Reuters48\n');
tic;
%k=200;           %Number of topics or latent factors
singularVals(:,2) = svds(A,k);
toc;
fprintf('sprank of Reuters48 : %d\n',sprank(A));
%csvwrite('s.csv',s);



%TDT2 with 30 classes
load TDT2_preprocessed_new.mat;
A=fea';
clear fea;
[d,n] = size(A);
fprintf('Sparsity(in %%): %d\n',(nnz(A)*100)/(d*n));
fprintf('Computing Singular values of TDT2\n');
tic;
%k=200;           %Number of topics or latent factors
singularVals(:,3) = svds(A,k);
toc;
fprintf('sprank of TDT2 : %d\n',sprank(A));
%csvwrite('s.csv',s);




%20News groups with 20 classes 
load Twentyng_fea_preprocessed_new.mat;
A=fea';
clear fea;
[d,n] = size(A);
fprintf('Sparsity(in %%): %d\n',(nnz(A)*100)/(d*n));

fprintf('Computing Singular values of 20ng\n');
tic;
%k=200;           %Number of topics or latent factors
singularVals(:,4) = svds(A,k);
toc;
fprintf('sprank of 20ng : %d\n',sprank(A));
%csvwrite('s.csv',s);


%yale with 15 classes
load yale32_A_4096.mat;
[d,n] = size(A);
fprintf('Sparsity(in %%): %d\n',(nnz(A)*100)/(d*n));

fprintf('Computing Singular values of yale faceD\n');
tic;
%k=200;           %Number of topics or latent factors
singularVals(:,5) = svds(A,k);
toc;
fprintf('sprank of yale faceD : %d\n',sprank(A));
%csvwrite('s.csv',s);

csvwrite('singularValues.csv',singularVals);
