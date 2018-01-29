function [anchor_indices]=heur_spa(A,~,k)
tic
disp('svd started\n');
%[U,D,~] = svds(sparse(A),k);
[~,~,V] = svds(sparse(A),k);
disp('svd done\n');
toc
tic
%A= (D\U')*A;
A = V';
toc
tic
clear V
disp('Calling SPA part');
anchor_indices = FastSepNMF(A,k,0);
toc
end
