function [anchor_indices]=pw_spa(A,~,k)

disp('svd started\n');
[U,D,~] = svds(sparse(A),k);
disp('svd done\n');

A= (D\U')*A;
anchor_indices = FastSepNMF(A,k,0);

end
