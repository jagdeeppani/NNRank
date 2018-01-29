% The function Un-normalized TSVD
function [M,W] = utsvd_nmf(A,k)
outpath = '/home/jagdeep/Desktop/NMF/Experiments/tsvd-code-birch-v2/tsvd_cplex/output/ClusteringDocs';
mkdir(outpath);

[M,~] = TSVD(A',outpath,k);
disp('M found');
W = solvelp(A',M); % solve lp or nnls, which is better
disp('W found');
end
