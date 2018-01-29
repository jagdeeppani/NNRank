
k=5;
noise_level=0.3;
load bbc.mat;
A=fea;

anchor_index_er=ER_spa_svds(A,noise_level,k)
anchor_index_spa=spa(A,noise_level,k)
anchor_index_xray=fast_hull(A,noise_level,k)
M_er=A(:,anchor_index_er);
M_spa=A(:,anchor_index_spa);
M_xray=A(:,anchor_index_xray);
W_er=nnlsHALSupdt(A,M_er);
W_spa=nnlsHALSupdt(A,M_spa);
W_xray=nnlsHALSupdt(A,M_xray);

Residual_er=sumabs(A-(M_er*W_er));
Residual_spa=sumabs(A-(M_spa*W_spa));
Residual_xray=sumabs(A-(M_xray*W_xray));

outpath_tsvd = '/home/jagdeep/Desktop/NMF/Experiments/tsvd-code-birch-v3/tsvd_cplex/test12/tsvd_output';
mkdir(outpath_tsvd);

[M_utsvd,cluster_id] = TSVD(A',outpath_tsvd,k);
W_utsvd=nnlsHALSupdt(A',M_utsvd);
Residual_utsvd=sumabs(A'-(M_utsvd*W_utsvd));

