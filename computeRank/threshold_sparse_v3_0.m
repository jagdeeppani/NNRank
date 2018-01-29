function [D, zeta1] = threshold_sparse_v3_0(A, eps1, alpha1, beta1,tolerance,outpath)
% Assuming input will be (d*w), that is transposed, so A will not be copied
% 0117: Does not require count vector, works on real matrix A
% The input is assumed to be a doc x word matrix

fprintf('Thresholding started \n');

addpath /home/jagdeep/Dropbox/sparsesubaccess/sparsesubaccess;
addpath /home/local/ANT/pjagdeep/Downloads/sparsesubaccess/sparsesubaccess;
subpath=sprintf('/time_taken_%f_%f_%f_%d.mat',eps1, alpha1, beta1,tolerance);
opath=strcat(outpath,subpath);

rtime=tic;

% A=A'; % After this operation A becomes a word x doc matrix

[d,w] = size(A); % d is the number of docs and w is the dictionary size

B = sort(A,1,'descend');
tmp11 = ceil(eps1*d);
nu1 = B (ceil(eps1*d),: );

nu_idx = find(nu1==0);

for i=1:length(nu_idx)
    tmp = B(:,nu_idx(i));
    
    min_tmp = min( nonzeros(tmp) );     % We assume each word is present in atleast one doc, so min_temp will not be empty.
%     if length(min_tmp)<1
%         save('/home/jagdeep/Desktop/debug/file.mat','min_tmp','nu1','nu_idx','i','A','B','tmp11');
%     end

    nu1( nu_idx(i) ) = min_tmp;
end

zeta1=  alpha1*nu1; % elements of nu1 can be zero.
clear B;

fprintf('Time taken to find threshold is %f \n',toc(rtime));
nzA=nnz(A);
nzD = 0;
fprintf('NNZ of A is %d \n',nzA);

id_cols = zeros(nzA, 1);
id_rows = zeros(nzA, 1);
values = zeros(nzA, 1);


for i=1:w

        ridx = find(A(:,i) >= zeta1(i));
        lg=length(ridx);
        
        id_rows(nzD+1:nzD+lg) = ridx;
        id_cols(nzD+1:nzD+lg) = i*ones(lg,1);
        values(nzD+1:nzD+lg) = sqrt(zeta1(i))*ones(lg,1);
%         values(nzD+1:nzD+lg) = 1*ones(lg,1);
        nzD = nzD + lg;
        
end

D = sparse(id_rows(1:nzD),id_cols(1:nzD),values(1:nzD),d,w);
clear id_rows id_cols values;
fprintf('Time taken to find initial D is %f \n',toc(rtime));
initial_nnzD = nnz(D);
fprintf('NNZ of initial D is %d \n',initial_nnzD);

W = cell(1,w);
dpw = zeros(w,1);
for i=1:w
    W{i} = find(D(:,i));       % D is asuumed to be doc x word matrix
    dpw(i) = length(W{i});   % # of docs per word i
end
fprintf('Time taken to find W is %f \n',toc(rtime));
[dpw_val,dpw_idx] = sort(dpw,'ascend');

dpw_id = find(dpw_val,1);

R = dpw_idx(dpw_id:end);

% Q = randperm(d);
% Q = Q(1:1000);       % We may tune this number 500
% Q=sort(Q);



% for l1=1:length(R)-1
l1=1;
lc=0;
Li_ln = zeros(w,1);
lR=length(R);
ub_t=beta1*d;

id_cols = zeros(initial_nnzD, 1);
id_rows = zeros(initial_nnzD, 1);
nzD =0;

fprintf('while loop started with lR = %d \n',lR);


while l1 <= lR-1
    i = R(l1);
    Li = W{i};
%   Li=intersect(Wi,Q);
%     Li = Wi;
    Li_ln(i) = length(Li);
    l2=l1+1;
    
    ub = dpw(i)+ub_t;
%   fprintf('Li length is %d \n',Li_ln(i));
    if (Li_ln(i) > 2*tolerance)
        while (l2 <= lR)
            id = R(l2);
            
            if  (dpw(id) >= ub)
                Wid = W{id};
                if sum(~ismember(Li, Wid))<=tolerance
                    
                    rest_docs = setdiff(Wid,Li);
                    lg=length(rest_docs);
                    
                    id_rows(nzD+1:nzD+lg) = rest_docs;
                    id_cols(nzD+1:nzD+lg) = id*ones(lg,1);
                    nzD = nzD + lg;
                    
                    R(l2) = [];
                    lR=lR-1;
                    l2=l2-1; % As the size of R decreases auto increment by for loop may miss the next word. so a decerement has been done
                    lc=lc+1;
                    fprintf('lc is %d  and l1 = %d, Li_ln(i) = %d \n',lc,l1,Li_ln(i))
                end
                
%                 fprintf('Inside inequality condition\n');
            end
            
            l2=l2+1;
%          fprintf('l2 is %d \n',l2)   
        end
        
    end
    l1=l1+1;
end
fprintf('Finished at : l1 is %d Li = %d \n',l1-1,Li_ln(i));
D = setsparse(D, id_rows(1:nzD), id_cols(1:nzD), zeros(nzD,1));

final_nnzD = nnz(D);
fprintf('NNZ of final D is %d \n',final_nnzD);
% save('/media/BM/Jagdeep/NMF/cplex/output_clustering/Li_length.mat','Li_ln');
% save('/media/BM/Jagdeep/NMF/cplex/output_clustering/W.mat','W');
fprintf('lc is %d\n',lc);
time_taken=toc(rtime);
fprintf('Time taken by threshold_sparse_v3 is %f secs \n',time_taken);

% save(opath,'time_taken','W','Li_ln','initial_nnzD','final_nnzD','Q','zeta1','D');
save(opath,'time_taken','W','Li_ln','initial_nnzD','final_nnzD','zeta1','D');

D=D';


end



