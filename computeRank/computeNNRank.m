function [lc, k] = computeNNRank(A, gamma, s,tolerance,edgeThreshold)

[d,n] = size(A);
A_sorted = sort(A,2,'descend');

%fprintf('Inside the function\n');
%tolerance=2*n/s;
%edgeThreshold=3*n/s;
%tolerance=20;
%edgeThreshold=500;

a = zeros(d,s);
thresh1 = zeros(1,d);
thresh2 = zeros(1,d);
threshQ = zeros(1,d);

interval=int16(floor(n/double(s)));
%fprintf('interval = %d \n',interval);
for j = 1:s
    a(:,j) = sum(A_sorted(:,(j-1)*interval+1:(j*interval)),2);
    b=j;
end    

for word = 1:d
    thresh1(word) = gamma*a(word,1);
    idx = find(a(word,:)<thresh1(word),1)-1;
    if length(idx) >0
        thresh2(word) = idx;
    else thresh2(word) = s;     
    end
    threshQ(word) = A_sorted(word, thresh2(word) * interval);
end
Q_mat = A >= repmat(threshQ.',1,n);

%csvwrite('tmp_A_mat.csv',A);
%csvwrite('tmp_A_sorted_mat.csv',A_sorted);
%csvwrite('tmp_Q_mat.csv',Q_mat);
%csvwrite('tmp_a.csv',a);

W = cell(1,d);
dpw = zeros(d,1);
for i=1:d
    W{i} = find(Q_mat(i,:));       % Q_mat is assumed to be doc x word binary matrix 
    dpw(i) = length(W{i});   % # of docs per word i
end

[dpw_val,dpw_idx] = sort(dpw,'ascend');
dpw_id = find(dpw_val,1);           % Removing zero rows

R = dpw_idx(dpw_id:end);

w=d;
l1=1;
lc=0; % Counter to track the number of words being removed from R
Li_ln = zeros(w,1);
lR=length(R);
ub_t=0;
qi_lowerlim = tolerance;
%fprintf('while loop started with lR = %d \n',lR);

% Step3 pruning starts from here.
while l1 <= lR-1
    i = R(l1);
    Li = W{i};
    Li_ln(i) = length(Li);
    l2=l1+1;
    ub = dpw(i)+ub_t;
   %fprintf('Li length is %d \n',Li_ln(i));
    if (Li_ln(i) > qi_lowerlim)
        while (l2 <= lR)
            id = R(l2);
            
            if  (dpw(id) >= ub)
                Wid = W{id};
                if sum(~ismember(Li, Wid))<=tolerance
                    
                    rest_docs = setdiff(Wid,Li);
                    lg=length(rest_docs);
                                        
                    R(l2) = [];
                    lR=lR-1;
                    l2=l2-1; % As the size of R decreases auto increment by for loop may miss the next word. so a decerement has been done
                    lc=lc+1;
                    %fprintf('lc is %d  and l1 = %d, Li_ln(i) = %d \n',lc,l1,Li_ln(i))
                end
                
%                 %fprintf('Inside inequality condition\n');
            end
            
            l2=l2+1;
%          %fprintf('l2 is %d \n',l2)   
        end
        
    end
    l1=l1+1;
end


%csvwrite('tsvdOutput/R_vec.csv',R);
% The new binary data matrix is A_filtered. This is expected to have fewer rows than the original matrix A.
A_filtered = double(Q_mat(R,:));
%singularVals=zeros(100,2);
%singularVals(:,1) = svds(A,100);
%singularVals(:,2) = svds(A_filtered,100);
%csvwrite('singularVals_Qmat.csv',singularVals);

intersectionMat = A_filtered * A_filtered';
adjacencyMat = intersectionMat >= edgeThreshold;
G = graph(adjacencyMat);
bins = conncomp(G);
k=bins(end);
%fprintf('Number of connected components is: %d\n', k);
