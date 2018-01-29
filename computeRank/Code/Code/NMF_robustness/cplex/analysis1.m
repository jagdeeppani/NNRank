 M_orig= dlmread('M');
 M_orig = M_orig';
[m,k]=size(M_orig);
cm=zeros(m,k);
cm_n=zeros(m,k);
idx=zeros(m,k);
for i=1:k
    cm(:,i)=M_orig(:,i);
    [val, ind] = sort(cm(:,i),'descend');
    idx(:,i)=ind;
    cm(:,i)= val;
    cm_n(:,i)=val./(sum(val));
end

unq_vec=unique(idx_vec);


% cm4=M_orig(:,4);
% cm4=sort(cm4,'descend');
% cm4_n=cm4./(sum(cm4));