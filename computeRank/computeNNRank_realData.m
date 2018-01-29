%--------------------------------
% Experiment with real data
%--------------------------------
load Reuters7285_new_preprocessed.mat;

k=10;           %Number of topics or latent factors
beta = 0.2; % Take 0 to 0.5 with 0.1 increment
normalize=  2;

%clearvars -except X Y;
%fea = X;
%gnd = Y;
%clear X Y;

A=fea';
[d,n] = size(A);
%s_arr = [15,20,30,40,50,60,70,80,90,100];
%gamma_arr = [0.1,0.3,0.5,0.7,0.9];
%edgeThreshold_arr = [5,10,20,30,50,75];
%tolerance_arr = 2* (n*ones(1,length(s_arr)))./s_arr;
%tolerance_arr = [66,50,30,20,10] 

%Test section
s_arr = [5,10];
gamma_arr = [0.1,0.5,0.9];
edgeThreshold_arr = [100];
tolerance_arr = [150];

fprintf('Total number of parameter configs is %d\n', length(s_arr)*length(gamma_arr)*length(edgeThreshold_arr)*length(tolerance_arr));
num_samples = 1;
allConfig = zeros(num_samples,7);
j=1;
for sample_idx =1:num_samples
    rankMat = zeros(length(s_arr),length(gamma_arr));
    %[A,A_orig,B,C,permute_vect] = generate_synthetic(d,n,k,c,etta0,etta2,beta,normalize);
    %[A,A_orig,B,C,permute_vect] = generate_dominant_multinomial(d,n,k,c,etta0,etta2,m);
    %[A_orig, A, W_orig, M_orig, anchor_indices_true,pi1] = synthetic_data_mizutani(m,n, k,beta);
    %rankA = rank(A);
    %rankA_orig = rank(A_orig);
    %A(A<0) = 0;
    for s_idx=1:length(s_arr)
        s = s_arr(s_idx);
        if s>n
            continue;
        end
        tolerance=2*(n/s);
        for gamma_idx=1:length(gamma_arr)
            gamma=gamma_arr(gamma_idx);
            for tolerance_idx=1:length(tolerance_arr)
                %tolerance=tolerance_arr(tolerance_idx);
                tolerance=2*(n/s);
                for edgeThreshold_idx=1:length(edgeThreshold_arr)
                    %edgeThreshold=edgeThreshold_arr(edgeThreshold_idx);
                    edgeThreshold=3*(n/s);

                    [lc, k1] = computeNNRank(A, gamma, s,tolerance,edgeThreshold);
                    rankMat(s_idx,gamma_idx) = k1;
                    fprintf('done for s=%f, gamma=%f , tolerance =%d, edgeThreshold=%f with lc= %d, k=%d\n',s,gamma,tolerance,edgeThreshold,lc,k1);   
                    %if k1>=(0.9*k) && k1<=(1.1*k)
                    allConfig(j,:) = [sample_idx, s, gamma, tolerance, edgeThreshold, k1,lc ];
                    j=j+1;
                    %end
                end
            end
        end
    end    
end
bestConfig = allConfig(allConfig(:,6)==k,:)
%bestConfig = allConfig(allConfig(:,6)<=k+1 & allConfig(:,6)>=k-1,:)
j
rankMat



%anchor_indices = spa(A', beta, k);
%if anchor_indices==0
%    break;
%end
%W = A(anchor_indices,:);
%M = nnlsHALSupdt(A',W'); % solving for 1 norm error and Rev(Inf,1) error are same. % solve lp or nnls, which is better
%M = M';
