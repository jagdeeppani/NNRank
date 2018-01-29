d = 500;       %Number of words or the dimension
n = 500;      %Number of documents or data points
k=10;           %Number of topics or latent factors
etta0= 0.4 ;
etta2= 1/(2*k);
beta = 0.1; % Take 0 to 0.5 with 0.1 increment
normalize=  2;
c = 10;
m=50;

%s_arr = [25,50,100,150,200,250,500];
%gamma_arr = linspace(0.1,0.9,5);
%edgeThreshold_arr = [10,20,30,40,50];
%tolerance_arr = [4,8,10,13,40,80];

%s_arr = [5,10,20,40,100,200,500];
%gamma_arr = [0.1,0.3,0.5,0.7,0.9];
%edgeThreshold_arr = [10,30,50,75,100];
%tolerance_arr = 2* (n*ones(1,7))./s_arr;

%s_arr = [20,40,100,200,250,500];
%gamma_arr = [0.1,0.3,0.5,0.7,0.9];
%edgeThreshold_arr = [30,50,75,100];
%tolerance_arr = 2* (n*ones(1,length(s_arr)))./s_arr;

s_arr = [15,20,30,40,50,60,70,80,90,100];
gamma_arr = [0.1,0.3,0.5,0.7,0.9];
edgeThreshold_arr = [5,10,20,30,50,75];
tolerance_arr = 2* (n*ones(1,length(s_arr)))./s_arr;
tolerance_arr = [66,50,30,20,10] 

%Test section
%s_arr = [];
%gamma_arr = [0.1,0.2];
%edgeThreshold_arr = [5,10];
%tolerance_arr = 2* (n*ones(1,length(s_arr)))./s_arr;

fprintf('Total number of parameter configs is %d\n', length(s_arr)*length(gamma_arr)*length(edgeThreshold_arr)*length(tolerance_arr));
num_samples = 10;
allConfig = zeros(num_samples,7);
j=1;
for sample_idx =1:num_samples
    rankMat = zeros(length(s_arr),length(gamma_arr));
    %[A,A_orig,B,C,permute_vect] = generate_synthetic(d,n,k,c,etta0,etta2,beta,normalize);
    %[A,A_orig,B,C,permute_vect] = generate_dominant_multinomial(d,n,k,c,etta0,etta2,m);
    [A_orig, A, W_orig, M_orig, anchor_indices_true,pi1] = synthetic_data_mizutani(m,n, k,beta);
    rankA = rank(A);
    rankA_orig = rank(A_orig);
    A(A<0) = 0;
    for s_idx=1:length(s_arr)
        s = s_arr(s_idx);
        if s>n
            continue;
        end
        tolerance=2*(n/s);
        for gamma_idx=1:length(gamma_arr)
            gamma=gamma_arr(gamma_idx);
            for tolerance_idx=1:length(tolerance_arr)
                tolerance=tolerance_arr(tolerance_idx);
                %tolerance=2*(n/s);
                for edgeThreshold_idx=1:length(edgeThreshold_arr)
                    edgeThreshold=edgeThreshold_arr(edgeThreshold_idx);

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
