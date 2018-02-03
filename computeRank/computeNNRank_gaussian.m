d = 1000;       %Number of words or the dimension
n = 2000;      %Number of documents or data points
k=20;           %Number of topics or latent factors
etta0= 0.1;     % Total mass of catchwords in topic
etta2= 1/(2*k);
normalize=  0; %normalize =0 implies element-wise Gaussian noise with sigma = 2*beta*sqrt(n)*B_il. normalize =2 implies column-wise Gaussian noise with l2 norm comparable to data.
c = 5;
m=50;


%s_arr = [75,80,85,90,95,100,105,110,115];
%s_arr = [50,100,150,200,250,300,350];
s_arr  = linspace(100,300,3);
gamma_arr = [0.4,0.5,0.6];
edgeThreshold_arr = [5];
tolerance_arr = [5];

noise_arr = [1,3,5] * 0.001118;
allNoise = cell(1,length(noise_arr));
fprintf('Total number of parameter configs is %d\n', length(s_arr)*length(gamma_arr)*length(edgeThreshold_arr)*length(tolerance_arr));
num_samples = 1;

for noise_idx=1:length(noise_arr)
    allConfig = zeros(num_samples,7);
    beta=noise_arr(noise_idx);
    fprintf('Noise is %f\n', beta);
    j=1;
    for sample_idx =1:num_samples
        rankMat = zeros(length(s_arr),length(gamma_arr));
        [A,A_orig,B,C,permute_vect] = generate_synthetic(d,n,k,c,etta0,etta2,beta,normalize);
        %[A,A_orig,B,C,permute_vect] = generate_dominant_multinomial(d,n,k,c,etta0,etta2,m);
        rankA = rank(A);
        rankA_orig = rank(A_orig);
        A(A<0) = 0;
        for s_idx=1:length(s_arr)
            s = s_arr(s_idx);
            if s>n
                continue;
            end
            for gamma_idx=1:length(gamma_arr)
                gamma=gamma_arr(gamma_idx);
                for tolerance_idx=1:length(tolerance_arr)
                    %tolerance=tolerance_arr(tolerance_idx);
                    tolerance=2*n/s;
                    for edgeThreshold_idx=1:length(edgeThreshold_arr)
                        %edgeThreshold=edgeThreshold_arr(edgeThreshold_idx);
                        edgeThreshold=3*(n/s);
                        [lc, k1] = computeNNRank(A, gamma, s,tolerance,edgeThreshold);
                        rankMat(s_idx,gamma_idx) = k1;
                        fprintf('Done s:%.0f, beta:%.4f, gamma:%.2f, tol:%.2f, edgeThr:%.2f, lc:%d, k:%d, k_out:%d\n',s,beta,gamma,tolerance,edgeThreshold,lc,k,k1);   
                        %if k1>=(0.9*k) && k1<=(1.1*k)
                        allConfig(j,:) = [sample_idx, s, gamma, tolerance, edgeThreshold, k1,lc ];
                        j=j+1;
                        %end
                    end
                end
            end
        end    
    end
allNoise{noise_idx}=allConfig;
fprintf('median k=%d\n',median(allConfig(:,6)));
end
bestConfig = allConfig(allConfig(:,6)==k,:)
%bestConfig = allConfig(allConfig(:,6)<=k+1 & allConfig(:,6)>=k-1,:)
bestConfig = allConfig(allConfig(:,2)==300,6)
j
rankMat
csvwrite('bestConfig_gaussian_0.01.csv',bestConfig);

for i=1:length(noise_arr)
res(i)=median(allNoise{i}(:,6));
end

res=0;
l=1;
for i =1:9
res(l) = median(allConfig(allConfig(:,2)==s_arr(i),6));
l=l+1;
end
