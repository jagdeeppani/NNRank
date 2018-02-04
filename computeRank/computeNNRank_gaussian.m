
%Parameter initializations
d = 5000;        %Number of words or the dimension
n = 10000;       %Number of documents or data points
k = 100;         %Number of topics or latent factors
eta0 = 0.05;     %Total mass of catchwords in topic
eta2 = 1/(1*k);  %Dirichlet parameter for C (in A~BC+N)
c = 20;          %number of catchwords per topic
s_arr = linspace(400,400,1);
gamma_arr = linspace(0.4,0.4,1);
edgeThreshold_arr = [3];
tolerance_arr = [2];
noise_arr=linspace(0,0.15,4);   %Computing rank for each configuration

%normalize = 0 implies element-wise Gaussian noise with sigma = 2*beta*sqrt(n)*B_il.
%normalize = 2 implies column-wise Gaussian noise with l2 norm comparable to data.
normalize = 0;


allNoise = cell(1,length(noise_arr));
fprintf('Total number of parameter configs is %d\n', length(s_arr)*length(gamma_arr)*length(edgeThreshold_arr)*length(tolerance_arr));
num_samples = 1;

for noise_idx=1:length(noise_arr)
    allConfig = zeros(num_samples,7);
    beta=noise_arr(noise_idx);
    j=1;
    for sample_idx =1:num_samples
        rankMat = zeros(length(s_arr),length(gamma_arr));
        [A,A_orig,B,C,permute_vect] = generate_synthetic_with_head(d,n,k,c,eta0,eta2,beta,normalize);
        %[A,A_orig,B,C,permute_vect] = generate_dominant_multinomial(d,n,k,c,etta0,etta2,m);
        for s_idx=1:length(s_arr)
            s = s_arr(s_idx);
            if s>n
                continue;
            end
            for gamma_idx=1:length(gamma_arr)
                gamma=gamma_arr(gamma_idx);
                for tolerance_idx=1:length(tolerance_arr)
                    %tolerance=tolerance_arr(tolerance_idx);
                    tolerance=tolerance_arr(tolerance_idx)*n/s;
                    for edgeThreshold_idx=1:length(edgeThreshold_arr)
                        %edgeThreshold=edgeThreshold_arr(edgeThreshold_idx);
                        edgeThreshold=edgeThreshold_arr(edgeThreshold_idx)*(n/s);
                        [lc, k1] = computeNNRank(A, gamma, s,tolerance,edgeThreshold);
                        rankMat(s_idx,gamma_idx) = k1;
                        fprintf('Done k:%d, k_out:%d, s:%.0f, beta:%.4f, gamma:%.2f, eta2:%.3f, tol:%.2f, edgeThr:%.2f, lc:%d\n',k,k1,s,beta,gamma,eta2,tolerance,edgeThreshold,lc);   
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
%fprintf('median k=%d\n',median(allConfig(:,6)));
end


%Appendix code
%bestConfig = allConfig(allConfig(:,6)<=k+2 & allConfig(:,6)>=k-2,:);
bestConfig = allConfig(allConfig(:,2)==300,6)
csvwrite('allConfig_gaussian_1k_10k_20_0.1.csv',allConfig);
csvwrite('bestConfig_gaussian_1k_10k_20_0.1.csv',bestConfig);

for i=1:length(noise_arr)
res(i)=median(allNoise{i}(:,6));
end

res=0;
l=1;
for i =1:length(s_arr)
res(l) = median(allConfig(allConfig(:,2)==s_arr(i),6));
res_std(l) = std(allConfig(allConfig(:,2)==s_arr(i),6));
l=l+1;
end

for i =1:length(noise_arr)
fprintf('At noise %f , %d +- %f\n is ',noise_arr(i),median(allNoise{i}(:,6)),std(allNoise{i}(:,6)) );
end
