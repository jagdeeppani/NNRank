clear all;
close all;


% The problem size of the synthetic data set
m = 100;   % m is the vocab size
n = 50;    % n is the number of documents
k = 10;      % number of topics

noise_arr = linspace(0.01,1,40);
d_arr = [0.01 0.25 1.0];
scaling = 1.0;   % Different scaling of the columns of Mtilde.
normalize = 1.0; % Different normalization of the Noise.
for type=1:2
    for density_idx=1:length(d_arr)
        density = d_arr(density_idx);
        for noise_idx=1:length(noise_arr)
            noise_level = noise_arr(noise_idx);
            %type = 2;        % 1=middlepoint (requires n >= r+r(r-1)/2), 2=Dirichlet.
            
            
            %density = d_arr(density_idx);   % Proportion of nonzero entries in the noise matrix (but at least one nonzero per column).
            
            for rpt=1:100
                [A_orig, A, W_orig, M_orig, ~, anchor_indices_true] = synthetic_data_gillis_lp(n, m, k,noise_level, type, scaling, density, normalize);
                
                if min(min(A_orig))<0
                    fprintf('-ve elements found in A_orig with setup, noise: %f, type: %d, density: %f, rpt: %d \n',noise_level,type,density,rpt);
                    break;
                end
                if min(min(A))<0
                    fprintf('-ve elements found in A with setup, type: %d, density: %f, noise: %f, rpt: %d \n',type,density,noise_level,rpt);
                    break;
                end
                
            end
            
        end
    end
end