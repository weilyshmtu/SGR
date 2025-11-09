clear
% clc
close all
addpath('funs\')
addpath('metric\')
addpath('algs\')
addpath('datasets\')

dataset = ["zoo","ORL_64x64","JAFFE_64x64","COIL20_32x32", "COIL100_32x32","MNIST_10k_Train_28x28"];

alpha_set = [5e-5, 0.0001, 0.0005, 0.001, 0.005, 0.01];
beta_set =  [5e-5, 0.0001, 0.0005, 0.001, 0.005, 0.01];
curr_fold = pwd;
rng(0)
for d_ind = 1 : length(dataset)
    data = dataset(d_ind);
    load(data)
    fprintf("Experiments on %s dataset.\n", data)
    
    X  = fea;
    labels = gnd;
    if min(gnd) == 0
        labels = labels + 1;
    end
    [n_samples, d] = size(X);

    % normalize data if required
    flag = 2; 
    X = data_normalize(X, flag,  n_samples);
    K = X*X';
    

    % options
    opts = [];
    opts.n_samples = n_samples;
    opts.d = d;
    opts.n_clusters = max(unique(labels));
    opts.k = 40;
    opts.labels = labels
    S_pre = constructW_PKN(X, opts.k, 1);
    opts.S_init = S_pre;
    W = S_pre;
    opts.L_W = diag(sum(W, 2)) - W;
    [F, ~] = eigs(S_pre, opts.n_clusters, "largestreal");
    opts.F_init = real(F);
    

    for a_ind = 1 : length(alpha_set)
        for b_ind = 1: length(beta_set)
            alpha = 1 * alpha_set(a_ind);
            beta = 1 * beta_set(b_ind);
            tic
            
            flag = 3;
            if flag == 1
                 [S, C, F, obj_val, acc_array1, acc_array2] = sgr(K, alpha, beta, opts);
            end
           
            if flag ==  2
                 [S, F, obj_val, acc_array1] = sgr_p1(K, alpha, opts);
            end
            
            if flag == 3
                 [S, C, F, obj_val, acc_array1, acc_array2] = sgr_p2(K, alpha, beta, opts);
            end
            time = toc

            predy = SpectralClustering(S, opts.n_clusters);
            curr_result1 = Clustering8Measure(labels, predy);

            acc1(a_ind, b_ind) = curr_result1(1)
            nmi1(a_ind, b_ind) = curr_result1(2)
            ari1(a_ind, b_ind) = curr_result1(3);
            fscore1(a_ind, b_ind) = curr_result1(4);
            purity1(a_ind, b_ind) = curr_result1(5);
            precision1(a_ind, b_ind) = curr_result1(6);
            recall1(a_ind, b_ind) = curr_result1(7);
            obj_array{a_ind, b_ind} = obj_val;
            acc_array1_cell{a_ind, b_ind} = acc_array1;
           
           
            if exist("C")
                predy = SpectralClustering(C, opts.n_clusters);
                curr_result2 = Clustering8Measure(labels, predy);
                acc2(a_ind, b_ind) = curr_result2(1)
                nmi2(a_ind, b_ind) = curr_result2(2)
                ari2(a_ind, b_ind) = curr_result2(3);
                fscore2(a_ind, b_ind) = curr_result2(4);
                purity2(a_ind, b_ind) = curr_result2(5);
                precision2(a_ind, b_ind) = curr_result2(6);
                recall2(a_ind, b_ind) = curr_result2(7);
            end
    
        end
        figure
        for c_ind = 1 : length(beta_set)
            subplot(1, length(beta_set), c_ind)
            plot( 1:length(obj_array{a_ind, c_ind}), obj_array{a_ind, c_ind})
        end

        figure
        for c_ind = 1 : length(beta_set)
            subplot(1, length(beta_set), c_ind)
            plot( 1:length(acc_array1_cell{a_ind, c_ind}), acc_array1_cell{a_ind, c_ind})
        end
        drawnow
       
    end

    
    

end

function normX = data_normalize(X, flag, n_samples)
% z-score
if flag == 1
    for  j = 1:n_samples
        normItem = std(X(j,:));
        if (0 == normItem)
            normItem = eps;
        end
        normX(j,:) = (X(j,:) - mean(X(j,:)))/normItem;
    end
end
% unit l2-norm
if flag == 2
    tX = X;
    for j=1:n_samples
        tX(j,:) = tX(j,:)/(eps + norm(tX(j,:),2));
    end
    normX = tX;
end 
% 0-1
if flag == 3
    normX = X./(max(max(X))+eps);
end

if flag == 0
    normX = X;
end


end