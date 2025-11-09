function [S, F, obj_val, acc_array1] = sgr_p1(K, alpha, opts)
% P1: \min ||x_i- x_j||S_ij + alpha*||s_i - s_j||W_ij 
% s.t. Se = e, S = S', S>=0, diag(S) = 0, rank(L_S) = n-c;
 
F = opts.F_init;
S = opts.S_init;
FFt = F*F';
L_W = opts.L_W;

n = opts.n_samples;
labels = opts.labels;

iter = 1;
max_Iter = 100;
stop_iter = 20;

tol1 = max(1e-3, 1/n)
lambda = 15;

acc_array1 = zeros(1, max_Iter);
predy = SpectralClustering(S, opts.n_clusters);
curr_result1 = Clustering8Measure(labels, predy);
acc_array1(iter) = max(curr_result1(1));



[obj, ~] = compute_obj(K, S, L_W, FFt, alpha, lambda, opts);
obj_val(iter) = obj;


while iter < max_Iter
    iter = iter + 1;

    [S, S_residual] = solve_S(K, S, L_W, FFt, alpha, lambda, opts);

    [F, lambda, bflag] = solve_F(S, F, lambda,  opts);
    FFt = F*F';

    predy = SpectralClustering(S, opts.n_clusters);
    curr_result1 = Clustering8Measure(labels, predy);
    acc_array1(iter) = (curr_result1(1));

    [obj, ~] = compute_obj(K, S, L_W, FFt, alpha, lambda, opts);
    obj_val(iter) = obj;

    obj_residual_t = abs(obj_val(iter) - obj_val(iter - 1))/abs(obj_val(iter));
    obj_residual(iter) = obj_residual_t;

    if iter > stop_iter
        if obj_residual(iter) <  tol1 && bflag
            obj_val = obj_val(1:iter);
            acc_array1 = acc_array1(1:iter);
            if bflag
               fprintf('the %d -th iteration -> end ...\n',iter)
            end
            break;
        end   
    end
end
end




function [S, S_residual] = solve_S(K, S_pre, L_W, FFt, alpha, lambda, opts)
n = opts.n_samples;
Y = zeros(n, n);
U = S_pre;
mu = 30;
rho = 1.5;
S = Y;
N = Y;

tol = 1e-6; 
iter = 0;
max_iter = 100;

while iter < max_iter

    iter = iter + 1;

    for i = 1 : n
        index = opts.S_init(:, i)>0;
        Ui = U(index, i);
        Yi = Y(index, i);
        L_Wi = L_W(index, index);
        m = Ui - (Yi + alpha*L_Wi*Ui - K(index, i) - lambda*FFt(index,i))/mu;
        S(index,i) = EProjSimplex_new(m);
        Si = S(index,i);

        N(index, i) = Si + (Yi - alpha*L_Wi'*Si)/mu;
    end
   
    U = N + N';
    U = U/2;
    U = U - diag(diag(U));

    Y = Y + mu*(S - U);
    mu = rho*mu;

    S_residual(iter) = norm(S - S_pre, "fro");
    
    if iter > 1 && S_residual(iter) < tol
        S_residual = S_residual(1:iter);
        break;
    else
        S_pre = S;
    end

end

end


function  [F, lambda, bflag] = solve_F(S, F, lambda,  opts)
    bflag = false;
    c = opts.n_clusters;
    zr = 1e-10;
    F_old = F;

    S = S + S';
    S = 0.5*S;
    L_S = diag(sum(S,2)) - S;
    [F, ev] = eigs(L_S, c+1, "smallestreal");
    F = real(F(:, 1:c));
    
    ev = diag(ev);
    ev = abs(ev);

    fn1 = abs(sum(ev(1:c)));
    fn2 = abs(sum(ev(1:c+1)));
    t = 2;
    if fn1 > zr
        lambda = t*lambda;
    elseif fn2 < zr
        lambda = lambda/t;  
        F = F_old;
    else
        bflag = true;
    end 
end



function [obj, obj1, obj2] = compute_obj(K, S, L_W, FFt, alpha, lambda, opts)
    L_S = eye(opts.n_samples) - S;

    obj1 = trace(K*L_S) + alpha * (trace(S'*L_W*S)) ;
    obj2 = lambda*trace(FFt*L_S);
    obj = obj1 + obj2;
end