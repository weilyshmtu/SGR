function [S, C, F, obj_val, acc_array1, acc_array2] = sgr_p2(K, alpha, beta, opts)
% P2: \min ||x_i- x_j||S_ij + alpha*||s_i - s_j||C_ij + beta*||S - C||_F^2
% s.t. Se = e, S = S', S>=0, diag(S) = 0, rank(L_S) = n-c;
%      Ce = e, C = C', C>=0, diag(C) = 0


labels = opts.labels;

S = opts.S_init;
F = opts.F_init;
FFt = F*F';
n = opts.n_samples;
C = ones(n,n)/n;

iter = 1;
max_iter = 100;

stop_iter = 20;

tol1 = max(1e-3, 1/n)
lambda = 15;

[obj_val(iter),~, obj2]= compute_obj(K, S, C, FFt, alpha, beta, lambda, opts);


acc_array1 = zeros(1, max_iter);
acc_array2 = zeros(1, max_iter);

predy = SpectralClustering(S, opts.n_clusters);
curr_result1 = Clustering8Measure(labels, predy);
acc_array1(iter) = max(curr_result1(1));


predy = SpectralClustering(C, opts.n_clusters);
curr_result2 = Clustering8Measure(labels, predy);
acc_array2(iter) = (curr_result2(1));



while iter < max_iter
    iter = iter + 1;
    
    % update S
    [S, S_res] = solve_S(K, S, C, FFt, alpha, beta, lambda, opts);
  
    % update C
    [C, C_res] = solve_C(S, C, alpha, beta, opts);

    % update F
    [F, lambda, bflag] = solve_F(S, F, lambda, opts);
    FFt = F*F';
    
    % update obj
    [obj_val(iter),~,obj2] = compute_obj(K, S, C, FFt, alpha, beta, lambda, opts);
    % lambda = 1/(sqrt(obj2)+eps);

 
    % 

    obj_residual_t = abs(obj_val(iter) - obj_val(iter - 1))/abs(obj_val(iter));
    obj_residual(iter) = obj_residual_t;
   


    if mod(iter, 10) == 0 
        predy = SpectralClustering(S, opts.n_clusters);
        curr_result1 = Clustering8Measure(labels, predy);
        acc_array1(iter) = (curr_result1(1));


        predy = SpectralClustering(C, opts.n_clusters);
        curr_result2 = Clustering8Measure(labels, predy);
        acc_array2(iter) = (curr_result2(1));
        disp(['iter = ', num2str(iter), ', obj = ', num2str(obj_residual(iter),  '%6.4f'), ...
            ', acc1 = ', num2str(acc_array1(iter), '%6.4f'), ...
            ', acc2 = ', num2str(acc_array2(iter), '%6.4f')]) 
    end
    if iter > stop_iter
        if obj_residual(iter) <  tol1 && bflag
            obj_val = obj_val(1:iter);
            acc_array1 = acc_array1(1:iter);
            acc_array2 = acc_array2(1:iter);

            if bflag
               fprintf('the %d -th iteration -> end ...\n',iter)
            end
            break;
        end
             
    end
    
end
end



function [S, S_residual] = solve_S(K, S_pre, C, FFt, alpha, beta, lambda, opts)
    n = opts.n_samples;
    Y = zeros(n, n);
    N = Y;
    S = Y;
    U = S_pre;
    I = eye(n);
    iter = 0;
    max_iter = 100;

    rho = 1.5;
    mu = 30;
    tol = 1e-6; 
    % 
    % CC = (C + C')/2;
    % L_C = diag(sum(CC, 2)) - CC;
    L_C = I - C;
    S_residual = zeros(1, max_iter);

    while iter < max_iter
        iter = iter + 1;

        % update S
        for i = 1:n
            index = opts.S_init(:, i)>0;
            % index = 1:n
            Ui = U(index, i);
            Yi = Y(index,i) ;
            Ai = alpha*L_C(index,index) + beta*I(index,index);
            Bi = K(index,i) + 2*beta*C(index,i) + lambda*FFt(index, i);
            m = Ui- (Yi + Ai*Ui - Bi)/mu;
     
            S(index,i) = EProjSimplex_new(m);
            Si = S(index,i);
            N(index,i) = Si + Yi/mu - (Ai*Si)/mu;
        end

     
        U = (N + N')/2;
        U = U - diag(diag(U));
       

        % update Y, mu
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

function [C, C_residual] = solve_C(S, C_pre, alpha, beta, opts)
    n = opts.n_samples;

    V = C_pre;
    Y = zeros(n, n);
    C = Y;
    St = S;

    iter = 0;
    max_iter = 100;
    mu = 30;
    rho = 1.5;
    tol = 1e-6; 
    I = eye(n);
    C_residual = zeros(1, max_iter);
    while iter < max_iter
        iter = iter + 1;

        % update C 
        
        for i = 1:n
            index = opts.S_init(:,i)>0;
            % index = 1:n;
            Ai = beta*I(index,index);
            Bi = alpha*(S(index,index)*St(index,i)) + 2*beta*S(index,i);
            Vi = V(index,i);
            Yi = Y(index,i);
            m =  Vi- (Yi + Ai*Vi - Bi)/mu;
            
            C(index,i) = EProjSimplex_new(m);
        end
    
        % update V
        M = C + Y/mu;
        V = (M + M')/2;
        V = V - diag(diag(V));

        % update Y, mu
        Y = Y + mu*(C - V);
        mu = mu*rho;
        
        C_residual(iter) = norm(C - C_pre, "fro");

        if iter > 1 && C_residual(iter) < tol
            C_residual = C_residual(1:iter);
            break;
        else
            C_pre = C;
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

function [obj, obj1, obj2] = compute_obj(K, S, C, FFt, alpha, beta, lambda, opts)
    L_S = eye(opts.n_samples) - S;
    L_C = eye(opts.n_samples) - C;
    obj1 = trace(K*L_S) + alpha * (trace(S'*L_C*S))  + beta*norm(S - C, "fro")^2;
    
    obj2 = lambda*trace(FFt*L_S);
    obj = obj1 + obj2;
    
end


