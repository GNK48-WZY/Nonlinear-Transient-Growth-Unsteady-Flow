% addpath(genpath('/home/zhw22003/YALMIP-master'))
% system('export PATH=$PATH:/home/zhw22003/mosek/11.0/tools/platform/linux64x86/bin')
% addpath(genpath('/home/zhw22003/mosek'))

% clear; clc;
% FD parameters
N  = 5;       % total FD grid points including boundaries
L  = 2;        % domain length [-1,1]
Re = 358;
flowType = 'couette';

% Spectral ranges (as in paper)
kx_list = logspace(-4, 0.48, 5);
kz_list = logspace(-2, 1.2, 5);

% Delta list (can be more values if needed)
delta_list = logspace(-6, 0, 1);
% T = 200;
% t0 = 0;
% dt = 0.1;
% t_steps = (T-t0)/dt;
results = cell(length(kx_list), length(kz_list), length(delta_list));
delete(gcp('nocreate'));
% parpool(4);

%% Main loop
for i_kx = 1:length(kx_list)
    [local_results] = Inside_loop(N, L, Re, kz_list, flowType, delta_list, i_kx, kx_list);
    results(i_kx,:) = local_results;
end

%% Save everything
disp('successful')
save('mu_full_scan_results.mat','results','kx_list','kz_list','delta_list')

function [local_results]= Inside_loop(N, L, Re, kz_list, flowType, delta_list,i_kx, kx_list)
     local_results = cell(1, length(kz_list));
     for j_kz = 1:length(kz_list)

        % Build operator for this (kx,kz)
        kx = kx_list(i_kx);
        kz = kz_list(j_kz);
        operator = build_operators_fd(N, L, Re, kx, kz, flowType);

        Ny = size(operator.B,2)/3;
        n  = size(operator.A,1);

%         for ind_delta = 1:length(delta_list)
%             delta = delta_list(ind_delta);

            %% LMI computation
            yalmip('clear');

             %% time-varying
%             for j1=1:t_steps
%                    P{j1}=sdpvar(n,n,'hermitian','complex');
%             end
%             sx = sdpvar(1,1); sy = sdpvar(1,1); sz = sdpvar(1,1);
%             gamma_H_inf_complex2 = sdpvar(1,1);
%             M = blkdiag(sx*eye(Ny), sy*eye(Ny), sz*eye(Ny));
% 
%             for j1=1:t_steps-1
%                 dP_dt = (P{j1 + 1}-P{j1})/dt;
%                 dV_ineq = [dP_dt + operator.A'*P{j1} + P{j1}*operator.A ...
%                         + sx*operator.C_grad_u'*operator.C_grad_u ...
%                         + sy*operator.C_grad_v'*operator.C_grad_v ...
%                         + sz*operator.C_grad_w'*operator.C_grad_w, ...
%                         P{j1}*operator.B;
%                         operator.B'*P{j1}, -gamma_H_inf_complex2*M];     
%                 F = [P{j1} - 1e-3*eye(n) >= 0, dV_ineq <= 0, sx >= 0, sy >= 0, sz >= 0];    
%             end
%             F=[F];

            %% time-independent
            P = sdpvar(n,n,'hermitian','complex');
            sx = sdpvar(1,1); sy = sdpvar(1,1); sz = sdpvar(1,1);
            gamma_H_inf_complex2 = sdpvar(1,1);

            M = blkdiag(sx*eye(Ny), sy*eye(Ny), sz*eye(Ny));

            dV_ineq = [operator.A'*P + P*operator.A ...
                        + sx*operator.C_grad_u'*operator.C_grad_u ...
                        + sy*operator.C_grad_v'*operator.C_grad_v ...
                        + sz*operator.C_grad_w'*operator.C_grad_w, ...
                        P*operator.B;
                        operator.B'*P, -gamma_H_inf_complex2*M];

            F = [P - 1e-3*eye(n) >= 0, dV_ineq <= 0, sx >= 0, sy >= 0, sz >= 0];

            sdp_options = sdpsettings('solver','mosek','verbose',0);
            bisection(F,gamma_H_inf_complex2,sdp_options);

            mu_LMI = sqrt(value(gamma_H_inf_complex2));

            %% === Mussv computation (pos/neg freq) ===
            C_all = [operator.C_grad_u;
             operator.C_grad_v;
             operator.C_grad_w];

            BlockStructure=[Ny,3*Ny;
            Ny,3*Ny;
            Ny,3*Ny];

            sys_grad = ss(operator.A, operator.B, C_all, zeros(size(C_all,1), size(operator.B,2)));
            [mu_frd_H_inf_grad, mu_info] = mussv(sys_grad, BlockStructure, 'Ufs');
            [mu_upper_H_inf_grad, max_ind] = max(squeeze(mu_frd_H_inf_grad(1).ResponseData));
            mu_upper_freq = mu_frd_H_inf_grad(1).Frequency(max_ind);
            % Negative frequency μ-analysis
            K_operator_discretized = eye(size(C_all,1));   % placeholder
            sys_grad_neg_freq = ss(-operator.A, -operator.B, K_operator_discretized*C_all, ...
                              zeros(size(C_all,1), size(operator.B,2)));
            [mu_frd_H_inf_grad_neg_freq, mu_info_neg] = mussv(sys_grad_neg_freq, BlockStructure, 'Ufs');
            [mu_upper_H_inf_grad_neg_freq, max_ind]   = max(squeeze(mu_frd_H_inf_grad_neg_freq(1).ResponseData));
            mu_upper_freq_neg_freq = -mu_frd_H_inf_grad_neg_freq(1).Frequency(max_ind);

            local_results{1, j_kz} = struct( ...
                'kx', kx, ...
                'kz', kz, ...
                'mu_LMI', mu_LMI, ...
                'P', value(P), ...
                'sx', value(sx), ...
                'sy', value(sy), ...
                'sz', value(sz), ...
                'mu_upper_H_inf_grad', mu_upper_H_inf_grad, ...
                'mu_upper_freq', mu_upper_freq, ...
                'mu_upper_H_inf_grad_neg_freq', mu_upper_H_inf_grad_neg_freq, ...
                'mu_upper_freq_neg_freq', mu_upper_freq_neg_freq ...
            );
%             local_results{i_kx,j_kz,ind_delta}.kx = kx;
%             local_results{i_kx,j_kz,ind_delta}.kz = kz;
%             local_results{i_kx,j_kz,ind_delta}.delta = delta;
% 
%             local_results{i_kx,j_kz,ind_delta}.mu_LMI   = mu_LMI;
%             local_results{i_kx,j_kz,ind_delta}.P        = value(P);
%             local_results{i_kx,j_kz,ind_delta}.sx       = value(sx);
%             local_results{i_kx,j_kz,ind_delta}.sy       = value(sy);
%             local_results{i_kx,j_kz,ind_delta}.sz       = value(sz);
% 
%             local_results{i_kx,j_kz,ind_delta}.mu_upper_H_inf_grad = mu_upper_H_inf_grad;
%             local_results{i_kx,j_kz,ind_delta}.mu_upper_freq    = mu_upper_freq;
%             local_results{i_kx,j_kz,ind_delta}.mu_upper_H_inf_grad_neg_freq = mu_upper_H_inf_grad_neg_freq;
%             local_results{i_kx,j_kz,ind_delta}.mu_upper_freq_neg_freq = mu_upper_freq_neg_freq;
        
    end
end
function operator = build_operators_fd(N, L, Re, kx, kz, flowType)
% Build state-space operators (A, B, C, C_grad_u/v/w)
% using finite-difference differentiation matrices.
%
% Inputs:
%   N        : total grid points (including boundaries)
%   L        : domain length in wall-normal direction
%   Re       : Reynolds number
%   kx, kz   : streamwise and spanwise wavenumbers
%   flowType : 'Couette' or 'Poiseuille'
%
% Output:
%   operator struct with fields:
%       A, B, C, C_grad_u, C_grad_v, C_grad_w

% ---- get finite-difference matrices ----
[x, Dy, Dyy, D4, ~] = finitediff(N, L);
Ny = length(x);   % interior grid points

% ---- base flow and derivatives ----
switch lower(flowType)
    case 'couette'
        U   = x;                % U(y) = y
        Uy  = ones(size(x));    % U'(y) = 1
        Uyy = zeros(size(x));   % U''(y) = 0
    case 'poiseuille'
        U   = 1 - x.^2;         % U(y) = 1 - y^2
        Uy  = -2*x;             % U'(y) = -2y
        Uyy = -2*ones(size(x)); % U''(y) = -2
    otherwise
        error('flowType must be ''Couette'' or ''Poiseuille''');
end


k2 = kx^2 + kz^2;
% Laplacians
Lap  = Dyy - k2*eye(Ny);                     % ∇^2
Lap2 = D4 - 2*k2*Dyy + (k2^2)*eye(Ny);       % ∇^4

% A operator
A11 = -1i*kx*diag(U)*Lap + 1i*kx*diag(Uyy) + (1/Re)*Lap2;
A12 = zeros(Ny);
A21 = -1i*kz*diag(Uy);
A22 = -1i*kx*diag(U) + (1/Re)*Lap;

LHS = blkdiag(Lap, eye(Ny));
RHS = [A11 A12; A21 A22];
operator.A = LHS \ RHS;

% B operator
Bmat = [ -1i*kx*Dy     -(k2)*eye(Ny)   -1i*kz*Dy;
          1i*kz*eye(Ny)  zeros(Ny)     -1i*kx*eye(Ny) ];
operator.B = LHS \ Bmat;

% C operator
Cmat = 1/k2 * [ 1i*kx*Dy   -1i*kz*eye(Ny);
                 k2*eye(Ny) zeros(Ny);
                 1i*kz*Dy   1i*kx*eye(Ny) ];
operator.C = Cmat;

% ---- build gradient-output operators ----
Grad_block = [1i*kx*eye(Ny); Dy; 1i*kz*eye(Ny)];
Grad_big   = blkdiag(Grad_block, Grad_block, Grad_block);

C_grad = Grad_big * operator.C;  % (9*Ny) × (2*Ny)

n = size(operator.C, 2); % = 2*Ny
operator.C_grad_u = C_grad(1:3*Ny, 1:n);
operator.C_grad_v = C_grad(3*Ny+1:6*Ny, 1:n);
operator.C_grad_w = C_grad(6*Ny+1:9*Ny, 1:n);

end



function [x, D1, D2, D4,w] =finitediff(N,L)

%Differentiation matrix using finite difference scheme.
%This is suitable for Dirichlet boundary condition v(x=L/2)=v(x=-L/2)=0 at
%the boundary and Neumann boundary condition v'(x=L/2)=v'(x=-L/2)=0. 

dx=L/N;%get grid spacing. 

x=linspace(-L/2,L/2,N);%get the grid point location

x=x(2:end-1); %delete the first and last points that are zero. 

w=dx*ones(1,N-2);%integration weighting 

N_diff=N-2;%The size of differentiation matrices

%First order derivative based on central difference
%f'(x_i)=(f(x_{i+1})-f(x_{i-1})/(2*dx)
%We also use the boundary condition that x_0=x_N=0
D1_stencil=diag(-1*ones(1,N_diff-1),1)+diag(ones(1,N_diff-1),-1);
D1=D1_stencil/(2*dx);

%Second order derivative based on central difference
%f''(x_i)=(f(x_{i+1})-2f(x_i)+f(x_{i-1})/(dx^2)
%We also use the boundary condition that x_0=x_N=0
D2_stencil=(diag(-2*ones(1,N_diff))+diag(ones(1,N_diff-1),1)+diag(ones(1,N_diff-1),-1));
D2=D2_stencil/dx^2;

%Forth order derivative based on central difference
%f''''(x_i)=(f(x_{i+2})-4f(x_{i+1})+6f(x_i)-4f(x_{i-1})+f(x_{i-2})/(dx^4)
%This differentiation matrix only go through x_1 up to x_{N-1}
D4_stencil=(diag(6*ones(1,N_diff))+diag(-4*ones(1,N_diff-1),1)+diag(-4*ones(1,N_diff-1),-1)...
    + diag(ones(1,N_diff-2),2)+diag(ones(1,N_diff-2),-2));

%Here, we use the Neumann boundary condition that x_0'=x_N'=0
%such that x_1=x_{-1} and x_{N-1}=x_{N+1} for the ghost points. Then we
%also use the condition x_0 and x_N=0 to express all values based on x_1 up
%to x_{N-1}
D4_stencil(1,1)=D4_stencil(1,1)+1;
D4_stencil(end,end)=D4_stencil(end,end)+1;

D4=D4_stencil/dx^4;
end
