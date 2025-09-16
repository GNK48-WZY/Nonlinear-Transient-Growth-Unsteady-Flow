% addpath(genpath('/home/zhw22003/YALMIP-master'))
% system('export PATH=$PATH:/home/zhw22003/mosek/11.0/tools/platform/linux64x86/bin')
% addpath(genpath('/home/zhw22003/mosek'))

% clear; clc;
% FD parameters
N  = 5;       % total FD grid points including boundaries
L  = 2;        % domain length [-1,1]
Re = 500;

% Spectral ranges (as in paper)
kx_list = logspace(-4, 0.48, 5);
kz_list = logspace(-2, 1.2, 5);

% delta_list = logspace(-6, 0, 1);
% T = 200;
% t0 = 0;
% dt = 0.1;
% t_steps = (T-t0)/dt;
results = cell(length(kx_list), length(kz_list));
delete(gcp('nocreate'));
% parpool(4);

%% Main loop
for i_kx = 1:length(kx_list)
    [local_results] = Inside_loop(N, L, Re, kz_list, i_kx, kx_list);
    results(i_kx,:) = local_results;
end

disp('successful')
save('mu_full_scan_results.mat','results','kx_list','kz_list','delta_list')

function [local_results]= Inside_loop(N, L, Re, kz_list, i_kx, kx_list)
     local_results = cell(1, length(kz_list));
     for j_kz = 1:length(kz_list)

        % Build operator for this (kx,kz)
        kx = kx_list(i_kx);
        kz = kz_list(j_kz);
        operator = build_operators_fd_and_LMI(N, L, Re, kx, kz);

            local_results{1, j_kz} = struct( ...
                'kx', kx, ...
                'kz', kz, ...
                'mu_LMI', operator.mu_LMI, ...
                'P', value(operator.P), ...
                'sx', value(operator.sx), ...
                'sy', value(operator.sy), ...
                'sz', value(operator.sz), ...
                'max_trans', operator.max_trans);        
    end
end


function operator = build_operators_fd_and_LMI(N, L, Re, kx, kz)

syms y t t2 n;
ttt1=0; %TIME START POINT
T=200;
dt=1;
t0=[0, 20, 40, 60, 80, 100];
t0=20;
[x, Dy, Dyy, D4, w] = finitediff(N, L);
Ny = length(x);   % interior grid points

h=eye(size(Dy));
h1=eye(size(Dy));
k2=(kx.^2+kz.^2)*h;
k=(kx.^2+kz.^2).^0.5*h1;
w1=w.^0.5;
W=diag(w1);
d=Dy+k;
WW=W*d;

%v
V=[WW,zeros(N-2,N-2);zeros(N-2,N-2),W]; %126*126
c=Dyy-k2;


%U
Cn=0;
NN=100;
k0 = 0.1;
Re= 500;
g=1-exp(-k0.*t); %Acc
g=exp(-k0.*t); %Dec
an = (pi*n).^2./Re;  %lately
m=diff(g,t);
a1=subs(m,t,0);
m2=diff(m,t);
a2=subs(m2,t,t2);
e=exp(an.*t2);
a3=simplify(e*a2);
a4=simplify(int(a3, t2, 0, t),'Steps',200);
a5=a4+a1;
aa=-2.*Re.*(-1).^n./(pi.*n).^3.*(a5)+Cn;
a6=simplify(aa.*exp(-an.*t).*sin(n.*y.*pi),'Steps',100);
b1=Re./6.*m.*(y.^3-y);
b2=g.*y;
bb=b1+b2;
a7 = 0;
for ii = 1:NN
    a7 = a7 + subs(a6, n, ii);
end
U=a7+bb;

mm=diff(U);
mm2=diff(mm);
% U_yi = (subs(U, y, yi(2:N)));
% m_yi = (subs(mm, y, yi(2:N)));
% m2_yi = (subs(mm2, y, yi(2:N)));

U_yi = (subs(U, y, x));
m_yi = (subs(mm, y, x));
m2_yi = (subs(mm2, y, x));
U_yi_function = matlabFunction(U_yi);
m_yi_function = matlabFunction(m_yi);
m2_yi_function = matlabFunction(m2_yi);


for j3 = 1:length(t0)
    %A
    t_steps = (T-t0(j3))/dt;
    A = zeros(2*(N-2), 2*(N-2), t_steps);%
    %U_value
    G = zeros(1, t_steps);
    AA = eye(size(V));
    for i = 1:t_steps  %step
        t_value=t0(j3)+i*dt;
%         U_yi_value = diag(U_yi_function(t_value));% t from t0 to t0+T
%         m_yi_value = diag(m_yi_function(t_value));
%         m2_yi_value = diag(m2_yi_function(t_value));
        k2 = kx^2 + kz^2;
        % Laplacians
        Lap  = Dyy - k2*eye(Ny);                     % ∇^2
        Lap2 = D4 - 2*k2*Dyy + (k2^2)*eye(Ny);       % ∇^4
        
        % A operator
        A11 = -1i.*kx.*diag(U_yi_function(t_value)).*Lap + 1i.*kx.*diag(m2_yi_function(t_value)) + (1/Re).*Lap2;
        A12 = zeros(Ny);
        A21 = -1i.*kz.*diag(m_yi_function(t_value));
        A22 = -1i.*kx.*diag(U_yi_function(t_value)) + (1/Re).*Lap;
        
        LHS = blkdiag(Lap, eye(Ny));
        RHS = [A11 A12; A21 A22];
        operator.A = LHS \ RHS;
        A(:, :, i) = L; 
        AA = expm(-1i*operator.A*dt)*AA;
        gh1=double(V*AA/V);
        G(i)=norm(gh1)^2;
    end

    G_storage{j3} = G;
    operator.max_trans=max(G);


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

    Ny = size(operator.B,2)/3;
    n  = size(operator.A,1);
   %% LMI computation
    yalmip('clear');

             %% time-varying
   for j1=1:t_steps
       P{j1}=sdpvar(n,n,'hermitian','complex');
       sx{j1} = sdpvar(1,1); sy{j1} = sdpvar(1,1); sz{j1} = sdpvar(1,1);

   end
%    sx = sdpvar(1,1); sy = sdpvar(1,1); sz = sdpvar(1,1);
%    gamma_H_inf_complex2 = sdpvar(1,1);
%    tt = sdpvar(1);
tt = 1;
   I = eye(n);
   scaling = 1;
   E=I*scaling;
   operator.A_new=operator.A*scaling;
   Gamma = sdpvar(1,1);
   F = [];
   for j1=1:t_steps-1
       M = blkdiag(sx{j1}*eye(Ny), sy{j1}*eye(Ny), sz{j1}*eye(Ny));
       dP_dt = (P{j1 + 1}-P{j1})/dt;
       dV_ineq = [dP_dt + operator.A_new'*P{j1}*E + E'*P{j1}*operator.A_new ...
                        + sx{j1}*operator.C_grad_u'*operator.C_grad_u ...
                        + sy{j1}*operator.C_grad_v'*operator.C_grad_v ...
                        + sz{j1}*operator.C_grad_w'*operator.C_grad_w, ...
                        P{j1}*operator.B;
                        operator.B'*P{j1}, -Gamma*M];     
        F = [F, P{j1} - tt*eye(n) >= 0, dV_ineq <= 0, sx{j1} >= 0, sy{j1} >= 0, sz{j1} >= 0];
%         F = [I <= P{j1} <= tt*I,  dV_ineq <= 0, sx >= 0, sy >= 0, sz >= 0];
   end
   F=[F];

    sdp_options = sdpsettings('solver','mosek');
     bisection(F,Gamma,sdp_options);
%     if diagnostics.problem == 0
        operator.P_optimal = value(P);
        operator.t_optimal = value(tt);
        operator.P = P;
        operator.sx =sx;
        operator.sy = sy;
        operator.sz = sz;
  
end

% 
% k2 = kx^2 + kz^2;
% % Laplacians
% Lap  = Dyy - k2*eye(Ny);                     % ∇^2
% Lap2 = D4 - 2*k2*Dyy + (k2^2)*eye(Ny);       % ∇^4
% 
% % A operator
% A11 = -1i*kx*diag(U_yi)*Lap + 1i*kx*diag(m2_yi) + (1/Re)*Lap2;
% A12 = zeros(Ny);
% A21 = -1i*kz*diag(m_yi);
% A22 = -1i*kx*diag(U_yi) + (1/Re)*Lap;
% 
% LHS = blkdiag(Lap, eye(Ny));
% RHS = [A11 A12; A21 A22];
% operator.A = LHS \ RHS;

% B operator


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
