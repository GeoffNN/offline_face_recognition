function labels = soft_hfs(X, Y, c_l, c_u, graph_param, laplacian_param)
% function [Y] = soft_hfs(X, Y, c_l, c_u, graph_param, laplacian_param)
%  a skeleton function to perform soft (unconstrained) HFS,
%  needs to be completed
%
%  Input
%  X:
%      (n x m) matrix of m-dimensional samples
%  Y:
%      (n x 1) vector with nodes labels [1, ... , num_classes] (0 is unlabeled)
%  c_l,c_u:
%      coefficients for C matrix
%  graph_param:
%      structure containing the graph construction parameters as fields
%  laplacian_param:
%      structure containing the laplacian construction parameters as fields
%
%  Output
%  labels:
%      class assignments for each (n) nodes

num_samples = size(X,1);
num_classes = length(unique(Y));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute the target y for the linear system                    %
% y = (n x num_classes) target vector                           %
% l_idx = (l x 1) vector with indices of labeled nodes          %
% u_idx = (u x 1) vector with indices of unlabeled nodes        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

l_idx = find(Y);
u_idx = find(Y==0);

n_l = length(l_idx);
n_u = length(u_idx);

y = zeros(num_samples, max(Y));
for i = 1:size(l_idx)(1)
  y(l_idx(i),Y(l_idx)(i)) = 1;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute the hfs solution, remember that you can use           %
%   build_laplacian_regularized and build_similarity_graph      %
% f = (n x 1) hfs solution for labeled nodes                    %
% C = (n x n) diagonal matrix with c_l for labeled samples      %
%             and c_u otherwise                                 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

C = 1/c_l * diag(Y !=0) + 1/c_u * diag(Y==0);
Q = build_laplacian_regularized(X, graph_param, laplacian_param);
f = inv(C*Q+eye(num_samples))*y;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute the labels assignment from the hfs solution           %
% label: (n x 1) class assignments [1, ... ,num_classes]        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[vals, labels] = max(f,[], 2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
