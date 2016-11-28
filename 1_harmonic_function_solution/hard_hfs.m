function [labels] = hard_hfs(X, Y, graph_param, laplacian_param)
% function [labels] = hard_hfs(X, Y, graph_param, laplacian_param)
%  a skeleton function to perform hard (constrained) HFS,
%  needs to be completed
%
%  Input
%  X:
%      (n x m) matrix of m-dimensional samples
%  Y:
%      (n x 1) vector with nodes labels [1, ... , num_classes] (0 is unlabeled)
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
% y = (l x c) target vector                                     %
% l_idx = (l x 1) vector with indices of labeled nodes          %
% u_idx = (u x 1) vector with indices of unlabeled nodes        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

l_idx = find(Y);
u_idx = find(Y==0);

n_l = length(l_idx);
n_u = length(u_idx);

y = -ones(n_l, num_classes - 1);
for i = 1:n_l
 y(i,Y(l_idx(i))) = 1;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute the hfs solution, remember that you can use           %
%   build_laplacian_regularized and build_similarity_graph      %
% f_l = (l x 1) hfs solution for labeled nodes                  %
% f_u = (u x 1) hfs solution for unlabeled nodes                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

f_l = y;
L = build_laplacian_regularized(X, graph_param, laplacian_param);
W = build_similarity_graph(X, graph_param);


f_u = inv(L(u_idx, u_idx))*(W(u_idx, l_idx)*f_l);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute the labels assignment from the hfs solution           %
% label: (n x 1) class assignments [1,2,...,num_classes]        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[vals_l, l_l] = max(f_l,[], 2);
[vals_l, l_u] = max(f_u,[], 2);
labels = zeros(num_samples, 1);
labels(l_idx) = l_l;
labels(u_idx) = l_u;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
