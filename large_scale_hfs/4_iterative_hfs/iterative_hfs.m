function [labels, accuracy] = iterative_hfs(T)
% function [labels, accuracy] = iterative_hfs(t)
% a skeleton function to perform HFS, needs to be completed
%  Input
%  T:
%      number of iterations to use for the iterative propagation

%  Output
%  labels:
%      class assignments for each (n) nodes
%  accuracy

% load the data

in_data = load('data/data_iterative_hfs_graph.mat');

W = in_data.W;
Y = in_data.Y;
Y_masked =  in_data.Y_masked;

num_samples = size(W,1);
num_classes = sum(unique(Y) ~= 0);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute the target y for the linear system                       %
% y = (num_samples x num_classes) target vector                    %
% l_idx = (l x num_classes) vector with indices of labeled nodes   %
% u_idx = (u x num_classes) vector with indices of unlabeled nodes %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


l_idx = find(Y_masked);
u_idx = find(Y_masked==0);
n_l = length(l_idx);
n_u = length(u_idx);
y = zeros(num_samples, num_classes);
for i = 1:n_l
  y(l_idx(i),Y_masked(l_idx(i))) = 1;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute the hfs solution, using iterated averaging            %
% remember that column-wise slicing is cheap, row-wise          %
% expensive and that W is already undirected                    %
% f_l = (l x num_classes) hfs solution for labeled              %
% f_u = (u x num_classes) hfs solution for unlabeled            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
D = sum(W, 2);
f = y;
accuracy = []
tic;
for t = 1:T
  for k = 1:n_u
    f(u_idx(k),:) = W(:, u_idx(k))'* f ./ D(u_idx(k));
  end
  [vals, labels] = max(f, [], 2);
  accuracy = [accuracy mean(labels == Y)];
end
toc

plot(1:length(accuracy),accuracy);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute the labels assignment from the hfs solution           %
% label: (n x 1) class assignments [1,2,...,num_classes]        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[vals, labels] = max(f, [], 2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

accuracy = mean(labels == Y)
