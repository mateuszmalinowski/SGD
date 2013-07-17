% generate a set of vectors in dimension d from a Gaussian mixture of nclasses, 
% with nex examples per class
% split equally in train and test, shuffle randomly
% 
% Slightly modified version from: http://lear.inrialpes.fr/src/jsgd/
% 

function [Xtrain, Ltrain, Xtest, Ltest] = generate_mixture_of_gaussians( ...
  d, nclass, nex, spread)
% 
% In:
%   d - dimensionality (number of features)
%   nclass - number of classes
%   nex - number of examples per class
%   spread - standard deviation of the distribution
% 
% Out:
%   Xtrain - training set; Xtrain \in R[d, nclass*nex/2]
%   Ltrain - training labels; Ltrain \in R[nclass*nex/2]
%   Xtest - test set; Xtest \in R[d,nclass*nex/2]
%   Ltest - test labels; Ltest \in R[nclass*nex/2]
% 

X = [];
labels = [];

% generate the data
for i = 1:nclass
  
  % centre in the unit hypercube
  mu = rand(d, 1); 
  
  % covariance matrix
  sigma = eye(d) * spread; 
  
  xi = sigma * randn(d, nex) + repmat(mu, 1, nex);
  
  X = [ X xi ]; 
  labels = [ labels (i * ones(1, nex)) ];
    
end

% split train / test (and shuffle the data randomly)
n = size(X, 2); 

perm = randperm(n); 

train = perm(1 : n / 2); 
Xtrain = X(:, train); 
Ltrain = labels(:, train); 

test = perm(n / 2 + 1 : end);
Xtest = X(:, test); 
Ltest = labels(:, test); 
