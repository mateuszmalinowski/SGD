function [cost, grad] = noisy_softmax_cost( ...
  theta, indices, lambda, numClasses, data, labels)
% Takes a trained theta and a training data set with labels,
% and returns cost and gradient of the softmax regression. 
% 
% In:
%   theta - parameter
%   indices - set of indices taken in the iteration to compute the (noisy)
%     gradient
%   numClasses - the number of classes 
%   lambda - weight decay parameter
%   data - the N x M input matrix, where each column data(:, i) 
%     corresponds to a single data point
%   labels - an M x 1 matrix containing the labels corresponding 
%     for the input data
%
% Out:
%   cost - cost of the softmax regression at a given point theta
%   grad - gradient of the softmax regression at a given point theta
%   
% Written by: Mateusz Malinowski
% Email: m4linka@gmail.com
% Created: 01.03.2012
%  

% pick up only interesting datapoints
data = data(:, indices);
labels = labels(indices);

inputSize = size(data, 1);

% unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);

numData = size(data, 2);
dataTermNormalization = 1.0 / numData;

groundTruth = full(sparse(labels, 1:numData, 1, numClasses, numData));

% compute the probability matrix probMatrix(r,c) = p(y^c=r | x^c; theta);
probMatrix = softmax_probability(theta, data);

regularizationTerm = 0.5 * lambda *  norm(theta(:), 2)^2;
dataTerm = -dataTermNormalization * sum(groundTruth .* log(probMatrix), 2);
dataTerm = sum(dataTerm);

cost = dataTerm + regularizationTerm;

negP = groundTruth - probMatrix;
thetagrad = -dataTermNormalization * negP * data' + lambda * theta;

% roll the gradient matrices into a vector for minFunc
grad = thetagrad(:);

end
