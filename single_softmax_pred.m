function pred = single_softmax_pred(theta, data)
% Predict class for the input data.
% 
% In:
%   theta - model trained using softmax_train;
%     it is a vector, theta \in R[nclasses * numfeatures]
%   data - data points;
%     data \in R[numfeatures, numdata]
% 
% Out:
%   pred - predicted class, where pred(i) is argmax_c P(y(c) | x(i))
% 

theta = reshape(theta, [], size(data, 1));

% class probability and prediction
probMatrix = prob(theta, data);
[~, pred] = max(probMatrix, [], 1);

end

function probMatrix = prob( theta, data )
%COMPUTEPROBMATRIX Computes probability function 
% p(y^k = class | x^k; theta_class) using exponential potentials.
% 
% theta - parameters
% data - current data (x); data(j,k) describes j-th feature of k-th
%   datapoint
% probMatrix - probMatrix(a,b) = p(y^b = a | x^b; theta_a)
% 

probMatrix = theta * data;
probMatrix = exp(bsxfun(@minus, probMatrix, max(probMatrix, [], 1)));
normalizationTerm = 1.0 ./ sum(probMatrix, 1);
probMatrix = bsxfun(@times, probMatrix, normalizationTerm);

end