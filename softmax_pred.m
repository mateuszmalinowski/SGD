function pred = softmax_pred(theta, data)
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
probMatrix = softmax_probability(theta, data);
[~, pred] = max(probMatrix, [], 1);

end

