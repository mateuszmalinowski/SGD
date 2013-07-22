function probMatrix = softmax_probability( theta, data )
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