function numgrad = compute_numerical_gradient(J, theta)
% Computes numerical gradient of the given function J at point theta.
% 
% In:
%   J - a function that outputs a real-number. 
%   theta - a vector of parameters
%     Calling y = J(theta) will return the function value at point theta. 
% 
% Out:
%   numgrad - numerical gradient of J at point theta
% 
% Written by: Mateusz Malinowski
% Email: m4linka@gmail.com
% Created: 22.02.2012
%  
  
EPSILON = 1e-7;

numOfEntries = length(theta);

denominator = 1.0 / (2.0 * EPSILON);

% % vectorized code - not really faster
% % standard basis multiplied by epsilon
% e = EPSILON * eye(numOfEntries);
% 
% f =@(a, b) J(a + b) - J(a - b);
% 
% numgrad = denominator * arrayfun(@(K) f(theta, e(:, K)), 1:numOfEntries)';

% unvectorized codes
% standard basis
e = zeros(numOfEntries, 1);

% here we store results
numgrad = zeros(numOfEntries, 1);

for k = 1:numOfEntries  
  e(k) = 1;
  
  numgrad(k) = ...
    denominator * ( J(theta + EPSILON * e) - J(theta - EPSILON * e) );
  
  e(k) = 0;
end

end
