function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

hypothesis = X * theta;
square_difference = (hypothesis - y).^2;
ssd = sum(square_difference);
J = (1/(2*m)) * ssd;
square_params = theta(2:end);
square_params = square_params .^ 2;
reg_value = sum(square_params);
reg_value = (lambda/(2*m))*reg_value;
J = J + reg_value;





gradient = (hypothesis - y)' * X;
grad = (1/m)*gradient;
grad(2:end) = grad(2:end) + ((lambda/m)*theta(2:end))';







% =========================================================================

grad = grad(:);

end
