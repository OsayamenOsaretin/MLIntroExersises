function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
%C = 1;
% sigma = 0.3;
C_set = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma_set = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_sigma_advantage = zeros(size(C_set));

for i=1:size(C_set,2)
for j=1:size(sigma_set,2)
model = svmTrain(X, y, C_set(i), @(x1, x2) gaussianKernel(x1,x2,sigma_set(j)));
prediction = svmPredict(model, Xval);
C_sigma_advantage(i,j) = mean(double(prediction ~= yval));
%fprintf('error for %f, %f: %f', i, j, C_sigma_advantage(i,j));
end
end

[column_min, column_index] = min(C_sigma_advantage);

[min_value, min_index] = min(column_min);
%fprintf('min_value: %f, min_index: %f', min_value, min_index);

C_index = column_index(min_index);
%fprintf('C_index: %f', C_index);
sigma_index = min_index;

C = C_set(C_index);
sigma = sigma_set(sigma_index);




% =========================================================================

end
