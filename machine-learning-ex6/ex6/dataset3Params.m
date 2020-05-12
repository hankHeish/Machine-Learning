function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;
steps = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
% steps = [0.01, 0.1, 1, 3, 30];
m = size(steps)(2);
bestCV = 1;
% cvMatrix = zeros(m, m);
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

for i = 1:m
    for j = 1:m
        C_ = steps(i); 
        sigma_ = steps(j); 
        % printf('Training (%d, %d): ', i, j);
        
        model= svmTrain(X, y, C_, @(x1, x2) gaussianKernel(x1, x2, sigma_));
        predictions = svmPredict(model, Xval);
        cv = mean(double(predictions ~= yval));
        % cvMatrix(i, j) = cv;
        
        if cv < bestCV
            bestCV = cv;
            C = C_;
            sigma = sigma_;
        end
        % printf('Update parameters: C = %f, sigma = %f\n', C, sigma);
    end
end

% =========================================================================

end
