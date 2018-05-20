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
alpha = 1;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
alpha = 1;
summ = 0;
hx = 0;
reg = 0;
for i = 1:m
	f = X(i,:);
	hx = theta' * f';
	%hx = theta(1,1) + (theta(2,1) * X(i,2)); %h(tt)
	hx = hx - y(i,1);  % h(tt) - y
	hx = hx * hx;    %( h(tt) - y)^2
    summ = summ + hx;  %sigma( h(tt) - y)^2
end
J = summ / (2 * m);


reg = sum(theta .* theta) - (theta(1,1) .* theta(1,1));
reg = lambda/(2*m);
J = J + reg;


[r no_of_features] = size(X);
%for iter = 1:1500

			
	hx_arr = zeros(no_of_features,1);
	summ_arr = zeros(no_of_features,1);
	theta_arr = zeros(no_of_features,1);
	
	for k = 1:no_of_features
		for l = 1:m
			f = X(l,:);
			hx_arr(k,1) = theta' * f';
			%hx_arr(k,1) = sigmoid(hx_arr(k,1));
			%	hx_arr(k,1) = theta(1,1) + (theta(2,1) * X(l,2));
			hx_arr(k,1) = hx_arr(k,1) - y(l,1);
			hx_arr(k,1) = hx_arr(k,1) * X(l,k);
			summ_arr(k,1) = summ_arr(k,1) + hx_arr(k,1);
		end
		summ_arr(k,1) = summ_arr(k,1)  *  (alpha /m);
		%theta_arr(k,1) = theta(k,1) - summ_arr(k,1);
		theta_arr(k,1) = summ_arr(k,1);
		%if(k>1)
		%theta_arr(k,1) = (lambda/m)*theta_arr(k,1)+theta(k,1);
		%endif
    %
	end
	
			
		
	
  for k = 2:no_of_features
       theta_arr(k,1) = theta_arr(k,1) + (lambda/m)*theta(k,1);
 end
  
  theta = theta_arr;
	grad = theta;




% =========================================================================

grad = grad(:);

end
