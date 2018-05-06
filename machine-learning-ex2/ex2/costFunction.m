function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
alpha = 1;
summ = 0;
hx = 0;
a=0;
b=0;
for i = 1:m
	f = X(i,:);
	hx = theta' * f';
	hx = sigmoid(hx);
	a = log(hx);
	a = -1*a*y(i,1);
	
	b = 1-hx;
	b = log(b);
	b=(1 - y(i,1))*b;
	
	a = a-b;
	%hx = theta(1,1) + (theta(2,1) * X(i,2)); %h(tt)
	%hx = hx - y(i,1);  % h(tt) - y
	%hx = hx * hx;    %( h(tt) - y)^2
    summ = summ + a;  %sigma( h(tt) - y)^2
end



J = summ / m;


%gradient descent
[r no_of_features] = size(X);
%for iter = 1:1500

			
	hx_arr = zeros(no_of_features,1);	
	summ_arr = zeros(no_of_features,1);
	theta_arr = zeros(no_of_features,1);
	
	for k = 1:no_of_features
		for l = 1:m
			f = X(l,:);
			hx_arr(k,1) = theta' * f';
			hx_arr(k,1) = sigmoid(hx_arr(k,1));
			%	hx_arr(k,1) = theta(1,1) + (theta(2,1) * X(l,2));
			hx_arr(k,1) = hx_arr(k,1) - y(l,1);
			hx_arr(k,1) = hx_arr(k,1) * X(l,k);
			summ_arr(k,1) = summ_arr(k,1) + hx_arr(k,1);
		end
		summ_arr(k,1) = summ_arr(k,1)  *  (alpha /m);
		%theta_arr(k,1) = theta(k,1) - summ_arr(k,1);
		theta_arr(k,1) = summ_arr(k,1);
	end
	theta = theta_arr;
	grad = theta;
% =============================================================

%{
htheta = sigmoid(X * theta);
for i = 1:size(theta, 1)
    grad(i) = 1 / m * sum((htheta - y) .* X(:, i));
end
%}
%end
end