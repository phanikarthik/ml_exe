function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
%mu = zeros(1, size(X, 2));% [0  0 ] 
%sigma = zeros(1, size(X, 2));% [0  0 ] 

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       

data = load('ex1data2.txt');
housesize = data(:,1);
bedroom = data(:,2);
means = mean(data)
stddev = std(data)

housesize = (housesize - means(1,1))/ stddev(1,1);
bedroom = (bedroom - means(1,2))/ stddev(1,2);

X_norm = [housesize bedroom];


fprintf(' x = [%f %f] \n', [X_norm(1:10,:)]');
mu(1,1) = means(1,1);
mu(1,2) = means(1,2);
sigma(1,1) = stddev(1,1);
sigma(1,2) = stddev(1,2);
%{

mu = mean(X_norm);
sigma = std(X_norm);

tf_mu = X_norm - repmat(mu,length(X_norm),1);
tf_std = repmat(sigma,length(X_norm),1);

X_norm = tf_mu ./ tf_std;

%}





% ============================================================

end
