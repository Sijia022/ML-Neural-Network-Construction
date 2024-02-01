function [a] = sigmoid(z)
%compute sigmoid function for input matrix z
a = 1./(ones(size(z))+exp(-1*z));

end

