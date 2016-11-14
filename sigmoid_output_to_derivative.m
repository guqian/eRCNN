function r = sigmoid_output_to_derivative(x)
r = x .* (1 - x);
end