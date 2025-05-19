function U2 = randomHaar(n)

% randomHaar generates an n x n unitary matrix uniformly distributed
% according to the Haar measure.
%
% A unitary matrix U (n x n) satisfies U'*U = I, where I is the identity
% matrix. The Haar measure is the unique invariant measure on the group of
% unitary matrices, meaning that matrices drawn from this distribution are
% "uniformly random" over the space of all unitaries.
%
% This function uses a common procedure based on the QR decomposition:
%
%   1. Generate a random complex matrix with independent Gaussian entries.
%      Each entry is a complex number whose real and imaginary parts are drawn
%      from a normal distribution with mean 0 and variance 1.
%
%   2. Perform a QR decomposition on the random matrix.
%      This factorizes the matrix U into Q and R, where:
%         - Q is an n x n unitary matrix.
%         - R is an n x n upper triangular matrix.
%
%   3. Correct the phases of Q using the diagonal elements of R.
%      The QR decomposition has an ambiguity in the phases (or signs) of Q because
%      R's diagonal elements can be arbitrary complex numbers. To ensure that Q
%      is uniformly distributed (Haar measure), we adjust Q by multiplying with a
%      diagonal matrix that contains the phase factors (i.e., each diagonal element
%      divided by its absolute value) of R.
%
% Inputs:
%   n - The dimension of the unitary matrix (resulting matrix is n x n).
%
% Output:
%   U2 - An n x n unitary matrix drawn uniformly from the Haar measure.
%
% Example:
%   U = randomHaar(4);  % Generates a 4x4 Haar-distributed random unitary matrix.

U = randn(n) + 1i .* randn(n);
[Q, R] = qr(U);
x = diag(R);
U2 = Q * diag(x ./ abs(x));

end
