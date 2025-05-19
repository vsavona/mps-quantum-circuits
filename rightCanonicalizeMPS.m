function mps = rightCanonicalizeMPS(mps)
% rightCanonicalizeMPS Casts an MPS into right-canonical form.
%
%   mps = rightCanonicalizeMPS(mps) takes a cell array 'mps' of tensors
%   representing a Matrix Product State (MPS) and returns a new MPS where
%   each tensor (except possibly the first one) satisfies the right-canonical
%   condition. In this form, when a tensor is reshaped by combining its
%   physical index with its right bond, the resulting matrix has orthonormal
%   rows, i.e., A * A^dagger = I.
%
%   The procedure works by moving from right to left and performing an
%   LQ decomposition at each step. (We implement LQ via QR on the transpose.)
%   The L factor from the decomposition is absorbed into the previous tensor.
%
% Input:
%   mps - Cell array of tensors defining the MPS.
%
% Output:
%   mps - The right-canonicalized MPS.
%
% Example:
%   mps = initMPS(d, chi, nt);
%   mps = rightCanonicalizeMPS(mps);

nt = length(mps);
for n = nt:-1:2
    % Extract the current tensor.
    A = mps{n};
    dims = size(A);
    
    % Reshape A into a matrix with its left bond as rows and the physical and 
    % right bond indices combined as columns.
    % For the rightmost tensor, dims is [chi, d] so A is (chi, d).
    % For middle tensors, dims is [chi, d, chi] so A is (chi, d*chi).
    A = reshape(A, dims(1), []);
    
    % To obtain an LQ factorization A = L * Q, we compute the QR decomposition
    % of the transpose: A' = Q * R. Then A = R' * Q'.
    % Here, L = R' (lower-triangular) and Q = Q' (unitary).
    [Q, R] = qr(A', 'econ');
    L = R';
    Q = Q';
    
    % The new bond dimension after the decomposition:
    newBond = size(L,2);
    
    % Reshape Q_new back into tensor form.
    if length(dims) == 2
        % For the rightmost tensor: reshape Q of size (newBond, d)
        mps{n} = reshape(Q, newBond, dims(2));
    else
        % For a middle tensor: reshape Q of size (newBond, d*chi) to (newBond, d, chi)
        mps{n} = reshape(Q, newBond, dims(2), dims(3));
    end
    
    % Absorb L into the previous tensor along its right bond.
    A = mps{n-1};
    dims_prev = size(A);
    
    if length(dims_prev) == 2
        % For the leftmost tensor: shape (d, chi)
        % Reshape it as (d, chi) and multiply on the right by L (chi x newBond)
        A = reshape(A, dims_prev(1), []);
        A = A * L;
        mps{n-1} = reshape(A, dims_prev(1), newBond);
    else
        % For a middle tensor: shape (chi, d, chi)
        % Reshape it to (chi*d, chi) and multiply on the right.
        A = reshape(A, dims_prev(1)*dims_prev(2), dims_prev(3));
        A = A * L;
        mps{n-1} = reshape(A, dims_prev(1), dims_prev(2), newBond);
    end
end
end
