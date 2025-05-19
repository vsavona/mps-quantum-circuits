function mps = leftCanonicalizeMPSpartial(mps,jtold)
% leftCanonicalizeMPS Casts an MPS into left-canonical form.
%
%   mps = leftCanonicalizeMPS(mps) takes a cell array 'mps' of tensors
%   representing a Matrix Product State (MPS) and returns a new MPS where
%   each tensor (except possibly the last one) satisfies the left-canonical
%   condition. 
%
%   In an MPS, canonical forms refer to representations where the tensors 
%   obey orthonormality conditions. In the left-canonical form, each tensor 
%   A (when reshaped as a matrix by combining its left bond and physical 
%   indices) satisfies A^dagger * A = I.
%
%   The function works by performing QR decompositions sequentially from
%   the leftmost tensor to the second-to-last one. The R factor from the QR
%   decomposition is absorbed into the next tensor.
%
% Input:
%   mps - Cell array of tensors defining the MPS.
%
% Output:
%   mps - The left-canonicalized MPS.
%
% Example:
%   mps = initMPS(d, chi, nt);
%   mps = leftCanonicalizeMPS(mps);

nt = length(mps);
for n = max(1,jtold-1):nt-1
    % Extract the current tensor.
    A = mps{n};
    dims = size(A);
    
    % Combine the left bond and physical indices into one index.
    % For the leftmost tensor, dims is [d, chi].
    % For middle tensors, dims is [chi, d, chi].
    A = reshape(A, [], dims(end));  % resulting size: (left*d, right)
    
    % QR factorization to obtain an orthonormal Q.
    [Q, R] = qr(A, 'econ');  % economy size
    
    newBond = size(Q, 2);
    
    % Reshape Q back to the original tensor structure with updated bond dimension.
    if length(dims) == 2
        % For leftmost tensor: reshape from (d, newBond)
        mps{n} = reshape(Q, dims(1), newBond);
    else
        % For middle tensors: reshape from (chi*d, newBond) to (chi, d, newBond)
        mps{n} = reshape(Q, dims(1), dims(2), newBond);
    end
    
    % Absorb R into the next tensor along its left bond.
    A = mps{n+1};
    dims_next = size(A);
    
    if length(dims_next) == 2
        % For rightmost tensor: shape (chi, d)
        A = reshape(A, dims_next(1), []);  % shape: (chi, d)
        A = R * A;
        mps{n+1} = reshape(A, size(R,1), dims_next(2));
    else
        % For a middle tensor: shape (chi, d, chi)
        % Reshape the tensor so that the left bond is the first index.
        A = reshape(A, dims_next(1), []);  % shape: (chi, d*chi_right)
        A = R * A;
        mps{n+1} = reshape(A, size(R,1), dims_next(2), dims_next(3));
    end
end
end
