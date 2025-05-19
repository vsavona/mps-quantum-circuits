function norm_val = mpsNorm(mps)
% mpsNorm Computes the norm of the state represented by an MPS.
%
%   norm_val = mpsNorm(mps) left-canonicalizes the MPS and then
%   computes the norm of the state as the Frobenius norm of the last tensor.
%
%   It is assumed that after left-canonicalization the first tensor is of
%   size [d, chi] and that the last tensor is in the form
%   [chi,d]. All other tensors have dimensions [chi1,d,chi2]
%
%   Example:
%       norm_val = mpsNormCanonical(mps);
%

    % Left-canonicalize the MPS.
    mps = leftCanonicalizeMPS(mps);
    
    % Get the last tensor.
    A = mps{end};
    
    % The state norm is given by the Frobenius norm of A.
    norm_val = norm(A, 'fro');
end
