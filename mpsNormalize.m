function mps = mpsNormalize(mps)
% mpsNormalize computes the norm of the state represented by an MPS, and
% returns the left-normalized MPS where the last tensor has been normalized
% accordingly. The resulting state is normalized, as explained in Sec. 4.4
% of the review by Schöllwock arXiv:1008.3477
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
    
    % Normalize the rightmost tensor, which for a left-canonical mps
    % corresponds to normalizing the state
    mps{end} = A ./ norm_val;
    
end
