function mps = mixedCanonicalizeMPSpartial(mps, n, nold)
% mixedCanonicalizeMPS Casts an MPS into a mixed canonical form.
%
%   mps = mixedCanonicalizeMPS(mps, n) returns an MPS in which the tensors
%   from site 1 to site n are left-canonical and the tensors from site n+1
%   to the end are right-canonical. The bond between site n and n+1 forms
%   the orthogonality center.
%
%   This function splits the MPS into two segments, canonicalizes each
%   segment independently, and then fixes the gauge on the shared bond via SVD.
%
% Input:
%   mps - Cell array containing the MPS tensors.
%   n   - Integer index such that 1 <= n < length(mps).
%
% Output:
%   mps - The mixed canonical form of the MPS.
%
% Example:
%   mps = initMPS(d, chi, nt);
%   mps = mixedCanonicalizeMPS(mps, n);
%
% Note:
%   The left-canonical form means that for each tensor A (when reshaped by
%   combining its left and physical indices) we have A^dagger * A = I.
%   The right-canonical form means that for each tensor B (when reshaped by
%   combining its physical and right indices) we have B * B^dagger = I.
%
%   This routine assumes that n is at least 1 and less than the total number
%   of sites so that a split into two non-empty segments is possible.

if n < 0 || n >= length(mps)
    error('n must be between 1 and length(mps)-1');
end

if n==0
    mps = rightCanonicalizeMPS(mps);
else
    % Split the MPS into left and right segments.
    left_mps  = mps(1:n);
    right_mps = mps(n+1:end);

    % Bring the left segment into left-canonical form.
    left_mps = leftCanonicalizeMPSpartial(left_mps, nold);

    % Bring the right segment into right-canonical form.
    % right_mps = rightCanonicalizeMPS(right_mps);

    % Now, adjust the connecting bond between left_mps{end} and right_mps{1}.
    % We perform an SVD on the bond of the last tensor of the left segment.
    A = left_mps{end};
    dimsA = size(A);
    % Reshape A into a matrix by combining all indices except the last one.
    A = reshape(A, [], dimsA(end));
    [U, S, V] = svd(A, 'econ');
    newBond = size(U,2);

    % Update the last tensor of the left segment.
    if numel(dimsA) == 2
        % Leftmost tensor shape: (d, chi)
        left_mps{end} = reshape(U, dimsA(1), newBond);
    else
        % Middle tensor shape: (chi, d, chi)
        left_mps{end} = reshape(U, dimsA(1), dimsA(2), newBond);
    end

    % Absorb S*V' into the first tensor of the right segment.
    B = right_mps{1};
    dimsB = size(B);
    B = reshape(B, dimsB(1), []);
    B = S * V' * B;
    if numel(dimsB) == 2
        % Rightmost tensor shape: (chi, d)
        right_mps{1} = reshape(B, newBond, dimsB(2));
    else
        % Middle tensor shape: (chi, d, chi)
        right_mps{1} = reshape(B, newBond, dimsB(2), dimsB(3));
    end

    % Combine the two segments back into a single MPS.
    mps = [left_mps, right_mps];
end

end
