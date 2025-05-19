function [mps, Sd] = applyTwoQubitGateNeighboring(mps, U2, jt, chi, svdtol)
% applyTwoQubitGateNeighboring Applies a 2-qubit gate between qubits in neighboring MPS sites.
%
%   [mps, Sd] = applyTwoQubitGateNeighboring(mps, U2, jt, chi, svdtol)
%
%   Inputs:
%     mps    - Cell array of MPS tensors in mixed canonical form centered at site jt.
%              * Left boundary tensor: size [d, chi]
%              * Middle tensors:      size [chi_left, d, chi_right]
%              * Right boundary tensor:size [chi, d]
%     U2     - d^2 × d^2 unitary matrix acting on two neighboring physical sites.
%     jt     - Site index of the first qubit (1 ≤ jt < nt).
%     chi    - Maximum bond dimension for SVD truncation.
%     svdtol - (optional) tolerance for singular values. If zero, truncate by chi only;
%              otherwise, drop singular values below svdtol (up to chi).
%
%   Outputs:
%     mps - Updated MPS after applying U2 and truncating the connecting bond.
%     Sd  - Vector of retained singular values (post-truncation).
%
%   Procedure:
%     1. Bring tensors A (site jt) and B (site jt+1) into uniform 3-index form.
%     2. Fuse A and B along their common bond into a single tensor T.
%     3. Apply U2 to the fused physical index of T.
%     4. Reshape T into a matrix M and perform SVD: M = U*S*V'.
%     5. Determine truncation rank r via chi and svdtol.
%     6. Truncate U, S, V and reshape into new tensors A_new and B_new.
%     7. Replace the original tensors in mps with A_new and B_new, restoring boundary shapes.

    % Number of sites in the MPS
    nt = length(mps);
    % Ensure jt is a valid site index for a neighbor pair
    if jt >= nt
        error('jt must be at most nt-1 (found jt=%d, nt=%d)', jt, nt);
    end

    %% Step 1: Prepare tensors A and B with 3-index shapes
    % Extract A = mps{jt} and reshape if it's a boundary tensor
    if jt == 1
        % Left boundary: original shape [d, chi]
        A = mps{jt};
        [d, chiA] = size(A);
        % Treat as 3-index: [1, d, chi]
        A = reshape(A, [1, d, chiA]);
    else
        % Middle tensor: already [chi_left, d, chi_mid]
        A = mps{jt};
    end
    dimsA = size(A);        % dimsA = [chi_left_A, d, chi_mid_A]
    d     = dimsA(2);       % Physical dimension
    chi1A = dimsA(1);       % Left bond dim of A

    % Extract B = mps{jt+1} and reshape if it's a boundary tensor
    if jt+1 == nt
        % Right boundary: original shape [chi, d]
        B = mps{jt+1};
        % Represent as [chi_mid, d, 1]
        dimsB = [size(B), 1];
    else
        % Middle tensor: already [chi_mid, d, chi_right]
        B = mps{jt+1};
        dimsB = size(B);
    end
    chi2B = dimsB(3);       % Right bond dim of B
    % By mixed-canonical form, chi_mid_A == chi_mid_B

    %% Step 2: Fuse A and B over their shared bond
    % tensorprod contracts A and B on A’s 3rd index and B’s 1st index
    T = tensorprod(A, B, 3, 1);
    % After contraction, T has shape [chi1A, d * d, chi2B]
    T = reshape(T, chi1A, d^2, chi2B);

    %% Step 3: Apply two-qubit gate U2 to the fused physical index
    % Move physical index to first dimension for matrix multiplication
    T = permute(T, [2, 1, 3]);           % [d^2, chi1A, chi2B]
    T = reshape(T, d^2, []);            % [d^2, chi1A * chi2B]
    T = U2 * T;                         % apply U2 on the d^2 space
    T = reshape(T, d^2, chi1A, chi2B);  % [d^2, chi1A, chi2B]
    % Return to [chi1A, d^2, chi2B]
    T = ipermute(T, [2, 1, 3]);

    %% Step 4: Reshape for SVD split
    % Merge chi1A and d along rows, and d and chi2B along columns
    M = reshape(T, chi1A * d, d * chi2B);

    %% Step 5: Perform SVD and determine truncation rank
    [U, S, V] = svd(M, 'econ');
    if svdtol == 0
        % Truncate purely by chi
        r = min(chi, size(S, 1));
    else
        % Determine first singular value below tolerance, then cap at chi
        Sd_all = diag(S);
        idx = find(Sd_all < svdtol, 1, 'first');
        if isempty(idx)
            r = min(length(Sd_all), chi);
        else
            r = min(idx, chi);
        end
    end

    % Truncate U, S, V to rank r
    U = U(:, 1:r);
    S = S(1:r, 1:r);
    V = V(:, 1:r);
    Sd = diag(S);    % Extract retained singular values

    %% Step 6: Build new tensors from truncated SVD factors
    % A_new: combine U back into a tensor of shape [chi1A, d, r]
    A_new = reshape(U, chi1A, d * r);
    % Then separate physical index: [chi1A, d, r]
    A_new = reshape(A_new, chi1A, d, r);

    % B_new: combine S and V' into a tensor of shape [r, d, chi2B]
    B_new = reshape(S * V', d * r, chi2B);
    B_new = reshape(B_new, r, d, chi2B);

    %% Step 7: Update mps with new tensors, restoring boundary shapes
    if jt == 1
        % Left boundary: reshape A_new [1, d, r] ➔ [d, r]
        mps{jt} = reshape(A_new, d, r);
    else
        % Middle: keep [chi_left_A, d, r]
        mps{jt} = A_new;
    end

    if jt+1 == nt
        % Right boundary: reshape B_new [r, d, 1] ➔ [r, d]
        mps{jt+1} = reshape(B_new, r, d);
    else
        % Middle: keep [r, d, chi_right_B]
        mps{jt+1} = B_new;
    end
end
