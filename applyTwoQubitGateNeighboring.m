function [mps, Sd] = applyTwoQubitGateNeighboring(mps, U2, jt, chi, gpu)
% applyTwoQubitGateNeighboring Applies a 2-qubit gate between qubits in neighboring groups.
%
%   mps = applyTwoQubitGateNeighboring2(mps, U2, jt, jq, kq, nq, chi)
%   applies the 4x4 unitary U2 to the qubit with local index jq in tensor jt and
%   the qubit with local index kq in tensor jt+1.
%
%   The procedure is as follows:
%
%     1. Cast the MPS in mixed canonical form (with tensors 1..jt left-canonical and
%        tensors jt+1..end right-canonical).
%
%     2. Prepare tensors A (site jt) and B (site jt+1) in a uniform 3-index form.
%        For the leftmost tensor (jt==1), reshape [d, chi] to [1,d,chi].
%        For the rightmost tensor (jt+1==nt), reshape [chi,d] to [chi,d,1].
%
%     3. Split target qubit from left tensor A using QR decomposition.
%
%     4. Contract the bond between the singled-out qubit A_mat and B to form T_fused.
%
%     5. Use applyTwoQubitGateTensor to apply U2 on the fused tensor.
%        In the fused tensor the left tensor’s target qubit is in positions 1 and the right
%        tensor’s qubits are in positions 2..nq+1. Thus, we apply U2 to qubit jq (left)
%        and qubit (1 + kq) (right). If B was a right–tensor, we call applyTwoQubitGateTensor
%        with the flag 'right'; otherwise, we use 'middle'.
%
%     6. Split the updated fused tensor, using SVD, into A_new with dimensions
%        [chi_left, 2*r] and B_new with dimensions [r, d, chi_right].
%
%     7. Update tensor A_left by contracting it with A_new. Apply inverse
%        permutation of qubits to restore original qubit order.
%
%     8. Update mps, converting back to the original tensor shapes.
%
%   Inputs:
%     mps       - Cell array of MPS tensors.
%     U2        - 4x4 unitary representing the two-qubit gate.
%     jt        - Index of the left tensor in the neighboring pair (must satisfy 1 <= jt <= nt-1).
%     jq        - Local qubit index (in tensor jt) to which U2 is applied.
%     kq        - Local qubit index (in tensor jt+1) to which U2 is applied.
%     nq        - Number of qubits per tensor (so that d = 2^nq).
%     chi       - Maximum bond dimension for truncation.
%
%   Output:
%     mps       - Updated MPS with the two-qubit gate applied.

nt = length(mps);
if jt >= nt
    error('jt must be at most nt-1 (found jt=%d, nt=%d)', jt, nt);
end

%% Step 1: Cast the MPS in mixed canonical form.
% mps = mixedCanonicalizeMPS(mps, jt);

%% Step 2: Prepare tensors A (site jt) and B (site jt+1) with uniform 3-index shape.
% For A:
if jt == 1
    % Left tensor: originally size [d, chi]. Treat as [1,d,chi].
    A = mps{jt};
    [dA, chiA] = size(A);
    A = reshape(A, [1, dA, chiA]);
else
    % Middle tensor: already size [chi, d, chi].
    A = mps{jt};
end
dimsA = size(A);  % dimsA = [chi_left, d, chi_mid]
dA = dimsA(2);
nqA = round(log2(dA));

% For B:
if jt+1 == nt
    % Right tensor: originally size [chi, d]. Treat as [chi, d, 1].
    B = mps{jt+1};
    dimsB = [size(B), 1];  % dimsB = [chi_mid, d, chi_right]
else
    % Middle tensor.
    B = mps{jt+1};
    dimsB = size(B);  % dimsB = [chi_mid, d, chi_right]
end
dB = dimsB(2);
nqB = round(log2(dB));

%% Step 3: Single out target qubit jq from left tensor
% A has dimensions [chi1, d, chi2], with d = 2^nq.
chi1A = dimsA(1);

%% Step 3.5: Single out target qubit kq from right tensor
% B has dimensions [chi1B, dB, chi2B], with dB = 2^nqB.
% By construction, chi2A = chi1B must hold
chi2B = dimsB(3);

%% Step 4: Contract the bond between the singled-out qubits A_mat and B_mat to form T_fused.
T = tensorprod(A, B, 3, 1);
% Reshape T to a 3-index tensor: [chimid, 2*d, chi_right]
T = reshape(T, chi1A, 4, chi2B);

%% Step 5: Apply the 2-qubit gate using applyTwoQubitGateTensor.
% The fused tensor is a 2-qubit tensor containing the two target qubits of U2.
T = permute(T,[2, 1, 3]);
T = reshape(T,4,[]);
T = U2 * T;
T = reshape(T, 4, chi1A, chi2B);
T = ipermute(T,[2, 1, 3]);

%% Step 6: Split the fused tensor using SVD.
% Reshape T into a matrix M where the left part combines [chimid, 2]
% and the right part combines [d, chi_right]:
M = reshape(T, chi1A*2, 2*chi2B);
if gpu==1
    M = gpuArray(M);
end
[U, S, V] = svd(M, 'econ');
% min(diag(S))
tt = find(diag(S)<1e-5,1,'first');
% [U, S, V] = svds(M, chi);
if gpu==1
    U = gather(U);
    S = gather(S);
    V = gather(V);
end
% r = min(chi, size(S,1));
if isempty(tt)
    r = min(chi, size(S,1));
else
    r = min([chi, tt, size(S,1)]);
end
U = U(:, 1:r);
S = S(1:r, 1:r);
Sd = diag(S);
V = V(:, 1:r);
% New left tensor (A_new) has shape: [chimid, 2*r]
A_new = reshape(U, chi1A, 2*r);
% New right tensor (B_new) has shape: [r, d, chi_right]
B_new = reshape(S * V', 2*r, chi2B);

%% Step 8: Update mps, converting back to the original tensor shapes.
if jt == 1
    % Left tensor: reshape A_new from [1, d, r] to [d, r]
    mps{jt} = reshape(A_new, dA, r);
else
    mps{jt} = reshape(A_new, chi1A, dA, r);
end
if jt+1 == nt
    % Right tensor: reshape B_new from [r, d, 1] to [r, d]
    mps{jt+1} = reshape(B_new, r, dB);
else
    mps{jt+1} = reshape(B_new, r, dB, chi2B);
end
end
