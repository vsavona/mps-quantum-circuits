function state_vec = mpsToState(mps)
% mpsToState Contracts an MPS into a full state vector.
%
%   state_vec = mpsToState(mps) contracts all bond indices of the MPS (given as a cell
%   array of tensors) and returns the full state vector as a column vector of size
%   2^(nt*nq) x 1.
%
%   The MPS is assumed to be in the following standard form:
%       - mps{1} is [d, chi]       (leftmost tensor)
%       - mps{i} for i=2:nt-1 is [chi, d, chi]  (middle tensors)
%       - mps{nt} is [chi, d]       (rightmost tensor)
%
%   Here, d = 2^(nq) is the physical dimension.
%
% Example:
%   state_vec = mpsToState(mps);
%

    nt = length(mps);
    % Start with the leftmost tensor.
    psi = mps{1}; % size: [d, chi]
    % Contract sequentially with the rest of the tensors.
    for i = 2:nt
        psi = contractLastFirst(psi, mps{i});
    end
    % After full contraction, psi has only physical indices.
    state_vec = reshape(psi, [], 1);
end

function R = contractLastFirst(A, B)
% contractLastFirst Contracts the last index of A with the first index of B.
%
%   R = contractLastFirst(A, B) takes tensor A with size
%         [s1, s2, ..., s_{n-1}, r]
%   and tensor B with size
%         [r, t1, t2, ... , t_m],
%   and returns the contraction over the index of dimension r. The result R
%   is reshaped to have size [s1, s2, ..., s_{n-1}, t1, t2, ... , t_m].

    A_size = size(A);
    B_size = size(B);
    r = A_size(end);
    if B_size(1) ~= r
        error('Mismatch in contraction dimensions: %d vs %d', r, B_size(1));
    end
    P = prod(A_size(1:end-1));  % product of all dimensions of A except the last.
    Q = prod(B_size(2:end));    % product of all dimensions of B except the first.
    A_mat = reshape(A, [P, r]);
    B_mat = reshape(B, [r, Q]);
    R_mat = A_mat * B_mat;
    R = reshape(R_mat, [A_size(1:end-1), B_size(2:end)]);
end
