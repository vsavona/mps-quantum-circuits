function innerp_val = mpsInnerProduct2(mps1, mps2)

% Computes the inner product between states represented by mps1 and mps2.
% Assumes that mps1 and mps2 are normalized. Assumes also that mps1 and
% mps2 have the exact same structure

nt = length(mps1);
innerp_val = 0;    % Initialize inner product


% Special treatment of the first tensor, as in Eq. (95) of the Schöllwock
% review arXiv:1008.3477
A = mps1{1};
dimA = size(A);
d = dimA(1);      % Must be the same physical dimensions for the two tensors of the bra and ket states
chiA = dimA(2);
B = mps2{1};
dimB = size(B);
chiB = dimB(2);
if d ~= dimB(1)
    disp('Physical dimensions of the two MPSs do not match');
    return
end

% Sum the matrix product of the j-th column of A' with the j-th row of B
% The resulting matrix C has dimensions equal to the right bond dimensions
% of A and B
C = zeros(chiA,chiB);
A = A';       % Take the adjoint matrix
for j=1:d
    C = C + A(:,j) * B(j,:);
end

% Always following Eq. (95) of arXiv:1008.3477, now loop on the middle
% tensors of the mps
for k=2:nt-1
    MA = mps1{k};
    dimA = size(MA);
    chilA = dimA(1);
    dA = dimA(2);
    chirA = dimA(3);
    MB = mps2{k};
    dimB = size(MB);
    chilB = dimB(1);
    dB = dimB(2);
    chirB = dimB(3);
    if dA ~= dB
        disp('Physical dimensions of the two MPSs do not match');
        return
    end
    % Build the matrices A and B from the j-th element of the physical
    % dimension of the tensors MA and MB. Sum the products A'*C*B over all
    % j's, as in Eq. (95). Here C is the result of the previous iteration
    C1 = zeros(chirA,chirB);
    for j=1:dA
        A = reshape(MA(:,j,:), chilA, chirA);
        B = reshape(MB(:,j,:), chilB, chirB);
        C1 = C1 + A' * C * B;
    end
    C = C1;   % Update C with the result of the new iteration
end

% Special treatment of the rightmost tensors in the two MPSs. Here the
% right bond dimension is 1 and is not explicit in the tensor structure, as
% the two tensors MA and MB are simply chi_left x d matrices.
MA = mps1{nt};
dimA = size(MA);
chilA = dimA(1);
dA = dimA(2);
MB = mps2{nt};
dimB = size(MB);
chilB = dimB(1);
dB = dimB(2);
if dA ~= dB
    disp('Physical dimensions of the two MPSs do not match');
    return
end
% Simply replace chirA and chirB with 1. Now the matrix product A'*C*B in
% the loop is simply a scalar, and the sum over j is the value of the inner
% product.
for j=1:dA
    A = reshape(MA(:,j,:), chilA, 1);
    B = reshape(MB(:,j,:), chilB, 1);
    innerp_val = innerp_val + A' * C * B;
end


end
