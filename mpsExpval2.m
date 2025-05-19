function expval = mpsExpval2(mps,U1,U2,jt1,jt2)

% Assumes mps is in mixed canonical form centered on jt1. Assumes jt2>=jt1


T = mps{jt1};
dims = size(T);
switch jt1
    case 1
        T1 = U1 * T;
        C = T' * T1;
    case length(mps)
        T1 = U1 * T.';
        T1 = T1.';
        C = T' * T1;
    otherwise
        T1 = permute(T,[2,1,3]);
        T1 = reshape(T1,2,[]);
        T1 = U1 * T1;
        T1 = reshape(T1,[2,dims(1),dims(3)]);
        T1 = ipermute(T1,[2,1,3]);
        C = reshape(T,dims(1)*dims(2),dims(3))' * reshape(T1,dims(1)*dims(2),dims(3));
end

% Always following Eq. (95) of arXiv:1008.3477, now loop on the middle
% tensors of the mps
for k=jt1+1:jt2-1
    MA = mps{k};
    dimA = size(MA);
    chilA = dimA(1);
    dA = dimA(2);
    chirA = dimA(3);
    % Build the matrices A and B from the j-th element of the physical
    % dimension of the tensors MA and MB. Sum the products A'*C*B over all
    % j's, as in Eq. (95). Here C is the result of the previous iteration
    A = reshape(MA, chilA * dA, chirA);
    A = A';
    B = reshape(MA, chilA, chirA * dA);
    C1 = C * B;
    C1 = reshape(C1,chilA * dA, chirA);
    C = A * C1; % Update C with the result of the new iteration
end

T = mps{jt2};
switch jt2
    case 1
        T1 = U2 * T;
    case length(mps)
        T1 = U2 * T.';
        T1 = T1.';
    otherwise
        T1 = permute(T,[2,1,3]);
        T1 = reshape(T1,2,[]);
        T1 = U2 * T1;
        T1 = reshape(T1,[2,dims(1),dims(3)]);
        T1 = ipermute(T1,[2,1,3]);
end
dimA = size(T);
chilA = dimA(1);
dA = dimA(2);
if jt2==length(mps)
    chirA = 1;
else
    chirA = dimA(3);
end
% Build the matrices A and B from the j-th element of the physical
% dimension of the tensors MA and MB. Sum the products A'*C*B over all
% j's, as in Eq. (95). Here C is the result of the previous iteration
A = reshape(T, chilA * dA, chirA);
A = A';
B = reshape(T1, chilA, chirA * dA);
C1 = C * B;
C1 = reshape(C1,chilA * dA, chirA);
expval = trace(A * C1); % Update C with the result of the new iteration


end
