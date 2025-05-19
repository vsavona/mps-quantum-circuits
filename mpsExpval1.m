function expval = mpsExpval1(mps,U1,jq)

% Assumes mps is in mixed canonical form centered on jt
T = mps{jq};
dims = size(T);
switch jq
    case 1
    T1 = U1 * T;
    expval = trace(T' * T1);
    case length(mps)
    T1 = U1 * T.';
    T1 = T1.';
    expval = trace(T' * T1);
    otherwise
    T1 = permute(T,[2,1,3]);
    T1 = reshape(T1,2,[]);
    T1 = U1 * T1;
    T1 = reshape(T1,[2,dims(1),dims(3)]);
    T1 = ipermute(T1,[2,1,3]);
    expval = trace(reshape(T,dims(1)*dims(2),dims(3))' * reshape(T1,dims(1)*dims(2),dims(3)));
end



end


