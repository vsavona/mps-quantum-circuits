function mps = applySingleQubitGate(mps, U1, jqubit)
% applySingleQubitGate Applies a single-qubit gate to a specified qubit of the MPS.

% Determine the tensor and its dimensions.
T = mps{jqubit};
nt = length(mps);
dims = size(T);

% Identify which dimension is the physical index.
% For leftmost tensor (jt==1): T has shape [d, chi] and physical index is 1.
% For rightmost tensor (jt==end): T has shape [chi, d] and physical index is 2.
% For a middle tensor: T has shape [chi, d, chi] and physical index is 2.
if jqubit == 1
    physIdx = 1;
else
    physIdx = 2;
end

% Get the physical dimension.
d = dims(physIdx);

switch jqubit
    case 1
        T = U1 * T;
        mps{jqubit} = T;
    case nt
        T = permute(T, [2, 1]);
        T = U1 * T;
        T = ipermute(T, [2, 1]);
        mps{jqubit} = T;
    otherwise
        T = permute(T, [2, 1, 3]);
        T = reshape(T, d, []);
        T = U1 * T;
        T = reshape(T, d, dims(1), dims(3));
        T = ipermute(T, [2, 1, 3]);
        mps{jqubit} = T;
end
        
