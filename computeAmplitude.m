function amp = computeAmplitude(mps, physIndices)
% computeAmplitude Computes the amplitude of the quantum state represented by an MPS.
%
%   amp = computeAmplitude(mps, physIndices) computes the amplitude for the 
%   computational basis state specified by physIndices.
%
%   For each tensor, the function "slices" the tensor at the
%   given physical index and contracts (multiplies) the resulting matrices.
%
%   The MPS is assumed to have the following structure:
%       - mps{1} is of size [d, chi]      (leftmost tensor)
%       - mps{end} is of size [chi, d]      (rightmost tensor)
%       - mps{i} for i=2,...,end-1 is of size [chi1, d, chi2] (middle tensors)
%
% Input:
%   mps    - Cell array containing the MPS tensors.
%   physIndices - An array of length equal to the number of tensors in the
%   MPS, defining the computational basis state whose amplitude is
%   computed. Example: physIndices = [1, 2, 2, 1, 1, 2, 1, 2]; for 8 qubits
%   (1 = 0, 2 = 1)
%
% Output:
%   amp - The computed amplitude (a scalar).
%

    numTensors = length(mps);
    
    if length(physIndices) ~= numTensors
        error('The length of the bit string must equal the number of tensors.');
    end
    
    % --- Contract the MPS ---
    % Process the leftmost tensor: dimensions [d, chi].
    T1 = mps{1};
    d1 = physIndices(1);
    % Select the row corresponding to d1: result is a 1 x chi row vector.
    C = T1(d1, :);
    
    % Process the middle tensors: dimensions [chi, d, chi].
    for i = 2:numTensors-1
        T = mps{i};
        d_val = physIndices(i);
        % Slice T along its physical index (the 2nd index).
        % T(:, d_val, :) is a 3D array of size [chi, 1, chi]; reshape it into [chi, chi].
        M = reshape(T(:, d_val, :), size(T,1), size(T,3));
        % Multiply: (1 x chi) * (chi x chi) = (1 x chi)
        C = C * M;
    end
    
    % Process the rightmost tensor: dimensions [chi, d].
    T_last = mps{numTensors};
    d_last = physIndices(numTensors);
    % Select the column corresponding to d_last: result is a chi x 1 vector.
    M_last = T_last(:, d_last);
    
    % Final contraction yields a scalar.
    amp = C * M_last;
end
