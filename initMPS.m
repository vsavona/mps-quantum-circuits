function mps = initMPS(d, chi, nt, U)
% initMPS Initialize a Matrix Product State (MPS) as a cell array of tensors.
%
%   mps = initMPS(d, chi, nt, U) returns a cell array 'mps' of length nt,
%   where each cell contains a tensor initialized to zeros.
%
%   The dimensions of the tensors are as follows:
%       - Leftmost tensor:  (d, chi)
%       - Rightmost tensor: (chi, d)
%       - Middle tensors:   (chi, d, chi)
%
% Input:
%   d   - Physical dimension on each tensor.
%   chi - Bond dimension.
%   nt  - Number of tensors/sites in the MPS.
%   U   - Single-site unitary, dimensions d x d 
%
% Output:
%   mps - Cell array containing the MPS tensors initialized to zeros.

% Preallocate cell array for MPS tensors
mps = cell(1, nt);

% Initialize the leftmost tensor with dimensions (d, chi)
mps{1} = zeros(d, chi);
mps{1}(1,1) = 1;

% Initialize the middle tensors (if any) with dimensions (chi, d, chi)
for i = 2:(nt-1)
    mps{i} = zeros(chi, d, chi);
    mps{i}(1,1,1) = 1;
end

% Initialize the rightmost tensor with dimensions (chi, d)
if nt > 1
    mps{nt} = zeros(chi, d);
    mps{nt}(1,1) = 1;
end

% Apply extra layer of single-size unitaries U

for jt = 1:nt
    mps = applySingleQubitGate(mps, U, jt);
end


end
