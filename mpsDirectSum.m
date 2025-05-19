function mps = mpsDirectSum(mps1, mps2, a, b)
% mpsDirectSum computes the direct sum of two matrix-product state (MPS) representations.
%
% This function takes two MPS's (mps1 and mps2) as input, where each MPS is a 
% cell array. Each cell contains a tensor (matrix) corresponding to one site (group)
% in the MPS. The physical index of each tensor corresponds to a quantum state of
% the local Hilbert space, while the auxiliary (or bond) indices connect the tensors.
%
% In the direct-sum construction, the new MPS is defined by taking, at each site,
% the direct sum (i.e., the block diagonal combination) of the two matrices (or tensors)
% from mps1 and mps2. This corresponds to an MPS sum where the matrices for the
% corresponding physical indices are joined into larger block–diagonal matrices.
%
% Note that the boundary tensors (at the left and right edges of the MPS) are treated
% differently from those in the bulk. This follows the guidelines of the work by Schollwöck,
% "The density-matrix renormalization group in the age of matrix product states"
% (see Section 4.3), where the left boundary is formed by a horizontal concatenation
% (i.e. combining the row vectors) while the right boundary is formed by a vertical
% concatenation (i.e. stacking the column vectors).
%
% Input:
%   mps1 - Cell array representing the first MPS.
%   mps2 - Cell array representing the second MPS.
%
% Output:
%   mps  - Cell array representing the MPS corresponding to the direct sum of mps1 and mps2.
%
% The procedure implemented in this function is as follows:
%   1. The left boundary tensor (site 1) is updated by concatenating the corresponding
%      matrices from mps1 and mps2 side-by-side (horizontal concatenation).
%
%   2. The right boundary tensor (last site) is updated by concatenating the corresponding
%      matrices from mps1 and mps2 vertically. (Note: the provided code concatenates mps1{end}
%      with itself. You may wish to replace one of these with mps2{end} depending on the desired behavior.)
%
%   3. For the tensors in the bulk (sites 2 to N-1):
%      a. Extract the tensors from both mps1 and mps2.
%      b. Determine the sizes (dimensions) of the tensors. In a typical MPS in grouped–qubit
%         form, the tensor for a bulk site has three indices: left bond, physical index, right bond.
%      c. Allocate a new tensor T of appropriate size such that its left bond dimension
%         is the sum of the left bond dimensions from mps1 and mps2, and similarly for
%         the right bond.
%      d. In order to build the block–diagonal structure for each physical index, we first
%         permute the tensor dimensions so that the physical index is the third dimension.
%      e. Loop over the physical indices and, for each value, use blkdiag to form the direct
%         sum (block diagonal matrix) of the corresponding matrices from mps1 and mps2.
%      f. Finally, invert the permutation to restore the original ordering of the indices.
%

    %% Determine the total number of sites in the MPS
    % Assume mps1 is a cell array where the number of elements indicates the number
    % of tensors (sites) in the MPS.
    dims = size(mps1);
    
    % Pre-allocate the output cell array "mps" with the same dimensions as mps1.
    mps = cell(dims);

    %% Process the left boundary tensor (site 1)
    % For the left boundary, the MPS convention (following Schollwöck) is to form
    % the direct sum via horizontal concatenation. This means combining the row vectors
    % from mps1{1} and mps2{1} into one matrix.
    mps{1} = horzcat(a .* mps1{1}, b .* mps2{1});
    
    %% Process the right boundary tensor (last site)
    % For the right boundary, the direct sum is taken via vertical concatenation, i.e.,
    % stacking the matrices on top of each other.
    mps{dims(2)} = vertcat(mps1{dims(2)}, mps2{dims(2)});

    %% Process the bulk tensors (sites 2 to N-1)
    % Loop over all bulk sites. For each such site, we will build a new tensor by taking
    % the direct sum (i.e., block diagonal concatenation) of the corresponding tensors
    % from mps1 and mps2.
    for j = 2:dims(2)-1
        % Extract the tensor for site j from both mps1 and mps2.
        T1 = mps1{j};  % Tensor from first MPS
        dims1 = size(T1);  % dims1 = [leftBond1, physicalDim, rightBond1]
        
        T2 = mps2{j};  % Tensor from second MPS
        dims2 = size(T2);  % dims2 = [leftBond2, physicalDim, rightBond2]
        
        % Allocate a new tensor T to hold the direct sum result.
        % New left bond dimension is (dims1(1) + dims2(1)) and new right bond dimension is (dims1(3) + dims2(3)).
        % The physical dimension remains the same (dims1(2)), which we assume equals dims2(2).
        T = zeros(dims1(1) + dims2(1), dims1(3) + dims2(3), dims1(2));
        
        % To facilitate forming block diagonal matrices for each slice in the physical index,
        % we first permute T1 and T2 so that the physical dimension is the third index.
        % The permutation [1, 3, 2] moves the current second index (physical) to the third position.
        T1 = permute(T1, [1, 3, 2]);
        T2 = permute(T2, [1, 3, 2]);
        
        % Loop over the physical index. For each value (each slice) we form the direct sum
        % by placing T1 and T2 on the diagonal in a block diagonal matrix.
        for k = 1:dims1(2)
            % blkdiag constructs a block diagonal matrix given T1(:,:,k) and T2(:,:,k).
            T(:, :, k) = blkdiag(T1(:, :, k), T2(:, :, k));
        end
        
        % After constructing the block diagonal matrices for every physical index,
        % we reverse the earlier permutation to restore the original ordering of the indices.
        mps{j} = ipermute(T, [1, 3, 2]);
    end

end
