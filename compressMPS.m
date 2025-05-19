function mps = compressMPS(mps, chi)
% compressMPS compresses a matrix-product state (MPS) to a fixed bond dimension.
%
%   mps = compressMPS(mps, chi)
%
% This function takes an MPS (a cell array of tensors) and a target maximum bond 
% dimension chi. It first casts the MPS into left–canonical form (by using 
% leftCanonicalizeMPS) and then compresses it by sweeping from the rightmost tensor
% leftward, truncating bonds via singular–value decomposition (SVD).
%
% IMPORTANT ASSUMPTIONS ON TENSOR SHAPES:
%   - Left boundary: the tensor is provided as a matrix [d, chi]. Internally it will 
%     be reshaped to [1, d, chi].
%   - Right boundary: the tensor is provided as a matrix [chi, d]. Internally it will 
%     be interpreted with an extra trailing dimension (i.e. size [chi, d, 1]). Note that 
%     MATLAB’s reshape does not change the apparent number of dimensions if the trailing 
%     dimensions are 1, so here we explicitly enforce a three–element size when needed.
%   - Bulk tensors: assumed to be 3D arrays with dimensions [chi_left, d, chi_right].
%
% The compression procedure at each bond (from rightmost bond to left) proceeds as follows:
%
%  1. For tensor M(i) (site i), first ensure its size is interpreted as 3–dimensional.
%     (If it is 2D, we append a trailing "1" so that size becomes [chi_left, d, chi_right].)
%
%  2. Reshape tensor M(i) into a matrix A by merging its left bond index into the row 
%     index and combining the physical and right bond indices as columns.
%
%  3. Compute the economy SVD: A = U*S*V' using svd(A, 'econ').
%
%  4. Truncate U, S, and V to the target bond dimension chi (or fewer if A has lower rank).
%     Optionally, one can renormalize S to preserve the norm.
%
%  5. Replace tensor M(i) with the reshaped truncated right factor (V_trunc').
%
%  6. Multiply the truncated left factor (U_trunc * S_trunc) into the left–neighbor tensor 
%     M(i-1). To do this, the left–neighbor tensor is reshaped appropriately, multiplied by 
%     the factor, and then reshaped back into a three–index tensor with the new bond dimension.
%
%  7. Iterate the procedure from the right boundary down to site 2.
%
% Inputs:
%   mps - Cell array of tensors representing the MPS.
%   chi - The target maximum bond dimension.
%
% Output:
%   mps - The compressed MPS with all bond dimensions (internal bonds) at most chi.
%
% See also:
%   leftCanonicalizeMPS

    %% Step 0: Cast the MPS into left-canonical form.
    % This improves stability and ensures consistency for the subsequent SVD truncations.
    mps = leftCanonicalizeMPS(mps);
    
    %% Step 1: Compress the MPS by sweeping from the rightmost tensor toward the left.
    L = length(mps); % Total number of sites
    
    for i = L:-1:2
        %---------------------------------------------------------------------
        % Step 1a: Ensure the current tensor mps{i} is interpreted as 3-dimensional.
        % For a boundary tensor stored as a 2D array (e.g. right boundary of size [chi, d])
        % we want to treat it as if its size were [chi, d, 1].
        dims_i = size(mps{i});
        if numel(dims_i) < 3
            % Append trailing ones so that dims_i becomes a three-element vector.
            dims_i = [dims_i, ones(1,3-numel(dims_i))];
            % We do not need to reshape the data because MATLAB’s internal data
            % already represent mps{i} as having the same number of elements.
            % We will use dims_i when reshaping below.
        end
        % dims_i is now: [chi_left_i, d_i, chi_right_i] (with chi_right_i==1 for right boundary).
        chi_left_i = dims_i(1);  % Bond dimension on the left of site i.
        d_i = dims_i(2);         % Physical dimension at site i.
        chi_right_i = dims_i(3); % Bond dimension on the right of site i.
        
        %---------------------------------------------------------------------
        % Step 2: Reshape tensor mps{i} into a 2D matrix A suitable for SVD.
        % We merge the left bond index into the rows, and fuse the physical and right bond
        % indices into the columns.
        % The resulting matrix A has size [chi_left_i, d_i*chi_right_i].
        A = reshape(mps{i}, [chi_left_i, d_i * chi_right_i]);
        
        %---------------------------------------------------------------------
        % Step 3: Perform the economy SVD on A.
        % That is, decompose A = U * S * V' (using svd(A, 'econ')).
        [U, S, V] = svd(A, 'econ');
        
        %---------------------------------------------------------------------
        % Step 4: Truncate U, S, and V to the target bond dimension chi.
        r_full = size(S, 1);      % Full rank of the decomposition.
        r = min(chi, r_full);       % New bond dimension (cannot exceed the intrinsic rank).
        
        % Retain only the first r columns/rows.
        U_trunc = U(:, 1:r);
        S_trunc = S(1:r, 1:r);
        V_trunc = V(:, 1:r);
        
        % Optionally, renormalize S_trunc to preserve the norm of the state.
        normS = norm(diag(S_trunc));
        if normS > 0
            S_trunc = S_trunc / normS;
        end
        
        %---------------------------------------------------------------------
        % Step 5: Update the current tensor at site i.
        % V_trunc is of size [d_i*chi_right_i, r]. We interpret V_trunc' (which has size [r, d_i*chi_right_i])
        % as the new factor and reshape it back into a 3-index tensor of size [r, d_i, chi_right_i].
        new_tensor_i = reshape(V_trunc', [r, d_i, chi_right_i]);
        mps{i} = new_tensor_i;
        
        %---------------------------------------------------------------------
        % Step 6: "Push" the truncated factor (U_trunc * S_trunc) into the left-neighbor tensor mps{i-1}.
        % First, prepare mps{i-1} as a 3-index tensor. For the left boundary,
        % mps{1} is provided as a matrix [d, chi] (and should be interpreted as [1,d,chi]).
        dims_left = size(mps{i-1});
        if numel(dims_left) < 3
            if i-1 == 1
                % Left boundary: reshape [d, chi] to [1, d, chi].
                dims_left = [1, dims_left];
                mps{i-1} = reshape(mps{i-1}, dims_left);
                dims_left = [size(mps{i-1}), 1];  % Ensure three elements.
            else
                % For a non-boundary tensor stored as 2D, append a trailing singleton.
                dims_left = [dims_left, ones(1, 3-numel(dims_left))];
                mps{i-1} = reshape(mps{i-1}, dims_left);
            end
        end
        % dims_left now is [chi_left_prev, d_prev, chi_right_prev],
        % where chi_right_prev is the bond connecting site i-1 to site i.
        chi_left_prev = dims_left(1);
        d_prev = dims_left(2);
        chi_right_prev = dims_left(3);
        
        % Reshape mps{i-1} into a matrix A_left. For site i-1,
        % merge its physical index and its right bond: [chi_left_prev * d_prev, chi_right_prev].
        A_left = reshape(mps{i-1}, [chi_left_prev * d_prev, chi_right_prev]);
        
        % U_trunc * S_trunc has size [chi_right_prev, r] (note: chi_right_prev should match chi_left_i).
        % Multiply A_left by the factor to incorporate the compression from site i.
        A_left_new = A_left * (U_trunc * S_trunc);
        
        % Reshape the updated left tensor back into its 3-index form.
        % It now has size [chi_left_prev, d_prev, r] where r is the new bond dimension.
        mps{i-1} = reshape(A_left_new, [chi_left_prev, d_prev, r]);
    end
    
    % After the loop, all bonds in the MPS have been compressed to a maximum dimension of chi.
    dims = size(mps{1});
    mps{1} = reshape(mps{1},dims(2),dims(3));
end
