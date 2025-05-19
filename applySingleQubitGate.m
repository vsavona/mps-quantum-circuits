function mps = applySingleQubitGate(mps, U1, jqubit)
% applySingleQubitGate Applies a single-qubit unitary gate to a specified qubit
%    in a matrix-product state (MPS).
%
%   mps = applySingleQubitGate(mps, U1, jqubit)
%
%   Inputs:
%     mps     - Cell array of tensors representing the MPS.
%               * Left boundary tensor has size [d, chi].
%               * Middle tensors have size [chi, d, chi].
%               * Right boundary tensor has size [chi, d].
%     U1      - d-by-d unitary matrix acting on one physical qubit.
%     jqubit  - Index of the tensor (site) where the gate is applied,
%               with 1 ≤ jqubit ≤ length(mps).
%
%   Output:
%     mps     - Updated MPS with U1 applied to the physical index of site jqubit.
%
%   This routine handles three cases separately:
%     1) Leftmost tensor (jqubit == 1)
%     2) Rightmost tensor (jqubit == nt)
%     3) Any middle tensor (2 ≤ jqubit ≤ nt−1)
%
%   In each case we:
%     - Identify which tensor index corresponds to the physical leg.
%     - Reshape or permute the tensor so that U1 multiplies directly on that leg.
%     - Undo the permutation/reshape to restore original tensor shape.

    % Extract the tensor at site jqubit and its size.
    T = mps{jqubit};
    nt = length(mps);      % Total number of sites in the MPS
    dims = size(T);        % dims indicates the dimensions of T

    % Determine which index of T corresponds to the physical index (dimension d).
    %   - If jqubit == 1 (left boundary): T is [d, chi], so physIdx = 1.
    %   - Otherwise (middle or right boundary): T is [chi, d, ...] or [chi, d],
    %     so physIdx = 2.
    if jqubit == 1
        physIdx = 1;
    else
        physIdx = 2;
    end

    % Read off the physical dimension d from the appropriate index.
    d = dims(physIdx);

    % Dispatch based on whether this is the left boundary, right boundary,
    % or a middle tensor, applying U1 to the correct mode.
    switch jqubit

        case 1
            % ----------------------------------------------------------------
            % Left boundary (site 1):
            %   T has shape [d, chi]. Multiply U1 on rows directly.
            %   Resulting T is still [d, chi].
            T = U1 * T;
            mps{jqubit} = T;

        case nt
            % ----------------------------------------------------------------
            % Right boundary (last site):
            %   T has shape [chi, d]. We need to apply U1 on the second index:
            %   1) Permute to [d, chi]
            %   2) Multiply U1 on rows (physical index)
            %   3) Permute back to [chi, d]
            T = permute(T, [2, 1]);    % now size [d, chi]
            T = U1 * T;                % apply on physical index
            T = ipermute(T, [2, 1]);   % back to [chi, d]
            mps{jqubit} = T;

        otherwise
            % ----------------------------------------------------------------
            % Middle tensor (sites 2..nt-1):
            %   T has shape [chi_left, d, chi_right].
            %   We must apply U1 on the second dimension:
            %   1) Permute so that physical index is first: [d, chi_left, chi_right]
            %   2) Reshape to a 2D matrix of size [d, chi_left*chi_right]
            %   3) Multiply U1 on rows
            %   4) Reshape back to [d, chi_left, chi_right]
            %   5) Inverse-permute to [chi_left, d, chi_right]
            T = permute(T, [2, 1, 3]);              % [d, chi_left, chi_right]
            T = reshape(T, d, []);                  % [d, chi_left*chi_right]
            T = U1 * T;                             % apply U1 on physical index
            T = reshape(T, d, dims(1), dims(3));    % [d, chi_left, chi_right]
            T = ipermute(T, [2, 1, 3]);             % back to [chi_left, d, chi_right]
            mps{jqubit} = T;
    end
end

