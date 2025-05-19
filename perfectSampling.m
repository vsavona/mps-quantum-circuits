function drawnSample = perfectSampling(mps)
% perfectSampling samples a bitstring from an MPS in right-canonical form.
%
%   drawnSample = perfectSampling(mps)
%
% Input:
%   mps - Cell array of tensors representing the MPS. Must be in right-canonical form,
%         so that "backward" contractions yield correct marginal probabilities.
%
% Output:
%   drawnSample - Column vector (nt x 1) of sampled physical indices (bit values) at each site.
%
% Algorithm overview:
%   We draw samples sequentially from left to right, computing the conditional
%   probability distribution of each qubit given previous samples. In right-canonical
%   MPS, the contraction from the right boundary inward yields orthonormality that
%   simplifies these marginals.

nt = length(mps);             % Number of sites (tensors) in the MPS
drawnSample = zeros(nt,1);   % Preallocate sample vector

%% 1) Sample the first qubit (site 1)
A = mps{1};                  % Leftmost tensor: size [d x chi]
dimsA = size(A);             % dimsA = [d, chi]

% Compute marginal probabilities p(j) = ||A(j,:)||^2 over the d physical outcomes
p = zeros(dimsA(1),1);
for j = 1:dimsA(1)
    row = A(j,:);            % extract the j-th row (size 1 x chi)
    p(j) = row * row';       % squared norm of row j
end

% Sample first qubit from the distribution p
cp = cumsum(p);
r = rand();
ind = find(r < cp, 1, 'first');
drawnSample(1) = ind;

% Initialize transfer operator Mt for subsequent conditional sampling
Mt = A(ind,:);               % 1 x chi row vector corresponding to chosen outcome

%% 2) Loop over middle tensors Loop over middle tensors (sites 2 to nt-1)
for jt = 2:nt-1
    A = mps{jt};                % Bulk tensor: size [chi_left x d x chi_right]
    dimsA = size(A);
    chiL = dimsA(1);
    d    = dimsA(2);
    chiR = dimsA(3);

    % Compute conditional marginal for each physical index j at site jt:
    %   p(j) = || Mt * A(:,:,j) ||^2
    p = zeros(d,1);
    for j = 1:d
        % Extract the slice Am = A(:,j,:)
        Am = reshape(A(:,j,:), [chiL, chiR]);  % [chiL x chiR]
        v  = Mt * Am;                         % [1 x chiR]
        p(j) = v * v';                        % squared norm
    end

    % Turn into a conditional distribution given previous sites
    p = p / sum(p);
    cp = cumsum(p);
    r = rand();
    ind = find(r < cp, 1, 'first');
    drawnSample(jt) = ind;

    % Update Mt by contracting with the chosen branch
    Mt = Mt * reshape(A(:,ind,:), [chiL, chiR]);  % [1 x chiR]
end

%% 3) Sample the last tensor (site nt)
A = mps{nt};                   % Right-boundary tensor: size [chi_left x d]
% Contract with Mt to get a [1 x d] vector
Amat = Mt * A;                 % [1 x d]

% Compute final conditional probabilities p(j) = |Amat(j)|^2
p = abs(Amat').^2;             % [d x 1]
p = p / sum(p);
cp = cumsum(p);
r = rand();
ind = find(r < cp, 1, 'first');
drawnSample(nt) = ind;

end
