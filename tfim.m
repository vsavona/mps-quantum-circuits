%% IBM Eagle quantum utility experiment

% Parameters
nqubits = 100;
d = 2;            % physical dimension for each tensor
dt = 0.05;
h = 1.0;
pbc = 1;
gpu = 0;

nstep_trotter = 100;   % Number of Trotter steps

chin = (300);
nchi = length(chin);

% Define one-qubit gates.
X = [0 1; 1 0];
Y = [0 -1i; 1i 0];
Z = [1 0; 0 -1];
H = [1 1; 1 -1] ./ sqrt(2);
SWAP = [1 0 0 0; 0 0 1 0; 0 1 0 0; 0 0 0 1];

U1 = expm(1i .* dt .* h .* X ./2);
U2 = expm(1i .* dt .* kron(Z,Z));

for jchi=1:nchi
    chi = chin(jchi);
    % Name of Savefile
    afile = strcat('TFIM_',sprintf('%d',nqubits),'_chi',sprintf('%d',chi),'_pbc',sprintf('%d',pbc),'_h',sprintf('%0.3f',h),'.mat');

    Mc = zeros(nqubits, nstep_trotter);
    Mt = zeros(nstep_trotter,1);

    % Initialize the MPS.
    mps = initMPS(d, 2, nqubits, H);

    for jtrotter = 1:nstep_trotter

        jtrotter

        tic
        % Apply single-qubit gates
        for jqubit = 1:nqubits
            mps = applySingleQubitGate(mps, U1, jqubit);
        end

        % Apply two-qubit gates to pairs of qubits in neighboring groups
        mps = mixedCanonicalizeMPS(mps, 0);
        for jqubit = 1:nqubits-1
            % jqubit
            mps = mixedCanonicalizeMPSpartial(mps, jqubit, jqubit-1);
            mps = applyTwoQubitGateNeighboring(mps, U2, jqubit, chi, gpu);
        end

        if pbc~=0
            tic
            mps1 = applySingleQubitGate(mps,Z,1);
            mps1 = applySingleQubitGate(mps1,Z,nqubits);
            % mpsNorm(mps1)
            mps = mpsDirectSum(mps, mps1, cos(dt), 1i .* sin(dt));
            mps = compressMPS(mps,chi);
            % mpsNorm(mps)
            toc
        end

        % Apply single-qubit gates
        for jqubit = 1:nqubits
            mps = applySingleQubitGate(mps, U1, jqubit);
        end

        mps = mpsNormalize(mps);
        toc

        tic
        % Compute observable expectation value
        % In this example, compute the full magnetization as the sum of the
        % expectation value of Z on each qubit, normalized by nqubits
        for jqubit=1:nqubits
            mps = mixedCanonicalizeMPSpartial(mps,jqubit-1,jqubit);
            temp = real(mpsExpval1(mps,X,jqubit));
            Mc(jqubit,jtrotter) = temp;
        end
        Mt(jtrotter) = sum(Mc(:,jtrotter)) ./ nqubits;
        plot(dt*jtrotter,Mt(jtrotter),'o')
        hold on
        drawnow
        toc

    end

    clear mps
    save(afile)

end

