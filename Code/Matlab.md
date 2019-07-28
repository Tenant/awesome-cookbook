## Matlab

### 1. Basic

```matlab
rng();
```



```matlab
mu = mean(Z,2);
zc = bsxfun(@minus, Z, mu);
```

```matlab
t = @(n,T) linspace(0,1,n) * 2 * pi * T;
Ztrue(1,:) = sin(t(n,T(1)));            % Sinusoid
Ztrue(2,:) = sign(sin(t(n,T(2))));      % Square
Ztrue(3,:) = sawtooth(t(n,T(3)));       % Sawtooth
```

```matlab
% Add some noise to make the signals "look" interesting
sigma = @(SNR,X) exp(-SNR / 20) * (norm(X(:)) / sqrt(numel(X)));
Ztrue = Ztrue + sigma(SNR,Ztrue) * randn(size(Ztrue));
```

```matlab
% Generate mixed signals
normRows = @(X) bsxfun(@rdivide,X,sum(X,2));
A = normRows(rand(d,3));
Zmixed = A * Ztrue;
```

```matlab
% Parse inputs
if ~exist('flag','var') || isempty(flag)
    % Default display flag
    flag = 1;
end
if ~exist('type','var') || isempty(type)
    % Default type
    type = 'kurtosis';
end
n = size(Z,2);
% Set algorithm type
if strncmpi(type,'kurtosis',1)
    % Kurtosis
    USE_KURTOSIS = true;
    algoStr = 'kurtosis';
elseif strncmpi(type,'negentropy',1)
    % Negentropy
    USE_KURTOSIS = false;
    algoStr = 'negentropy';
else
    % Unsupported type
    error('Unsupported type ''%s''',type);
end
```

