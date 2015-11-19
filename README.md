# Instrumental Variables

[![Build Status](https://travis-ci.org/gragusa/InstrumentalVariables.jl.svg?branch=master)](https://travis-ci.org/gragusa/InstrumentalVariables.jl)

`InstrumentalVariables` is a Julia package for instrumental variables estimation.

At the moment the API is pretty limited, but all typical functionality are covered. In particular, the package is interfaced to `CovarianceMatrices` so that inference can be based on robust (to heteroskedasticity and/or autocorrelation) variance covariance estimators.

# Install

At the moment the package is not registered, so it can be installed by cloning the github's repository:
```
Pkg.clone("")
```

# Usage

A the moment you can estimate an IV model by
```
iv = rndiv(x, z, y);
```
where `x` is the `Matrix{Float64}` of regressors in the original reduced form model, and `z` is the `Matrix{Float64}` of instruments. A show method gives
```
show(iv)
```
which is the output of a `coeftable` method call.

## Robust variances

A different variance from the one to which `InstrumentalVariables` defaults to --- heteroskedasticity robust of type HC0 --- can be obtained by
```
vcov(iv, ::CovarianceMatrices::RobustVariance)
```

So for instance,
```
vcov(iv, HC1())
```
gives the variance based on HC1 type heteroskedasticity robust variance matrices (which differs from `HC0` by a scala factor).

Other heteroskedastic robust variances are `HC2`, `HC3`, `HC4`, `HC4m`, `HC5`.
