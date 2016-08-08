# Instrumental Variables

[![Build Status](https://travis-ci.org/gragusa/InstrumentalVariables.jl.svg?branch=master)](https://travis-ci.org/gragusa/InstrumentalVariables.jl)

[![Coverage Status](https://coveralls.io/repos/gragusa/InstrumentalVariables.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/gragusa/InstrumentalVariables.jl?branch=master) [![codecov.io](https://codecov.io/github/gragusa/InstrumentalVariables.jl/coverage.svg?branch=master)](https://codecov.io/github/gragusa/InstrumentalVariables.jl?branch=master)

`InstrumentalVariables` is a Julia package for instrumental variables estimation.

At the moment the API is pretty limited, but all typical functionality are covered. In particular, the package is interfaced to `CovarianceMatrices` so that inference can be based on robust (to heteroskedasticity and/or autocorrelation) variance covariance estimators.

## Install

At the moment the package is not registered, so it can be installed by cloning the github's repository:
```
Pkg.clone("https://github.com/gragusa/InstrumentalVariables.jl.git")
```

## Quick usage guide

A IV model can be estimated by

```
ivreg(x, z, y)
```
where `x` is the `Matrix{Float64}` of regressors in the original reduced form model, `z` is the `Matrix{Float64}` of instruments, and `y` is a `Array{Float64, 1}`.

A different variance from the one to which `InstrumentalVariables` defaults to --- heteroskedasticity robust of type HC0 --- can be obtained by
```
vcov(iv, ::CovarianceMatrices.RobustVariance)
```

So for instance,
```
vcov(iv, HC1())
```
gives the variance based on HC1 type heteroskedasticity robust variance matrices (which differs from `HC0` by a scale factor).

Other heteroskedastic robust variances are `HC2`, `HC3`, `HC4`, `HC4m`, `HC5`.

## Example

We consider estimating the tax elasticity of alcohol demand.

```
using RDatasets
using DataFramesMeta
cig = dataset("Ecdat", "Cigarette")
cig[:lrincome] = @with(cig, log(:Income./:POP./:CPI));
cig[:lrprice] = @with(cig, log(:AvgPrs./:CPI));
cig[:tdiff] = @with(cig, (:Taxs - :Tax)./:CPI)
cig[:rtax] = @with(cig, :Tax./:CPI)
cig95 = @ix(cig, :Year .== 1995)
y = convert(Array, log(cig95[:PackPC]));
x = convert(Array, [ones(size(cig95,1)) cig95[:lrincome] cig95[:lrprice]])
z = convert(Array, [ones(size(cig95,1)) cig95[:lrincome] cig95[:tdiff] cig95[:rtax]])
```

The model can be estimated by

```
iv = ivreg(x, z, y)
```

By default, the `show` method of `LinearIvModel` will print a
`coeftable` with standard errors obtained by assuming conditional
heteroskedasticity. That is, the output of 

```
iv = ivreg(x, z, y)
```
is equivalent to the output of 
```
coeftable(iv)
```

If robust standard errors are sought, an interface with the
`CovarianceMatrices` package allows to obtain them easily. For instance,
the `HC1` type standard errors (these are the one calculated by default
in STATA) are obtained by

```
coeftable(HC1())
```

If one only needs the variances (or the standard errors) 
```
vcov(iv, HC1())
stderr(iv, HC1())
```

## To do

- [ ] Improve API
- [ ] Add formula interface
- [ ] Add first stage information
- [ ] Add test statistics for validity of instruments
- [ ] Add support for sparse IV
