using InstrumentalVariables
using Base.Test
# write your own tests here
#@test 1 == 1

function randiv(;n::Int64        = 100,
                m::Int64         = 5,
                k::Int64         = 1,
                theta0::Float64  = 0.0,
                rho::Float64     = 0.9,
                CP::Int64        = 20)
    randiv(n, m, k, theta0, rho, CP)
end

function randiv(n::Int64        = 100,
                m::Int64         = 5,
                k::Int64         = 1,
                theta0::Float64  = 0.0,
                rho::Float64     = 0.9,
                CP::Int64        = 20)
    tau     = fill(sqrt(CP/(m*n)), m)
    z       = randn(n, m)
    vi      = randn(n, 1)
    eta     = randn(n, 1)
    epsilon = rho*eta+sqrt(1-rho^2)*vi
    BLAS.gemm!('N', 'N', 1.0, z, tau, 1.0, eta)
    BLAS.gemm!('N', 'N', 1.0, eta, [theta0], 1.0, epsilon)
    return epsilon, eta, z
end

srand(1)
y, x, z = randiv(n = 500, k = 3, m = 15);
cl = repmat(collect(1:25), 20)

iivv = ivreg(x,z,reshape(y, 500))

mut = [0.18402423026700765,-0.18069781054811473,-0.10666434926774973,
       0.1716025154131859,-0.0794834970843437,0.1728389042278303,
       -0.19455903354864845,-0.04328538690391654,0.3573882542662863]

mu = predict(iivv)
@test maximum(mu[1:9] - mut)<1e-10

ut = [0.06743220393049221,-0.33965003523101844,-0.2636323254480044,
      0.41789439202294165,0.09369677251530516,1.357234945075299,
      -1.1479591516308787,-0.34120715739919666,0.012435837904900104]

u = residuals(iivv)
@test maximum(u[1:9] - ut) < 1e-10


betat = [0.26050875643847554]
@test maximum(betat - coef(iivv)) < 1e-09

@test_approx_eq stderr(iivv) [0.15188237754059922]
@test_approx_eq stderr(iivv, HC1()) [0.16119631706495913]
@test_approx_eq stderr(iivv, HC2()) [0.16159542810823724]
@test_approx_eq stderr(iivv, HC3()) [0.16216172762549416]
@test_approx_eq stderr(iivv, HC4()) [0.16295579476374497]
@test_approx_eq stderr(iivv, HC4m()) [0.16242050216392523]
@test_approx_eq stderr(iivv, HC5()) [0.16437294390550466]

data = readtable("../../InstrumentalVariables/test/iv_test.csv")

y = convert(Array, data[:y])
x = convert(Array, data[:x])
x = reshape(x, length(x), 1)
z = convert(Array, data[3:17])
w = convert(Array, data[:weight])
cl = round(Int, convert(Array, data[:cluster]))
iivv = ivreg(x,z,reshape(y, 500), wts = w)

@test_approx_eq sqrt(vcov(iivv, HC1()))  [0.16634761896913675]
@test_approx_eq sqrt(vcov(iivv, HC3()))  [0.16744380411206328]


@test_approx_eq sqrt(vcov(iivv, CRHC0(cl)))  [0.17650498286330474]
@test_approx_eq sqrt(vcov(iivv, CRHC1(cl)))  [0.18014464378074402]
@test_approx_eq sqrt(vcov(iivv, CRHC2(cl)))  [0.18100382220522054]
@test_approx_eq sqrt(vcov(iivv, CRHC3(cl)))  [0.18961072793672662]

coeftable(iivv)
coeftable(iivv, HC1())
coeftable(iivv, CRHC1(cl))

# srand(1)
# y, x, z = randiv(n = 25000, k = 3, m = 15);
# add_x = randn(25000, 20)
# x = [x add_x]
# z = [z add_x]
# cl = repmat([1:50], 50)
# ww = rand(25000)

# println("Timing of ivreg")
# @time iivv = ivreg(x,z,reshape(y, 25000), wts = ww)




## srand(7875)

## y = randn(100); x = randn(100,3); z = randn(100, 10);

## function ivreg(y, x, z)
##     zz = PDMat(z'z)
##     Pz = X_invA_Xt(zz, z)
##     xPz= x'*Pz
##     reshape(xPz*x\xPz*y, size(x)[2])
## end

## bas = ivreg(y, x, z);
## out = iv(x,z,y);
## vcov(out)
## using CovarianceMatrices
## vcov(out, HC0())

# data = readtable("../../InstrumentalVariables/test/iv_test.csv")

# y = convert(Array, data[:y])
# x = convert(Array, data[:x])
# x = reshape(x, length(x), 1)
# z = convert(Array, data[3:17])
# w = convert(Array, data[:weight])
# cl = convert(Array, data[:cluster])
# iivv = ivreg(x,z,reshape(y, 500), wts = w)

# xw = x.*sqrt(w)
# zw = z.*sqrt(w)
# yw = y.*sqrt(w)

# a = inv((xw'zw)*inv(zw'zw)*(zw'xw))
# b = (xw'zw)*inv(zw'zw)*(zw'yw)

# bread_ = inv((xw'zw)*inv(zw'zw)*(zw'xw))
# u = yw-xw*(a*b)
# meat_  = (xw'zw)*inv(zw'zw)*(zw.*u)'
# meat_  = meat_*meat_'
# sqrt(bread_*(meat_)*bread_)





# k = HC0()
# u = yw-xw*(a*b)
# X = copy(ModelMatrix(iivv))
# Z = X.*u
# CovarianceMatrices.adjfactor!(u, iivv, k)
# Base.LinAlg.At_mul_B(Z, Z.*u)
