module InstrumentalVariables

using Reexport
@reexport using CovarianceMatrices
using GLM
using StatsBase

import GLM: Cholesky, FP, ModelMatrix, delbeta!, predict
import StatsBase: residuals, coeftable, stderr, vcov
import Distributions: ccdf, FDist, Chisq, Normal
import CovarianceMatrices: CRHC, adjresid!, clusterize!, wrkresidwts, RobustVariance, HC, meat

typealias FPVector{T<:AbstractFloat} DenseArray{T,1}
typealias BlasReal Union{Float32,Float64}

type IVResp{V<:FPVector} <: GLM.ModResp  # response in a linear model
    mu::V                                # mean response
    offset::V                            # offset added to predictor
    wts::V                               # prior weights (may have length 0)
    wrkresid::V                          # residual
    y::V                                 # response
    function IVResp(y::V, mu::V, off::V, wts::V)
        n = length(y); length(mu) == n || error("mismatched lengths of mu and y")
        ll = length(off); ll == 0 || ll == n || error("length of offset is $ll, must be $n or 0")
        ll = length(wts); ll == 0 || ll == n || error("length of wts is $ll, must be $n or 0")
        new(mu, off, wts, similar(y), y)
    end
end

# function IVResp{V<:FPVector}(y::V)
#     IVResp(fill!(similar(y), zero(eltype(V))), similar(y, 0),
#               similar(y, 0), similar(y, 0), y)
# end
#
# function IVResp{V<:FPVector}(y::V, off::V, wts::V)
#     IVResp(fill!(similar(y), zero(eltype(V))), off, wts,
#               similar(y, 0), y)
# end

type LinearIVModel{T<:LinPred} <: LinPredModel
    rr::IVResp
    pp::T
end

type DenseIVPredChol{T <: AbstractFloat, C} <: GLM.DensePred
    X::Matrix{T}                   # model matrix
    Z::Matrix{T}                   # instrument matrix
    beta0::Vector{T}               # base vector for coefficients
    delbeta::Vector{T}
    Xp::Matrix{T}
    chol::C
end

function DenseIVPredChol{T<:AbstractFloat}(X::Matrix{T}, Z::Matrix{T},
                         beta0::Vector{T}, wt::Vector{T})
    n,p = size(X); length(beta0) == p || error("dimension mismatch")
    if length(wt) > 0
        src = similar(Z)
        Zw = broadcast!(*, src, Z, sqrt(wt))
        src = similar(X)
        Xw = broadcast!(*, src, X, sqrt(wt))
        cholfac_Z = cholfact(Zw'Zw)
        a  = Xw'Zw
        b  = copy(Zw)
    else
        cholfac_Z = cholfact(Z'Z)
        a  = X'Z
        b  = copy(Z)
    end
    Base.LinAlg.A_rdiv_B!(a, cholfac_Z[:UL])
    XX = A_mul_Bt(a,a)
    Base.LinAlg.A_rdiv_B!(b, cholfac_Z[:UL])
    Xt = A_mul_Bt(b, a)
    DenseIVPredChol(X, Z, beta0, beta0, Xt, cholfact(XX))
end

function DenseIVPredChol{T<:AbstractFloat}(X::Matrix{T}, Z::Matrix{T})
    DenseIVPredChol(X, Z, zeros(T,size(X,2)), similar(X, 0))
end

function DenseIVPredChol{T<:AbstractFloat}(X::Matrix{T}, Z::Matrix{T}, wt::Vector{T})
    scr = similar(Z)
    DenseIVPredChol(X, Z, zeros(T, size(X,2)), wt)
end



deviance(r::IVResp) = length(r.wts) == 0 ? sumsqdiff(r.y, r.mu) : wsumsqdiff(r.wts, r.y, r.mu)

residuals!(r::IVResp) = r.wrkresid = length(r.wts) == 0 ? r.y - r.mu :  (r.y-r.mu)
residuals(r::IVResp) = r.wrkresid
residuals(l::LinearIVModel) = residuals(l.rr)
residuals(l::LinearIVModel, k::RobustVariance) = wrkresidwts(l.rr)

function wrkresidwts(r::IVResp)
    a = wrkwts(r)
    u = copy(wrkresid(r))
    length(a) == 0 ? u : broadcast!(*, u, u, sqrt(a))
end

function updatemu!{V<:FPVector}(r::IVResp{V}, linPr::V)
    n = length(linPr); length(r.y) == n || error("length(linPr) is $n, should be $(length(r.y))")
    length(r.offset) == 0 ? copy!(r.mu, linPr) : broadcast!(+, r.mu, linPr, offset)
end
updatemu!{V<:FPVector}(r::IVResp{V}, linPr) = updatemu!(r, convert(V,vec(linPr)))

function StatsBase.fit{T<:AbstractFloat, V<:FPVector,
                       LinPredT<:LinPred}(::Type{LinearIVModel{LinPredT}},
                                          X::Matrix{T}, Z::Matrix{T}, y::V;
                                          dofit::Bool=true,
                                          wts::V=similar(y, 0),
                                          offset::V=similar(y, 0), fitargs...)
    size(X, 1) == size(y, 1) || DimensionMismatch("number of rows in X and y must match")
    size(X, 1) == size(Z, 1) || DimensionMismatch("number of rows in X and Z must match")
    size(X, 2) < size(Z, 1)  || DimensionMismatch("number of instruments is not sufficient for identification")
    n = length(y)
    wts = T <: typeof(y) ? copy(wts) : convert(typeof(y), wts)
    rr = IVResp{typeof(y)}(y, similar(y), offset, wts)
    if length(wts) > 0
        pp = LinPredT(X, Z, wts)
        updatemu!(rr, linpred(delbeta!(pp, rr.y, rr.wts), 0.))
    else
        pp = LinPredT(X, Z)
        updatemu!(rr, linpred(delbeta!(pp, rr.y), 0.))
    end
    residuals!(rr)
    LinearIVModel(rr, pp)
end

function StatsBase.fit(::Type{LinearIVModel}, X::Matrix, Z::Matrix,
                       y::Vector; kwargs...)
    StatsBase.fit(LinearIVModel{DenseIVPredChol}, X, Z, y; kwargs...)
end

function wrkresp(r::IVResp)
    if length(r.offset) > 0
        return map1!(Add(), map(Subtract(), r.y, r.offset), r.wrkresid)
    end
    map(Add(), r.y, r.wrkresid)
end


function delbeta!{T<:AbstractFloat}(p::DenseIVPredChol{T}, r::Vector{T})
    A_ldiv_B!(p.chol, At_mul_B!(p.delbeta, p.Xp, r))
    p
end

function delbeta!{T<:AbstractFloat}(p::DenseIVPredChol{T}, r::Vector{T}, wt::Vector{T})
    rp = broadcast(*, r, sqrt(wt))
    A_ldiv_B!(p.chol, At_mul_B!(p.delbeta, p.Xp, rp))
    p
end

ivreg(X, Z, y; kwargs...) = fit(LinearIVModel, X, Z, y; kwargs...)


cholfactors(c::Cholesky) = c.factors
Base.cholfact{T<:FP}(p::DenseIVPredChol{T}) = (c = p.chol; Cholesky(copy(cholfactors(c)), c.uplo))

Base.LinAlg.cholfact!{T}(p::DenseIVPredChol{T}) = p.chol

function GLM.scale(m::LinearIVModel, sqr::Bool=false)
    resid = residuals(m)
    wts   = m.rr.wts
    s = zero(eltype(resid))
    if length(wts) == 0
        @inbounds @simd for i = 1:length(resid)
            s += abs2(resid[i])
        end
    else
        @inbounds @simd for i = 1:length(resid)
            s += wts[i]*abs2(resid[i])
        end
    end
    s /= df_residual(m)
    sqr ? s : sqrt(s)
end

function coeftable(mm::LinearIVModel)
    cc = coef(mm)
    se = stderr(mm)
    tt = cc ./ se
    CoefTable(hcat(cc,se,tt,ccdf(Normal(0, 1), abs2(tt))),
              ["Estimate","Std.Error","t value", "Pr(>|t|)"],
              ["x$i" for i = 1:size(mm.pp.X, 2)], 4)
end

function coeftable(mm::LinearIVModel, vv::RobustVariance)
    cc = coef(mm)
    se = stderr(mm, vv)
    tt = cc ./ se
    CoefTable(hcat(cc,se,tt,ccdf(Normal(0, 1), abs2(tt))),
              ["Estimate","Std.Error","t value", "Pr(>|t|)"],
              ["x$i" for i = 1:size(mm.pp.X, 2)], 4)
end

ModelMatrix(x::LinearIVModel) = x.pp.Xp

#wrkresidwts(r::IVResp) = wrkresid(r)

CovarianceMatrices.wrkwts(r::IVResp) = (w = r.wts; length(w) == 0 ? ones(length(r.y)) : w)


wrkwts(l::LinearIVModel) = l.rr.wts
wrkwts(r::IVResp)        = r.wts

wrkresid(l::LinearIVModel) = l.rr.wrkresid
wrkresid(r::IVResp)        = r.wrkresid

# function bread(l::LinearIVModel)
#     inv(cholfact(l.pp))
# end

# function CovarianceMatrices.meat(l::LinearIVModel, k::RobustVariance)
#     u = copy(residuals(l))
#     w = l.rr.wts
#     length(w) > 0 ? broadcast!(*, u, u, sqrt(w)) : u
#     X = copy(ModelMatrix(l))
#     broadcast!(*, X, X, u)
#     CovarianceMatrices.adjfactor!(u, l, k)
#     Base.LinAlg.At_mul_B(X, X.*u)
# end

function CovarianceMatrices.meat(x::LinearIVModel, v::CRHC)
    idx = sortperm(v.cl)
    cls = v.cl[idx]
    ichol = inv(x.pp.chol)
    X = ModelMatrix(x)[idx,:]
    e = wrkresid(x.rr)[idx]
    w =  wrkwts(x.rr)
    if length(w) > 0
        #e = e.*sqrt(w[idx])
        broadcast!(*, e, e, sqrt(w[idx]))
    end
    bstarts = [searchsorted(cls, j[2]) for j in enumerate(unique(cls))]
    adjresid!(v, X, e, ichol, bstarts)
    M = zeros(size(X, 2), size(X, 2))
    clusterize!(M, X.*e, bstarts)
    scale!(M, 1/nobs(x))
end


function vcov(l::LinearIVModel)
    e = residuals(l)
    w = wrkwts(l)
    if length(w) > 0
        broadcast!(*, e, e, sqrt(w))
    end
    sigma = sumabs2(e)
    B = l.pp.chol
    sigma*inv(B)/df_residual(l)
end

stderr(l::LinearIVModel) = sqrt(diag(vcov(l)))


GLM.nobs(obj::LinearIVModel) = length(obj.rr.y)::Int64

export ivreg, residuals, coeftable, stderr, vcov, predict, coef

end # module
