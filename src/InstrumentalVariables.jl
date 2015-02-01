module InstrumentalVariables

using Reexport
using NumericExtensions

@reexport using GLM
@reexport using StatsBase

import GLM: BlasReal, Cholesky, FP, WtResid, Add, Subtract, Multiply, DispersionFun, ModelMatrix, result_type, delbeta!
import StatsBase: residuals, coeftable
import Distributions: ccdf, FDist, Chisq, Normal

typealias FPVector{T<:FloatingPoint} DenseArray{T,1}

type IVResp{V<:FPVector} <: GLM.ModResp  # response in a linear model
    mu::V                                # mean response
    offset::V                            # offset added to linear predictor (may have length 0)
    wts::V                               # prior weights (may have length 0)
    wrkresid::V                          # residual
    y::V                                 # response
    function IVResp(mu::V, off::V, wts::V, wrkresid::V, y::V)
        n = length(y); length(mu) == n || error("mismatched lengths of mu and y")
        ll = length(off); ll == 0 || ll == n || error("length of offset is $ll, must be $n or 0")
        ll = length(wts); ll == 0 || ll == n || error("length of wts is $ll, must be $n or 0")
        new(mu,off,wts,wrkresid,y)
    end
end

IVResp{V<:FPVector}(y::V) = IVResp{V}(fill!(similar(y), zero(eltype(V))), similar(y, 0), similar(y, 0), similar(y, 0), y)

IVResp{V<:FPVector}(y::V, off::V, wts::V) = IVResp{V}(fill!(similar(y), zero(eltype(V))), off, wts, similar(y, 0), y)

type LinearIVModel{T<:LinPred} <: LinPredModel
    rr::IVResp
    pp::T
end


deviance(r::IVResp) = length(r.wts) == 0 ? sumsqdiff(r.y, r.mu) : wsumsqdiff(r.wts, r.y, r.mu)
residuals!(r::IVResp) = r.wrkresid = length(r.wts) == 0 ? r.y - r.mu :  (r.y-r.mu).*sqrt(r.wts)
residuals(r::IVResp) = r.wrkresid
residuals(l::LinearIVModel) = residuals(l.rr)



function updatemu!{V<:FPVector}(r::IVResp{V}, linPr::V)
    n = length(linPr); length(r.y) == n || error("length(linPr) is $n, should be $(length(r.y))")
    length(r.offset) == 0 ? copy!(r.mu, linPr) : map!(Add(), r.mu, linPr, r.offset)
end
updatemu!{V<:FPVector}(r::IVResp{V}, linPr) = updatemu!(r, convert(V,vec(linPr)))

function StatsBase.fit{T<:FloatingPoint, V<:FPVector, LinPredT<:LinPred}(::Type{LinearIVModel{LinPredT}},
                                                                         X::Matrix{T}, Z::Matrix{T}, y::V;
                                                                         dofit::Bool=true,
                                                                         wts::V=similar(y, 0),
                                                                         offset::V=similar(y, 0), fitargs...)
    size(X, 1) == size(y, 1) || DimensionMismatch("number of rows in X and y must match")
    size(X, 1) == size(Z, 1) || DimensionMismatch("number of rows in X and Z must match")
    size(X, 2) < size(Z, 1)  || DimensionMismatch("number of instruments is not sufficient for identification")
    n = length(y)
    ## lw = length(wts)
    ## length(wts) == n || error("length(wts) = $lw should be 0 or $n")
    wts = T <: Float64 ? copy(wts) : convert(typeof(y), wts)
    rr = IVResp(y, offset, wts)
    pp = LinPredT(X, Z)
    scratch = similar(pp.X)
    if length(wts) > 0
        updatemu!(rr, linpred(delbeta!(pp, rr.y, rr.wts, scratch)))
    else
        updatemu!(rr, linpred(delbeta!(pp, rr.y)))
    end     
    residuals!(rr)
    LinearIVModel(rr, pp)
end

function wrkresp(r::IVResp)
    if length(r.offset) > 0
        return map1!(Add(), map(Subtract(), r.y, r.offset), r.wrkresid)
    end
    map(Add(), r.y, r.wrkresid)
end

function StatsBase.fit(::Type{LinearIVModel}, X::Matrix, Z::Matrix,
                       y::Vector; kwargs...)
    StatsBase.fit(LinearIVModel{DenseIVPredChol}, X, Z, y; kwargs...)
end 

type DenseIVPredChol{T<:BlasReal} <: DensePred
    X::Matrix{T}                   # model matrix
    Z::Matrix{T}                   # instrument matrix
    beta0::Vector{T}               # base vector for coefficients
    delbeta::Vector{T}
    Pz::Matrix{T}
    Xp::Matrix{T}
    Zp::Matrix{T}
    chol::Cholesky{T}           
    function DenseIVPredChol(X::Matrix{T}, Z::Matrix{T}, beta0::Vector{T})
        n,p = size(X); length(beta0) == p || error("dimension mismatch")
        cholfac_Z = cholfact(Z'Z)
        Pz = Z*inv(cholfac_Z)*Z'  ## No need to store this probably
        Xt = Pz*X
        new(X, Z, beta0, beta0, Pz, Xt, Pz*Z, cholfact(Xt'Xt))
    end
end
function DenseIVPredChol{T<:BlasReal}(X::Matrix{T}, Z::Matrix{T})
    DenseIVPredChol{T}(X, Z, zeros(T,size(X,2)))
end 

function delbeta!{T<:BlasReal}(p::DenseIVPredChol{T}, r::Vector{T})
    A_ldiv_B!(p.chol, At_mul_B!(p.delbeta, p.Xp, r))
    p
end

function delbeta!{T<:BlasReal}(p::DenseIVPredChol{T}, r::Vector{T}, wt::Vector{T}, scr::Matrix{T})
    vbroadcast!(Multiply(), scr, p.X, wt, 1)
    cholfact!(At_mul_B!(p.chol.UL, scr, p.X), :U)
    A_ldiv_B!(p.chol, At_mul_B!(p.delbeta, scr, r))
    p
end

ivreg(X, Z, y; kwargs...) = fit(InstrumentalVariables.LinearIVModel, X, Z, y; kwargs...)

if VERSION >= v"0.4.0-dev+122"
    #cholfact{T<:FP}(p::DenseIVPredQR{T}) = Cholesky{T,Matrix{T},:U}(p.qr[:R])
    Base.cholfact{T<:FP}(p::DenseIVPredChol{T}) = (c = p.chol; typeof(c)(copy(c.UL)))
else
    #cholfact{T<:FP}(p::DenseIVPredQR{T}) = Cholesky(p.qr[:R], 'U')
    Base.cholfact{T<:FP}(p::DenseIVPredChol{T}) = (c = p.chol; Cholesky(copy(c.UL),c.uplo))
end

function GLM.scale(m::LinearIVModel, sqr::Bool=false)
    if length(m.rr.wts) == 0
        s = sumsq(residuals(m))/df_residual(m)
    else
        s = sum(DispersionFun(), m.rr.wts, residuals(m))/df_residual(m)
    end 
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


ModelMatrix(x::LinearIVModel) = x.pp.Xp

export iv, residuals, coeftable

end # module
