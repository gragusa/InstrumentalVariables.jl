using InstrumentalVariables
using PDMats

# write your own tests here
#@test 1 == 1

srand(7875)

y = randn(100); x = randn(100,3); z = randn(100, 10);

function ivreg(y, x, z)
    zz = PDMat(z'z)
    Pz = X_invA_Xt(zz, z)
    xPz= x'*Pz
    reshape(xPz*x\xPz*y, size(x)[2])
end

bas = ivreg(y, x, z);

out = iv(x,z,y);

vcov(out)

using CovarianceMatrices

vcov(out, HC0())


