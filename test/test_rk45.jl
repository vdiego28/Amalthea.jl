using TestItems

@testitem "Rk45" tags=[:physics] begin
import FFTW
import Luna: RK45
import Test: @test

function testinit()
    trange = 32
    ωrange = 160
    samples = 2^(ceil(Int, log2(trange*ωrange/2π)))
    δt = trange/samples
    δω = 2π/trange
    n = collect(range(0, length=samples))
    t = @. (n-samples/2)*δt
    ω = @. (n-samples/2)*δω
    N = 5
    z0 = π/2
    zmax = z0*2
    
    At = @. N*sech(t)
    Aω = FFTW.fftshift(FFTW.fft(At))

    FT = FFTW.plan_fft!(Aω)
    IFT = FFTW.plan_ifft!(Aω)
    f! = let FT=FT, IFT=IFT, ω=ω
        function f!(out, Aω, z)
                out .= Aω
                IFT*out
                out .= abs2.(out).*out
                FT*out
                @. out = im*out - im/2*ω^2*Aω
                return out
        end
    end

    Lin = @. -im/2*ω^2
    Linfunc = let ω=ω
        function Linfunc(out, z)
            @. out = -im/2*ω^2
        end
    end

    fnl! = let FT=FT, IFT=IFT
        function fnl!(out, Aω, z)
                out .= Aω
                IFT*out
                out .= abs2.(out).*out
                FT*out
                @. out = im*out
                return out
        end
    end

    return t, ω, zmax, Aω, f!, Lin, fnl!, Linfunc
end



_, ω, zmax, Aω, f!, Lin, fnl!, Linfunc = testinit()
z = 0
dz = 1e-3

zarr, Aarr = RK45.solve(f!, copy(Aω), z, dz, zmax, rtol=1e-8, output=true, outputN=501)
_, Aarrp = RK45.solve_precon(fnl!, Lin, copy(Aω), z, dz, zmax,
                                 rtol=1e-8, output=true, outputN=501)
_, Aarrpf = RK45.solve_precon(fnl!, Linfunc, copy(Aω), z, dz, zmax, 
                                   rtol=1e-8, output=true, outputN=501)
# Is the initial spectrum restored after 2 soliton periods? (without preconditioner)
@test isapprox(abs2.(Aarr[:, 1]), abs2.(Aarr[:, end]), rtol=1e-4)
# Is the initial spectrum restored after 2 soliton periods? (with preconditioner)
@test isapprox(abs2.(Aarrp[:, 1]), abs2.(Aarrp[:, end]), rtol=1e-3)
# Is the initial spectrum restored after 2 soliton periods?
# (with preconditioner and z-dependent linear part)
@test isapprox(abs2.(Aarrpf[:, 1]), abs2.(Aarrpf[:, end]), rtol=1e-3)
# Is there a difference if the linear part is a function (but constant)?
@test all(abs2.(Aarrp) .== abs2.(Aarrpf))


end
