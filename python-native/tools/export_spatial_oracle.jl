# Independent physical spatial synthesis/projection; no propagation acceptance.
for key in ("NATIVE","STEPPER","DISPERSION","IONISATION","RAMAN")
    ENV["AMALTHEA_USE_RUST_"*key]="0"
end
using Amalthea, DelimitedFiles, TOML, LinearAlgebra, Cubature
const MO=Amalthea.Modes;const PD=Amalthea.PhysData
const A=125e-6
root=abspath(ARGS[1]);mkpath(root)
radius(z)=A*(1+.1sin(3z)+.05z^2)
struct CartesianOracle <: MO.AbstractMode
    index::Int
    tapered::Bool
end
MO.dimlimits(m::CartesianOracle;z=0.)=(:cartesian,(-radius(z),-radius(z)/2),(radius(z),radius(z)/2))
function MO.field(m::CartesianOracle,xy;z=0.)
    x,y=xy./(radius(z),radius(z)/2)
    out=m.index==1 ? [1+x,.2*(1-y^2)] : [.3*(1-x^2),y]
    m.tapered ? out.*(1-x^2) : out
end
function MO.N(m::CartesianOracle;z=0.)
    integral=if m.tapered
        m.index==1 ? 256/105+.04*256/225 : .09*512/315+32/45
    else
        m.index==1 ? 16/3+.04*32/15 : .09*32/15+4/3
    end
    .5sqrt(PD.ε_0/PD.μ_0)*(radius(z)^2/2)*integral
end
MO.neff(m::CartesianOracle,w;z=0.)=1.0001+1e-7im+(m.index-1)*2e-6+z*1e-5
complexfile(path,x)=writedlm(path,hcat(real.(vec(x)),imag.(vec(x))))
for (name,spec,full,components) in [
    ("radial",[(1,1,:HE,0.),(1,2,:HE,0.)],false,:y),
    ("radial-full",[(1,1,:HE,0.),(1,2,:HE,0.)],true,:y),
    ("polarized",[(1,1,:HE,0.),(1,1,:HE,π/2),(1,2,:HE,0.)],false,:xy),
    ("mixed",[(2,1,:HE,.12),(2,2,:HE,.4),(0,1,:TE,0.),(0,1,:TM,0.)],true,:xy),
    ("x-only",[(2,1,:HE,.12),(1,2,:HE,π/2)],true,:x),
    ("cartesian",[],true,:xy),
    ("cartesian-reduced",[],false,:xy)]
    tapered=name=="cartesian-reduced"
    modes=isempty(spec) ? [CartesianOracle(1,tapered),CartesianOracle(2,tapered)] :
        [Capillary.MarcatiliMode(radius,:Ar,2.;n,m,kind,ϕ=phi) for (n,m,kind,phi) in spec]
    ts=MO.ToSpace(modes;components)
    field=ComplexF64[sin(.3i+.2j)+.2im*cos(.17i-.4j) for i=1:5,j=1:length(modes)]
    dir=joinpath(root,name);mkpath(dir)
    open(joinpath(dir,"parameters.toml"),"w") do io
        TOML.print(io,Dict("full"=>full,"components"=>String(components),"tapered"=>tapered,
            "modes"=>[Dict("n"=>n,"m"=>m,"kind"=>String(kind),"phi"=>phi) for (n,m,kind,phi) in spec]))
    end
    complexfile(joinpath(dir,"field.txt"),field)
    for (zi,z) in enumerate((0.,.137))
        dl=MO.dimlimits(modes[1];z);kind,ll,ul=dl
        points=kind==:polar ? [0. .17radius(z) .71radius(z) .99radius(z) radius(z); .2 .7 1.3 5.9 2.] :
            [0. -.4radius(z) .8radius(z) radius(z) 0.;0. .17radius(z) -.3radius(z) 0. radius(z)/2]
        writedlm(joinpath(dir,"points-$zi.txt"),points')
        writedlm(joinpath(dir,"normalization-$zi.txt"),[MO.N(m;z) for m in modes])
        er=zeros(ComplexF64,size(field,1),ts.npol)
        all_fields=zeros(ComplexF64,size(er)...,size(points,2))
        all_projections=zeros(ComplexF64,size(field)...,size(points,2))
        for p in axes(points,2)
            MO.to_space!(er,field,Tuple(points[:,p]),ts;z)
            all_fields[:,:,p].=er
            polarization=er.*sum(abs2,er;dims=2)
            all_projections[:,:,p].=polarization*transpose(ts.Ems)
        end
        complexfile(joinpath(dir,"synthesis-$zi.txt"),all_fields)
        complexfile(joinpath(dir,"nodes-$zi.txt"),all_projections)
        function integrand(xs,fval)
            for i in axes(xs,2)
                x1=xs[1,i];x2=full ? xs[2,i] : 0.
                if x1<=ll[1] || x1>=ul[1]
                    fval[:,i].=0.;continue
                end
                MO.to_space!(er,field,(x1,x2),ts;z)
                # Independent synthetic complete-array response, before temporal wiring.
                p=er.*sum(abs2,er;dims=2)
                projection=p*transpose(ts.Ems)
                pre=kind==:polar ? (full ? x1 : 2π*x1) : 1.
                fval[:,i].=pre.*reinterpret(Float64,vec(projection))
            end
        end
        for (tag,tol) in (("coarse",1e-4),("fine",1e-10),("refined",1e-12))
            integration=full ? Cubature.hcubature_v : Cubature.pcubature_v
            val,err=integration(length(field)*2,integrand,full ? ll : (ll[1],),full ? ul : (ul[1],);
                                reltol=tol,abstol=0.,maxevals=2_000_000,error_norm=Cubature.L2)
            @assert norm(err)<=tol*norm(val)
            complexfile(joinpath(dir,"projection-$zi-$tag.txt"),reshape(reinterpret(ComplexF64,val),size(field)))
            writedlm(joinpath(dir,"error-$zi-$tag.txt"),[norm(err),tol*norm(val)])
        end
    end
    println("exported ",name);flush(stdout)
end
