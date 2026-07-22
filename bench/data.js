window.BENCHMARK_DATA = {
  "lastUpdate": 1784729798899,
  "repoUrl": "https://github.com/vdiego28/Amalthea.jl",
  "entries": {
    "Benchmark": [
      {
        "commit": {
          "author": {
            "email": "vdiego28@yahoo.es",
            "name": "vdiego28",
            "username": "vdiego28"
          },
          "committer": {
            "email": "vdiego28@yahoo.es",
            "name": "vdiego28",
            "username": "vdiego28"
          },
          "distinct": true,
          "id": "118f35da8f825289c9d640bafc1e4d5d344defc6",
          "message": "Fix native-radial deadlock: force FFTW 1-D plans to nthreads=1\n\nCI's rust/sim-propagation jobs hung indefinitely (6h timeout) under\nJULIA_NUM_THREADS=auto. Root cause: Julia's Utils.FFTWthreads() raises\nFFTW's process-global internal thread count (4*Threads.nthreads()) before\nRust dlopen's the same libfftw3.so, so every 1-D plan native.rs creates for\nthe rayon-threaded per-r-column radial RHS inherits that thread count.\nFFTW's \"execute is reentrant against one shared plan with distinct\nbuffers\" guarantee only holds for plans built with nthreads=1 — a\nmultithreaded plan dispatches to FFTW's own internal worker pool on\nexecute, so concurrent execute calls from multiple rayon workers on the\nsame plan deadlock (reproduced locally: hangs deterministically on the\n~5th-9th rhs_radial call under -t 4/-t 8, confirmed via /proc thread\nstates — all threads parked on futex_do_wait, zero CPU progress).\n\nFix: wrap every 1-D FFTW plan-creation call (ComplexFft1d, RealFft1d,\nSplitComplexFft1d, SplitRealFft1d) in FftwApi::with_single_threaded_plan,\nwhich forces fftw_plan_with_nthreads(1) for the duration of planning and\nrestores the prior value afterward. The 3-D plans (RealFft3d/ComplexFft3d,\nused by the free-space geometry's single joint transform, never called\nconcurrently) are untouched.\n\nAlso bootstraps the missing gh-pages branch so the native-path benchmark\njob's github-action-benchmark step can push/fetch history instead of\nfailing with \"couldn't find remote ref gh-pages\".",
          "timestamp": "2026-07-12T13:00:28-04:00",
          "tree_id": "ef63c167845f5e88c209812aacf0b30933602878",
          "url": "https://github.com/vdiego28/Luna-Rust.jl/commit/118f35da8f825289c9d640bafc1e4d5d344defc6"
        },
        "date": 1783878717593,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "native mode-avg+plasma per-step (fixed dt)",
            "value": 2.960685,
            "unit": "ms/step"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vdiego28@yahoo.es",
            "name": "vdiego28",
            "username": "vdiego28"
          },
          "committer": {
            "email": "vdiego28@yahoo.es",
            "name": "vdiego28",
            "username": "vdiego28"
          },
          "distinct": true,
          "id": "3b33302514d2443e661641a0ccf71babb5405736",
          "message": "Point benchmark-action at bench/ instead of the default dev/bench\n\nAvoids colliding with Documenter's default gh-pages:dev/ deploy\nfolder, which it clears on every deploy of push-to-main docs. The\nexisting tracked history was already migrated on gh-pages itself\n(dev/bench -> bench).",
          "timestamp": "2026-07-12T14:23:43-04:00",
          "tree_id": "103309b13a640722dbf63f707ddc629fabf7182a",
          "url": "https://github.com/vdiego28/Luna-Rust.jl/commit/3b33302514d2443e661641a0ccf71babb5405736"
        },
        "date": 1783880872152,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "native mode-avg+plasma per-step (fixed dt)",
            "value": 2.988912,
            "unit": "ms/step"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "53799316+vdiego28@users.noreply.github.com",
            "name": "vdiego28",
            "username": "vdiego28"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "00a4821a84e568054e0ab023499b68a7b137b5b5",
          "message": "Merge pull request #57 from vdiego28/imgbot\n\n[ImgBot] Optimize images",
          "timestamp": "2026-07-12T16:28:45-04:00",
          "tree_id": "49b401f44949028adde3bcd1d59e2b0672e6ce93",
          "url": "https://github.com/vdiego28/Luna-Rust.jl/commit/00a4821a84e568054e0ab023499b68a7b137b5b5"
        },
        "date": 1783888606294,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "native mode-avg+plasma per-step (fixed dt)",
            "value": 2.975871,
            "unit": "ms/step"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vdiego28@yahoo.es",
            "name": "vdiego28",
            "username": "vdiego28"
          },
          "committer": {
            "email": "vdiego28@yahoo.es",
            "name": "vdiego28",
            "username": "vdiego28"
          },
          "distinct": true,
          "id": "0280cb463f8b3111fb25c1c29e2e4d6722cc1b88",
          "message": "Rename package from Luna-Rust.jl to Amalthea.jl\n\nGives the fork an independent Julia package identity (new name and\nUUID, distinct from upstream Luna.jl's) and repo branding, ahead of\nregistering it as its own package in the General registry and cutting\na v1.0.0 release. Historical CHANGELOG/REVIEW entries are kept as\n\"formerly Luna-Rust.jl\" rather than rewritten.\n\nCo-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>",
          "timestamp": "2026-07-12T17:06:52-04:00",
          "tree_id": "bf1da31a527a5cc6a1de6bbcb1f33d8c370e026b",
          "url": "https://github.com/vdiego28/Amalthea.jl/commit/0280cb463f8b3111fb25c1c29e2e4d6722cc1b88"
        },
        "date": 1783890683393,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "native mode-avg+plasma per-step (fixed dt)",
            "value": 2.953231,
            "unit": "ms/step"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vdiego28@yahoo.es",
            "name": "vdiego28",
            "username": "vdiego28"
          },
          "committer": {
            "email": "vdiego28@yahoo.es",
            "name": "vdiego28",
            "username": "vdiego28"
          },
          "distinct": true,
          "id": "74bd6f644ae4b97cedb879cfdb4f76b41af2a67b",
          "message": "Bump minimum Julia version to 1.10\n\nProject.toml declared julia = \"1.9\" but DSP = \"0.8\", and DSP >=0.8.0\nitself requires Julia >=1.10 — an unsatisfiable requirement at the\ndeclared floor. AutoMerge's Pkg.add on Julia 1.9.4 caught this on the\nAmalthea registration PR (JuliaRegistries/General#160997). Raising the\nfloor to 1.10 (already covered by CI's 'lts'/'1'/'pre' matrix) resolves\nit without touching the DSP compat bound.\n\nCo-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>",
          "timestamp": "2026-07-12T17:19:28-04:00",
          "tree_id": "4c1e14b49d1c6bfd8ae4109e82c45d6b1daf2584",
          "url": "https://github.com/vdiego28/Amalthea.jl/commit/74bd6f644ae4b97cedb879cfdb4f76b41af2a67b"
        },
        "date": 1783891307248,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "native mode-avg+plasma per-step (fixed dt)",
            "value": 2.956111,
            "unit": "ms/step"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vdiego28@yahoo.es",
            "name": "vdiego28",
            "username": "vdiego28"
          },
          "committer": {
            "email": "vdiego28@yahoo.es",
            "name": "vdiego28",
            "username": "vdiego28"
          },
          "distinct": true,
          "id": "440f368d269eb7bb94622ffa1ab99914b7dfed00",
          "message": "Fix Windows prebuilt-download race and Documenter cross-references\n\n- deps/build.jl: download the prebuilt library and SHA256SUMS.txt into\n  a real temp directory instead of writing/deleting them directly in\n  the live luna-rust/target/release/ dir. On Windows that directory\n  can still be locked by the preceding `cargo build --release` CI\n  step (or antivirus), causing an EBUSY unlink error on cleanup. Only\n  a single atomic `mv` of the verified library now touches target/release/.\n  First exposed by real Windows release binaries existing for v1.0.0.\n\n- Documenter build was failing on 6 unresolved @ref links (all\n  pre-existing, unrelated to the rename):\n  - ZDepLinopMarcatili / ZDepLinopFree structs had rationale as plain\n    comments, not docstrings — added proper docstrings.\n  - prop_capillary_args's docstring was textually attached to the\n    wrong function (_prop_capillary_args, defined right after it) —\n    moved to the correct binding.\n  - 3 cross-module @ref links (LinearOps.make_linop_free_gradient,\n    Capillary.gradient, NonlinearRHS.norm_free_gradient) failed to\n    resolve from another module's @autodocs page; fully-qualified\n    them (Amalthea.<Module>.<name>) which Documenter resolves\n    regardless of the page's CurrentModule.\n\nVerified locally: `julia --project=docs docs/make.jl` now completes\nwith no cross-reference errors.\n\nCo-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>",
          "timestamp": "2026-07-12T17:52:13-04:00",
          "tree_id": "abfe187e405fb97f0af47ed69f0dcaad54099823",
          "url": "https://github.com/vdiego28/Amalthea.jl/commit/440f368d269eb7bb94622ffa1ab99914b7dfed00"
        },
        "date": 1783893359212,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "native mode-avg+plasma per-step (fixed dt)",
            "value": 2.968935,
            "unit": "ms/step"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vdiego28@yahoo.es",
            "name": "vdiego28",
            "username": "vdiego28"
          },
          "committer": {
            "email": "vdiego28@yahoo.es",
            "name": "vdiego28",
            "username": "vdiego28"
          },
          "distinct": true,
          "id": "ca72e9f693bb4c97ecd7f4c2d06aee98855e5666",
          "message": "Add mocked unit tests for LunaOutput's dispatch logic\n\ntest_output.py had exactly one test (__getitem__'s KeyError wrapping)\nexercising the real LunaOutput class -- everywhere else in the suite\nuses a separate, simpler MockLunaOutput that bypasses it entirely. So\n_to_python's isa-dispatch, __contains__, and keys() (3-5 branches\neach) had zero fast/mocked coverage; their only exercise was\nincidental, via the real integration tests hitting a subset of paths.\n\nAdds mocked tests for the reachable keys()/__contains__ branches\n(Dict, MemoryOutput, HDF5Output, AbstractOutput, and the no-match\nfallback), including a direct regression test for the HDF5 file-close\nfix in the prior commit -- confirmed it actually catches the\nregression by temporarily reintroducing the leak and watching it fail.\n\nThe HDF5.Group/File and generic AbstractOutput-.data-fallback\nbranches are left untested/unremoved: MemoryOutput and HDF5Output are\nthe only AbstractOutput subtypes in this codebase, so those branches\nare currently unreachable through the public API -- harmless\ndefensive code, not worth testing or deleting.\n\nCo-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>",
          "timestamp": "2026-07-13T11:20:25-04:00",
          "tree_id": "a3820d90ce1fb3399927fa9f9dd093d5b7e40220",
          "url": "https://github.com/vdiego28/Amalthea.jl/commit/ca72e9f693bb4c97ecd7f4c2d06aee98855e5666"
        },
        "date": 1784077182751,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "native mode-avg+plasma per-step (fixed dt)",
            "value": 2.959503,
            "unit": "ms/step"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vdiego28@yahoo.es",
            "name": "vdiego28",
            "username": "vdiego28"
          },
          "committer": {
            "email": "vdiego28@yahoo.es",
            "name": "vdiego28",
            "username": "vdiego28"
          },
          "distinct": true,
          "id": "60daa6e67ab9290dbefebbacd4fbe0c276a4911a",
          "message": "Extend high-level API to ZeisbergerMode/VincettiMode/StepIndexMode; add native-support matrix\n\nprop_capillary's makemode_s now accepts prebuilt AbstractMode(s) via modes=,\nletting ZeisbergerMode/VincettiMode reuse the existing gas/pressure pipeline.\nStepIndexMode gets its own prop_stepindex entry point (mirrors prop_gnlse),\nsince it has no gas/density concept. Adds docs/dev/native-port/NATIVE_SUPPORT_MATRIX.md\ndocumenting what runs natively vs falls back.\n\nCo-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>",
          "timestamp": "2026-07-16T18:17:45-04:00",
          "tree_id": "bd292f23d33e88a2ba87eb5edc71ed69f7ce7d8b",
          "url": "https://github.com/vdiego28/Amalthea.jl/commit/60daa6e67ab9290dbefebbacd4fbe0c276a4911a"
        },
        "date": 1784240683043,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "native mode-avg+plasma per-step (fixed dt)",
            "value": 2.929755,
            "unit": "ms/step"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vdiego28@yahoo.es",
            "name": "vdiego28",
            "username": "vdiego28"
          },
          "committer": {
            "email": "vdiego28@yahoo.es",
            "name": "vdiego28",
            "username": "vdiego28"
          },
          "distinct": true,
          "id": "32ae9dde0005d02be6962d5d438846ce1d2d70b6",
          "message": "Correct amalthea/README.md's stale hardware-dispatcher claims\n\ndispatch.rs's HardwarePath/SimulationEngine is detection-only and\nunreferenced outside its own unit tests -- no RHS kernel or the real\nGPU path (CudaNativeSim/cuda.rs) uses it (see BACKLOG.md S5.2, S1\nitem 4). Replaces the old \"multi-branch dispatcher\" description with\nwhat's actually true: CPU throughput comes from target-cpu=native +\nLLVM auto-vectorization (verified via objdump), the one hand-written\nSIMD lane is raman.rs::solve_avx2 (needed for its sequential\nrecurrence), and GPU offload runs through CudaNativeSim independently.\nAlso records this session's measured CPU-vs-GPU numbers on real\nhardware (RTX 5060 Ti): GPU ~20-30x slower with plasma active, ~5-27x\nfaster for Kerr-only above n≈16k.\n\nCo-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>",
          "timestamp": "2026-07-16T19:31:17-04:00",
          "tree_id": "e371bb1642b3184f15eb8290dbdc3869a1a8cc53",
          "url": "https://github.com/vdiego28/Amalthea.jl/commit/32ae9dde0005d02be6962d5d438846ce1d2d70b6"
        },
        "date": 1784244966965,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "native mode-avg+plasma per-step (fixed dt)",
            "value": 2.419185,
            "unit": "ms/step"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vdiego28@yahoo.es",
            "name": "vdiego28",
            "username": "vdiego28"
          },
          "committer": {
            "email": "vdiego28@yahoo.es",
            "name": "vdiego28",
            "username": "vdiego28"
          },
          "distinct": true,
          "id": "28630151d4fa75797e47a84fd65159414d2da4d3",
          "message": "Add measured problem-size dispatch threshold for the GPU-resident stepper\n\nAMALTHEA_NATIVE_GPU=off/on/auto (Config.jl's new gpu_dispatch field) layers\na dispatch policy on top of AMALTHEA_USE_RUST_CUDA_NATIVE's existing master\nopt-in. Benchmarked CPU-vs-GPU native_step directly on real hardware (RTX\n5060 Ti) before choosing a threshold: Kerr-only crosses over around n=8-16k\nand wins up to 27x at n=262k (cuFFT-dominated), but Kerr+plasma is 20-30x\nslower than CPU at every size tested up to n=131k and gets worse with n\n(single-thread sequential plasma-scan kernels, a documented V1 tradeoff) --\ntwo different regimes, not one crossover. `auto` (new default) requires a\nplasma-free config at n >= 16384; `on` restores the old unconditional\nbehavior; `off` forces CPU. RK45._gpu_native_eligible split into a pure\nconfig-shape check (_gpu_kernel_supports) and the new size/policy-aware\n3-arg eligibility function. Full measured table lives in\nRK45._GPU_KERR_ONLY_N_THRESHOLD's docstring.\n\nExisting GPU equivalence tests pinned to AMALTHEA_NATIVE_GPU=on (they test\nraw kernel correctness at small/known configs, independent of the dispatch\nheuristic). New test/test_native_gpu_dispatch.jl covers the off/on/auto\ndecision matrix directly.\n\nCo-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>",
          "timestamp": "2026-07-16T19:58:41-04:00",
          "tree_id": "fdd1fbc7e3e89eafc49a8a082b06b8628a154da9",
          "url": "https://github.com/vdiego28/Amalthea.jl/commit/28630151d4fa75797e47a84fd65159414d2da4d3"
        },
        "date": 1784246676802,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "native mode-avg+plasma per-step (fixed dt)",
            "value": 2.935759,
            "unit": "ms/step"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vdiego28@yahoo.es",
            "name": "vdiego28",
            "username": "vdiego28"
          },
          "committer": {
            "email": "vdiego28@yahoo.es",
            "name": "vdiego28",
            "username": "vdiego28"
          },
          "distinct": true,
          "id": "c80338d0ff471689971abdc037fe1f8c99f0e7ca",
          "message": "Docs: correct stale ~1e-13 phase8 endpoint tolerance comment to measured ~1.6e-11\n\nNative-vs-Julia endpoint agreement for the eligible config is ~1.6e-11\n(measured, printed by the test), not ~1e-13. Comment/println only; the\n<1e-8 assertion is unchanged. Flagged during S5 dense-output review.\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>",
          "timestamp": "2026-07-19T17:40:25-04:00",
          "tree_id": "e05c193d353c3dda307d2a37db62a7b1dec094bb",
          "url": "https://github.com/vdiego28/Amalthea.jl/commit/c80338d0ff471689971abdc037fe1f8c99f0e7ca"
        },
        "date": 1784497424000,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "native mode-avg+plasma per-step (fixed dt)",
            "value": 2.935014,
            "unit": "ms/step"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vdiego28@yahoo.es",
            "name": "vdiego28",
            "username": "vdiego28"
          },
          "committer": {
            "email": "vdiego28@yahoo.es",
            "name": "vdiego28",
            "username": "vdiego28"
          },
          "distinct": true,
          "id": "051feb824a35bf2209f5aaeab4420949c75831ce",
          "message": "S2 Phase 4 (modal): thread the native modal RHS over cubature nodes\n\nParallelizes the per-node modal integrand (`modal_pointcalc`) across rayon\nworkers when `n_threads > 1`, the third of S2's four threading seams (after\nradial FFT+plasma and radial Raman).\n\nMeasured first (temp `Instant` counters, reverted): the integrand loop is\n90.3% (full=false, 1 mode) / 95.6% (full=true) / 82.8% (2-mode) of `rhs_modal`\nwall time — well above the proceed bar (radial was 38-61%; S1.6 parked ~2%).\n\nRefactor: `rhs_modal_pointcalc` (a `&mut self` method scribbling on ~13 shared\n`self.modal_*`/`raman_*` scratch buffers) became a free associated fn\n`modal_pointcalc(&ModalRO, &mut ModalScratch, r, θ, out)` — read-only sim state\nin a `Sync` `ModalRO` view (all `&[..]`/`Copy`/`Option<&Plan>`, FFT wrappers\nalready `Sync`), every written buffer in a per-worker `ModalScratch` pooled on\n`self.modal_scratch_pool` (entry 0 = sequential path). Both paths share the one\nfunction body. Nodes split into <= n_threads contiguous groups; each group's\n`out[p*fdim..]` is disjoint with no cross-node reduction => bit-identical\nn_threads=1-vs-4. Raman-modal threaded too: each worker owns a cloned\n`TimeDomainRamanSolver` + Hilbert scratch (solve() resets state at entry =>\nclone == shared; Hilbert FFT plan shared read-only). No new GC-root hazard —\nthe solver is Rust-owned/cloned, not a persistent raw pointer into Julia memory.\n\nVerified:\n- bit-identical n_threads=1 vs 4 across Kerr full=false/full=true/2-mode/npol=2\n  and Raman :N2 (test/test_native_modal_threading.jl, + forced-GC.gc() stress)\n- native-vs-Julia parity unchanged (~2e-16 Kerr, ~1e-6 Raman ADE-vs-FFT floor)\n- wall-clock speedup 1->4 threads: full=false 1.31x/1.52x (1/2-mode),\n  full=true 2.64x — proves the parallel branch actually engages\n- full 7-group gate green: rust 42160/42160, sim-multimode 33/33,\n  sim-propagation 18/18, physics 1657/1657, sim-interface 314/314, io 2302/2302,\n  fields 334/334; 70/70 Rust unit tests; clean -D warnings build\n\nDocs: BACKLOG S2 item 3 + PLAN_S2_THREADING.md Phase 4 (modal done; only\nfree-space 3-D FFT threading remains open). Also folded in a stale-doc fix\nmarking S6 item 2 (native scan HDF5 writer) done (commit 05c4a4e).\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>",
          "timestamp": "2026-07-20T12:18:19-04:00",
          "tree_id": "f2f27ff1150c5f1bdb4bffd74d7f9a359d0f58b6",
          "url": "https://github.com/vdiego28/Amalthea.jl/commit/051feb824a35bf2209f5aaeab4420949c75831ce"
        },
        "date": 1784564508794,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "native mode-avg+plasma per-step (fixed dt)",
            "value": 2.92089,
            "unit": "ms/step"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "49699333+dependabot[bot]@users.noreply.github.com",
            "name": "dependabot[bot]",
            "username": "dependabot[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "6bdae032bce194261dde022a25ff8df7314d4a88",
          "message": "build(deps): bump softprops/action-gh-release from 2 to 3 (#59)\n\nBumps [softprops/action-gh-release](https://github.com/softprops/action-gh-release) from 2 to 3.\n- [Release notes](https://github.com/softprops/action-gh-release/releases)\n- [Changelog](https://github.com/softprops/action-gh-release/blob/master/CHANGELOG.md)\n- [Commits](https://github.com/softprops/action-gh-release/compare/v2...v3)\n\n---\nupdated-dependencies:\n- dependency-name: softprops/action-gh-release\n  dependency-version: '3'\n  dependency-type: direct:production\n  update-type: version-update:semver-major\n...\n\nSigned-off-by: dependabot[bot] <support@github.com>\nCo-authored-by: dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>",
          "timestamp": "2026-07-20T18:12:02-04:00",
          "tree_id": "69c3ca2e073e85def53c2a19c3a40addfa3b43ff",
          "url": "https://github.com/vdiego28/Amalthea.jl/commit/6bdae032bce194261dde022a25ff8df7314d4a88"
        },
        "date": 1784585582189,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "native mode-avg+plasma per-step (fixed dt)",
            "value": 2.920886,
            "unit": "ms/step"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "49699333+dependabot[bot]@users.noreply.github.com",
            "name": "dependabot[bot]",
            "username": "dependabot[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "7f76d49acb04c04c72a496c329763efc02bbf6e1",
          "message": "build(deps): bump actions/upload-artifact from 4 to 7 (#62)\n\nBumps [actions/upload-artifact](https://github.com/actions/upload-artifact) from 4 to 7.\n- [Release notes](https://github.com/actions/upload-artifact/releases)\n- [Commits](https://github.com/actions/upload-artifact/compare/v4...v7)\n\n---\nupdated-dependencies:\n- dependency-name: actions/upload-artifact\n  dependency-version: '7'\n  dependency-type: direct:production\n  update-type: version-update:semver-major\n...\n\nSigned-off-by: dependabot[bot] <support@github.com>\nCo-authored-by: dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>",
          "timestamp": "2026-07-20T18:13:31-04:00",
          "tree_id": "53b0a2aaeedd9efadeb6e5d80d448d1a19c1df19",
          "url": "https://github.com/vdiego28/Amalthea.jl/commit/7f76d49acb04c04c72a496c329763efc02bbf6e1"
        },
        "date": 1784585708504,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "native mode-avg+plasma per-step (fixed dt)",
            "value": 2.927187,
            "unit": "ms/step"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "vdiego28@yahoo.es",
            "name": "vdiego28",
            "username": "vdiego28"
          },
          "committer": {
            "email": "vdiego28@yahoo.es",
            "name": "vdiego28",
            "username": "vdiego28"
          },
          "distinct": true,
          "id": "e0612a18854e9e61fb5451b69b05a31d5a6e7d35",
          "message": "Merge worktree-agent-a29df789be3b26da4: S6.3 CLI plan docs\n\nDocs-only (`docs/dev/native-port/PLAN_S6_3_CLI.md` + BACKLOG S6.3 status),\nso no gate required.\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>",
          "timestamp": "2026-07-22T10:12:47-04:00",
          "tree_id": "5f6afe59235cb9cb1c4c83d603ed5b7d168582d6",
          "url": "https://github.com/vdiego28/Amalthea.jl/commit/e0612a18854e9e61fb5451b69b05a31d5a6e7d35"
        },
        "date": 1784729798157,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "native mode-avg+plasma per-step (fixed dt)",
            "value": 2.950042,
            "unit": "ms/step"
          }
        ]
      }
    ]
  }
}