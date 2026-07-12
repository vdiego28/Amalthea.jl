window.BENCHMARK_DATA = {
  "lastUpdate": 1783893359477,
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
      }
    ]
  }
}