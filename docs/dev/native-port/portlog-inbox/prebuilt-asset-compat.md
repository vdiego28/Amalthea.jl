# Inbox note — prebuilt-binary asset-name compatibility (BACKLOG resume-queue item 4)

Agent: worktree `agent-ad38fdad15bd1fbf9`. Date: 2026-07-25.
Scope: **local half only** of "make prebuilt-binary installation work despite
the release asset-name mismatch." Exclusive file zone: `deps/build.jl`
(+ this new inbox file). Did **not** touch `.github/workflows/release.yml`,
any GitHub release asset, `PORT_LOG.md`, `BACKLOG.md`, `ARCHIVE.md`,
`SUGGESTIONS.md`, or `README.md`.

## 1. Confirmed problem (read-only `gh` inspection)

`gh release view v1.0.0 --repo vdiego28/Amalthea.jl` shows the real,
currently-published assets:

```
asset:  libluna_rust-aarch64-apple-darwin.dylib
asset:  libluna_rust-x86_64-pc-windows-msvc.dll
asset:  libluna_rust-x86_64-unknown-linux-gnu.so
asset:  SHA256SUMS.txt
```

`Project.toml`'s `version` is `1.0.0` — so `deps/build.jl`'s
`try_download_prebuilt` (pre-fix) requests `libamalthea-<triple>`, misses
every asset in the manifest, and falls straight to `cargo build --release`
for every installer today. Confirmed the repo has exactly two releases
(`v0.7.0`, `v1.0.0`); only `v1.0.0` carries this legacy naming (it's the
first release published post-rename, but before `release.yml` was updated to
match).

Fetched the real `SHA256SUMS.txt` for v1.0.0 directly
(`curl -sL .../v1.0.0/SHA256SUMS.txt | cat -A`) to verify the line format my
parser assumes, rather than inferring it from the current workflow (which
post-dates that release and could differ): confirmed two-space `sha256  name`
separator on macOS/Linux lines and single-space + CRLF on the Windows line —
Julia's `split(line)` (whitespace-collapsing, and `eachline` already strips
`\r\n`) parses all three correctly with no code change needed for that.

## 2. What I changed — `deps/build.jl`

- **`_LAST_LEGACY_NAMED_VERSION = v"1.0.0"`, `_LEGACY_LIBNAME_PREFIX =
  "libluna_rust"`** (new consts, `deps/build.jl:19-32`).
- **`_prebuilt_asset_candidates(triple, ext, version)`** (new function,
  `deps/build.jl:46-61`): returns `["libamalthea-<triple><ext>"]` always
  first, appending `"libluna_rust-<triple><ext>"` only when
  `version <= _LAST_LEGACY_NAMED_VERSION`.
- **`try_download_prebuilt(rust_dir; base_url=nothing)`** (rewritten,
  `deps/build.jl:82-143`):
  - Added a `base_url` keyword (default `nothing` → unchanged production
    GitHub URL) purely so tests can redirect it at a local HTTP server. The
    real call site (`deps/build.jl:147`) still calls it with zero arguments,
    so production behavior/URL is byte-identical to before when there's no
    legacy asset in play.
  - Downloads `SHA256SUMS.txt` **once** per release, then loops over the
    candidate asset names against that same manifest (canonical first) —
    not a fresh `SHA256SUMS.txt` fetch per candidate.
  - Per candidate: if the name isn't in the manifest, or its own download
    404s/network-errors, `continue` to the next candidate (this is the
    "this release just doesn't have that name" case). If the name *is* in
    the manifest but the downloaded bytes don't match the checksum,
    **`return false` immediately** — does not try the next candidate. See
    "Decision 2" below for why this is a deliberate asymmetry.
  - Everything is still inside the outer `try`/`catch` that never
    `rethrow`s, still inside `mktempdir() do ... end` so temp files are
    always cleaned up, and the `mv` into
    `amalthea/target/release/<libname>` only happens after a verified match,
    same as before.
- `AMALTHEA_RUST_SKIP_DOWNLOAD=1` check is untouched (first line of the
  function, still short-circuits before any network I/O).

## 3. Decisions and reasoning

**Decision 1 — bound the legacy fallback to `<= v1.0.0`, not open-ended.**
`release.yml` (checked, see §4) already stages `libamalthea-*` for every tag
going forward, and v1.0.0 is the *only* release that ever used the old name
(v0.7.0 predates any binary-asset publishing at all, per the earlier
`hygiene.md` inbox note's finding that no release carried assets before
v1.0.0). An unconditional "try `libluna_rust-<triple>` whenever
`libamalthea-<triple>` is missing" would silently paper over a *future*
regression — e.g. if `release.yml` ever reverts the asset name, or an
upload step fails and only partially populates a release — by having the
installer succeed anyway via a name that was never supposed to exist for
that version. Bounding it to versions at or before the last legacy release
means: the fallback only ever fires for the exact historical situation it
was built for, and any future miss surfaces honestly as a download failure
(→ falls to `cargo build --release`, which always works if a toolchain is
present) rather than a masked one.

**Decision 2 — checksum mismatch fails the whole attempt, does not cascade
to the next candidate.** Originally I had mismatch `continue` like the
other two failure modes, symmetric with "not in manifest"/"download
failed". Reviewer feedback (external second opinion) changed this: "not in
manifest" and "download failed" both mean *this asset doesn't exist under
this name for this release* — trying the next name candidate is exactly
right. A checksum mismatch means the manifest *does* have an entry for that
name, but the bytes don't match it — a tamper/corruption signal about the
release itself, not a naming question. Continuing to the legacy candidate
after a canonical-name checksum mismatch could mask a genuinely compromised
or corrupted release behind an unrelated fallback path succeeding. So a
mismatch now aborts the whole `try_download_prebuilt` call
(`return false`) rather than trying legacy next. This satisfies the task's
literal requirement ("checksum mismatch on the legacy asset is rejected and
falls back to source without installing") either way, since legacy is the
last candidate in every currently-real scenario — but it's a real behavior
choice for the hypothetical "canonical asset exists but is corrupted, AND a
legacy asset also exists" case, recorded here per the task's ask to state
the reasoning.

**Decision 3 — `base_url` keyword, not a bigger refactor.** Minimal
surface: the resolved production URL computation is untouched string
interpolation, just conditionally overridable. No new file, no
config/env-var indirection for the URL itself (env-var override felt like
scope creep for a test-only seam, and would add a second, undocumented way
to redirect the download in production).

## 4. `.github/workflows/release.yml` — checked, unchanged

Confirmed lines 25/28/31 (matrix `libname`) and 55/64/67 (asset staging)
already produce `libamalthea-<triple>.<ext>` for all three platforms,
including Windows (staged as `libamalthea-<triple>.dll`, i.e. still with the
`lib` prefix despite the `.dll` extension, consistent with Unix). This
already matches the canonical name `deps/build.jl` requests first. **No
change made** — future tagged releases already produce correctly-named
assets; only the already-published `v1.0.0` needed the compatibility
fallback.

## 5. Verification

All commands run from the worktree root
(`/home/diego/Documents/fernando_luz/Luna-Rust.jl/.claude/worktrees/agent-ad38fdad15bd1fbf9`).
Scratch scripts and fixtures live under
`/tmp/claude-1000/.../scratchpad/` (`make_fixtures.jl`, `test_prebuilt.jl`,
`test_real_release.jl`, `webroot/`) — **not** committed to the repo.

### 5a. Confirmed the real release's manifest line format
```
curl -sL https://github.com/vdiego28/Amalthea.jl/releases/download/v1.0.0/SHA256SUMS.txt | cat -A
```
→ 3 lines, `hash<space><space>name` (macOS/Linux) and `hash<space>name<CR>`
(Windows). Verified in a real Julia session that `split(line)` over
`eachline(path)` parses all three into exactly `["hash", "name"]` — no
special-casing needed.

### 5b. Real end-to-end run against the actual (unmodified) v1.0.0 release
Built `amalthea/target/release/libamalthea.so` locally (`cargo build
--release`, needed as a same-triple binary is not otherwise present in a
fresh worktree; not otherwise touched by this task). Loaded only the
function-definition prefix of `deps/build.jl` (via `include_string` with an
explicit filename so `@__DIR__` still resolves correctly, stopping before
the executable tail so no real `cargo build`/production
`try_download_prebuilt` call happens as a side effect of loading the file),
then called `try_download_prebuilt(mktempdir())` with **no `base_url`
override** — the real production code path, hitting the real GitHub v1.0.0
release:

```
$ julia test_real_release.jl <repo_root>
[ Info: No checksum entry for libamalthea-x86_64-unknown-linux-gnu.so in SHA256SUMS.txt; trying next candidate.
[ Info: Downloaded prebuilt amalthea library (libluna_rust-x86_64-unknown-linux-gnu.so, v1.0.0), skipping cargo build.
Installed size: 1116024 bytes
Test Summary:                       | Pass  Total  Time
real v1.0.0 release, production URL |    4      4  0.0s
```
4/4 assertions: download succeeded, file installed at the canonical
`<rust_dir>/target/release/libamalthea.so` path, non-empty, and no leftover
temp files in the scratch `rust_dir`. This is the actual failure this task
targets, fixed and observed fixed against the real artifact — not just a
local fixture.

### 5c. Local HTTP-server fixture suite (branch-logic coverage)
Following the S6-item-1 precedent (a local server, not a repo test —
per this task's own instructions and to avoid adding an HTTP.jl test
dependency), built 4 fixture release layouts under
`scratchpad/webroot/{happy,badsum,both,miss}/`, each a real copy of the
locally-built `libamalthea.so` (or a 1-byte-perturbed copy for `both`, to
prove which asset actually gets installed) plus a real `SHA256SUMS.txt`
line, served via `python3 -m http.server 8792`. Ran
`scratchpad/test_prebuilt.jl` (same `include_string`-prefix-only loading
trick) with `base_url` pointed at each fixture subpath:

```
Test Summary:                       | Pass  Total  Time
prebuilt asset legacy-name fallback |   20     20  1.6s
```

Scenarios (all passed):
1. **Legacy-name happy path** — only `libluna_rust-<triple>.so` present,
   correct checksum → downloads, verifies, installs to the canonical path;
   no leftover `.download`/`SHA256SUMS.txt` files.
2. **Checksum mismatch on legacy asset** — manifest lists a wrong hash for
   the only present asset → `try_download_prebuilt` returns `false`, no
   file installed, no temp files left.
3. **Canonical wins when both present** — fixture has both names in the
   manifest with correct checksums but *different bytes* (canonical fixture
   is the real file + 1 trailing byte) → installed file's bytes match the
   canonical fixture exactly, not the legacy one, proving priority order.
4. **Total miss** — manifest exists (simulating a real release for a
   different platform/version) but neither asset name appears in it →
   returns `false`; a pre-existing sentinel file at the destination path is
   left byte-for-byte and mtime-unchanged (checked with a >1s sleep before
   comparing `mtime`); no temp files left.
5. **`AMALTHEA_RUST_SKIP_DOWNLOAD=1`** still short-circuits to `false`
   before any network call, even against the known-good `happy` fixture.

## 6. What remains for the user

- **The actual republish decision is explicitly not mine to make** — I did
  not run `gh release upload`/`delete`/`edit` or touch any asset. The code
  now works correctly against the *existing* legacy-named v1.0.0 assets as
  they stand today, so no republish is strictly required for
  `try_download_prebuilt` to succeed. If the user separately decides to
  republish v1.0.0's assets under the canonical `libamalthea-*` names later
  (e.g. for consistency, or to shed the legacy branch sooner), that's fine
  and this code will simply take the canonical fast path afterward — the
  legacy fallback becomes dead code for that release but stays harmless
  (and still bounded) for as long as it's kept.
- If a future release is ever tagged with a version `> 1.0.0` and its
  `libamalthea-<triple>` asset is missing for some unrelated reason (a CI
  failure, a partial upload), this code will **not** attempt the legacy
  name for it — by design (Decision 1) — and will fall straight to
  `cargo build --release`. That's the intended behavior, not a gap, but
  worth knowing if a future maintainer wonders why the fallback "stopped
  working" for a newer version.
- No test was added to the repo's own test suite (`test/`) — per this
  task's instructions, the verification lived in scratchpad scripts since
  standing up an HTTP server + fixture binaries as a permanent, always-run
  test would be a heavier addition than the S6-item-1 precedent used, and
  a repo test can't safely hit the real GitHub release either. If the lead
  wants a permanent regression test, `scratchpad/test_prebuilt.jl`'s
  approach (fixture webroot + `base_url` override) is directly liftable
  into `test/` behind a `:rust`-style tag or an opt-in env var, since the
  `base_url` seam already exists for exactly this purpose.
