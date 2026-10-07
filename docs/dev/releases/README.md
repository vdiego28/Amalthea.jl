# Release operation

Design: [PLANS §32](../native-port/PLANS.md#32-launch-preparation-and-exact-source-release-artifacts-2026-09-29).
Current status and blockers: [BACKLOG resume queue](../BACKLOG.md#start-here--current-resume-queue-2026-09-06).
Prepared release body: [v1.1.0.md](v1.1.0.md).

Read candidate versions from the package manifests and the matching release
body. This checklist describes the procedure; current commits, completed
gates, blockers and the published release belong in the linked backlog.
Neither candidate metadata nor a successful preparation run establishes
publication.

## Before delivery

1. Identify the intended dependency and repair changes. Review their current
   PR heads and completed checks before integration, then record the exact
   final `main` SHA. Preserve failure artifacts and distinguish branch-head,
   PR-merge and final-main evidence. A successful preparation run for an older
   commit cannot accept a newer candidate.
2. Check candidate metadata and release tooling at that SHA. Run
   `python3 test/release.py metadata` and
   `python3 test/test_release.py -v`.
3. Require fresh exact-candidate complete CPU/FFI, installed/offline sixteen-cell
   Python, real Apple and ARM64, and documentation gates. Rebuilds caused by
   version/README metadata changes are new artifacts. Repeat the recorded
   minimum-glibc 2.28 runtime gate for CPython 3.11–3.14; preserve its manifest,
   artifact and source hashes. Confirm that its jobs actually completed and
   its results match the candidate; old running-job notes are not acceptance.
   This separate runtime gate is not supplied by the release-preparation
   workflow's successful assembly. After downloading the candidate or draft
   assets, run `test/release.py verify-glibc` with the four completed helper
   outputs, their retained builds, matching references and candidate checkout.
   Require its passing report to bind the runtime-tested source wheels to the
   actual release bytes; retain that report alongside the unmodified inputs.
   Directory layout and invocation are in the
   [minimum-glibc gate](../native-port/TESTING.md#linux-minimum-glibc-runtime-gate).
   Canonical numerical commands and acceptance
   criteria are in [TESTING](../native-port/TESTING.md#5-commands).
4. After the candidate reaches `main`, manually dispatch **Prepare Amalthea
   release** at that commit. This waits for successful exact-source push CI
   (including documentation, whose push trigger covers `main` and tags), runs locked Cargo
   tests on four platforms, rechecks installed wheel evidence and stages the
   tested source-rebuilt wheels. Download the `release-candidate` artifact and
   verify `sha256sum -c SHA256SUMS.txt` (or platform-equivalent checks).
5. Review the complete inventory: four CPU libraries, sixteen wheels, one
   source archive, Python provenance/evidence, release manifest and checksums.
   Check all four interpreter versions and platform tags. Install a downloaded
   candidate with `--only-binary=:all:` in a fresh environment, run both public
   propagation APIs, and inspect NPZ/HDF5 results. Use the maintained offline
   and numerical gate for acceptance; a two-call smoke alone is insufficient.
6. Review the tracked release body, upgrade notes, support boundary and citation.
   Public installation links continue to name the currently published release
   until publication. PyPI distribution-name availability must be rechecked
   if that route is chosen;
   the GitHub wheel route requires no PyPI account or credentials.

## Tag and draft, after explicit authorization

The lead authorizes the exact tested commit and matching version tag. Tag
creation/push starts fresh tagged CI; the release workflow refuses a tag that differs from
`Project.toml`. Its tag path creates a **draft**, after complete exact-source
tests/documentation and all artifact checks. Manual dispatch never publishes
a release. Review the draft's bytes, manifest revision and release notes.

Publication is a separate lead action. After publication, verify the release
is public, download and checksum its actual assets, install the tagged Julia
package through its prebuilt path with Cargo absent, and install a published
Python wheel outside the checkout. Run complete offline examples. Update
README/manual current-release examples, CHANGELOG's Unreleased heading,
CITATION's actual release date and independently verified version DOI, then
record evidence in PORT_LOG and current status in BACKLOG. Do not infer
publication from a tag, draft or successful build.

## Retry or hold

A failed/skipped/missing required CI job, incomplete wheel cell, expired
reference artifact, version mismatch or checksum/inventory error prevents
draft assembly. Retain the failure and rerun the affected workflow at the same
source when appropriate; regenerate expired oracle artifacts with their tests.
Fixes require a new candidate and its own acceptance. Never reuse old wheel
bytes under a new version or retarget a published tag. If post-publication
installation fails, retain the previous release as an available fallback and
prepare a new correction release rather than replacing validated assets in place.
