# NEF–QVF Diffusion Explorer

Interactive browser laboratory for the five NEF–QVF reference diffusions used in the accompanying study: Gaussian, Poisson, Gamma, binomial, and negative-binomial. The generalized-hyperbolic-secant family is intentionally excluded.

The explorer lets you choose a family, baseline law, data law (including atoms), sample size, and maximum noising time. It animates incremental noising, simulated paths, deterministic moment flow, and a log-scaled density map. Linear and logarithmic time displays are available independently.

Live version: https://nef-qvf-diffusion-explorer.che55e.chatgpt.site

## Local installation

Node.js 22.13 or newer is required. Linux is not required to install or run the
explorer.

### macOS

If a compatible Node.js installation is not already available, install Node 22
with Homebrew. The formula is keg-only, so add it to `PATH` in each new shell
(or put the `export` line in your shell profile):

```bash
brew install node@22
export PATH="$(brew --prefix node@22)/bin:$PATH"
node --version
```

The reported version must be at least `v22.13.0`. Both Apple Silicon and Intel
Homebrew installations are supported because `brew --prefix` supplies the
correct installation path.

### Install project dependencies

From the repository root, install the exact dependency versions in
`package-lock.json`:

```bash
cd explorer
npm ci
```

No Python package, compiler toolchain, database, or GNU command-line utilities
are required for the local explorer.

## Run the development server

```bash
npm run dev
```

Open http://localhost:5173 in a browser. Stop the server with Ctrl-C.

If a new macOS shell reports `node: command not found`, run the `export PATH=...`
command from the installation section again or add it to `~/.zprofile`.

## Build for deployment

Build the deployable Cloudflare Worker/Vinext bundle locally on macOS or Linux
with:

```bash
npm run build:local
```

The output is written to `dist/`.

The hosted Linux environment uses bounded install and build helpers:

```bash
npm run install:ci
npm run build
```

These two CI commands intentionally require GNU `timeout`; the install helper
additionally relies on Linux `flock` and `/proc`. Use plain `npm ci` and
`npm run build:local` for ordinary local development, including on macOS.

The principal implementation is in `app/page.tsx`, with styling in `app/globals.css`.

## Preserved release

This source snapshot corresponds to the preserved ChatGPT Sites release, version 13 (24 August 2026). The public Sites URL may later point to a newer release; the Git history in this repository provides the immutable record.
