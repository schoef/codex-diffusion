# NEF–QVF Diffusion Explorer

Interactive browser laboratory for the five NEF–QVF reference diffusions used in the accompanying study: Gaussian, Poisson, Gamma, binomial, and negative-binomial. The generalized-hyperbolic-secant family is intentionally excluded.

The explorer lets you choose a family, baseline law, data law (including atoms), sample size, and maximum noising time. It animates incremental noising, simulated paths, deterministic moment flow, and a log-scaled density map. Linear and logarithmic time displays are available independently.

Live version: https://nef-qvf-diffusion-explorer.che55e.chatgpt.site

## Run locally

Requirements: Node.js 22.13 or newer and a Linux environment with GNU `timeout`.

```bash
cd explorer
npm ci
npm run dev
```

Build the deployable Cloudflare Worker/Vinext bundle with:

```bash
npm run build
```

The principal implementation is in `app/page.tsx`, with styling in `app/globals.css`.

## Preserved release

This source snapshot corresponds to the preserved ChatGPT Sites release, version 13 (24 August 2026). The public Sites URL may later point to a newer release; the Git history in this repository provides the immutable record.
