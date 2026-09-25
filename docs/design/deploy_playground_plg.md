# Deploying the playground at plg.duetlabs.co

duetlabs.co is on Vercel (www CNAME -> vercel-dns) with DNS at Namecheap (registrar-servers). Two deployables:

## A. Static playground (viewer only) — recommended first
1. `.venv/bin/python scripts/playground_export_static.py --out data/playground/static_export` (self-contained: index.html, app.js, JSON, small MP4s; `vercel.json` included).
2. `cd data/playground/static_export && npx vercel --prod` (or import the folder as a new Vercel project "duet-playground"; framework: Other; no build step).
3. Vercel project → Settings → Domains → add `plg.duetlabs.co`.
4. Namecheap → Advanced DNS → add CNAME record: host `plg`, value `cname.vercel-dns.com`.
5. Vercel issues TLS automatically. Videos need HTTP Range: Vercel static hosting supports it (verified locally with a range-capable server that the offsets hold).

Re-deploy after each new export. Keep episodes to a few hundred MB total; large sets should go to S3/CloudFront with the same folder layout.

## B. Full pipeline (run stages from the browser)
FastAPI + GPU perception cannot run on Vercel. Run `scripts/playground.py serve --host 0.0.0.0 --port 8791` on a GPU box (an AWS g5/g6 instance from the Activate credits, or the NVIDIA desktop used for ZED capture) behind Caddy with basic auth, and point `plg-run.duetlabs.co` (CNAME/A record) at it. Uploads: add an `/api/upload` route or sync raw episodes to the box with rsync/S3.
