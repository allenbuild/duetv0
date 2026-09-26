# Deploying the playground at plg.duetlabs.co

Status (2026-09-25): the static playground is **live** on Vercel; the custom domain is attached and **waiting on two DNS records at Namecheap**.

| item | value |
|---|---|
| live now | https://duet-playground.vercel.app |
| Vercel project | `duet-playground` (scope `idhantranjans-projects`, Idhant's account) |
| custom domain | `plg.duetlabs.co` — added to the project, state `pending_domain_verification` |
| why pending | `duetlabs.co` is registered in Allen's Vercel account (deployed from his GitHub). A different Vercel account may serve a subdomain only after proving control of the apex with a TXT record. |

## Step 1 — two records at Namecheap (Advanced DNS for duetlabs.co)

| type | host | value | TTL |
|---|---|---|---|
| TXT | `_vercel` | `vc-domain-verify=plg.duetlabs.co,5605f4c7c1b28ff43412` | automatic |
| CNAME | `plg` | `cname.vercel-dns.com` | automatic |

Namecheap shows only the host part; do not type `.duetlabs.co` after it. Whoever has the Namecheap login (Allen) adds these; nothing else on the domain changes, `www` keeps pointing at the main site.

## Step 2 — verify (usually within 5–10 minutes of the records appearing)

```bash
dig +short _vercel.duetlabs.co TXT
dig +short plg.duetlabs.co CNAME
cd data/playground/static_export && vercel domains verify plg.duetlabs.co
curl -sI https://plg.duetlabs.co/ | head -3
```

Vercel issues the TLS certificate itself once the CNAME resolves. Expect a 200 from the last command.

### Alternative if Allen prefers to keep everything in his account
Allen adds `plg.duetlabs.co` to a new project in **his** Vercel account and either (a) imports the folder `data/playground/static_export` from a repo he controls (framework: Other, no build step), or (b) invites `idhantranjan` to his Vercel team, after which the existing project can be transferred there. In that path only the CNAME is needed, no TXT.

## Redeploy after changes

```bash
scripts/deploy_playground.sh
```

That script re-exports the static site from the local episodes and pushes it to production. The `.vercel/` folder inside `static_export` links the folder to the project; it is git-ignored along with the rest of `data/playground`.

Videos need HTTP Range for synced multi-camera playback. Verified on the live site: `curl -I -H "Range: bytes=0-99" .../eidon_10004/streams/ego.mp4` returns `206` with `content-range`. Keep the export to a few hundred MB total; larger episode sets belong on S3/CloudFront with the same folder layout.

## B. Full pipeline (upload a video, run the stages from the browser)
FastAPI plus GPU perception cannot run on Vercel. Run `scripts/playground.py serve --host 0.0.0.0 --port 8791` on a GPU box (an AWS g5/g6 instance from the Activate credits, or the NVIDIA desktop used for ZED capture) behind Caddy with basic auth, and point `plg-run.duetlabs.co` (CNAME/A record) at it. The upload route `/api/upload` already exists; uploaded episodes start processing immediately and appear in the viewer when done.
