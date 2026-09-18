# Spend the absolute minimum of your $200 DigitalOcean credit

**The only real cost is keeping the heavy torch/video stack running 24/7 — so don't.**
Split the app by resource profile:

| part | needs | where it lives | cost |
|---|---|---|---|
| Models A/B calculator (torch-free, 1.6 MB) | <256 MB RAM | **smallest droplet, always on** | **$4–6/mo** |
| Live video (SigLIP+CLAP+Whisper, ~4 GB) | 4 GB RAM, 30–120 s/clip | **on demand only** (snapshot or laptop) | **~$0** |

> DigitalOcean bills a droplet **even when powered off** — only *destroying* it stops the meter.
> That's why the video box is snapshot-and-destroy, not "turn off when idle."

---

## Tier 1 — always-on calculator (the cheap public URL)

Smallest droplet that runs the slim app. **`s-1vcpu-1gb` ≈ $6/mo** (comfortable), or
**`s-1vcpu-512mb-10gb` ≈ $4/mo** (add the 4 GB swap from `setup_droplet.sh`).

```bash
# 1. create the smallest droplet (Ubuntu 24.04), then:
ssh root@<ip> 'bash -s' < deploy/setup_droplet.sh          # installs Docker
rsync -az heroku_app deploy root@<ip>:/opt/shouldipost/    # tiny upload (~2 MB)
ssh root@<ip> 'cd /opt/shouldipost/deploy && docker compose -f slim.docker-compose.yml up -d --build'
# -> http://<ip>/   (manual entry, Model A pre-post, Model B day-1)
```

**Burn:** $6/mo × 12 = **~$72** of your $200 over the whole credit window → **~$128 left unused**.
At $4/mo it's ~$48. Effectively free for the year.

---

## Tier 2 — live video, at ~zero idle cost

Pick one:

**A. Laptop (simplest, $0).** Run `./start.sh` locally during the demo. Your Mac's MPS is *faster*
than a CPU droplet anyway. The public URL stays the cheap Tier-1 calculator; you drive video live.

**B. Snapshot-on-demand (serves video from a real URL, ~$0.30–0.50/mo idle).** Build the heavy
image once, snapshot it, destroy the droplet. Recreate from the snapshot only for a demo — it boots
in ~1 min with the image built and models cached, then you destroy it again.

```bash
# one-time: build the full app on a 4 GB droplet, then snapshot + destroy
doctl compute droplet create sip-video --size s-2vcpu-4gb --image ubuntu-24-04-x64 \
  --region <r> --ssh-keys <id> --wait
# ...provision + rsync full repo + `docker compose up -d --build` (see DEPLOY_DIGITALOCEAN.md)...
doctl compute droplet-action snapshot <id> --snapshot-name sip-video --wait
doctl compute droplet delete <id> --force        # meter stops; only the snapshot is billed (~$0.06/GiB-mo)

# per demo: recreate from the snapshot, use it, destroy it
doctl compute droplet create sip-video --image <snapshot-id> --size s-2vcpu-4gb --region <r> \
  --ssh-keys <id> --wait                          # ~$0.036/hr → a 2-hour demo ≈ $0.07
# ... demo at http://<new-ip>/ ...
doctl compute droplet delete sip-video --force
```

**Burn:** snapshot storage ~$0.30–0.50/mo + a few cents per demo → **cents per month**.

---

## Bottom line

- **Cheapest that's always reachable:** Tier 1 only ($4–6/mo), video on your laptop. ~$50–70/yr.
- **Cheapest that also serves video from a URL:** Tier 1 + Tier 2-B snapshot. ~$55–75/yr.
- Either way you finish the 12-month credit window having spent well under half of the $200.

**Don't:** run the 4 GB video droplet 24/7 ($24/mo = ~$288/yr, blows the credit), or use App Platform
(cheapest dynamic tier ~$5/mo *and* reintroduces the timeout/RAM limits for video).
