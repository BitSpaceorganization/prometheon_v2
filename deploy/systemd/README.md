# systemd units for a validator

Two timers, and you need both. The cycle scores a day; the re-post keeps that
score counting until the next one.

| Unit | Cadence | Cost |
|---|---|---|
| `prometheon-validator.timer` | daily, 04:00 UTC | labelling + evaluation |
| `prometheon-resubmit.timer` | hourly | one extrinsic |

**Running only the first one earns your miners nothing for most of the day.**
Weights stop counting toward consensus once `activity_cutoff` passes — 720
blocks, about 2.4 hours on netuid 108 — while the cycle runs every 24. For the
remaining ~21 hours the validator's row is masked out, and every miner it
weighted sits at zero incentive and zero emission. `validator resubmit` re-sends
the allocation the last cycle computed, without re-labelling or re-evaluating.

## Install

```bash
sudo cp prometheon-*.service prometheon-*.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now prometheon-validator.timer prometheon-resubmit.timer
systemctl list-timers 'prometheon-*'
```

## Adjust before enabling

- `WorkingDirectory` — where you cloned this repository.
- `ExecStart` — the absolute path to `uv`. `which uv` often gives
  `~/.local/bin/uv` rather than `/usr/local/bin/uv`, and systemd does not expand
  `~`.
- `--config` — your TOML, copied from `configs/mainnet.example.toml`.
- `EnvironmentFile` — a `0600` file holding `OPENAI_API_KEY`,
  one `KEY=value` per line. No `export`, no quotes: systemd
  parses this file itself, it is not a shell script.

## Checking on it

```bash
systemctl list-timers 'prometheon-*'
journalctl -u prometheon-validator.service -n 50
journalctl -u prometheon-resubmit.service -n 20
```

A re-post that lands inside the 100-block (~20 min) weights rate limit — which
happens when the hourly timer fires just after the daily cycle — reports
`rate limit not clear for another N blocks; nothing sent` and exits `0`. That is
the expected outcome, not a failure.
