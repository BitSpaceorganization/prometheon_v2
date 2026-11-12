# systemd units for a validator

**`prometheon-validator.service` assumes a score-mode host with one GPU of 24 GB, for example an RTX 4090.** A cycle downloads and runs every eligible paid model. A box without CUDA falls back to CPU and is far too slow: its models time out and get no score from it. For copy mode, remove `--extra wrapper` from its `ExecStart`. See [`docs/validator.md`](../../docs/validator.md#copy-mode).

There are three timers. You need the first two: the cycle scores a day, and the re-post keeps that score counting until the next one. The third looks for a newer release, and does nothing unless `[update] auto = true`.

| Unit | Cadence | Cost |
|---|---|---|
| `prometheon-validator.timer` | daily, 04:00 UTC | labelling and evaluation |
| `prometheon-resubmit.timer` | every 30 minutes, on :00 and :30 | one extrinsic |
| `prometheon-update.timer` | every 30 minutes, on :15 and :45 | a `git fetch`; an install when a release is newer |

**Running only the first one earns the miners you weighted nothing for most of the day.** Weights stop counting toward consensus once the subnet's `activity_cutoff` passes, while the cycle runs every 24 hours. `validator resubmit` re-sends the vector the last cycle saved, without labelling or evaluating anything. Check that 30 minutes sits between `weights_set_rate_limit` and `activity_cutoff` for the subnet (`btcli subnet hyperparameters --netuid <netuid>`).

## Install

The units run from `/opt/prometheon/current`, the side-by-side layout that updates use. [`docs/validator.md`](../../docs/validator.md#updating) shows how to set it up.

```bash
sudo cp prometheon-*.service prometheon-*.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now prometheon-validator.timer prometheon-resubmit.timer
sudo systemctl enable --now prometheon-update.timer      # if you turn on [update] auto
systemctl list-timers 'prometheon-*'
```

## Adjust before enabling

- **`WorkingDirectory`:** `/opt/prometheon/current`, or wherever your install's `current` link is.
- **`ExecStart`:** the absolute path to `uv`. `which uv` often gives `~/.local/bin/uv` rather than `/usr/local/bin/uv`, and systemd doesn't expand `~`.
- **`--config`:** your TOML, copied from `configs/finney.example.toml` with the netuid, wallet, hub and labelling settings filled in.
- **`EnvironmentFile`:** a `0600` file holding `OPENAI_API_KEY`, one `KEY=value` per line. No `export` and no quotes: systemd parses this file itself; it isn't a shell script. Copy mode needs no key.

## Checking on it

```bash
systemctl list-timers 'prometheon-*'
journalctl -u prometheon-validator.service -n 50
journalctl -u prometheon-resubmit.service -n 20
journalctl -u prometheon-update.service -n 20
```

A re-post that lands inside the weights rate limit, for example just after the daily cycle sent its vector, reports `rate limit not clear for another N blocks; nothing sent` and exits `0`. That's expected, not a failure.

A failed daily cycle is retried by `Restart=on-failure` after 15 minutes. A finished day isn't run again. A day whose vector was saved but not sent is sent on the next run, without being computed again.

## Updates

With `[update] auto = true`, `prometheon-update.service` installs a newer release tag beside the running one, has it check itself, and switches `current` to it between cycles. Every unit starts from `current`, so the next run of each is the new release, and nothing needs restarting. If you run a long-lived process, such as `validator schedule` under a unit of its own, name that unit in `[update] restart_units`.

A failed update leaves the running release current, and says why in `journalctl -u prometheon-update.service`. `uv run prometheon update rollback --config /etc/prometheon/prometheon.toml` switches back to the previous release.
