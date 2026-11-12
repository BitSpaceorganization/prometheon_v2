# systemd units for a validator

**`prometheon-validator.service` assumes a score-mode host with one GPU of 24 GB, for example an RTX 4090.** A cycle downloads and runs every eligible paid model. A box without CUDA falls back to CPU and is far too slow: its models time out and get no score from it. For copy mode, remove `--extra wrapper` from its `ExecStart`. See [`docs/validator.md`](../../docs/validator.md#copy-mode).

There are two timers, and you need both. The cycle scores a day; the re-post keeps that score counting until the next one.

| Unit | Cadence | Cost |
|---|---|---|
| `prometheon-validator.timer` | daily, 04:00 UTC | labelling and evaluation |
| `prometheon-resubmit.timer` | every 30 minutes, on :00 and :30 | one extrinsic |

**Running only the first one earns the miners you weighted nothing for most of the day.** Weights stop counting toward consensus once the subnet's `activity_cutoff` passes, while the cycle runs every 24 hours. `validator resubmit` re-sends the vector the last cycle saved, without labelling or evaluating anything. Check that 30 minutes sits between `weights_set_rate_limit` and `activity_cutoff` for the subnet (`btcli subnet hyperparameters --netuid <netuid>`).

## Install

```bash
sudo cp prometheon-*.service prometheon-*.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now prometheon-validator.timer prometheon-resubmit.timer
systemctl list-timers 'prometheon-*'
```

## Adjust before enabling

- **`WorkingDirectory`:** where you cloned this repository.
- **`ExecStart`:** the absolute path to `uv`. `which uv` often gives `~/.local/bin/uv` rather than `/usr/local/bin/uv`, and systemd doesn't expand `~`.
- **`--config`:** your TOML, copied from `configs/finney.example.toml` with the netuid, wallet, hub and labelling settings filled in.
- **`EnvironmentFile`:** a `0600` file holding `OPENAI_API_KEY`, one `KEY=value` per line. No `export` and no quotes: systemd parses this file itself; it isn't a shell script. Copy mode needs no key.

## Checking on it

```bash
systemctl list-timers 'prometheon-*'
journalctl -u prometheon-validator.service -n 50
journalctl -u prometheon-resubmit.service -n 20
```

A re-post that lands inside the weights rate limit, for example just after the daily cycle sent its vector, reports `rate limit not clear for another N blocks; nothing sent` and exits `0`. That's expected, not a failure.

A failed daily cycle is retried by `Restart=on-failure` after 15 minutes. A finished day isn't run again. A day whose vector was saved but not sent is sent on the next run, without being computed again.
