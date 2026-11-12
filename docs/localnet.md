# A whole cycle on a local chain

The team doesn't test on testnet. Every end-to-end run happens on a **local subtensor chain**, with real extrinsics and nothing public: the opentensor localnet image, on one machine. `tools/localnet/cycle.py` drives a complete day through the same commands an operator uses, then reads the weights back from the chain.

## What the run proves

1. A subnet is registered, and two miners and two validators join it.
2. Each miner buys a credit with `prometheon credits buy`, a real `transfer_stake` to a local treasury coldkey. The script checks the treasury's stake afterwards.
3. Each miner commits a real Hugging Face model with `prometheon model commit`, and checks it with `model verify`:
   - `HuggingFaceTB/SmolLM2-135M`
   - `Qwen/Qwen2.5-0.5B`
4. A local hub (the in-repo `FakeHub`, served over HTTP) freezes the day into a signed epoch from what the chain says: neurons, coldkeys, commitment blocks and weights fingerprints. A local labeller answers from a fixed truth table.
5. `prometheon validator run` does the rest:
   - fetches and verifies the epoch, and downloads the files
   - labels the day, one post per request, and runs both models on the CPU
   - reads each model's parameter count from its safetensors headers
   - scores, sets weights on chain, and publishes its signed record
6. The script reads the weights back from the chain and requires all four of these to agree:
   - the weights on chain
   - the vector the validator saved before sending
   - the record it published
   - the spec's split worked out by hand from the record's counts, with each model's size penalty recomputed and the places decided by the one-point rule
7. A second validator runs `prometheon validator run` in copy mode. It fetches the owner validator's record from the hub's weights route, checks it, and sets the same weights. The script reads them back from the chain and requires them to equal the owner validator's.

## Run it

```bash
# 1. The chain. Reuse the container if it is already running: others may be using it.
docker ps --filter name=prometheon-localnet
docker run -d --name prometheon-localnet -p 127.0.0.1:9945:9944 \
    ghcr.io/opentensor/subtensor-localnet:devnet-ready

# 2. The code, with the evaluation runtime.
uv sync --extra wrapper

# 3. One day, end to end.
uv run --extra wrapper python tools/localnet/cycle.py \
    --endpoint ws://127.0.0.1:9945 --workdir /tmp/prometheon-localnet
```

The script needs network access to Hugging Face. It uses ports 8600 for the hub and 8601 for the labeller (`--hub-port`, `--labeller-port`). Add `--min-allowed-weights 3` to make the chain demand more weights than the day earns. The run then also checks, by hand, that the vector was shaped to the subnet's limits and that the chain accepted it (see [`scoring.md`](./scoring.md#the-subnets-weight-limits)). It prints each CLI command's output and ends with:

```text
[localnet] units sent:      {...}
[localnet] u16 vector sent: {...}
[localnet] read from chain: {...}
[localnet] copy-mode validator (uid 3) on chain: {...}
[localnet] PASS: chain weights = the vector computed, saved and published = the spec by hand
[localnet] PASS: the copy-mode validator set the owner validator's weights, unchanged
```

The workdir keeps everything for inspection:

- the wallets, under `home/.bittensor`
- the four configs, made from `configs/localnet.example.toml` with only the run's values filled in
- both validators' state files

## What it changes on the chain

It's a development chain, but it may be shared, so here is everything the run writes:

- **A new subnet every run**, owned by `//Alice`, with:
  - commit-reveal off
  - no weights rate limit
  - tempo 20
- **The chain-wide admin freeze window.** The image's tempo is shorter than its admin freeze window, so no hyperparameter can change while the window is in force. The script reads the window, sets it to 0 for the three changes above, and puts the original value back straight afterwards.
- **Registrations and stake:**
  - `//Bob` and `//Charlie` register a miner hotkey each, and `//Dave` registers a second validator hotkey, the one in copy mode.
  - `//Alice` and `//Dave` stake 1,000 TAO each on their validator hotkeys.
  - Each miner stakes in 100 TAO steps until its hotkey holds 55 alpha.
  - Each miner then pays one credit to `//Eve`: the published fee of 50 alpha. The fee isn't a setting, so the run pays the real amount.

The dev accounts' keys are public by design, and none of this exists anywhere but the local chain. The script never stops the container.

## Things the localnet does differently

- **Short forks are routine.** The image runs two block authors with fast blocks. The SDK reads state again after a write, at a block hash it took before the write, and that read occasionally fails with `State already discarded` although the extrinsic is in. The script checks each setup write by its effect, such as the stake or the registration, rather than by the SDK's report. `prometheon credits buy` does the same for the payment itself: if the SDK reports a failure, the command reads the treasury's stake and reports what actually happened.
- **The scoring floor is lowered for the run.** `model_min_scored_items` is set to 10, because the day holds a few dozen items rather than thousands. Every other `[scoring]` value is the shipped one.
- **CPU, float32.** Fine for two small models. It isn't a measure of what a real day costs; see [`validator.md`](./validator.md#hardware).

## Doing it by hand

The script is the reference, but every step is an ordinary command. With a local chain running:

1. Create a subnet with `btcli subnet create`, and register hotkeys.
2. Fill in `configs/localnet.example.toml` with the netuid, a hub URL and its key, and a labeller.
3. Use `prometheon credits buy`, `model commit` and `model verify` as a miner would.
4. Run `prometheon validator run` against a hub.

For the hub, `prometheon.hub.fake.serve(FakeHub(...))` serves the whole contract on a local port, and [`hub-contract.md`](./hub-contract.md) describes it.
