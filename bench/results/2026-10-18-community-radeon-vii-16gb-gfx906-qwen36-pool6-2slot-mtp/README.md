# Community benchmark: AMD Radeon VII 16 GB (gfx906), Qwen3.6-35B-A3B UD-Q4_K_XL, `--pool-workers 6`

Measured on 2026-10-18 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: AMD Radeon VII 16 GB (gfx906, Vega 20), one card; the card also drives the desktop
- PCIe: 3.0 x16 (current and max: 8.0 GT/s, width 16)
- Power limit: 250 W (`power1_cap` = `power1_cap_max`)
- ROCm: container image `mixa3607/rocm-gfx906:10.0-complete` (hipcc HIP 7.15.26333, clang 23.0.0git), `HSA_OVERRIDE_GFX_VERSION=9.0.6`
- CPU: Intel(R) Xeon(R) CPU E5-2676 v4 @ 2.40GHz (16 cores, 16 threads, AVX2, no AVX-512); installed RAM: 62.7 GiB; storage: SSD (the model on a SATA SSD)
- OS: Linux 6.12.111+deb13-amd64 (Debian 13); GPU driver: kernel amdgpu (in-tree)
- Strata: project 0.1.45, commit 477ac13c; engine built from source with `-DSTRATA_HIP_GFX906=ON` (gfx906)
- Background workloads: the KDE desktop, Firefox, the DeepSeek Harness and deepseek v4.1 flash were running during the suite; nothing else ran.

## Model and configuration

- Model and quantization: qwen3.6-35b-a3b-ud-q4_k_xl (Unsloth's MTP GGUF, sha256 `55983c5a75a1ab969824077b3bb3de4146e82a9234072b48ad4e8f92ad3fe9f1`, 22,853,663,008 bytes)
- Context: 32768 tokens; parallel slots: 2 (docs/QWEN36.md's rule for a 16 GB card at 32K)
- Experts: hybrid — 62 of 256 per layer in VRAM (4.54 GiB), all 18.75 GiB of experts pinned in host RAM, the rest of a batch of up to 128 rows on 7 CPU threads
- **`--pool-workers 6` is the important setting.** Changing only the worker count, at the same 32K context, the same slots and the same 62 of 256 experts in VRAM: **13.5 tok/s chat decode at 15 workers** against **34.0 at 6**; at one slot, 15.4 against 35.2 (the four rows are in the table at the end). 15 is the engine's own default on this 16-core CPU (one worker per physical core except the first; setup writes no `--pool-workers` on a CPU whose cores are all alike) — leaving the flag out entirely, the engine's log prints the same 16 CPU threads and the suite measures 17.3 tok/s at one slot. At 15 workers the GPU was busy 33% of the time and the engine's own log shows the pool's threads then spin at 100% on every core. On a warm 128-token request 4, 6 and 8 workers all land at ~64-67 tok/s; 15 lands at ~26.
- **MTP needs an explicitly greedy request.** With `--mtp 4 --batch 2 --batch-mtp` the engine still reports `drafts accepted 0 of 0` for a request that sends only `temperature: 0` (this suite's requests, and setup's own config, do exactly that). Adding `presence_penalty: 0, frequency_penalty: 0, repeat_penalty: 1` makes the drafts appear (`drafts accepted 43 of 84`) and the same 128-token request decodes at **66-84 tok/s** instead of 34. So the numbers below are the engine *without* MTP drafts, at the fixed thread count.
- Settings: the install's own configuration, unchanged (hip backend); thinking off for the suite; temperature 0

```text
strata --native Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf --max-context 32768 --prefill auto --mtp 4 --pool-workers 6 --batch 2 --batch-mtp
```

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 34.0 (26.3-39.7) | 0.47 (0.44-0.53) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1150.3 (1081.1-1156.2) | not measured | 3.51 (3.49-3.73) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1040.9 (1040.4-1041.4) | not measured | 15.47 (15.46-15.47) |
| agent session, follow-up turns | 14520 (14517-14521) at the last turn | see below | up to 64 | 3 | see below | 34.5 (30.7-38.3) | 8.72 (3.13-15.52) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 0.0 % (0.0-0.0); earlier-turn tokens read again per session: 30073 (30055-30110) (earlier turns that the cache did not hold; the 13697 (13697-13697) new tokens of the files are read once); first turn (nothing cached) first token: 1.56 s (1.56-1.58).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 387.5 | 1.581 | 26.3 |
| 1 | 3753 | 0 | 3753 | 1162.5 | 3.25 | 34.0 |
| 2 | 5269 | 0 | 5269 | 1075.7 | 4.915 | 35.9 |
| 3 | 8684 | 0 | 8684 | 998.3 | 8.722 | 34.5 |
| 4 | 11580 | 0 | 11580 | 1077.4 | 10.774 | 34.2 |
| 5 | 14521 | 0 | 14521 | 949.5 | 15.32 | 30.7 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt`.

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded.

Not tested: pictures (no `mmproj` was installed), `--clients`, `--long` (the context is 32K), a second build for comparison, and any long soak. The suite's own requests do not send the zeroed penalties that MTP needs (see above), so the MTP-active speed is a separate manual measurement here, not a suite number.

The same card, the same model, the same suite, 32K context and the whole card (no `--expert-vram` cap), varying
one setting at a time. Every row below ran on the same day on the same build:

| `--pool-workers` | 1 slot (81/256 experts in VRAM) | 2 slots, `--batch 2 --batch-mtp` (62/256) |
| ---: | ---: | ---: |
| not passed (the engine's own choice) | 17.3 chat / 21.7 agent | not run |
| 15 | 15.4 chat / 19.0 agent | 13.5 chat / 17.2 agent |
| 6 | 35.2 chat / 36.5 agent | 34.0 chat / 34.5 agent |

Leaving `--pool-workers` out entirely gives the same result as 15: the engine's log then prints the same 16 CPU
threads (15 workers plus the host), and the suite measures 17.3 chat decode against the explicit 15's 15.4.
`--pool-workers` is the setting that matters here: 15 to 6 is 2.3-2.6x, while the slot count and the experts the
extra slot costs (81 against 62 of 256 in VRAM) move the result by 1-2 tok/s at most. The results above are the 2-slot, 6-worker row of this table.

Other settings I ran on the same card and model, for the record (measured the same day on the same build):

| Settings | chat decode | agent decode |
| --- | ---: | ---: |
| `--pool-workers 15`, 1 slot, experts capped to leave 3.4 GiB for the desktop | 15.3 | 13.7 |
| `--pool-workers 15`, `--batch 4 --batch-mtp` (22/256 experts in VRAM) | 11.0 | 12.3 |
