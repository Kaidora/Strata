# Community benchmark: AMD Radeon VII 16 GB (gfx906), Qwen3.8-Flash-Next Q2_0, `--pool-workers` sweep

Measured on 2026-10-21 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are
the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness
check was run.

## Hardware and software

- GPU and VRAM: AMD Radeon VII 16 GB (gfx906, Vega 20), one card; the card also drives the desktop
- PCIe: 3.0 x16 (current and max: 8.0 GT/s, width 16)
- Power limit: 250 W (`power1_cap` = `power1_cap_max`)
- ROCm: container image `mixa3607/rocm-gfx906:10.0-complete` (hipcc HIP 7.15.26333, clang 23.0.0git), `HSA_OVERRIDE_GFX_VERSION=9.0.6`
- CPU: Intel(R) Xeon(R) CPU E5-2676 v4 @ 2.40GHz (16 cores, 16 threads, AVX2, no AVX-512); all 16 cores are alike, no efficiency cores; installed RAM: 62.7 GiB; storage: SSD
- OS: Linux 6.12.111+deb13-amd64 (Debian 13); GPU driver: kernel amdgpu (in-tree)
- Strata: release `v0.2.1.4` (tag `v0.2.1.4`, commit 9de04cd4), which carries engine 0.1.46.3; the engine was built
  from source for gfx906 with
  `-DSTRATA_HIP_GFX906=ON -DCMAKE_HIP_ARCHITECTURES=gfx906 -DSTRATA_GGML_DIR=<llama.cpp at 3cf03257>` and no local
  patches
- Background workloads: the KDE desktop, Firefox and the DeepSeek Harness were running during the suite; nothing else ran

## Model and configuration

- Model: Qwen3.8-Flash-Next Q2_0 (GSQ-RCO, two GGUF parts) with MTP; `--expert-cache auto`, `--spec 4 --spec-min-p 0.5`, `--vram-reserve-mib 1024`
- Context: 65536 and 262144 tokens, each measured on its own; `--kv int8 --kv-resident 32768`
  - at 65536 the KV stays in VRAM;
  - at 262144 the engine streams it (`KV streaming: 32768 of 262144 cells per QSA layer in VRAM, the K/V in 3.09 GiB of pinned RAM`)
- Experts: whole card, no `--expert-vram` cap. This engine keeps the experts of the whole model in a VRAM cache
  (`--expert-cache auto`): 8052 slots, 10.37 GiB of VRAM at 65536; 7417 slots at 262144 (the streamed KV takes
  the rest).
- **The sweep: `--pool-workers` left out against 10 and 6,** everything else the same within a context. Left out,
  the engine picks 15 workers on this 16-core CPU and says so at start-up (`CPU pool: 15 workers by default (16
  physical cores, 16 logical processors; bounded by every physical core but the host's)`, `participating threads:
  16`); with the flag it prints the same lines with the smaller count (11 and 7 threads).

```text
strata --serve --pack <q2_0 pack> --native <part 1 of 2.gguf> --ple-gguf <part 2 of 2.gguf>
       --expert-profile <expert-profile.bin> --expert-cache auto --prefill auto
       --spec 4 --spec-min-p 0.5 --mtp <mtp dir>
       --max-context <65536|262144> --kv int8 --kv-resident 32768 --vram-reserve-mib 1024
       [--pool-workers <6|10>]
```

## Method

`python tools/strata_bench.py` — the quick suite of `docs/COMMUNITY_BENCHMARKS.md`, which is what `bench.sh` runs — against the
HTTP API of a running Strata server; model loading is not timed. This is **not** the draft standard protocol
(`strata bench run --suite standard-v1`, `docs/BENCHMARK_PROTOCOL.md`): that one fixes its own prompts, thinking and sampling
for results meant to be compared across machines, while this sweep needs one setting changed at a time with everything else
held fixed. The protocol is still a draft, and these numbers are not standard-protocol results. One
warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache
cannot help the fresh-prompt tests. Within a context only `--pool-workers` differs between arms: the model and pack,
the spec and MTP settings, the KV settings and the machine are the same.

## Results

| context | `--pool-workers` | chat decode | chat first token | 4K prompt | 16K prompt | agent decode | agent cache |
| --- | --- | --- | --- | ---: | ---: | --- | ---: |
| 65536 | not passed (= 15) | 38.9 (32.1-42.6, n=9) | 0.79 s (0.67-0.98) | 494.4 | 519.0 | 45.1 (21.3-53.0, n=15) | 68.37 % |
| 65536 | 10 | 45.3 (40.4-53.7, n=9) | 0.66 s (0.55-0.74) | 494.8 | 506.2 | 47.1 (9.5-60.5, n=15) | 68.52 % |
| 65536 | 6 | 43.2 (37.1-52.5, n=9) | 0.69 s (0.57-0.80) | 489.4 | 504.9 | 48.9 (18.2-58.9, n=15) | 68.57 % |
| 262144 | not passed (= 15) | 37.3 (31.6-41.6, n=9) | 0.86 s (0.66-0.94) | 489.8 | 497.4 | 38.3 (21.8-51.6, n=15) | 68.53 % |
| 262144 | 10 | 45.8 (40.3-53.5, n=9) | 0.68 s (0.57-0.76) | 489.1 | 504.5 | 49.4 (16.8-57.3, n=15) | 68.57 % |
| 262144 | 6 | 44.7 (37.5-52.8, n=9) | 0.71 s (0.58-0.81) | 491.3 | 504.6 | 50.5 (27.9-61.1, n=15) | 68.58 % |

Ranges are the lowest and highest of the runs; `n` counts requests that returned a token. Per-request data:
`65536-default/results.json`, `65536-pool10/results.json`, `65536-pool6/results.json`,
`262144-default/results.json`, `262144-pool10/results.json`, `262144-pool6/results.json`, and the repeat runs
`65536-default-r2/results.json`, `65536-default-r3/results.json`, `262144-default-r2/results.json` and
`262144-default-r3/results.json`.

### Repeats, and whether the gap is noise

The default arm (the engine's own 15 workers) was run three times at each context, so the effect can be compared with
the run-to-run spread of one unchanging configuration:

| context | default, run 1 | run 2 | run 3 | 10 | 6 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 65536 | 38.9 | 40.0 | 39.0 | 45.3 | 43.2 |
| 262144 | 37.3 | 38.8 | 35.0 | 45.8 | 44.7 |

The three default runs span 2.8 % at 65536 and 10.2 % at 262144. Against the pooled default (27 requests per
context) the smaller pools are +16.2 % (10) and +10.8 % (6) at 65536, +22.8 % and +19.8 % at 262144; permutation
tests over the 9-request medians give p < 0.0001 for 10 in both contexts and p = 0.0016 (65536) / 0.0001 (262144) for
6. At 65536 that is 3.8-5.7x the replicate spread; at 262144 the spread is larger (10.2 %), and the effect is 1.9-2.2x
it — still outside the noise, but the 65536 comparison is the cleaner one.

What the sweep shows: with the engine's own 15 workers, chat decode is 38.9 (65536) and 37.3 (262144); with 10 it is
45.3 and 45.8, with 6 it is 43.2 and 44.7, so the default is 11-19 % behind the two smaller pools and 10 and 6 are
within noise of each other. Prompt processing does not move (489-519 tok/s at 4K and 16K in every arm), so the
difference is in decode only. The engine's log says the pool is PCIe-bound on this path (`the CPU pool takes
136 us per missed expert (10.2 GB/s of expert bytes), the link 13.5 GB/s`). The agent numbers carry more noise (a
9.5 outlier in the 65536/10 arm), so the chat medians are the cleaner signal.

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little
between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power
limits are not recorded.

Not tested: pictures (no `mmproj` was installed), `--clients`, `--long` (the contexts are 64K and 256K), a second
build for comparison, `--pool-workers` between 7 and 9, and any long soak. The two contexts are not comparable with
each other (65536 keeps the KV in VRAM, 262144 streams it); only the arms within one context are. MTP is configured
but the suite's requests send only `temperature: 0`, so these are what the engine does with those requests, not the
MTP-active speed.
