#!/usr/bin/env python3
"""
Fixed-step dots.tts baseline.

Synthesizes every prompt in the benchmark JSON, saves float32 WAVs named by
prompt ID, and writes config.json, manifest.json, requirements.lock.txt.

Usage (GPU):
    python run_baseline.py \
        --model dots-studio/dots.tts-soar \
        --num-steps 10 \
        --out-dir baselines/soar_steps10_seed42

Usage (CPU / Mac, slower):
    python run_baseline.py \
        --model dots-studio/dots.tts-soar \
        --num-steps 10 \
        --precision float32 \
        --out-dir baselines/soar_steps10_seed42
"""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
import importlib.metadata as meta
from pathlib import Path

import soundfile as sf
import torch
from dots_tts.runtime import DotsTtsRuntime

PROMPTS_DEFAULT = Path(__file__).parent / "benchmark_prompts_v1.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seed_all(s: int) -> None:
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="dots-studio/dots.tts-soar")
    p.add_argument("--revision", default=None,
                   help="Pin a HF commit hash for reproducibility. None = latest.")
    p.add_argument("--prompts", default=str(PROMPTS_DEFAULT),
                   help="Path to benchmark_prompts_v1.json")
    p.add_argument("--num-steps", type=int, default=10)
    p.add_argument("--guidance-scale", type=float, default=1.2)
    p.add_argument("--speaker-scale", type=float, default=1.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--precision", default="bfloat16",
                   help="float32 required on CPU; bfloat16 raises on CPU.")
    p.add_argument("--optimize", action="store_true",
                   help="torch.compile warmup (~3 min). CUDA only.")
    p.add_argument("--out-dir", required=True,
                   help="e.g. baselines/soar_steps10_seed42")
    a = p.parse_args()

    if not torch.cuda.is_available() and a.precision == "bfloat16":
        print("No CUDA - switching precision to float32.")
        a.precision = "float32"

    out = Path(a.out_dir)
    wav_dir = out / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    prompts = json.loads(Path(a.prompts).read_text(encoding="utf-8"))
    print(f"Loaded {len(prompts)} prompts from {a.prompts}")

    rt = DotsTtsRuntime.from_pretrained(
        a.model,
        revision=a.revision,
        precision=a.precision,
        optimize=a.optimize,
    )

    # VERIFY: resolve_sampling_options assumed to return (ode_method, num_steps, guidance_scale).
    # Fixed-step MF artifacts ignore num_steps/guidance_scale overrides.
    ode, steps, cfg = rt.resolve_sampling_options(
        ode_method=None,
        num_steps=a.num_steps,
        guidance_scale=a.guidance_scale,
    )
    nfe = steps
    print(f"Resolved: ode={ode}  num_steps={steps}  guidance_scale={cfg}  NFE={nfe}")

    gen_kwargs = dict(
        prompt_audio_path=None,
        prompt_text=None,
        ode_method=ode,
        num_steps=steps,
        guidance_scale=cfg,
        speaker_scale=a.speaker_scale,
    )

    print("Warming up (discarded)...")
    seed_all(a.seed)
    rt.generate(text=prompts[0]["text"], **gen_kwargs)

    rows = []
    sample_rate = None
    for entry in prompts:
        seed_all(a.seed)
        r = rt.generate(text=entry["text"], **gen_kwargs)

        sample_rate = r["sample_rate"]
        wav_path = wav_dir / f"{entry['id']}.wav"
        sf.write(wav_path, r["audio"].float().cpu().squeeze().numpy(),
                 sample_rate, subtype="FLOAT")

        rows.append(dict(
            id=entry["id"],
            text=entry["text"],
            length_bucket=entry["length_bucket"],
            category=entry["category"],
            wav=str(wav_path),
            wav_sha256=sha256(wav_path),
            audio_s=round(r["audio"].shape[-1] / sample_rate, 4),
            time_s=round(r["time_used"], 4),
            rtf=round(r["rtf"], 6),
        ))
        print(f"[{entry['id']}] {entry['length_bucket']:6}  {entry['category']:25}  RTF={r['rtf']:.3f}")

    # VERIFY: pretrained_path, max_generate_length, vocoder_merge_steps
    # are assumed attributes on DotsTtsRuntime - replace UNVERIFIED if they differ.
    config = dict(
        model=a.model,
        revision_requested=a.revision,
        snapshot_path=str(getattr(rt, "pretrained_path", "UNVERIFIED")),
        ode_method=ode,
        num_steps=steps,
        guidance_scale=cfg,
        nfe_estimate=nfe,
        speaker_scale=a.speaker_scale,
        seed=a.seed,
        precision=a.precision,
        optimize=a.optimize,
        max_generate_length=getattr(rt, "max_generate_length", "UNVERIFIED"),
        vocoder_merge_steps=getattr(rt, "vocoder_merge_steps", "UNVERIFIED"),
        sample_rate=sample_rate,
        prompt_audio=None,
        prompt_text=None,
        n_prompts=len(prompts),
        prompts_file=str(Path(a.prompts).resolve()),
        prompts_sha256=sha256(Path(a.prompts)),
        prompts_version=prompts[0].get("version", "unknown"),
        dots_tts_version=meta.version("dots.tts"),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        python=sys.version,
        platform=platform.platform(),
        argv=sys.argv,
    )

    (out / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False)
    )
    (out / "manifest.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False)
    )
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True, text=True,
    ).stdout
    (out / "requirements.lock.txt").write_text(freeze)

    print(f"\nSaved {len(rows)} WAVs to {wav_dir}/")
    print(f"Config:    {out / 'config.json'}")
    print(f"Manifest:  {out / 'manifest.json'}")
    print(f"Lock file: {out / 'requirements.lock.txt'}")
    print("\nReproducibility check: run twice and compare WAV SHA256s in manifest.json.")


if __name__ == "__main__":
    main()
