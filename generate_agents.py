"""
generate_agents.py
==================
Phase 2: SGLang 기반 소비자 에이전트 ~15,000명 대량 생성

이전 버전(vLLM Qwen3-32B-AWQ 고정)을 다음과 같이 개선:
  * vLLM → **SGLang** OpenAI 호환 엔드포인트
  * 3-way 모델 선택: qwen32b / qwen9b / exaone  (sglang_client.MODELS)
  * 프롬프트를 **5-layer 구조**로 재배치 (prompt_layers.build_layers)
    → 공유 prefix를 앞으로 모아 SGLang RadixAttention 캐시 적중률 극대화

사전 조건:
  1. SGLang 서버 기동 (별도 venv 권장):
       bash scripts/serve_qwen32b.sh   # 또는 serve_qwen9b.sh / serve_exaone.sh
  2. pip install -r requirements.txt   # openai, tqdm

사용법:
  python generate_agents.py                        # 기본 모델(qwen32b)
  python generate_agents.py --model qwen9b         # 개발용 9B
  python generate_agents.py --model exaone         # 대회용 EXAONE
  python generate_agents.py --resume               # 중단 지점부터 재개
  python generate_agents.py --limit 20 --dry-run   # 프롬프트만 확인
"""

import argparse
import asyncio
import json
import random
import re
import time
from pathlib import Path
from typing import Any

from prompt_layers import SYSTEM_PROMPT, build_layers
from sglang_client import (
    DEFAULT_BASE_URL,
    MODELS,
    generate_chat,
    get_spec,
    make_client,
    resolve_mode,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
STATS_DIR = Path(__file__).parent / "output" / "stats"
OUTPUT_DIR = Path(__file__).parent / "output" / "agents"

MAX_RETRIES = 3
TEMPERATURE = 0.85
MAX_TOKENS = 2000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="SGLang 기반 에이전트 대량 생성")
    p.add_argument(
        "--model",
        choices=sorted(MODELS.keys()),
        default=None,
        help=f"모델 모드 (default: env LLM_MODE > qwen32b). 선택: {sorted(MODELS)}",
    )
    p.add_argument(
        "--base-url",
        default=None,
        help=f"SGLang OpenAI 호환 엔드포인트 (default: env SGLANG_BASE_URL > {DEFAULT_BASE_URL})",
    )
    p.add_argument("--stats-dir", type=Path, default=STATS_DIR)
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--max-concurrent", type=int, default=8,
                   help="동시 LLM 요청 수 (default: 8)")
    p.add_argument("--resume", action="store_true",
                   help="이전 중단 지점부터 재개")
    p.add_argument("--limit", type=int, default=0,
                   help="처리할 그룹 수 제한 (0=전체)")
    p.add_argument("--dry-run", action="store_true",
                   help="실제 LLM 호출 없이 프롬프트 레이어만 출력")
    return p.parse_args()


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def extract_json_from_text(text: str) -> list[dict]:
    """LLM 응답에서 JSON 배열 또는 객체를 추출 (think 태그, 코드펜스 처리)."""
    text = (text or "").strip()

    # <think>...</think> 태그 제거 (Qwen thinking 모드가 새어나올 경우)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # ```json ... ``` 블록 추출
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            if "agents" in parsed:
                return parsed["agents"]
            return [parsed]
    except json.JSONDecodeError:
        pass

    # 여러 JSON 객체가 연속된 경우
    objects = []
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    objects.append(json.loads(text[start:i + 1]))
                except json.JSONDecodeError:
                    pass
                start = None
    return objects


# ---------------------------------------------------------------------------
# Group ordering (for prefix-cache locality)
# ---------------------------------------------------------------------------
def order_keys_for_cache(keys: list[str]) -> list[str]:
    """`{adm8}_{gender}_{age}` 키들을 (dong, cohort) 순으로 정렬.

    같은 동의 모든 코호트가 연속으로 처리되어 L1+L2+L3 prefix가 반복 적중하고,
    그 안에서 같은 코호트가 인접해 L4까지 적중한다. 첫 호출만 cold miss.
    """
    def parts(k: str) -> tuple[str, str, str]:
        adm8, gender, age = k.rsplit("_", 2)
        return adm8, gender, age
    return sorted(keys, key=parts)


# ---------------------------------------------------------------------------
# LLM call per group
# ---------------------------------------------------------------------------
async def generate_group(
    client,
    mode: str,
    group_key: str,
    count: int,
    profiles: dict,
    dong_ctx: dict,
    workplace_flow: dict,
    global_dist: dict,
    agg_stats: dict,
    consump_detail: dict,
    sem: asyncio.Semaphore,
) -> tuple[str, list[dict]]:
    """한 그룹에 대해 layered 프롬프트로 LLM 호출 → 에이전트 리스트 반환."""
    async with sem:
        parts = group_key.rsplit("_", 2)
        if len(parts) != 3:
            return group_key, []
        adm8, gender, age = parts

        profile = profiles.get(group_key, {})
        d_ctx = dong_ctx.get(adm8)
        wf = workplace_flow.get(adm8)
        demo_key = f"{gender}_{age}"
        agg = agg_stats.get(demo_key)
        cd = consump_detail.get(group_key)

        sys_prompt, user_prompt, _ = build_layers(
            group_key=group_key,
            count=count,
            gender=gender,
            age=age,
            profile=profile,
            dong_ctx=d_ctx,
            wf=wf,
            global_dist=global_dist,
            agg_stats=agg,
            consump_detail=cd,
        )

        for attempt in range(MAX_RETRIES):
            try:
                raw = await generate_chat(
                    client,
                    mode,
                    sys_prompt,
                    user_prompt,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                )
                agents = extract_json_from_text(raw)

                if not agents:
                    if attempt < MAX_RETRIES - 1:
                        continue
                    return group_key, []

                # agent_id 및 키 필드 보정
                for i, agent in enumerate(agents):
                    expected_id = f"AGT_{adm8}_{gender}_{age}_{i + 1:03d}"
                    agent["agent_id"] = expected_id
                    agent.setdefault("residence", {})["dong_code"] = adm8
                    personal = agent.setdefault("personal", {})
                    personal["gender"] = gender
                    personal["age_group"] = age

                return group_key, agents[:count]

            except Exception:
                wait = 2 ** attempt + random.random()
                await asyncio.sleep(wait)

        return group_key, []


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
async def run(args):
    mode = resolve_mode(args.model)
    spec = get_spec(mode)
    print(f"Model: {mode}  ({spec.hf_id})")
    print(f"  {spec.description}")

    # --- Load stats ---
    print("Loading stats...")
    profiles = load_json(args.stats_dir / "agent_profiles.json")
    allocation = load_json(args.stats_dir / "agent_allocation.json")
    dong_ctx = load_json(args.stats_dir / "dong_context.json")
    workplace_flow = load_json(args.stats_dir / "workplace_flow.json")
    global_dist = load_json(args.stats_dir / "global_distributions.json")
    agg_stats = load_json(args.stats_dir / "aggregate_stats.json")

    cd_path = args.stats_dir / "consumption_detail.json"
    consump_detail = load_json(cd_path) if cd_path.exists() else {}

    total_agents = sum(allocation.values())
    total_groups = len([v for v in allocation.values() if v > 0])
    print(f"  Total: {total_agents:,} agents, {total_groups:,} groups")

    # --- Key list (prefix-cache 친화 순서) ---
    keys = order_keys_for_cache([k for k, v in allocation.items() if v > 0])
    if args.limit > 0:
        keys = keys[:args.limit]
        print(f"  Limited to {args.limit} groups")

    # --- Resume ---
    done_keys: set = set()
    existing_agents: list = []
    partial_dir = args.output_dir / "partial"

    if args.resume and partial_dir.exists():
        for pf in sorted(partial_dir.glob("batch_*.json")):
            try:
                batch_data = load_json(pf)
                existing_agents.extend(batch_data.get("agents", []))
                done_keys.update(batch_data.get("completed_keys", []))
            except Exception:
                pass
        print(f"  Resume: {len(existing_agents)} agents loaded, {len(done_keys)} groups done")

    remaining_keys = [k for k in keys if k not in done_keys]
    print(f"  Remaining: {len(remaining_keys)} groups")

    # --- Dry-run: 레이어 출력 후 종료 ---
    if args.dry_run:
        print("\n=== Dry-run: 첫 그룹의 5-layer 프롬프트 ===")
        if remaining_keys:
            k = remaining_keys[0]
            adm8, gender, age = k.rsplit("_", 2)
            _, user_prompt, debug = build_layers(
                group_key=k,
                count=allocation[k],
                gender=gender,
                age=age,
                profile=profiles.get(k, {}),
                dong_ctx=dong_ctx.get(adm8),
                wf=workplace_flow.get(adm8),
                global_dist=global_dist,
                agg_stats=agg_stats.get(f"{gender}_{age}"),
                consump_detail=consump_detail.get(k),
            )
            for name, text in debug:
                print("-" * 60)
                print(f"[{name}]  ({len(text)} chars)")
                preview = text if name != "L1_system" else text[:400] + "..."
                print(preview)
            print("-" * 60)
            print(f"[final user_prompt]  {len(user_prompt)} chars")
        return

    # --- LLM client ---
    client = make_client(args.base_url)
    sem = asyncio.Semaphore(args.max_concurrent)

    print(f"\nStarting generation (concurrent={args.max_concurrent})")
    start_time = time.time()

    all_agents = list(existing_agents)
    batch_num = len(done_keys)

    chunk_size = args.max_concurrent * 2
    for chunk_start in range(0, len(remaining_keys), chunk_size):
        chunk_keys = remaining_keys[chunk_start:chunk_start + chunk_size]

        tasks = [
            generate_group(
                client, mode, k, allocation[k],
                profiles, dong_ctx, workplace_flow,
                global_dist, agg_stats, consump_detail, sem,
            )
            for k in chunk_keys
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        chunk_agents = []
        chunk_done = []
        for result in results:
            if isinstance(result, Exception):
                continue
            gk, agents = result
            if agents:
                chunk_agents.extend(agents)
            chunk_done.append(gk)

        all_agents.extend(chunk_agents)
        done_keys.update(chunk_done)

        if chunk_agents:
            batch_num += 1
            partial_dir.mkdir(parents=True, exist_ok=True)
            save_json(
                {"agents": chunk_agents, "completed_keys": chunk_done},
                partial_dir / f"batch_{batch_num:04d}.json",
            )

        elapsed = time.time() - start_time
        done_count = chunk_start + len(chunk_keys)
        pct = len(all_agents) / max(total_agents, 1) * 100
        eta = (elapsed / max(done_count, 1)) * (len(remaining_keys) - done_count)
        print(f"  [{done_count}/{len(remaining_keys)}] "
              f"{len(all_agents):,}/{total_agents:,} agents ({pct:.1f}%) | "
              f"{elapsed:.0f}s elapsed | ETA {eta:.0f}s")

    # --- Final save ---
    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_path = args.output_dir / "agents_final.json"
    save_json(all_agents, final_path)

    elapsed = time.time() - start_time
    print(f"\nDone: {len(all_agents):,} agents in {elapsed:.1f}s")
    print(f"Output: {final_path}")

    if len(all_agents) != total_agents:
        print(f"Warning: target {total_agents:,} != actual {len(all_agents):,}")
        print("  -> python generate_agents.py --resume  로 재시도 가능")


def main():
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
