"""
Build the inspector page for a set of runs (no model calls).

    python -m reports.pilot_page experiments/reports/sushi_roll_pilot15.json \\
        --out static/reports/sushi_roll_pilot15.html

The manifest names the runs (one experiment per condition) and the versions
to show (which run folder each condition used in that version; a missing
entry means the newest finished run under the experiment's current settings):

    {
      "title": "Sushi Roll Pilot Facets",
      "eyebrow": "sushi_roll · 15-video pilot",
      "intro": ["Paragraph with **bold** text.", "..."],
      "budget": 10, "seed": 0,
      "taus": [0.2, 0.4, 0.55, 0.6],
      "measure_settings": {"refinement_threshold": 0.55, "max_schema_facets": 2},
      "compare": {"from": "v1", "to": "v2", "title": "Original vs revised prompts"},
      "runs": [{"key": "a0", "kind": "pipeline", "experiment": "sushi_roll_step_pilot15_a0",
                "label": "Batch, one request", "short": "a0"}, ...],
      "versions": [{"id": "v1", "label": "Original prompts",
                    "results": {"a0": "2026-10-11_140210__sushi_roll_step_pilot15_a0", ...}}, ...]
    }

Each run gets: its factors (values, counts, example segments, coverage), the
ledger, the full transcript (``reports.transcripts``), the tree at each tau,
its design space in the feedback format, and every measure from
``analysis.measures``.
"""

from __future__ import annotations

import argparse
import dataclasses
import html
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from analysis.measures import is_applicable, run_measures
from contextual.evaluation import PoolEvaluator
from contextual.models import refinement_edges
from contextual.reporting import summarize, tau_sweep, total_usage
from facets.prompts.formatters import schemas_to_message_content
from optimization import count_unique_units

from .runs import LoadedRun, load_run
from .transcripts import transcript

TEMPLATE = Path(__file__).parent / "template.html"
EXAMPLES_PER_VALUE = 4


def _facets(run: LoadedRun, titles: Dict[str, str]) -> List[Dict[str, Any]]:
    result = run.result
    items = {i.id: i for i in run.corpus}
    kept = {s.facet_ids[0] for s in result.accepted_schemas if len(s.facet_ids) == 1}
    redundant = {
        s.facet_ids[0]: {"by": [titles.get(b, b) for b in s.rejected_by_facet_ids],
                         "score": round(s.refinement_score, 3)}
        for s in result.rejected_schemas if len(s.facet_ids) == 1
    }
    out = []
    for f in result.pool:
        segments = defaultdict(list)
        for (item_id, fid), label in result.annotations.items():
            if fid == f.id and item_id in items:
                segments[label].append(items[item_id].text)
        counts = Counter({label: len(texts) for label, texts in segments.items()})
        total = sum(counts.values())
        values = [{"label": v.label, "definition": v.definition, "count": counts.get(v.label, 0),
                   "examples": segments.get(v.label, [])[:EXAMPLES_PER_VALUE]}
                  for v in f.vocabulary]
        known = {v.label for v in f.vocabulary}
        values += [{"label": label, "definition": "(label not in vocabulary)", "count": c,
                    "examples": segments[label][:EXAMPLES_PER_VALUE]}
                   for label, c in counts.items() if label not in known]
        applicable = sum(v["count"] for v in values if is_applicable(v["label"]))
        out.append({
            "id": f.id, "title": f.title, "plural": f.title_plural,
            "question": f.user_question, "scope": f.scope, "kind": f.kind or "",
            "definition": f.definition, "justification": f.justification,
            "guidelines": f.guidelines, "values": values, "labeled": total,
            "tree": "kept" if f.id in kept else ("redundant" if f.id in redundant else "n/a"),
            "coverage": round(applicable / total, 3) if total else 0,
            "redundant_with": redundant.get(f.id),
        })
    return out


def _ledger(run: LoadedRun, titles: Dict[str, str]) -> List[Dict[str, Any]]:
    pool_ids = {f.id for f in run.result.pool}
    return [{
        "attempt": r.attempt_index, "status": r.status.value, "title": r.title,
        "scope": r.scope, "definition": r.definition, "note": r.note,
        "duplicate_of": titles.get(r.duplicate_of) if r.duplicate_of else None,
        "replaced": titles.get(r.replaced_facet_id) if r.replaced_facet_id else None,
        "similarity": round(r.nearest_similarity, 3) if r.nearest_similarity is not None else None,
        "nearest": titles.get(r.nearest_id) if r.nearest_id else None,
        "tree_nodes": r.tree_quality.tree_nodes if r.tree_quality else None,
        "in_pool": r.facet_id in pool_ids,
    } for r in run.result.ledger]


def _design_space(run: LoadedRun) -> Dict[str, Any]:
    """The run's DAG re-evaluated at its own settings, rendered like the feedback."""
    result, settings = run.result, run.settings
    evaluation = PoolEvaluator(config=settings, focused_kb=run.focused).evaluate_pool(
        result.pool, result.annotations)
    edges = refinement_edges(type("S", (), {"tree": evaluation.tree})())
    text = "\n".join(b["text"] for b in schemas_to_message_content(
        evaluation.accepted_schemas, evaluation.rejected_schemas, result.pool,
        result.annotations, run.focused, edges,
        refinement_threshold=settings.refinement_threshold).content)
    levels = Counter(len(n.facet_ids) for n in evaluation.tree.nodes)
    return {
        "dag": text[text.index("Design space: kept"):],
        "kept_singles": levels.get(1, 0), "kept_pairs": levels.get(2, 0),
        "edges": len(evaluation.tree.edges), "pruned": len(evaluation.rejected_schemas),
    }


def _sees_history(run: LoadedRun) -> bool:
    cfg = run.result.config
    if run.kind == "pipeline":
        return "history" in (cfg.get("context") or [])
    # older runs pushed a full history section; since 2026-10-11 pool and
    # schemas feedback push history changes ("Not added to the pool")
    markers = ("Tried factors that were not kept", "Added to the pool:", "Not added to the pool")
    return any(m["role"] == "tool" and any(k in m.get("content", "") for k in markers)
               for m in run.result.runner_state.get("transcript", []))


def collect_run(spec: Dict[str, Any], run: LoadedRun, taus: List[float],
                measure_settings: Dict[str, Any]) -> Dict[str, Any]:
    result = run.result
    titles = {r.facet_id: r.title for r in result.ledger if r.facet_id}
    measures = run_measures(
        run.focused, result.pool, result.annotations, result.ledger,
        dataclasses.replace(run.settings, **measure_settings), embeddings=result.embeddings)
    by_id = {f.id: f.title for f in result.pool}
    measures["unit_coverage"]["per_factor"] = {
        by_id[fid]: round(c, 3) for fid, c in measures["unit_coverage"]["per_factor"].items()}
    facets = _facets(run, titles)
    coverage = [f["coverage"] for f in facets]
    s = summarize(result)
    cfg = result.config
    return {
        **{k: spec[k] for k in ("key", "kind", "label", "short")},
        "strategy": cfg.get("strategy"), "context": cfg.get("context"),
        "feedback": cfg.get("feedback"), "sees_history": _sees_history(run),
        "stop": s.stop_reason, "attempts": s.attempts, "proposals": s.proposals,
        "pool": s.pool_facets, "tree_nodes": s.tree_nodes, "statuses": s.status_counts,
        "usage": total_usage(result), "usage_by_stage": result.usage,
        "turns": result.runner_state.get("turns"),
        "tool_calls": result.runner_state.get("tool_calls"),
        "taus": tau_sweep(result, run.knowledge_base, run.settings, taus),
        "rescored": result.runner_state.get("rescored") or result.runner_state.get("rescored_dag"),
        "run_tau": run.settings.refinement_threshold,
        "max_schema_facets": run.settings.max_schema_facets,
        **_design_space(run),
        "facets": facets, "ledger": _ledger(run, titles),
        "measures": measures, "run": run.path.name,
        "mean_coverage": round(sum(coverage) / len(coverage), 3) if coverage else 0,
        "low_coverage": sum(c < 0.2 for c in coverage),
        "kinds": dict(Counter(f["kind"] or "unknown" for f in facets)),
    }


def collect(manifest: Dict[str, Any]) -> Dict[str, Any]:
    taus = manifest.get("taus", [0.2, 0.4, 0.6])
    measure_settings = manifest.get("measure_settings", {})
    versions, corpus, models, transcripts = [], None, None, {}
    for version in manifest["versions"]:
        runs = []
        for spec in manifest["runs"]:
            run = load_run(spec["kind"], spec["experiment"],
                           version.get("results", {}).get(spec["key"]))
            if corpus is None:
                kb = run.corpus
                corpus = {"videos": len(kb.by_video()), "segments": len(kb),
                          "units": count_unique_units(kb),
                          "total_videos": len(run.knowledge_base.by_video()),
                          "video_ids": sorted(kb.by_video())}
                models = {"proposal": run.settings.proposal_model.model,
                          "labeling": run.settings.labeling_model.model}
            runs.append(collect_run(spec, run, taus, measure_settings))
            if run.path.name not in transcripts:   # versions can share a run
                transcripts[run.path.name] = transcript(run)
        versions.append({"id": version["id"], "label": version["label"], "runs": runs})
    return {"corpus": corpus, "models": models, "budget": manifest.get("budget"),
            "seed": manifest.get("seed"), "taus": taus, "measure_settings": measure_settings,
            "compare": manifest.get("compare"), "versions": versions,
            "transcripts": transcripts}


def _paragraphs(paragraphs: List[str]) -> str:
    out = []
    for text in paragraphs:
        text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html.escape(text))
        out.append(f'    <p class="lede soft">{text}</p>')
    return "\n".join(out)


def render(manifest: Dict[str, Any], data: Dict[str, Any]) -> str:
    page = TEMPLATE.read_text()
    page = page.replace("__TITLE__", html.escape(manifest.get("title", "Runs")))
    page = page.replace("__EYEBROW__", html.escape(manifest.get("eyebrow", "")))
    page = page.replace("__INTRO__", _paragraphs(manifest.get("intro", [])))
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return page.replace("__DATA__", payload)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("manifest")
    parser.add_argument("--out", help="HTML file (default: static/reports/<manifest>.html)")
    parser.add_argument("--data", help="also write the collected data as JSON here")
    args = parser.parse_args(argv)
    manifest = json.loads(Path(args.manifest).read_text())
    data = collect(manifest)
    out = Path(args.out or f"static/reports/{Path(args.manifest).stem}.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(manifest, data))
    if args.data:
        Path(args.data).write_text(json.dumps(data, ensure_ascii=False))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")
    for version in data["versions"]:
        for r in version["runs"]:
            a = r["measures"]["at_run_tau"]
            print(f"{version['id']} {r['short']:<10} pool {r['pool']:>2}  singles {a['kept_singles']:>2} "
                  f"pairs {a['kept_pairs']}  vendi {a['vendi_nmi']:.2f}  {r['run']}")


if __name__ == "__main__":
    main()
