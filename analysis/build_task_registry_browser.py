#!/usr/bin/env python3
"""Build a static HTML browser for auditing task prompts and ICL examples."""

from __future__ import annotations

import argparse
import csv as csv_module
import html
import inspect
import json
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "analysis" / "task_registry_browser" / "index.html"
GROUP_ORDER = ["base_tasks", "simple_icl_tasks", "compositional_tasks", "textfrct_tasks"]
GROUP_LABELS = {
    "base_tasks": "Base Tasks",
    "simple_icl_tasks": "Simple ICL Variants",
    "compositional_tasks": "Compositional Variants",
    "textfrct_tasks": "TextFRCT Variants",
}

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasks.registry import get_task, get_task_info, list_all_tasks


def serialize_value(value: Any) -> Any:
    """Convert task data to JSON-safe values."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): serialize_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [serialize_value(v) for v in value]
    return repr(value)


def determine_num_shots(task: Any) -> int:
    """Find the effective default number of shots for build_prompt()."""
    signature = inspect.signature(task.build_prompt)
    param = signature.parameters.get("num_shots")
    if param is None or param.default is inspect._empty:
        return 5
    default = param.default
    if isinstance(default, int) and default >= 0:
        return default
    if default == -1 and hasattr(task, "num_shots"):
        return int(getattr(task, "num_shots"))
    return 5


def determine_num_shots_from_data(task: Any, num_shots: int) -> int:
    """Determine default num_shots from task's data, for CSV-based tasks."""
    try:
        config = getattr(task, "config", None)
        if config and getattr(config, "data_format", None) == "csv":
            data_path = getattr(config, "data_path", None)
            if data_path:
                csv_file = ROOT / data_path
                if csv_file.exists():
                    with open(csv_file, 'r', encoding='utf-8') as f:
                        reader = csv_module.DictReader(f)
                        csv_rows = list(reader)
                        if csv_rows:
                            # Use min of (default_num_shots, floor(total_rows / 2))
                            max_shots_from_data = max(1, len(csv_rows) // 2)
                            return min(num_shots, max_shots_from_data)
    except Exception:
        pass
    return num_shots


def safe_get_icl_examples(task: Any, num_shots: int) -> List[Dict[str, Any]]:
    if num_shots <= 0 or not hasattr(task, "get_icl_examples"):
        return []

    if hasattr(task, "reset_icl_tracking"):
        task.reset_icl_tracking()

    try:
        examples = task.get_icl_examples(
            num_examples=num_shots,
            shuffle=True,
            seed=42,
            fresh=False,
        )
    except TypeError:
        examples = task.get_icl_examples(num_examples=num_shots)
    except Exception:
        examples = []

    formatted = []
    for example in examples:
        if isinstance(example, dict):
            formatted.append({
                "input": serialize_value(example.get("input", "")),
                "output": serialize_value(example.get("output", "")),
                "raw": serialize_value(example),
            })
        else:
            formatted.append({"input": "", "output": repr(example), "raw": repr(example)})
    return formatted


def safe_build_prompt(task: Any, instance: Dict[str, Any], num_shots: int) -> str:
    if instance is None:
        return ""
    if hasattr(task, "reset_icl_tracking"):
        task.reset_icl_tracking()
    try:
        prompt = task.build_prompt(instance, num_shots=num_shots)
    except TypeError:
        prompt = task.build_prompt(instance)
    return prompt if isinstance(prompt, str) else repr(prompt)


def collect_task_records() -> List[Dict[str, Any]]:
    all_tasks = list_all_tasks()
    records: List[Dict[str, Any]] = []

    for group_name in GROUP_ORDER:
        for display_name in all_tasks[group_name]:
            spaced = display_name.endswith(" (spaced)")
            task_name = display_name.replace(" (spaced)", "")
            base_name = task_name.split(":", 1)[0]
            
            # Skip base "simple_icl" and "compositional" tasks (keep only variants with ":")
            if base_name in ("simple_icl", "compositional") and ":" not in task_name:
                continue

            task = get_task(task_name, spaced=spaced)
            task_info = get_task_info(base_name)
            rows = task.get_split("test")
            example_instance = rows[0] if rows else None
            default_num_shots = determine_num_shots(task)
            
            # For CSV-based tasks, adjust default shots based on data size
            default_num_shots = determine_num_shots_from_data(task, default_num_shots)
            
            example_prompt = safe_build_prompt(task, example_instance, 0)
            icl_examples = safe_get_icl_examples(task, default_num_shots)
            full_prompt = safe_build_prompt(task, example_instance, default_num_shots)

            config = getattr(task, "config", None)
            record = OrderedDict(
                id=len(records),
                display_name=display_name,
                task_name=task_name,
                base_name=base_name,
                group=group_name,
                group_label=GROUP_LABELS[group_name],
                spaced=spaced,
                class_name=task_info.get("class", task.__class__.__name__),
                module=task_info.get("module", task.__class__.__module__),
                docstring=(task_info.get("docstring", "") or "").strip(),
                dataset_size=len(rows),
                default_num_shots=default_num_shots,
                supports_icl=bool(getattr(task, "supports_icl", False)),
                config={
                    "name": getattr(config, "name", ""),
                    "data_path": getattr(config, "data_path", ""),
                    "data_format": getattr(config, "data_format", ""),
                    "input_column": getattr(config, "input_column", ""),
                    "output_column": getattr(config, "output_column", ""),
                },
                example_instance=serialize_value(example_instance),
                example_prompt=example_prompt,
                icl_examples=icl_examples,
                fully_filled_prompt=full_prompt,
            )
            records.append(record)

    return records


def render_html(records: List[Dict[str, Any]]) -> str:
    data_json = json.dumps(records, ensure_ascii=False)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    title = "Task Registry Browser"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #f5f1e8;
      --panel: #fffaf1;
      --panel-2: #efe6d7;
      --text: #1f1c17;
      --muted: #6c6258;
      --accent: #0f766e;
      --accent-2: #c2410c;
      --border: #d8ccb9;
      --shadow: 0 18px 50px rgba(69, 42, 10, 0.10);
      --mono: "IBM Plex Mono", "SFMono-Regular", Consolas, monospace;
      --sans: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Palatino, serif;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: var(--sans);
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(15,118,110,0.12), transparent 30%),
        radial-gradient(circle at top right, rgba(194,65,12,0.10), transparent 28%),
        linear-gradient(180deg, #f8f3ea 0%, var(--bg) 100%);
    }}
    .error-banner {{
      background: #ffe0e0;
      border: 2px solid #ff6b6b;
      color: #c00;
      padding: 16px;
      margin: 16px;
      border-radius: 12px;
      font-family: var(--mono);
      white-space: pre-wrap;
      word-break: break-word;
      max-height: 400px;
      overflow: auto;
    }}
    .app {{
      display: grid;
      grid-template-columns: 340px minmax(0, 1fr);
      min-height: 100vh;
    }}
    .sidebar {{
      border-right: 1px solid var(--border);
      background: rgba(255,250,241,0.92);
      backdrop-filter: blur(10px);
      padding: 20px 18px 28px;
      position: sticky;
      top: 0;
      height: 100vh;
      overflow: auto;
    }}
    .brand {{ margin-bottom: 18px; }}
    .eyebrow {{
      font-family: var(--mono);
      text-transform: uppercase;
      letter-spacing: 0.14em;
      color: var(--accent);
      font-size: 12px;
    }}
    h1 {{ margin: 8px 0 8px; font-size: 28px; line-height: 1.05; }}
    .meta {{ color: var(--muted); font-size: 14px; line-height: 1.45; }}
    .search {{ margin: 18px 0 14px; }}
    .search input {{
      width: 100%;
      border: 1px solid var(--border);
      background: #fff;
      border-radius: 14px;
      padding: 12px 14px;
      font: inherit;
      color: var(--text);
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.75);
    }}
    .toc-group {{ margin: 18px 0; }}
    .toc-title {{
      font-family: var(--mono);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.10em;
      color: var(--muted);
      margin-bottom: 8px;
    }}
    .toc-list {{ display: grid; gap: 6px; }}
    .toc-item {{
      display: block;
      text-decoration: none;
      color: var(--text);
      border: 1px solid transparent;
      border-radius: 12px;
      padding: 9px 10px;
      background: transparent;
      transition: 120ms ease;
      font-size: 14px;
      line-height: 1.35;
    }}
    .toc-item:hover {{ background: var(--panel-2); border-color: var(--border); }}
    .toc-item.active {{ background: #fff; border-color: var(--accent); box-shadow: var(--shadow); }}
    .toc-count {{ color: var(--muted); font-family: var(--mono); font-size: 12px; }}
    main {{ padding: 28px clamp(18px, 3vw, 40px) 48px; }}
    .hero {{
      display: grid;
      gap: 16px;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      margin-bottom: 22px;
    }}
    .stat {{
      background: rgba(255,250,241,0.85);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 16px 18px;
      box-shadow: var(--shadow);
    }}
    .stat-label {{ color: var(--muted); font-family: var(--mono); font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; }}
    .stat-value {{ font-size: 28px; margin-top: 6px; }}
    .panel {{
      background: rgba(255,250,241,0.92);
      border: 1px solid var(--border);
      border-radius: 24px;
      padding: 22px;
      box-shadow: var(--shadow);
      margin-bottom: 18px;
    }}
    .task-header {{ display: flex; justify-content: space-between; gap: 16px; align-items: flex-start; flex-wrap: wrap; }}
    .task-title {{ margin: 0; font-size: clamp(28px, 4vw, 42px); line-height: 0.98; }}
    .badges {{ display: flex; gap: 8px; flex-wrap: wrap; margin-top: 12px; }}
    .badge {{
      background: var(--panel-2);
      color: var(--text);
      border-radius: 999px;
      padding: 6px 10px;
      font-family: var(--mono);
      font-size: 12px;
      border: 1px solid var(--border);
    }}
    .nav-buttons {{ display: flex; gap: 10px; flex-wrap: wrap; }}
    button {{
      border: 1px solid var(--border);
      background: #fff;
      color: var(--text);
      border-radius: 999px;
      padding: 10px 14px;
      font: inherit;
      cursor: pointer;
    }}
    button:hover {{ border-color: var(--accent); }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px; }}
    .kv {{ background: #fff; border: 1px solid var(--border); border-radius: 16px; padding: 14px; }}
    .kv-label {{ color: var(--muted); font-family: var(--mono); font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px; }}
    .kv-value {{ font-size: 15px; line-height: 1.5; word-break: break-word; }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: var(--mono);
      font-size: 13px;
      line-height: 1.55;
      background: #fff;
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 16px;
      max-height: 520px;
      overflow: auto;
    }}
    .section-title {{ margin: 0 0 12px; font-size: 20px; }}
    .section-subtitle {{ margin: 0 0 14px; color: var(--muted); }}
    .icl-list {{ display: grid; gap: 12px; }}
    .icl-card {{ background: #fff; border: 1px solid var(--border); border-radius: 18px; padding: 16px; }}
    .icl-card h4 {{ margin: 0 0 10px; font-size: 16px; }}
    .icl-field {{ margin-bottom: 12px; }}
    .icl-field:last-child {{ margin-bottom: 0; }}
    .icl-field-label {{ color: var(--muted); font-family: var(--mono); font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px; }}
    .empty {{ color: var(--muted); font-style: italic; }}
    .footer-note {{ color: var(--muted); font-size: 13px; margin-top: 8px; }}
    @media (max-width: 980px) {{
      .app {{ grid-template-columns: 1fr; }}
      .sidebar {{ position: static; height: auto; border-right: 0; border-bottom: 1px solid var(--border); }}
    }}
  </style>
</head>
<body>
  <div id="init-error" class="error-banner" style="display: none;"></div>
  <div class="app">
    <aside class="sidebar">
      <div class="brand">
        <div class="eyebrow">Prompt Audit</div>
        <h1>Task Registry Browser</h1>
        <div class="meta">Generated {html.escape(generated_at)}. Browse task prompts, inspect current ICL examples, and compare fully rendered prompts side by side.</div>
      </div>
      <div class="search">
        <input id="search" type="search" placeholder="Filter tasks by name, group, or module">
      </div>
      <div id="toc"></div>
    </aside>
    <main>
      <section class="hero" id="summary"></section>
      <section class="panel" id="task-root"></section>
    </main>
  </div>
  <script>
    const GROUP_ORDER = ["base_tasks", "simple_icl_tasks", "compositional_tasks", "textfrct_tasks"];
    const GROUP_LABELS = {{"base_tasks": "Base Tasks", "simple_icl_tasks": "Simple ICL Variants", "compositional_tasks": "Compositional Variants", "textfrct_tasks": "TextFRCT Variants"}};
    
    const TASKS = {data_json};
    
    const state = {{
      filtered: TASKS.slice(),
      selectedId: TASKS.length ? TASKS[0].id : null,
      query: "",
    }};

    const tocEl = document.getElementById("toc");
    const summaryEl = document.getElementById("summary");
    const taskRootEl = document.getElementById("task-root");
    const searchEl = document.getElementById("search");

    function escapeHtml(value) {{
      const str = String(value);
      return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#39;");
    }}

    function taskSlug(task) {{
      return `${{task.id}}-${{task.display_name.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "")}}`;
    }}

    function byId(id) {{
      return TASKS.find(task => task.id === id) ?? null;
    }}

    function selectedTask() {{
      return byId(state.selectedId) ?? state.filtered[0] ?? TASKS[0] ?? null;
    }}

    function updateHash() {{
      const task = selectedTask();
      if (task) window.location.hash = "#" + taskSlug(task);
    }}

    function selectTask(id) {{
      state.selectedId = id;
      updateHash();
      renderToc();
      renderTask();
    }}

    function parseHashSelection() {{
      const hash = window.location.hash.replace(/^#/, "");
      if (!hash) return;
      const task = TASKS.find(item => taskSlug(item) === hash);
      if (task) state.selectedId = task.id;
    }}

    function filteredGroups() {{
      const groups = new Map();
      for (const name of GROUP_ORDER) groups.set(name, []);
      for (const task of state.filtered) groups.get(task.group).push(task);
      return groups;
    }}

    function renderSummary() {{
      const groups = filteredGroups();
      const cards = [
        {{ label: "Visible Tasks", value: state.filtered.length }},
        {{ label: "Base Tasks", value: groups.get("base_tasks").length }},
        {{ label: "Subtask Variants", value: state.filtered.length - groups.get("base_tasks").length }},
        {{ label: "Current Selection", value: selectedTask() ? selectedTask().display_name : "None" }},
      ];
      summaryEl.innerHTML = cards.map(card => `
        <article class="stat">
          <div class="stat-label">${{escapeHtml(card.label)}}</div>
          <div class="stat-value">${{escapeHtml(card.value)}}</div>
        </article>
      `).join("");
    }}

    function renderToc() {{
      const groups = filteredGroups();
      tocEl.innerHTML = GROUP_ORDER.map(groupName => {{
        const tasks = groups.get(groupName);
        if (!tasks.length) return "";
        const showGroupLabel = !["simple_icl_tasks", "compositional_tasks"].includes(groupName);
        return `
          <section class="toc-group">
            ${{showGroupLabel ? `<div class="toc-title">${{escapeHtml(GROUP_LABELS[groupName])}} <span class="toc-count">${{tasks.length}}</span></div>` : ""}}
            <div class="toc-list">
              ${{tasks.map(task => `
                <a class="toc-item ${{task.id === state.selectedId ? "active" : ""}}" href="#${{taskSlug(task)}}" data-id="${{task.id}}">
                  <div>${{escapeHtml(task.display_name)}}</div>
                  <div class="toc-count">${{escapeHtml(task.class_name)}} · ${{task.dataset_size}} rows</div>
                </a>
              `).join("")}}
            </div>
          </section>
        `;
      }}).join("");

      tocEl.querySelectorAll(".toc-item").forEach(link => {{
        link.addEventListener("click", event => {{
          event.preventDefault();
          selectTask(Number(link.dataset.id));
        }});
      }});
    }}

    function renderIclExamples(task) {{
      if (!task.icl_examples.length) return `<div class="empty">No explicit ICL examples were returned for this task at its default shot count.</div>`;
      const items = task.icl_examples.map((example, index) => {{
        return `<article class="icl-card"><h4>Example ${{index + 1}}</h4><div class="icl-field"><div class="icl-field-label">Input</div><pre>${{escapeHtml(example.input)}}</pre></div><div class="icl-field"><div class="icl-field-label">Output</div><pre>${{escapeHtml(example.output)}}</pre></div></article>`;
      }});
      return `<div class="icl-list">${{items.join("")}}</div>`;
    }}

    function renderTask() {{
      const task = selectedTask();
      if (!task) {{
        taskRootEl.innerHTML = `<div class="empty">No tasks match the current filter.</div>`;
        renderSummary();
        return;
      }}

      const currentIndex = state.filtered.findIndex(item => item.id === task.id);
      const prevTask = currentIndex > 0 ? state.filtered[currentIndex - 1] : null;
      const nextTask = currentIndex >= 0 && currentIndex < state.filtered.length - 1 ? state.filtered[currentIndex + 1] : null;

      taskRootEl.innerHTML = `
        <div class="task-header">
          <div>
            <div class="eyebrow">${{escapeHtml(task.group_label)}}</div>
            <h2 class="task-title">${{escapeHtml(task.display_name)}}</h2>
            <div class="badges">
              <span class="badge">${{escapeHtml(task.class_name)}}</span>
              <span class="badge">${{task.dataset_size}} test rows</span>
              <span class="badge">${{task.default_num_shots}} default shots</span>
              <span class="badge">${{task.supports_icl ? "ICL enabled" : "ICL disabled"}}</span>
            </div>
          </div>
          <div class="nav-buttons">
            <button id="prev-task" ${{prevTask ? "" : "disabled"}}>Prev Task</button>
            <button id="next-task" ${{nextTask ? "" : "disabled"}}>Next Task</button>
          </div>
        </div>
        <p class="section-subtitle">${{escapeHtml(task.docstring || "No docstring available.")}}</p>

        <div class="grid">
          <div class="kv"><div class="kv-label">Module</div><div class="kv-value">${{escapeHtml(task.module)}}</div></div>
          <div class="kv"><div class="kv-label">Config Name</div><div class="kv-value">${{escapeHtml(task.config.name || "")}}</div></div>
          <div class="kv"><div class="kv-label">Data Path</div><div class="kv-value">${{escapeHtml(task.config.data_path || "")}}</div></div>
          <div class="kv"><div class="kv-label">Columns</div><div class="kv-value">${{escapeHtml(`${{task.config.input_column || "?"}} → ${{task.config.output_column || "?"}}` )}}</div></div>
        </div>

        <div class="panel">
          <h3 class="section-title">Example Prompt</h3>
          <p class="section-subtitle">The query prompt with zero-shot formatting for the first example instance.</p>
          <pre>${{escapeHtml(task.example_prompt || "")}}</pre>
        </div>

        <div class="panel">
          <h3 class="section-title">ICL Examples</h3>
          <p class="section-subtitle">Examples returned by the task's current get_icl_examples() implementation at its default shot count.</p>
          ${{renderIclExamples(task)}}
        </div>

        <div class="panel">
          <h3 class="section-title">Fully Filled Prompt</h3>
          <p class="section-subtitle">The first example instance rendered through build_prompt() using the task's default shot count.</p>
          <pre>${{escapeHtml(task.fully_filled_prompt || "")}}</pre>
        </div>

        <div class="panel">
          <h3 class="section-title">Example Instance</h3>
          <p class="section-subtitle">Raw first-row payload after task loading/filtering.</p>
          <pre>${{escapeHtml(JSON.stringify(task.example_instance, null, 2))}}</pre>
        </div>

        <div class="footer-note">Prev/next navigation only walks the currently filtered list from the table of contents.</div>
      `;

      const prevButton = document.getElementById("prev-task");
      const nextButton = document.getElementById("next-task");
      if (prevButton && prevTask) prevButton.addEventListener("click", () => selectTask(prevTask.id));
      if (nextButton && nextTask) nextButton.addEventListener("click", () => selectTask(nextTask.id));
      renderSummary();
    }}

    function applyFilter(query) {{
      state.query = query.trim().toLowerCase();
      if (!state.query) {{
        state.filtered = TASKS.slice();
      }} else {{
        state.filtered = TASKS.filter(task => {{
          const haystack = [task.display_name, task.group_label, task.module, task.class_name].join(" ").toLowerCase();
          return haystack.includes(state.query);
        }});
      }}
      if (!state.filtered.some(task => task.id === state.selectedId)) {{
        state.selectedId = state.filtered[0]?.id ?? null;
      }}
      renderToc();
      renderTask();
    }}

    searchEl.addEventListener("input", event => applyFilter(event.target.value));
    window.addEventListener("hashchange", () => {{
      parseHashSelection();
      renderToc();
      renderTask();
    }});

    try {{
      const errorBanner = document.getElementById('init-error');
      console.log("Initializing task registry browser...");
      console.log("TASKS array length:", TASKS.length);
      console.log("Initial state:", state);
      
      parseHashSelection();
      console.log("After parseHashSelection, selectedId:", state.selectedId);
      
      renderSummary();
      console.log("renderSummary completed");
      
      renderToc();
      console.log("renderToc completed, tocEl innerHTML length:", tocEl.innerHTML.length);
      
      renderTask();
      console.log("renderTask completed");
      console.log("✓ Initialization successful!");
    }} catch (error) {{
      const errorBanner = document.getElementById('init-error');
      const message = `ERROR: ${{error.message}}\\n\\nStack trace:\\n${{error.stack}}`;
      console.error("Initialization error:", error);
      console.error(error.stack);
      if (errorBanner) {{
        errorBanner.style.display = 'block';
        errorBanner.textContent = message;
      }}
      alert("Task Registry Browser failed to initialize. Check the page for error details.");
    }}
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Where to write the browser HTML (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    records = collect_task_records()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_html(records), encoding="utf-8")
    print(f"Wrote {len(records)} task entries to {output_path}")


if __name__ == "__main__":
    main()
