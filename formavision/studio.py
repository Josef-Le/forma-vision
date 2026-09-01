"""Forma Studio — full web UI over the training pipeline. No folder-dropping:
upload, label, harvest, scrape and train from the browser.

    python -m formavision.studio --data ./dataset
    # open http://localhost:7860

Runs anywhere Python runs: your machine, a GitHub Codespace (free tier works —
port 7860 auto-forwards to your phone's browser), or a cloud box with a GPU.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
from collections import deque
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_from_directory

POSES = ["", "Front", "Side", "Back", "Front Relaxed", "Front Double Biceps",
         "Front Lat Spread", "Side Chest", "Side Triceps", "Back Double Biceps",
         "Rear Lat Spread", "Most Muscular (Crab)"]
METHODS = ["", "DEXA", "Caliper", "BIA", "Navy", "estimate"]

JOB = {"name": None, "running": False, "log": deque(maxlen=500)}


def _labels_path(root: Path) -> Path:
    for n in ("labels.json", "labels.yaml", "labels.yml", "labels.jsonl"):
        p = root / n
        if p.exists():
            return p
    return root / "labels.json"


def _load(root: Path) -> list[dict]:
    p = _labels_path(root)
    if not p.exists():
        return []
    if p.suffix in (".yaml", ".yml"):
        import yaml
        return yaml.safe_load(p.read_text()) or []
    if p.suffix == ".jsonl":
        return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    return json.loads(p.read_text() or "[]")


def _save(root: Path, records: list[dict]):
    (root / "labels.json").write_text(json.dumps(records, indent=1), encoding="utf-8")


def _run_job(name: str, argv: list[str]):
    if JOB["running"]:
        return False
    JOB.update(name=name, running=True)
    JOB["log"].clear()

    def worker():
        JOB["log"].append(f"$ {' '.join(argv)}")
        try:
            proc = subprocess.Popen(argv, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in proc.stdout:
                JOB["log"].append(line.rstrip())
            proc.wait()
            JOB["log"].append(f"[exit {proc.returncode}]")
        except Exception as e:  # noqa: BLE001
            JOB["log"].append(f"[error] {e}")
        JOB["running"] = False

    threading.Thread(target=worker, daemon=True).start()
    return True


def create_app(data_dir: Path) -> Flask:
    app = Flask(__name__)
    data_dir = Path(data_dir).resolve()

    @app.before_request
    def _csrf_guard():
        # All mutating routes require a custom header, which cross-site HTML
        # forms and simple fetches cannot set — blocks CSRF from other sites.
        if request.method == "POST" and request.headers.get("X-Forma-Studio") != "1":
            abort(403)
    (data_dir / "images").mkdir(parents=True, exist_ok=True)

    @app.get("/img/<path:name>")
    def img(name):
        return send_from_directory(data_dir / "images", name)

    @app.get("/api/state")
    def state():
        recs = _load(data_dir)
        return jsonify({
            "records": len(recs),
            "with_bf": sum(1 for r in recs if r.get("bf") is not None),
            "with_pose": sum(1 for r in recs if r.get("pose")),
            "with_judge": sum(1 for r in recs if r.get("judge_score") is not None),
            "job": {"name": JOB["name"], "running": JOB["running"],
                    "log": list(JOB["log"])[-60:]},
        })

    @app.get("/api/records")
    def records():
        recs = _load(data_dir)
        page = int(request.args.get("page", 0))
        only = request.args.get("filter", "")
        if only == "unlabeled":
            recs = [r for r in recs if not r.get("pose") and r.get("bf") is None]
        return jsonify({"total": len(recs), "page": page,
                        "items": recs[page * 24:(page + 1) * 24]})

    @app.post("/api/records/<path:image>")
    def save_record(image):
        recs = _load(data_dir)
        body = request.get_json(force=True)
        for r in recs:
            if r["image"] == image:
                if body.get("delete"):
                    recs.remove(r)
                    try:
                        (data_dir / "images" / image).unlink(missing_ok=True)
                    except OSError:
                        pass
                    break
                for k in ("pose", "bf", "bf_method", "judge_score"):
                    v = body.get(k)
                    if v in ("", None):
                        r.pop(k, None)
                    else:
                        r[k] = float(v) if k in ("bf", "judge_score") else v
                break
        else:
            abort(404)
        _save(data_dir, recs)
        return jsonify({"ok": True})

    @app.post("/api/upload")
    def upload():
        recs = _load(data_dir)
        seen = {r["image"] for r in recs}
        pose = request.form.get("pose") or None
        n = 0
        for f in request.files.getlist("photos"):
            name = Path(f.filename).name
            if not name or name in seen:
                continue
            f.save(data_dir / "images" / name)
            recs.append({"image": name, **({"pose": pose} if pose else {})})
            seen.add(name)
            n += 1
        _save(data_dir, recs)
        return jsonify({"ok": True, "uploaded": n})

    @app.post("/api/import-bundle")
    def import_bundle():
        f = request.files.get("bundle")
        if not f:
            abort(400)
        tmp = data_dir / "_bundle.json"
        f.save(tmp)
        _run_job("import-bundle", [sys.executable, "-m", "formavision.prep_forma_export",
                                   "--export", str(tmp), "--out", str(data_dir)])
        return jsonify({"ok": True})

    @app.post("/api/harvest")
    def harvest():
        q = request.form.get("query", "").strip()
        limit = request.form.get("limit", "25")
        argv = [sys.executable, "-m", "formavision.datasets.commons",
                "--out", str(data_dir), "--limit", limit]
        if q:
            argv += ["--query", q]
        _run_job("harvest", argv)
        return jsonify({"ok": True})

    @app.post("/api/scrape")
    def scrape():
        _run_job("scrape", [sys.executable, "-m", "formavision.datasets.contest_results",
                            "--olympia", "--out", str(data_dir / "contest_results.csv")])
        return jsonify({"ok": True})

    @app.post("/api/train")
    def train():
        epochs = request.form.get("epochs", "30")
        backbone = request.form.get("backbone", "convnext_tiny")
        _run_job("train", [sys.executable, "-m", "formavision.train", "--data",
                           str(data_dir), "--epochs", epochs, "--backbone", backbone])
        return jsonify({"ok": True})

    @app.get("/")
    def index():
        return PAGE

    return app


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Forma Studio</title>
<style>
:root{color-scheme:dark; --bg:#131210; --surface:#1c1b18; --surface2:#24221e; --ink:#f2efe8;
 --ink2:#bcb7a9; --muted:#8a867a; --line:#2e2c27; --line2:#3a372f; --accent:#e8a33c; --on:#131210; --good:#0ca30c}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--ink);
 font:15px/1.5 system-ui,-apple-system,'Segoe UI',sans-serif;padding:16px}
.wrap{max-width:980px;margin:0 auto}
h1{font-size:1.4rem;letter-spacing:.12em;text-transform:uppercase} h1 b{color:var(--accent)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:14px}
.card h2{font-size:.85rem;letter-spacing:.1em;text-transform:uppercase;color:var(--ink2);margin:0 0 8px}
.tiles{display:flex;gap:16px;flex-wrap:wrap;margin:12px 0}
.tile b{font-size:1.6rem} .tile span{color:var(--muted);font-size:.78rem;display:block}
button,.btn{background:var(--accent);color:var(--on);border:none;border-radius:10px;
 padding:9px 14px;font-weight:700;cursor:pointer;font-family:inherit}
button.ghost{background:var(--surface2);color:var(--ink);border:1px solid var(--line2)}
input,select{background:var(--surface2);color:var(--ink);border:1px solid var(--line2);
 border-radius:8px;padding:8px;font-family:inherit;max-width:100%}
form{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
#log{background:#0c0b0a;border:1px solid var(--line);border-radius:10px;padding:10px;
 font:12px/1.5 ui-monospace,monospace;height:180px;overflow:auto;white-space:pre-wrap}
.recs{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:10px;margin-top:12px}
.rec{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:8px}
.rec img{width:100%;height:200px;object-fit:cover;border-radius:8px}
.rec .f{display:flex;gap:5px;margin-top:6px} .rec .f>*{flex:1;min-width:0}
.rec .nm{font-size:.68rem;color:var(--muted);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.ok{color:var(--good);font-size:.8rem}
.note{color:var(--muted);font-size:.8rem}
</style></head><body><div class="wrap">
<h1>For<b>ma</b> Studio</h1>
<div class="tiles" id="tiles"></div>
<div class="grid">
 <div class="card"><h2>Add photos</h2>
  <form onsubmit="return F.post(event, '/api/upload')">
   <input type="file" name="photos" multiple accept="image/*" required>
   <select name="pose" id="pose-default"></select>
   <button>Upload</button></form>
  <p class="note">Or import the app's training bundle (Lab → Export bundle):</p>
  <form onsubmit="return F.post(event, '/api/import-bundle')">
   <input type="file" name="bundle" accept=".json" required><button class="ghost">Import bundle</button></form></div>
 <div class="card"><h2>Harvest free-licensed images</h2>
  <form onsubmit="return F.post(event, '/api/harvest')">
   <input name="query" placeholder="blank = curated pose set" style="flex:1">
   <input name="limit" type="number" value="25" style="width:80px">
   <button>Harvest Commons</button></form>
  <p class="note">CC0 / PD / CC-BY(-SA) only; attribution.csv is written automatically.</p>
  <form onsubmit="return F.post(event, '/api/scrape')"><button class="ghost">Scrape contest results (Olympia)</button></form></div>
 <div class="card"><h2>Train</h2>
  <form onsubmit="return F.post(event, '/api/train')">
   <input name="epochs" type="number" value="30" style="width:80px">
   <select name="backbone"><option>convnext_tiny</option><option>convnext_small</option>
    <option>vit_base_patch16_384</option><option>efficientnet_b3</option></select>
   <button>Start training</button></form>
  <p class="note">Checkpoints land in runs/&lt;timestamp&gt;/; watch the job log below.</p></div>
</div>
<div class="card" style="margin-top:12px"><h2>Job log <span id="job" class="ok"></span></h2><div id="log"></div></div>
<div class="card" style="margin-top:12px"><h2>Label editor</h2>
 <button class="ghost" onclick="F.filter='';F.page=0;F.loadRecs()">All</button>
 <button class="ghost" onclick="F.filter='unlabeled';F.page=0;F.loadRecs()">Unlabeled</button>
 <button class="ghost" onclick="F.page++;F.loadRecs()">Next page ›</button>
 <span class="note" id="pageinfo"></span>
 <div class="recs" id="recs"></div></div>
</div><script>
const POSES = ["","Front","Side","Back","Front Relaxed","Front Double Biceps","Front Lat Spread",
 "Side Chest","Side Triceps","Back Double Biceps","Rear Lat Spread","Most Muscular (Crab)"];
const METHODS = ["","DEXA","Caliper","BIA","Navy","estimate"];
const opt = (list, sel) => list.map(p=>`<option ${p===(sel||"")?"selected":""}>${p}</option>`).join("");
document.getElementById("pose-default").innerHTML = opt(POSES, "");
const F = { page:0, filter:"",
 async post(ev, url){
  ev.preventDefault();
  const fd = new FormData(ev.target);
  await fetch(url, {method:"POST", body: fd, headers: {"X-Forma-Studio":"1"}});
  ev.target.reset(); F.loadState(); F.loadRecs();
  return false;
 },
 async loadState(){
  const s = await (await fetch("/api/state")).json();
  document.getElementById("tiles").innerHTML =
   `<div class=tile><b>${s.records}</b><span>records</span></div>`+
   `<div class=tile><b>${s.with_pose}</b><span>pose labeled</span></div>`+
   `<div class=tile><b>${s.with_bf}</b><span>bf labeled</span></div>`+
   `<div class=tile><b>${s.with_judge}</b><span>judge scored</span></div>`;
  document.getElementById("job").textContent = s.job.running ? ("running: "+s.job.name) : (s.job.name ? s.job.name+" done" : "");
  const log = document.getElementById("log");
  log.textContent = s.job.log.join("\\n"); log.scrollTop = log.scrollHeight;
 },
 async loadRecs(){
  const d = await (await fetch(`/api/records?page=${F.page}&filter=${F.filter}`)).json();
  document.getElementById("pageinfo").textContent = ` page ${d.page+1} · ${d.total} total`;
  document.getElementById("recs").innerHTML = d.items.map(r=>`<div class=rec>
   <img src="/img/${encodeURIComponent(r.image)}" loading=lazy alt="">
   <div class=nm>${r.image}</div>
   <div class=f><select id="p-${cssId(r.image)}">${opt(POSES,r.pose)}</select></div>
   <div class=f><input id="b-${cssId(r.image)}" type=number step=0.1 placeholder="bf %" value="${r.bf??""}">
    <select id="m-${cssId(r.image)}">${opt(METHODS,r.bf_method)}</select>
    <input id="j-${cssId(r.image)}" type=number step=0.1 placeholder="judge" value="${r.judge_score??""}"></div>
   <div class=f><button onclick="F.save('${esc(r.image)}')">Save</button>
    <button class=ghost onclick="F.del('${esc(r.image)}')">✕</button></div></div>`).join("");
 },
 async save(image){
  const id = cssId(image);
  await fetch("/api/records/"+encodeURIComponent(image), {method:"POST",
   headers:{"Content-Type":"application/json","X-Forma-Studio":"1"},
   body: JSON.stringify({ pose:document.getElementById("p-"+id).value,
    bf:document.getElementById("b-"+id).value, bf_method:document.getElementById("m-"+id).value,
    judge_score:document.getElementById("j-"+id).value })});
  F.loadState();
 },
 async del(image){
  if(!confirm("Delete "+image+"?")) return;
  await fetch("/api/records/"+encodeURIComponent(image), {method:"POST",
   headers:{"Content-Type":"application/json","X-Forma-Studio":"1"}, body:JSON.stringify({delete:true})});
  F.loadRecs(); F.loadState();
 }};
function cssId(s){ return s.replace(/[^a-zA-Z0-9]/g,"_"); }
function esc(s){ return s.replace(/'/g,"\\\\'"); }
F.loadState(); F.loadRecs(); setInterval(()=>F.loadState(), 2500);
</script></body></html>"""


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args(argv)
    app = create_app(Path(args.data))
    print(f"Forma Studio -> http://{args.host}:{args.port}  (dataset: {args.data})")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
