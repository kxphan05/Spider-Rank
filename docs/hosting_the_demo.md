# Hosting the demo so other people can try it

Two front ends exist for the identical `starter.agent.Agent`:

- `demo/streamlit_app.py` — the original, quickest to deploy on Streamlit
  Community Cloud.
- `demo/web_app.py` — a FastAPI backend plus a hand-built chat UI at
  `demo/web/index.html` (chat bubbles, animated attribute chips, a live
  product grid, an override banner). Same agent underneath, nicer to look at
  and to share for a People's Choice vote. Run it with:

  ```bash
  uv run --with fastapi --with "uvicorn[standard]" uvicorn demo.web_app:app --port 8000
  ```

  Then open <http://localhost:8000>. It reads the catalog the same way the
  Streamlit app does (`TECHJAM_CATALOG`, `data/catalog.jsonl`, or
  `TECHJAM_CATALOG_URL`), and reports which components (BM25, dense, phrase,
  classifiers) are actually live rather than hiding a degrade. Deploy it
  anywhere that runs a Python ASGI app (Render, Fly.io, a VM behind a
  tunnel) — it needs `demo/requirements-web.txt` plus the root
  `requirements.txt` for the full agent.

The hosting options below were written for the Streamlit app; the FastAPI one
follows the same catalog and memory constraints, just with `uvicorn` instead
of `streamlit run`.

---

Three options, cheapest first. Pick based on whether you want the **full**
agent or the **lite** one.

| | lite (keyword + phrase) | full (adds neural components) |
|---|---|---|
| memory needed | ~500 MB | ~2 GB |
| install | `streamlit`, `numpy` | + `torch`, `sentence-transformers` |
| start-up | ~30 s | ~2 min (downloads 129 MB of weights) |
| honest? | yes — see below | yes |

**Lite is not a crippled demo.** Our own ablation measured the dense leg to
add **no hit rate** on this benchmark — identical to four decimal places
whether it is on or off. It only reorders results. The phrase leg, which needs
nothing but SQLite, is the component that actually finds products and is our
largest measured win. The UI shows which components are live, so nobody has to
take that on trust.

---

## First: the catalog problem

The catalog is 58 MB of organizer-provided Amazon metadata and is **not
committed**, so any fresh clone — including every hosting option below — starts
without it. You have two choices.

**Point the app at private storage.** Add a secret and the app downloads it
once on first boot:

```toml
# Streamlit Cloud: Manage app -> Settings -> Secrets
catalog_url = "https://your-private-bucket/catalog.jsonl.gz"
```

`DATA_ATTRIBUTION.md` requires following the source dataset's terms. Use
storage you control. **Do not publish the catalog to a public repository or a
public bucket** — that is redistribution, and it is the one part of this that
could actually get you in trouble.

**Or commit it to a private repo.** Streamlit Cloud can deploy from a private
GitHub repo. Simplest, but the file is 58 MB, so use Git LFS.

---

## Option 1 — Streamlit Community Cloud (easiest)

Best for the **lite** agent. The free tier's memory is tight for the full one.

1. Push this repository to GitHub (private, if you commit the catalog).
2. Go to <https://share.streamlit.io> and click **New app**.
3. Set:
   - **Main file path:** `demo/streamlit_app.py`
   - **Python version:** 3.12
4. Under **Advanced settings -> Secrets**, add your `catalog_url`.
5. Deploy.

Streamlit Cloud picks up `demo/requirements.txt` automatically, which installs
`streamlit` and `numpy` only. No torch, so the build is quick and the
container stays small.

The app will report itself as running **lite** in the sidebar, and explain
why. That is intentional — do not hide it during a demo.

---

## Option 2 — Hugging Face Spaces (best for the full agent)

Free, and gives **16 GB of RAM**, which is far more headroom than Streamlit
Cloud. It is also the natural home for the model weights, since `bge-small`
downloads from Hugging Face anyway.

1. Create a Space at <https://huggingface.co/new-space>, SDK **Streamlit**.
2. Push this repository to it.
3. Add `app_file: demo/streamlit_app.py` to the Space's `README.md` front
   matter.
4. Use the root `requirements.txt` (the full set) instead of
   `demo/requirements.txt`.
5. Add `catalog_url` as a Space **secret**.

The first boot downloads the encoder and builds the dense index, so expect a
few minutes. After that it is cached in the Space's storage.

---

## Option 3 — your own machine, shared over a tunnel

Fastest way to let one friend poke at it right now, with zero deployment.

```bash
uv run --with streamlit streamlit run demo/streamlit_app.py
```

Then in a second terminal, expose it:

```bash
# Cloudflare, no account needed
cloudflared tunnel --url http://localhost:8501
```

You get a public URL that stops existing when you press Ctrl-C. Good for a
stress test, bad for anything you want to leave running — it is your laptop
serving it, and anyone with the link can reach it.

---

## Notes for stress testing

**It is safe to have several people on it at once.** Streamlit serves each
visitor from a worker thread and they share one cached `Agent`. That was
broken until recently: the SQLite index was created with the default
`check_same_thread=True`, so every keyword and phrase query raised from a
worker thread — and because the retrieval layer catches per-leg errors and
continues, the app kept returning ten products from the dense leg alone
without surfacing anything. Fixed with a shared lock. If you had stress-tested
before that fix you would have seen silently worse results, not an error.

**Session state is capped.** The shared `Agent` never evicted finished
sessions, which is fine for the evaluator (200 sessions, then exit) and a slow
leak for a server. The demo now prunes to the 200 most recent.

**Per-turn latency is roughly 100-300 ms** for lite and higher for full, and
turns are serialised by the index lock. That lock is not a bottleneck at demo
scale — queries are sub-millisecond — but it does mean throughput is bounded
by one core.

**What to ask your friend to try**, in rough order of how likely it is to
break something interesting:

- a vague opener: *"I want something for the summer"* (should route
  **browsing**)
- a specific one: *"waterproof hiking boots under $80"* (should route
  **buying**)
- a mid-chat pivot: *"actually forget that, show me jewellery"* (the sidebar
  should show the override firing and the constraints clearing)
- refusing to answer: *"I don't have a preference"*
- nonsense, emoji, an empty message, a 2000-character paste
- the same question ten times, to see the slate stop changing
