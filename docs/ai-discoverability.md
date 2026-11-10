# AI-assisted discovery strategy for revive-me

Research date: 2026-11-10. Scope: documentation strategy only. No release, visibility change, publication, outreach, or implementation is authorized by this report. The initial implementation provider was OpenCode; the user later selected Sonnet 5.5 High when OpenCode's free quota was exhausted.

> **Implementation status note (added 2026-11-10, updated after the documentation phase; the research below is unchanged and describes the state at research time).**
> The five defects listed under "Establish the evidence boundary" were fixed in commit `9a734a9`; the parent verified 67 passing tests at that commit. The decorator-first API (`@revive_me.resumable` with an injected `run` handle) is in `bdcd0dd`, where the parent verified 113 tests on one macOS arm64 CPU environment (Python 3.12.7, torch 2.15.0). The package name is still `revive_me`; "Torchback" remains an unadopted proposal. The GitHub repository remains private and the package is not published on PyPI. The documentation phase prepared a docs index, three tutorials and a comparison page (see [README.md](README.md)); the parent checked 36 local links, parsed four Python blocks and ran the example followed by a completed-run skip. No AI ranking, citation, recommendation, reboot, GPU or Linux/systemd outcome has been measured; the measurement protocol below is still unrun. A historical local save-time benchmark is described separately in the project README.

## Decision

Fix recovery defects first, then make one accurate, publicly readable explanation easy to retrieve and verify. A memorable name supports identification; it does not establish recommendation quality. Treat retrieval eligibility, citation, and an appropriate recommendation as three separate outcomes. There is no universal AI ranking demonstrated by these sources and no guaranteed citation: Google and Bing explicitly disclaim guaranteed inclusion or visibility. [Google](https://developers.google.com/search/docs/appearance/ai-features), [Bing](https://www.bing.com/webmasters/help/bing-webmaster-guidelines-30fba23a).

## What Ravex demonstrates—and does not

The query `ravex PyTorch checkpoint decorator preemption_handler train_loop` surfaced its PyPI page during this investigation. This is **branded discovery**, not evidence of unbranded recommendation performance. The opened page states its PyTorch checkpoint/resume purpose immediately, supplies installation instructions and a complete CPU example, describes state coverage and limitations, and links to supporting material. Those are observable publishing choices, not independently verified correctness claims. [Ravex on PyPI](https://pypi.org/project/ravex/).

Inference: a self-contained package page gives retrieval systems concrete text to match and readers concrete claims to check. Replicate that clarity, not its feature claims. Neither this observation nor the user's reported DeepSeek recommendation establishes why DeepSeek chose Ravex, its current GitHub star count, or any star threshold. No Ravex implementation or tests were audited.

## Platform boundaries

- **ChatGPT search:** OAI-SearchBot supports appearance in search results. GPTBot concerns content that may be used for foundation-model training; the controls are independent. Allowing search while declining training is possible. OpenAI recommends permitting OAI-SearchBot and its published IP ranges. ChatGPT-User handles certain user-triggered visits and does not determine search eligibility. These are access controls, not ranking guarantees. [OpenAI crawler documentation](https://developers.openai.com/api/docs/bots).
- **Google Search AI Overviews / AI Mode:** supporting pages need indexing and snippet eligibility. Ordinary SEO applies; important information should be textual and internally linked. Google requires no special AI file or schema and does not guarantee crawling, indexing, or serving. This documentation does **not** establish rules for every Gemini app or mode. [Google Search Central](https://developers.google.com/search/docs/appearance/ai-features).
- **Bing grounding:** discoverable, accurate, clearly structured content supports eligibility across Bing, Copilot, and grounding results. Its guidance supports consistent entity names, canonical URLs, useful external links, and sitemaps, without guaranteeing citations. It prohibits artificial promotion, keyword stuffing, and prompt manipulation. This does not prove that any particular DeepSeek or Gemini response uses Bing. [Bing guidelines](https://www.bing.com/webmasters/help/bing-webmaster-guidelines-30fba23a).
- **DeepSeek:** its official site announces API web-search and fetch support. That does not reveal the ranking or retrieval path behind the user's earlier recommendation. No verified publisher inclusion formula was established here; do not invent a DeepSeek crawler directive or assume a search supplier. Evaluate the exact product and search mode empirically. [DeepSeek](https://www.deepseek.com/en/).

**Retrieval versus memorization:** the OpenAI distinction demonstrates why search accessibility and possible training use must be tracked separately. A newly accessible page is not proof of training inclusion, model memorization, or future recommendation. An uncited answer alone does not identify its provenance. [OpenAI](https://developers.openai.com/api/docs/bots). The last two sentences are methodological cautions, not claims about proprietary ranking internals.

## Minimal plan

### 1. Establish the evidence boundary before promotion

The following are **task-provided project facts, not independently inspected or executed in this research**:

- The GitHub repository is private; the initial snapshot is on `main`.
- The current API is explicit: `Checkpointer`, constructor registration, `restore`, `commit`, and sampler integration. A decorator is desired but not implemented; its interface remains under consultation.
- Known defects: corrupt-generation rotation prunes a good fallback; failed saves are retried on every immediate commit; sampler-free cursor restoration does not position an ordinary DataLoader; checkpoint temporary files can remain orphaned; library-level writer exclusion is absent.
- Forty CPU tests reportedly passed. True reboot recovery, GPU behavior, and Linux systemd integration have not been validated.

Recommendation: have OpenCode resolve all five defects and retain reproducible regression evidence before promoting reliability. Validate restart behavior separately from true reboot, GPU state, and service restart behavior. Record versions, platform, storage, saved step, restored step, data position, and comparison against an uninterrupted run. Do not convert a CPU test count into claims of production safety, exact replay, concurrency safety, automatic reboot recovery, or decorator support.

### 2. Decide what can become public

GitHub private repositories restrict access to authorized people; a private repository is therefore not a public documentation surface. [GitHub repository visibility](https://docs.github.com/en/repositories/creating-and-managing-repositories/about-repositories).

Recommendation, subject to a later owner decision: either release an audited public repository with documentation, or keep source private and publish a separate public documentation site with truthful availability/access instructions. Prefer the first only when the project is ready for public use. Public docs can explain a private project, but must not imply that readers can install inaccessible code. Do not submit private URLs for indexing or give crawlers credentials. No visibility change is made here.

### 3. Use a stable identity and an executable explanation

Illustrative positioning only:

> **Torchback — PyTorch checkpoint and resume**
>
> Explicit checkpoint management for custom PyTorch training loops.

Alternatively retain **revive-me — PyTorch checkpoint and resume**. Neither proposal asserts name, trademark, domain, import, or PyPI availability. Check those before choosing; avoid a rename solely for supposed AI ranking benefits.

Use the chosen brand consistently across GitHub, documentation, and any later PyPI distribution. Clearly map distribution name to import name if different. Python packaging supports a short project description, a longer README description, and project URLs for documentation, repository, and issues. [Python Packaging User Guide](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/). Consistency is a recommendation informed by Bing's entity guidance, not a measured ranking effect. [Bing](https://www.bing.com/webmasters/help/bing-webmaster-guidelines-30fba23a).

Prepare one quickstart after API consultation: exact supported Python/PyTorch versions; an install command that works for its intended audience; a complete synthetic-data CPU script; constructor registration; restore placement; commit placement; supported sampler usage; checkpoint location; two invocations showing interruption and restoration; expected output; reset instructions; and precise limitations. Execute the exact published script in a clean environment before calling it runnable. Do not fabricate decorator syntax or a currently available `pip install` command. If a decorator is later implemented, replace the example only after validation.

### 4. Answer real problems and compare fairly

Prepare three focused tutorials, each linked from the quickstart:

- **“Resume PyTorch training after reboot.”** Explain recovery from the latest durable checkpoint, potential lost work since that checkpoint, and the separate need to launch the process again. Gate project-specific reboot instructions on an actual reboot test; do not advertise validated systemd behavior yet.
- **“tmux vs reboot: what keeps training recoverable?”** tmux holds session state in a running server process. Inference: terminal detachment is different from persisting training state across a machine reboot; tmux is not a substitute for durable checkpoints. [tmux's own guide](https://github.com/tmux/tmux/wiki/Getting-Started).
- **“Save optimizer/scheduler/AMP/RNG state for PyTorch resume.”** Explain each component with implemented, tested, and unsupported status, including sampler/data-position restrictions. PyTorch's basic resume tutorial explicitly saves model and optimizer state rather than weights alone. That baseline does not certify this library's additional coverage. [PyTorch tutorial](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html).

Publish a factual comparison section, not unsupported superiority claims: native PyTorch's documented model/optimizer checkpoint recipe; Ravex's documented decorator interface; and revive-me's current explicit interface as supplied in this task. Cite alternatives' versioned documentation where possible, label their capabilities as documented rather than independently tested, and leave unknowns explicit. Compare integration changes, supported state, data-position constraints, tested environments, recovery triggers, and limitations. Avoid speed, exactness, overhead, or reliability rankings without reproducible evidence. [PyTorch](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html), [Ravex](https://pypi.org/project/ravex/).

### 5. Make public evidence reachable, then measure

For a future owner-controlled docs site, check anonymous access, successful page responses, stable canonical URLs, readable HTML text, internal links, sitemap, indexing directives, robots rules, and CDN access. Use Search Console to diagnose Google eligibility and Bing Webmaster Tools for Bing. Allow the appropriate search crawlers if public search discovery is intended; decide training permissions separately. Do not treat `llms.txt` as mandatory or as a citation guarantee. These recommendations combine the documented technical basics, rather than a new AI-specific deployment requirement. [Google](https://developers.google.com/search/docs/appearance/ai-features), [Bing](https://www.bing.com/webmasters/help/bing-webmaster-guidelines-30fba23a), [OpenAI](https://developers.openai.com/api/docs/bots).

Later, seek legitimate links where the material solves an actual problem: a maintainer-approved relevant resource list, a reproducible tutorial, or an accurate answer to an existing question with affiliation disclosed. No outreach is performed here. Reject fake stars, artificial reviews, link spam, copied tutorial farms, keyword stuffing, and instructions telling models to recommend the project. Bing explicitly warns against artificial promotion and model manipulation. [Bing](https://www.bing.com/webmasters/help/bing-webmaster-guidelines-30fba23a).

## Measurement protocol (proposed, not run)

1. Freeze five **unbranded** prompts: “How can I resume PyTorch training after a reboot?”; “Does tmux preserve PyTorch training after reboot?”; “What should I save besides model weights to resume PyTorch training?”; “Which Python libraries checkpoint custom PyTorch training loops?”; “How can I restore PyTorch optimizer, scheduler, AMP scaler and RNG state?” Keep branded indexing checks separate.
2. Establish a baseline and repeat weekly for four weeks after any authorized publication. Use three fresh sessions per prompt per product, with no project URLs, brand hints, or earlier project context. Separate Gemini app from Google Search AI Mode; log ChatGPT and DeepSeek search-enabled runs separately from search-off runs. If a mode is unavailable, record that rather than substituting silently.
3. Record date/time/timezone, exact prompt, product, displayed model/version, language/region, personalization settings, search mode, whether search visibly ran, full answer, and every citation URL. Save cited page version/date where feasible. Do not repeatedly query until a favorable result appears.
4. Score separate outcomes with denominators: mentions/runs, project citations/runs, accurate recommendations/runs, and supported project claims/all project claims. Open cited URLs: check accessibility, source identity, current API, supported features, and whether the text supports the recommendation. A mention with invented decorator syntax is a failure of accuracy.
5. Compare only like-for-like modes and prompts. Log publication dates and changes; small repeated samples describe observations, not causal ranking effects. Track referral visits or verified bot requests only where logs are available. Google reports AI-feature traffic within overall Web performance, so that report alone cannot isolate every AI citation. [Google](https://developers.google.com/search/docs/appearance/ai-features).

## Evidence ledger

**Inspected/executed:** opened official OpenAI, Google, GitHub, Python Packaging, PyTorch, tmux, DeepSeek, and Ravex pages; ran a branded Ravex search that returned PyPI; checked local paths and Git status. Bing's direct page returned a JavaScript shell, so its substantive guidance was inspected through search-indexed excerpts from Bing's own URL. No secondary-source claims were used. Created only this findings file; no library tests, install, reboot, GPU, systemd, or assistant recommendation trials were executed.

**Supplied, not independently verified:** repository privacy/push status, current API, five defects, and the 40-test CPU result. No library source-code correctness claims are established by this report.

**Inferred/proposed:** clear public evidence should improve the opportunity for appropriate retrieval; consistent naming should reduce ambiguity; the minimal publishing plan and measurement protocol are research recommendations. Their effect on Gemini, ChatGPT, and DeepSeek recommendations remains unmeasured. No ranking causality, name availability, or feature correctness is inferred from Ravex's page.

## Prioritized checklist

1. [x] **P0:** Give OpenCode the five-defect list; require reproducible recovery evidence before promotion. *(Status: defects fixed locally at `9a734a9`, 67 tests verified; CPU/macOS only, not published.)*
2. [ ] **P1:** Settle the API and public-docs/public-repo decision; keep unsupported claims out of positioning.
3. [ ] **P2:** Choose a checked name, execute the quickstart, and prepare the three problem tutorials.
4. [ ] **P3:** After separate publication authorization, check crawl/index access and add legitimate contextual links.
5. [ ] **P4:** Run the fixed-query baseline and weekly accuracy/citation measurements.

Immediate next action (under two minutes): choose the final public-facing name and whether to make the repository public.

## Release readiness and measurement checklist (added 2026-11-10)

Facts only; each unchecked box needs an owner decision or a run that has not happened.

### Local evidence that exists

- [x] Version history: `b676143` initial package, `9a734a9` recovery-correctness fixes, `bdcd0dd` decorator API. Use Git to check current synchronization with the private remote.
- [x] At `bdcd0dd`: 113 tests passed with two expected corruption warnings (about 43 seconds), `pip check` clean, whitespace check clean. Environment: macOS arm64, CPU, Python 3.12.7, torch 2.15.0, local APFS.
- [x] Process-kill walkthroughs in [the reboot tutorial](resume-pytorch-training-after-reboot.md) were executed in temporary directories (direct `SIGKILL` and resume; launcher and training process both killed, then the launcher started again). A process kill is not a reboot test.

### Not done

- [ ] A real server reboot with a boot-started run.
- [ ] GPU/CUDA recovery.
- [ ] Linux and systemd (the unit file is an untested template).
- [ ] NFS or other network filesystems (`flock` may be advisory there).
- [ ] A production-length run at the default five-minute interval.

### Owner decisions required (not made, not authorized)

- [ ] Make the repository public, or publish a separate public docs site. Until then the docs are not publicly reachable; they describe an unpublished, private package and must not imply a working `pip install`.
- [ ] Publish to PyPI (and check name and import availability first).
- [ ] Final name. "Torchback" was a suggestion only; keep `revive-me` / `revive_me` consistent until decided. `pyproject.toml`'s `description` still says "Explicit, durable PyTorch checkpointing..."; reword at the time of that decision.
- [ ] Update the README "Availability" paragraph and `docs/README.md` availability sentence when status changes.

### Measurement protocol parameters to fix before any run

- [ ] Prompts: the five unbranded prompts above, verbatim, no project name, URL or hint; separate branded checks.
- [ ] Fresh session per run, three per prompt per product; no memory or earlier project context.
- [ ] Per run, record: date, time and timezone; exact prompt; product and displayed model/version; language and region; personalization state; search mode on/off and whether search visibly ran; full answer; every citation URL.
- [ ] Treat **Google Search AI Mode / AI Overviews** and the **Gemini app** as different products with separate logs; Google's documentation covers Search features only.
- [ ] Denominators stated for every score: mentions per runs, project citations per runs, accurate recommendations per runs, supported claims per all project claims.
- [ ] No universal "AI ranking" is reported: results are per product, mode, prompt and date.
- [ ] Citation experiments run so far: **none**. Baseline cannot be taken until pages are publicly reachable.
- [ ] Source-backed strategy (platform boundaries, plan, evidence ledger) is the sections above; this checklist adds no new platform claims.
