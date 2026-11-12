"""Builds docs/presentation.pdf: charts from the local lakehouse -> HTML slides -> PDF (headless Edge/Chrome)."""
import base64
import io
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from peek import rd  # noqa: E402

OUT_HTML = os.path.join(ROOT, "docs", "presentation.html")
OUT_PDF = os.path.join(ROOT, "docs", "presentation.pdf")
COL = {"brussels": "#2563eb", "antwerp": "#f59e0b", "amsterdam": "#dc2626"}
VCOL = {"baseline": "#9ca3af", "v1": "#1f2937", "v2": "#7c3aed", "v3": "#059669"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})


def img(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


silver = rd("silver_listings")
preds = rd("gold_predictions")
metrics = rd("gold_model_metrics")
city_m = rd("gold_city_metrics").set_index("city")
ok = silver[silver.price_status == "ok"]

# --- fig: missingness pattern ----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.4, 3.0))
rows = []
for c in ["brussels", "antwerp"]:
    d = silver[silver.city == c]
    rows.append((c, (d.price_status == "missing")[d.availability_365 == 0].mean(),
                 (d.price_status == "missing")[d.availability_365 > 0].mean()))
x = np.arange(2)
ax.bar(x - .18, [r[1] * 100 for r in rows], .36, color="#374151", label="availability_365 = 0")
ax.bar(x + .18, [r[2] * 100 for r in rows], .36, color="#d1d5db", label="availability_365 > 0")
for i, r in enumerate(rows):
    ax.text(i - .18, r[1] * 100 + 2, f"{r[1]:.0%}", ha="center"); ax.text(i + .18, r[2] * 100 + 2, f"{r[2]:.0%}", ha="center")
ax.set_xticks(x); ax.set_xticklabels(["Brussels", "Antwerp"]); ax.set_ylabel("% listings without price"); ax.set_ylim(0, 110)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
f_missing = img(fig)

# --- fig: model vs baseline (A, B) ---------------------------------------------------------
m1 = metrics[(metrics.model_version == "v1") & (metrics.evaluation_set == "test split")].set_index("city")
cities = ["brussels", "antwerp", "ALL TRAINING CITIES"]
fig, ax = plt.subplots(figsize=(6, 3.2))
x = np.arange(3)
ax.bar(x - .18, m1.loc[cities, "mae_baseline"], .36, color=VCOL["baseline"], label="baseline: median by room type x guests")
ax.bar(x + .18, m1.loc[cities, "mae_model"], .36, color=VCOL["v1"], label="model v1 (gradient boosting)")
for i, c in enumerate(cities):
    ax.text(i - .18, m1.loc[c, "mae_baseline"] + 1, f"€{m1.loc[c, 'mae_baseline']:.1f}", ha="center", fontsize=9)
    ax.text(i + .18, m1.loc[c, "mae_model"] + 1, f"€{m1.loc[c, 'mae_model']:.1f}", ha="center", fontsize=9, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(["Brussels", "Antwerp", "both"]); ax.set_ylabel("test MAE, EUR"); ax.set_ylim(0, 68)
ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
f_model = img(fig)

# --- fig: predicted vs actual (v1 on A/B test, v1 and v3 on Amsterdam test) ---------------------
ev = preds[(preds.price_status == "ok") & (preds.split == "test")]
fig, axs = plt.subplots(1, 3, figsize=(11, 3.5), sharex=True, sharey=True)
panels = [("v1", ["brussels", "antwerp"], "v1 on Brussels + Antwerp"),
          ("v1", ["amsterdam"], "v1 on Amsterdam (first run)"),
          ("v3", ["amsterdam"], "v3 on Amsterdam (after changes)")]
for a, (ver, cs, title) in zip(axs, panels):
    for c in cs:
        d = ev[(ev.model_version == ver) & (ev.city == c)]
        a.scatter(d.price, d.predicted_price, s=5, alpha=.35, color=COL[c], label=c.title())
    a.plot([20, 2500], [20, 2500], color="#111", lw=.8)
    a.set_xscale("log"); a.set_yscale("log"); a.set_title(title, fontsize=11); a.set_xlabel("actual price (EUR)")
    a.legend(frameon=False, fontsize=8, markerscale=3, loc="upper left")
axs[0].set_ylabel("predicted price (EUR)")
f_scatter = img(fig)

# --- fig: the gap (Amsterdam test split) -------------------------------------------------------
mt = metrics[metrics.evaluation_set == "test split"].set_index(["model_version", "city"])
bars = [("v1 on A+B\n(home)", mt.loc[("v1", "ALL TRAINING CITIES"), "mae_model"], VCOL["v1"]),
        ("baseline A+B\non Amsterdam", mt.loc[("v1", "amsterdam"), "mae_baseline"], VCOL["baseline"]),
        ("v1 on Amsterdam\n(first run*)", mt.loc[("v1", "amsterdam"), "mae_model"], "#ef4444"),
        ("baseline built\non Amsterdam", 111.07, VCOL["baseline"]),
        ("v2: relative price\n(A+B only)", mt.loc[("v2", "amsterdam"), "mae_model"], VCOL["v2"]),
        ("v3: + retrained\nwith Amsterdam", mt.loc[("v3", "amsterdam"), "mae_model"], VCOL["v3"])]
fig, ax = plt.subplots(figsize=(10, 3.6))
for i, (lab, v, c) in enumerate(bars):
    ax.bar(i, v, color=c, width=.65); ax.text(i, v + 3, f"€{v:.1f}", ha="center", fontweight="bold")
ax.set_xticks(range(len(bars))); ax.set_xticklabels([b[0] for b in bars], fontsize=9.5)
ax.set_ylabel("MAE, EUR (test split)"); ax.set_ylim(0, 235)
f_gap = img(fig)


# ================================================================================================
# 7-minute deck: one slide per evaluation criterion. Timings and spoken text: docs/presentation_script.md
# ================================================================================================
SLIDES = []
S = SLIDES.append


def kick(n: int, label: str) -> str:
    """Opens a slide: <section> plus the evaluation criterion it covers."""
    return f'<section><p class="crit"><span>{n}/9</span>{label}</p>'


# 1 ---- title (0:20) ---------------------------------------------------------------------------
S("""<section class="title"><div>
<p class="kicker">Group assignment 2 · Big Data · Region 12: Benelux (EUR) · Team BoberRMDT</p>
<h1>Our pipeline ran a new city without a single error,<br>and that was the problem</h1>
<p class="sub">Lakehouse and price model built on Brussels (A) and Antwerp (B), then run unchanged on the new client Amsterdam (C)</p>
<p class="team">Matvii Shumylovych · Dmytro Hrebenyuk</p></div></section>""")

# 2 ---- exploratory analysis (1:00) ------------------------------------------------------------
S(kick(2, "Exploratory analysis") + f"""<h2>Price is missing for a reason, outliers are mostly placeholders, and four columns are the price</h2>
<div class="cols"><div><img src="{f_missing}" style="max-height:300px">
<p class="cap"><b>Missing, explained:</b> no price = no bookable night, so no quote. That is not random, so we <b>don't impute</b>:
the rows stay in silver for supply metrics and are left out of training.</p>
<p class="cap"><b>Outliers, explained:</b> above €1,000 (0.5 %) are mostly 0-1-review placeholders (a 1-guest room at €1,500+);
they are <b>flagged, not deleted</b>. A 25-bedroom, 2-guest flat is a typo: value set to NULL, row kept.</p></div>
<div><h3>Leakage review: found by exact equality, not by correlation</h3>
<table class="tight"><tr><th>column</th><th>relation to price</th><th>verdict</th></tr>
<tr><td>price_quote_price_per_night</td><td>= price (99.96 %)</td><td class="bad">leak</td></tr>
<tr><td>price_quote_total_price</td><td>= price × nights</td><td class="bad">leak</td></tr>
<tr><td>price_quote_raw</td><td>JSON that contains the price</td><td class="bad">leak</td></tr>
<tr><td>estimated_revenue_l365d</td><td>= price × occupancy</td><td class="bad">leak</td></tr>
<tr><td>availability_*, occupancy</td><td>a consequence of the price</td><td class="warn">excluded</td></tr>
<tr><td>neighbourhood</td><td>target-encoded inside CV folds only</td><td class="good">used</td></tr></table>
<p class="cap">Reproducible: the notebook <code>00_eda_brussels_antwerp</code> rebuilds every number from the raw files and never loads Amsterdam.</p></div></div></section>""")

# 3 ---- architecture and pipeline (1:00) -------------------------------------------------------
S(kick(3, "Architecture and pipeline") + """<h2>Three layers, each with one job, and one pipeline where the city is the only parameter</h2>
<div class="arch">
<div class="lane"><div class="box src">Inside Airbnb<br><small>city → latest snapshot URL, found on the index page</small></div></div>
<div class="arrow">→</div>
<div class="lane"><div class="box bronze"><b>Bronze</b><br><i>keep it as it came</i><br><small>raw file in a UC volume, all strings, lineage</small></div></div>
<div class="arrow">→</div>
<div class="lane"><div class="box silver"><b>Silver</b><br><i>make it trustworthy</i><br><small>column contract, try_cast, price parsing, dedup, outlier flags, DQ checks</small></div></div>
<div class="arrow">→</div>
<div class="lane"><div class="box gold"><b>Gold</b><br><i>answer questions</i><br><small>features, city / neighbourhood / room-type metrics, predictions, model metrics</small></div></div>
<div class="arrow">→</div>
<div class="lane"><div class="box out">Dashboard<br>Genie</div></div>
</div>
<div class="cols"><ul class="small">
<li><b>Job <code>airbnb_city_etl(city)</code></b>: bronze → silver → gold for one city</li>
<li><b>Job <code>airbnb_lakehouse(cities)</code></b>: for-each city ETL → train (optional) → for-each score → report</li>
<li>Each city overwrites only its own partition (<code>replaceWhere</code>): runs are idempotent and cities run in parallel</li></ul>
<div class="proof"><b>Proof that it really runs per city:</b> Amsterdam was onboarded with one parameter value,<br>
<code>cities=amsterdam, retrain=false, model_version=v1</code><br>
No code change: <code>git diff week1-final</code> was empty before the run.</div></div></section>""")

# 4 ---- price model (0:50) ---------------------------------------------------------------------
S(kick(4, "Price model") + f"""<h2>On held-out listings the model beats the baseline by 30 % in both cities</h2>
<div class="cols"><div><img src="{f_model}"></div><div><ul class="small">
<li><b>Honest test set</b>: 20 % of listings, picked by a hash of the id, set aside <i>before</i> any choice; the model was chosen by 5-fold CV on the other 80 %</li>
<li><b>Baseline</b>: median training price for the same room type and number of guests</li>
<li><b>Candidates (CV MAE)</b>: ridge €42.9, random forest €38.6, <b>gradient boosting €37.9</b>; target is log(price)</li>
<li><b>Features we can justify</b>: everything a host knows <i>before</i> setting a price: guests, bedrooms, room type, minimum nights, amenities, reviews, distance to centre, neighbourhood</li>
<li><b>No leakage</b>: the four leak columns and availability are excluded, the neighbourhood encoder is fitted inside CV</li>
<li><b>Weak spot</b>: above €400 the error is €173 and prices are under-predicted</li></ul></div></div></section>""")

# 5 ---- city C first run (0:50) ----------------------------------------------------------------
S(kick(5, "City C: the first run") + f"""<h2>Amsterdam, first run: exit code 0, no warnings, and 95 % of homes under-priced</h2>
<div class="tiles">
<div class="tile"><b>0</b><span>errors or warnings</span></div>
<div class="tile red"><b>€158.6</b><span>MAE on Amsterdam<br>(€34.9 on Brussels + Antwerp)</span></div>
<div class="tile red"><b>−€155</b><span>mean error: 95 % under-priced;<br>no prediction above €601</span></div>
<div class="tile"><b>43 %</b><span>missing prices passed silently<br>(16-20 % in A/B)</span></div>
<div class="tile"><b>100 %</b><span>neighbourhoods the model<br>has never seen</span></div></div>
<img src="{f_scatter}" class="wide" style="max-height:300px">
<p class="cap">Gradient-boosted trees can't predict above the prices they were trained on, so v1 flattens at Belgian levels (middle panel).
The pipeline coped with changes in <i>format</i> but didn't notice changes in <i>distribution</i>.</p></section>""")

# 6 ---- city C changes classified (0:40) -------------------------------------------------------
S(kick(6, "City C: changes, classified") + """<h2>Every change since the freeze, sorted into pipeline fixes and model changes</h2>
<table class="changes"><tr><th>#</th><th>kind</th><th>change</th><th>why</th></tr>
<tr><td>P1</td><td class="pipe">pipeline</td><td>outlier ceiling = 8× the city's median (was a fixed €1,000)</td><td>the rule encoded Belgian prices; it deleted 103 genuine Amsterdam homes</td></tr>
<tr><td>P2</td><td class="pipe">pipeline</td><td>data-quality expectations → <code>dq_metrics</code> pass / warn</td><td>a 43 % missing target went through without a word</td></tr>
<tr><td>P3</td><td class="pipe">pipeline</td><td>scoring coverage: unseen city, unseen neighbourhoods, price level</td><td>the model has to say when it is extrapolating</td></tr>
<tr><td>P4-P6</td><td class="pipe">pipeline</td><td>DQ per layer; <code>a+b+c</code> city lists; job dependency fix</td><td>found while deploying, not caused by Amsterdam</td></tr>
<tr><td>M1</td><td class="mod">model</td><td>relative target: log(price / city price index)</td><td>learn structure, not Belgian euros</td></tr>
<tr><td>M2</td><td class="mod">model</td><td>drop the one-hot <code>city</code></td><td>it means nothing for an unseen city</td></tr>
<tr><td>M3</td><td class="mod">model</td><td>retrain with Amsterdam's 80 % training split (v3)</td><td>its structure differs; its 20 % test split stays untouched</td></tr></table>
<p class="cap">Left unchanged because it held up on unseen data: URL resolution, download, column contract, try_cast, parsing, dedup, the split.
Every change is tagged in the code and listed by <code>git diff week1-final</code>.</p></section>""")

# 7 ---- city C gap explained (0:50) ------------------------------------------------------------
S(kick(7, "City C: the gap explained") + f"""<h2>Most of the gap is price level; what's left is the same 25 % error on prices twice as high</h2>
<img src="{f_gap}" class="wide" style="max-height:290px">
<div class="cols3">
<div class="pt"><b>≈ €60: price level</b><br>Rescaling v1 by Amsterdam's median gives €112.7. v2 gets €108.8, but a baseline built on Amsterdam alone scores €111.1</div>
<div class="pt"><b>≈ €26: price structure</b><br>Bigger entire-home premium, steeper drop with distance. v3 learns it: €82.9, 25 % below that baseline</div>
<div class="pt"><b>The rest: proportional</b><br>Relative error is 26 % for v3 on Amsterdam vs 23 % for v1 at home. Same relative error, on prices 2.2× higher</div></div>
<p class="cap">* €171.8 = v1 after P1 puts the 103 expensive homes back. The original first run (€158.6) looked better only because those homes had been dropped.</p></section>""")

# 8 ---- dashboard and Genie, live demo (1:10) --------------------------------------------------
S(kick(8, "Dashboard and Genie: live demo") + """<h2>Live demo: each dashboard page opens with its answer, and Genie returns the numbers we expect</h2>
<div class="cols"><div><h3>AI/BI dashboard: page titles state the finding</h3>
<ol class="small">
<li><b>City overview</b>: “Amsterdam costs 2.3× Belgium (median €285 vs €125-130), and 42 % of its calendars are closed”</li>
<li><b>Cross-city comparison</b>: “Brussels and Antwerp price alike; what differs is the mix (Antwerp: 84 % entire homes, 61 % multi-listing hosts)”</li>
<li><b>Model</b>: “v1 beats the baseline by 30 % at home but under-priced Amsterdam by €168; the relative model retrained on Amsterdam (v3) cuts its MAE from €172 to €83”, plus the pipeline warnings table</li></ol></div>
<div><h3>Genie: question → expected answer</h3>
<table class="tight"><tr><th>we ask</th><th>it should answer</th></tr>
<tr><td>Median price of entire homes in City A?</td><td>€138 (Brussels, 3,509 homes)</td></tr>
<tr><td>Neighbourhoods in City B with the most listings?</td><td>Historisch Centrum 397, Amandus-Atheneum 219, Theaterbuurt-Meir 194</td></tr>
<tr><td>How did the model do on the new city?</td><td>v1 €172 → v2 €109 → v3 €83</td></tr></table>
<p class="cap">How we got it right: 5 pre-aggregated gold tables, “City A/B/C” mapped in the instructions,
a trusted SQL example for each question, UC comments on every column, and test-split-only evaluation built into the instructions.</p></div></div></section>""")

# 9 ---- takeaway + team (0:20) -----------------------------------------------------------------
S(kick(9, "Takeaway") + """<h2>A crash is easy to spot. We fixed the run that succeeded and was still wrong</h2>
<div class="cols"><div><ul>
<li>The week-1 pipeline handled <b>format</b> changes and was blind to <b>distribution</b> changes</li>
<li>Now it <b>warns</b> when a city doesn't look like the training cities (P2, P3)</li>
<li>The model learns <b>price relative to the city</b>, which transfers (M1, M2)</li>
<li>Production routing: v1 for Brussels and Antwerp, v3 for Amsterdam</li></ul></div>
<div><table><tr><th>member</th><th>contribution</th></tr>
<tr><td>Matvii Shumylovych</td><td>EDA and leak review, medallion pipeline, Databricks bundle and jobs, Amsterdam first run, pipeline fixes P1-P6, Genie</td></tr>
<tr><td>Dmytro Hrebenyuk</td><td>Price model v1 and model cards, model changes M1-M3 (v2, v3), city C report, AI/BI dashboard, presentation</td></tr></table>
<p class="cap">Everything is in the repo: <code>docs/city_c/report.md</code>, model cards, logs of each run.</p></div></div></section>""")

CSS = """
@page { size: 1280px 720px; margin: 0 }
* { box-sizing: border-box }
body { margin: 0; font-family: 'Segoe UI', system-ui, sans-serif; color: #111827; background: #fff }
section { width: 1280px; height: 720px; padding: 44px 60px; page-break-after: always; position: relative; overflow: hidden }
section:after { content: ''; position: absolute; left: 60px; right: 60px; bottom: 14px; border-top: 1px solid #e5e7eb }
h1 { font-size: 46px; line-height: 1.15; margin: 0 0 22px } h2 { font-size: 29px; margin: 0 0 20px; letter-spacing: -.3px }
h3 { font-size: 18px; margin: 0 0 8px; color: #374151 }
.title { display: flex; align-items: center; background: #0f172a; color: #f8fafc } .title .sub { font-size: 22px; color: #cbd5e1; max-width: 1000px }
.kicker { text-transform: uppercase; letter-spacing: 2px; font-size: 14px; color: #94a3b8 } .team { margin-top: 40px; color: #94a3b8 }
.cols { display: grid; grid-template-columns: 1fr 1fr; gap: 36px; align-items: start }
ul { margin: 0; padding-left: 22px; font-size: 20px; line-height: 1.45 } li { margin-bottom: 10px } ul.small { font-size: 16.5px } ul.small li { margin-bottom: 7px }
img { max-width: 100%; } img.wide { display: block; max-height: 252px; margin: 0 auto 4px }
.big img { max-height: 330px }
.cap { font-size: 15.5px; color: #4b5563; margin-top: 8px }
table { border-collapse: collapse; font-size: 16px; width: 100% } th, td { text-align: left; padding: 8px 9px; border-bottom: 1px solid #e5e7eb }
th { color: #6b7280; font-weight: 600 } .bad { color: #b91c1c; font-weight: 600 } .warn { color: #b45309 } .good { color: #047857 }
.changes td { font-size: 16.5px; padding: 11px 9px } .pipe { color: #1d4ed8; font-weight: 600 } .mod { color: #047857; font-weight: 600 }
code { background: #f3f4f6; padding: 1px 4px; border-radius: 3px; font-size: .92em }
.arch { display: flex; align-items: stretch; gap: 8px; margin: 10px 0 26px } .lane { flex: 1; display: flex }
.box { flex: 1; border-radius: 10px; padding: 16px; font-size: 16px; line-height: 1.35; border: 1px solid #e5e7eb }
.arrow { align-self: center; font-size: 26px; color: #9ca3af } .src { background: #f9fafb } .bronze { background: #fdf0e3; border-color: #f0c79b }
.silver { background: #f1f5f9; border-color: #cbd5e1 } .gold { background: #fef9e3; border-color: #f3d36b } .out { background: #ecfdf5; border-color: #a7f3d0 }
small { color: #6b7280 }
.crit { margin: 0 0 6px; font-size: 13px; letter-spacing: 1.5px; text-transform: uppercase; color: #6b7280 }
.crit span { display: inline-block; background: #0f172a; color: #fff; border-radius: 4px; padding: 1px 7px; margin-right: 10px; letter-spacing: 0 }
ol { margin: 0; padding-left: 22px; line-height: 1.45 } ol.small { font-size: 16.5px } ol.small li { margin-bottom: 10px }
table.tight { font-size: 15px } table.tight td, table.tight th { padding: 6px 8px }
.proof { background: #ecfdf5; border: 1px solid #a7f3d0; border-radius: 10px; padding: 16px 18px; font-size: 16.5px; line-height: 1.6 }
.tiles { display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; margin: 4px 0 14px }
.tile { border: 1px solid #e5e7eb; border-radius: 10px; padding: 10px 14px; background: #f9fafb }
.tile b { display: block; font-size: 30px; line-height: 1.1 } .tile span { font-size: 13.5px; color: #4b5563 }
.tile.red { background: #fef2f2; border-color: #fecaca } .tile.red b { color: #b91c1c }
.cols3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 18px; margin-top: 6px }
.pt { font-size: 15.5px; line-height: 1.4; border-left: 3px solid #059669; padding-left: 12px }
"""
html = f"<!doctype html><html><head><meta charset='utf-8'><title>Benelux Airbnb lakehouse</title><style>{CSS}</style></head><body>{''.join(SLIDES)}</body></html>"
with open(OUT_HTML, "w", encoding="utf-8") as fh:
    fh.write(html)

browser = next(p for p in [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                           r"C:\Program Files\Google\Chrome\Application\chrome.exe"] if os.path.exists(p))
subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                f"--print-to-pdf={OUT_PDF}", "file:///" + OUT_HTML.replace("\\", "/")], check=True, timeout=120)
print("wrote", OUT_PDF, os.path.getsize(OUT_PDF))
