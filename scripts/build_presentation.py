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

# --- fig: price distributions (A, B) -------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 2.6))
for c in ["brussels", "antwerp"]:
    p = ok.loc[ok.city == c, "price"]
    ax.hist(np.log10(p), bins=50, alpha=.55, color=COL[c], label=f"{c.title()}  (median €{p.median():.0f})")
ax.set_xticks(np.log10([25, 50, 100, 200, 400, 1000])); ax.set_xticklabels(["25", "50", "100", "200", "400", "1000"])
ax.set_xlabel("price per night, EUR (log scale)"); ax.set_yticks([]); ax.legend(frameon=False)
f_price = img(fig)

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

# --- fig: Amsterdam structure differences -------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(10, 3.2))
pp = ok.assign(ppg=ok.price / ok.accommodates)
for c in ["brussels", "antwerp", "amsterdam"]:
    d = pp[pp.city == c]
    g = d.groupby(d.accommodates.clip(upper=8)).price.median()
    axs[0].plot(g.index, g.values, marker="o", color=COL[c], label=c.title())
    med = d.price.median()
    band = pd.cut(d.latitude * 0, [-1, 1])  # placeholder to keep pandas quiet
feats = rd("gold_listing_features")
fo = feats[feats.price_status == "ok"]
for c in ["brussels", "antwerp", "amsterdam"]:
    d = fo[fo.city == c]
    rel = d.price / d.price.median()
    g = rel.groupby(pd.cut(d.dist_centre_km, [0, 1, 2, 3, 5, 20]), observed=True).median()
    axs[1].plot(range(len(g)), g.values, marker="o", color=COL[c], label=c.title())
axs[0].set_xlabel("guests (8 = 8+)"); axs[0].set_ylabel("median price, EUR"); axs[0].legend(frameon=False)
axs[0].set_title("price level", fontsize=11)
axs[1].set_xticks(range(5)); axs[1].set_xticklabels(["<1", "1-2", "2-3", "3-5", "5+"]); axs[1].set_xlabel("km from centre")
axs[1].set_ylabel("price / city median"); axs[1].set_title("price structure: distance gradient", fontsize=11)
axs[1].axhline(1, color="#999", lw=.6)
f_struct = img(fig)

cm = city_m
SLIDES = []
S = SLIDES.append

S(f"""<section class="title"><div>
<p class="kicker">Group assignment 2 · Big Data · Region 12: Benelux (EUR)</p>
<h1>The third city is where you find out<br>whether you built a pipeline or a prototype</h1>
<p class="sub">A Databricks lakehouse and price model for Brussels (A) and Antwerp (B), then run unchanged on the new client Amsterdam (C)</p>
<p class="team">Team: <i>names</i></p></div></section>""")

S("""<section><h2>The task, and what we built</h2>
<div class="cols"><div>
<ul>
<li><b>Client cities A and B</b>: Brussels (snapshot 2026-09-20, 4,850 listings) and Antwerp (2026-09-23, 2,463)</li>
<li><b>New client C</b>: Amsterdam (2026-09-16, 10,612). Not downloaded until the A/B pipeline and model were frozen (git tag <code>week1-final</code>)</li>
<li>Why Amsterdam: the largest and most expensive Benelux market, another country, another neighbourhood system. The hardest realistic transfer</li>
</ul></div><div>
<ul>
<li><b>Lakehouse</b>: bronze, silver and gold Delta tables in Unity Catalog, on Free Edition serverless (€0)</li>
<li><b>One pipeline, city as the only parameter</b>: a Databricks job for one city plus an orchestrator with for-each</li>
<li><b>Price model</b>: gradient boosting on log(price), against the required baseline</li>
<li><b>AI/BI dashboard</b> (3 pages) and a <b>Genie space</b> over the gold tables</li>
<li><b>City C report</b>: first run, every fix classified, the gap explained</li>
</ul></div></div></section>""")

S(f"""<section><h2>What the data told us: price is missing for a reason, and four columns are the price</h2>
<div class="cols"><div><img src="{f_missing}"><p class="cap">Price is missing <b>not at random</b>: it is missing for listings with no bookable night (no quote).
We exclude those rows from training (they are kept in silver for supply metrics) instead of inventing a price.</p></div>
<div><table><tr><th>column</th><th>relation to price</th><th>verdict</th></tr>
<tr><td>price_quote_price_per_night</td><td>= price (99.96 %/100 %)</td><td class="bad">leak</td></tr>
<tr><td>price_quote_total_price</td><td>= price x nights</td><td class="bad">leak</td></tr>
<tr><td>price_quote_raw</td><td>JSON containing the price</td><td class="bad">leak</td></tr>
<tr><td>estimated_revenue_l365d</td><td>= price x occupancy (100 %)</td><td class="bad">leak</td></tr>
<tr><td>availability_*, occupancy</td><td>consequence of price</td><td class="warn">excluded</td></tr>
<tr><td>neighbourhood</td><td>target-encoded inside CV only</td><td class="good">used</td></tr></table>
<p class="cap">Also: 15 columns are 100 % empty in this snapshot; the long-stay (min. 28 nights) price is half (monthly discount).</p>
<img src="{f_price}" style="max-height:185px"><p class="cap">Prices are log-normal, so the model is trained on log(price); A and B have almost the same price level.</p></div></div></section>""")

S(f"""<section><h2>Architecture: medallion lakehouse, one parameterised pipeline</h2>
<div class="arch">
<div class="lane"><div class="box src">Inside Airbnb<br><small>index page: city to latest snapshot URL</small></div></div>
<div class="arrow">→</div>
<div class="lane"><div class="box bronze"><b>Bronze</b><br>raw volume /city/date/<br>bronze_listings<br><small>all strings + lineage</small></div></div>
<div class="arrow">→</div>
<div class="lane"><div class="box silver"><b>Silver</b><br>silver_listings · dq_metrics<br><small>column contract, try_cast, price parse,<br>dedup, outlier flags</small></div></div>
<div class="arrow">→</div>
<div class="lane"><div class="box gold"><b>Gold</b><br>listing_features · city / neighbourhood /<br>room-type metrics · predictions ·<br>model_metrics</div></div>
<div class="arrow">→</div>
<div class="lane"><div class="box out">AI/BI dashboard<br>Genie space</div></div>
</div>
<ul class="small">
<li><b>Job airbnb_city_etl</b>(city): bronze → silver → gold. <b>Job airbnb_lakehouse</b>(cities): for-each city ETL → train (optional) → for-each score → report</li>
<li>Per-city writes overwrite only that city's partition (<code>replaceWhere</code>), so runs are idempotent and cities run in parallel. <b>Adding a city = one parameter value.</b></li>
<li>All logic lives in one Python package (<code>src/airbnb_lakehouse</code>); the notebooks parse widgets and call it, and the same code runs locally for tests</li>
<li>Model versions are immutable files in a UC volume (+ MLflow), so "score Amsterdam with v1" really means the week-1 model</li></ul></section>""")

S(f"""<section><h2>The model beats the baseline by 30 % in Brussels and Antwerp</h2>
<div class="cols"><div><img src="{f_model}"></div><div>
<ul><li><b>Target</b>: log(price), converted back with exp. <b>Split</b>: deterministic 20 % test per listing (xxhash of id), held out before any choice</li>
<li><b>Choice</b>: 5-fold CV on the training split. Ridge €42.9, random forest €38.6, <b>gradient boosting €37.9</b>, baseline €51.7</li>
<li><b>Features</b> (all known before setting a price): guests, bedrooms, bathrooms, room/property type, minimum nights and long-stay flag,
amenities, reviews, host, distance to the city centre, target-encoded neighbourhood, city</li>
<li>Most important: guests, bedrooms, minimum nights, city, room type</li>
<li>Weak spot: the expensive tail. Above €400 the MAE is €173 and prices are under-predicted</li></ul></div></div></section>""")

S(f"""<section><h2>City C, first run: no errors, no warnings, and useless prices</h2>
<div class="cols"><div>
<ul><li>The frozen pipeline found Amsterdam on the index, downloaded it, typed and cleaned it: <b>exit code 0</b></li>
<li>Same 90 columns, EUR prices: every format check passed</li>
<li><b>MAE €158.60</b> on 5,927 priced listings (A+B: €34.91; A+B baseline on C: €195.54)</li>
<li>95 % of listings under-priced, mean error <b>−€155</b>; highest prediction €601</li>
<li>Silently passed through: <b>43 %</b> missing prices (A/B: 16-20 %), <b>42 %</b> closed calendars,
<b>100 %</b> unseen neighbourhoods, 2.2x price level, and 112 genuine homes flagged as "outliers"</li></ul></div>
<div><table><tr><th></th><th>Brussels</th><th>Antwerp</th><th>Amsterdam</th></tr>
<tr><td>median price</td><td>€{cm.loc['brussels','median_price']:.0f}</td><td>€{cm.loc['antwerp','median_price']:.0f}</td><td><b>€{cm.loc['amsterdam','median_price']:.0f}</b></td></tr>
<tr><td>% calendars closed</td><td>13 %</td><td>12 %</td><td><b>42 %</b></td></tr>
<tr><td>% multi-listing hosts</td><td>53 %</td><td>61 %</td><td><b>19 %</b></td></tr>
<tr><td>% long-stay</td><td>8 %</td><td>7 %</td><td><b>1 %</b></td></tr>
<tr><td>price per guest</td><td>€48</td><td>€45</td><td><b>€108</b></td></tr></table>
<p class="cap">The pipeline was robust to <i>format</i> changes and blind to <i>distribution</i> changes.</p></div></div></section>""")

S(f"""<section><h2>Why: a price level the trees never saw, and a different price structure</h2>
<img src="{f_scatter}" class="wide">
<img src="{f_struct}" class="wide">
<p class="cap">Trees cannot extrapolate: v1 predictions flatten around Belgian levels. A single oracle rescaling factor would halve v1's error. Amsterdam also
differs in structure: per-guest prices are higher, and price falls faster away from the centre.</p></section>""")

S("""<section><h2>What we changed, and which kind of change it was</h2>
<table class="changes"><tr><th>#</th><th>kind</th><th>change</th><th>why</th></tr>
<tr><td>P1</td><td class="pipe">pipeline</td><td>outlier ceiling = 8x the city's median (was a fixed €1,000)</td><td>the rule encoded Belgian prices; it removed 103 genuine Amsterdam homes</td></tr>
<tr><td>P2</td><td class="pipe">pipeline</td><td>data-quality expectations to dq_metrics (pass/warn)</td><td>43 % missing target went through silently; it now warns</td></tr>
<tr><td>P3</td><td class="pipe">pipeline</td><td>scoring coverage report: unseen city, unseen neighbourhoods, price level</td><td>the model must say when it extrapolates</td></tr>
<tr><td>P4/P5</td><td class="pipe">pipeline</td><td>dq_metrics per layer; city lists as <code>a+b+c</code> via a plan task</td><td>two writers to one table; the CLI splits params on commas</td></tr>
<tr><td>M1</td><td class="mod">model</td><td>relative target: log(price / city price index), with the index taken from the city's training split</td><td>learn structure, not Belgian euros</td></tr>
<tr><td>M2</td><td class="mod">model</td><td>drop the one-hot <code>city</code></td><td>meaningless for an unseen city</td></tr>
<tr><td>M3</td><td class="mod">model</td><td>retrain with Amsterdam's 80 % training split (v3)</td><td>its structure differs; its 20 % test split stays untouched</td></tr></table>
<p class="cap">Unchanged because they held up: URL resolution, download, column contract, try_cast typing, parsing, dedup, split.</p></section>""")

S(f"""<section><h2>The numbers: Amsterdam's MAE falls from €172 to €83, and the rest of the gap is proportional</h2>
<img src="{f_gap}" class="wide">
<div class="cols"><div><ul class="small">
<li><b>*</b> €171.8 = v1 on Amsterdam's test split once the outlier fix puts the 103 expensive homes back. The original first run scored <b>€158.6</b> (all priced) / €156.0 (test split)</li>
<li><b>Price level ≈ €60 of the gap</b>: v1 x (Amsterdam median / A+B median) gives €112.7; v2 gives €108.8, no better than a trivial baseline built on Amsterdam's own data (€111.1)</li></ul></div>
<div><ul class="small"><li><b>Structure ≈ €26</b>: v3 learns Amsterdam's room-type premiums and distance gradient, 25 % better than that baseline</li>
<li><b>The rest is proportional</b>: MAPE is 26 % for v3 on Amsterdam and 23 % for v1 at home, so the same relative error lands on 2.2x higher prices</li>
<li>Cost at home: v3 is €35.1 on Brussels (v1: €32.9), so in production we route A/B to v1 and C to v3</li></ul></div></div></section>""")

S("""<section><h2>Dashboard and Genie (live demo)</h2>
<div class="cols"><div><h3>AI/BI dashboard: 3 pages, each titled with a finding</h3>
<ul class="small"><li><b>City overview</b>: "Amsterdam costs 2.3x Belgium, and 42 % of its calendars are closed". Listings, price, availability, ratings, room mix, neighbourhoods by price</li>
<li><b>Cross-city comparison</b>: "Brussels and Antwerp price alike; what differs is the mix". Side-by-side metrics, price distributions, room-type prices, revenue</li>
<li><b>Model</b>: MAE table for every version and city, predicted vs actual, error by room type and neighbourhood, top under/over-priced listings, pipeline warnings</li></ul></div>
<div><h3>Genie: the tricks we used to configure it</h3>
<ul class="small"><li>Only 5 pre-aggregated gold tables; medians precomputed with the dashboard's outlier rule, so answers match the dashboard</li>
<li>Instructions map "City A/B/C" to slugs, "entire home" to <code>Entire home/apt</code>, and fix the sign of <code>error</code> (negative = under-priced)</li>
<li>The honest-evaluation filter is written out (test split or new city) so Genie never reports training error</li>
<li>Production routing (v1 for A/B, v3 for C) as the default model version</li>
<li>UC comments on every column; trusted example SQL for each sample question</li></ul></div></div></section>""")

S("""<section><h2>Challenges, and who did what</h2>
<div class="cols"><div><ul class="small">
<li><b>Holdout discipline</b>: freezing a git tag and a model hash before touching Amsterdam, so the first run is provably unchanged</li>
<li><b>Leaks hidden in new columns</b> (<code>price_quote_*</code>, revenue): found only by checking exact equalities, not correlations</li>
<li><b>Missing target</b>: 16-43 % of listings have no price, for structural reasons; deciding not to impute it</li>
<li><b>Free Edition constraints</b>: serverless only, ANSI SQL (hence try_cast everywhere), no cluster libraries (%pip per notebook)</li>
<li><b>Silent success</b> is harder to catch than a crash: a stack trace would have been easier to deal with than a confident wrong table</li></ul></div>
<div><table><tr><th>member</th><th>contribution</th></tr>
<tr><td><i>name</i></td><td><i>fill in</i></td></tr><tr><td><i>name</i></td><td><i>fill in</i></td></tr><tr><td><i>name</i></td><td><i>fill in</i></td></tr></table></div></div></section>""")

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
"""
html = f"<!doctype html><html><head><meta charset='utf-8'><title>Benelux Airbnb lakehouse</title><style>{CSS}</style></head><body>{''.join(SLIDES)}</body></html>"
with open(OUT_HTML, "w", encoding="utf-8") as fh:
    fh.write(html)

browser = next(p for p in [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                           r"C:\Program Files\Google\Chrome\Application\chrome.exe"] if os.path.exists(p))
subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                f"--print-to-pdf={OUT_PDF}", "file:///" + OUT_HTML.replace("\\", "/")], check=True, timeout=120)
print("wrote", OUT_PDF, os.path.getsize(OUT_PDF))
