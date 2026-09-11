# SGP2 Extension Notes

Generated from repo evidence on 2026-04-05 UTC.

## What I checked

I reviewed the comparable BTCUSDT evaluation artifacts already saved in the repo, not just the Word draft:

- March 19, 2026 cross-timeframe deployment summary:
  - `finetune_csv/runs_requested_by_user/conference_report_20260319`
- Legacy multi-timeframe leaderboard used by the earlier paper draft:
  - `reports/srcac2025_paper/best_runs_summary_multitimeframe.csv`
- Daily March 13, 2026 model screening:
  - `finetune_csv/runs_requested_by_user/research_1m_multistep_from_epoch4_20260311/close_to_close_best_runs_1d_to_2026-03-13/summary.json`
- Later exploratory runs after the conference snapshot:
  - `research_1h_ce_sweep_20260320/run_summary.json`
  - `research_1d_generalization_research_sweep_20260401/run_summary.json`
  - `research_1d_context_window_mag_20260329_151639/.../metrics.json`
  - `research_1d_context_window_mag_full2025ckpt_20260329_153041/.../metrics.json`
  - `eval_1d_2026_to_2026-03-29_cons_only_lr3e6_last6_adapter_metrics.json`
- Website prototype:
  - `webui/app.py`
  - `webui/templates/index.html`
  - `webui/README.md`

## Stronger conclusion

The repo supports a stronger but narrower conclusion than the current draft:

1. The best model depends on the horizon.
2. The cleanest evidence is at `1h`, where `50%` pruning clearly improved both accuracy and trading outcome.
3. The strongest economic result is at `1d`, but later March and April experiments show that daily generalization is still fragile.
4. The `5min` winner is positive, but the advantage is modest and should be framed cautiously.

### 5-minute

The native 5-minute direction-head run is still the best saved `5min` deployment result in the repo:

- Accuracy: `51.40%`
- Baseline: `50.16%`
- Margin: `+1.24 pp`
- PnL: `+5,836.58 USDT`
- Window: `2026-01-01` to `2026-03-09`

Why it wins:

- It is trained natively on the same timeframe.
- It keeps the simplest objective: direct one-step direction classification.
- It beats the transferred `pruned50` runner-up on absolute USDT PnL, even though the two models are almost tied on hit rate.

Why the paper should stay careful here:

- The `5min` result is not dominant in the same way as the `1h` and `1d` winners.
- Older consecutive `5min` baselines stayed at or below majority baseline on the shorter `2026-01-01` to `2026-02-16` benchmark.
- A defensible wording is: `native 5-minute direction-head fine-tuning produced the best saved short-horizon 5-minute result, but the edge remains modest.`

### 1-hour

This is the strongest and cleanest section of the whole story.

- Best model: `direction_head_pruned50_actualfeed_1h`
- Accuracy: `53.67%`
- Baseline: `50.73%`
- Margin: `+2.94 pp`
- PnL: `+6,069.51 USDT`
- Window: `2026-01-01` to `2026-03-04`

Why `pruned50` is the right winner:

- It is the highest-accuracy hourly variant in the pruning ladder.
- It is the only profitable member of the dense / `25%` / `50%` / `75%` hourly sweep.
- It materially reduced long-bias relative to the dense checkpoint:
  - Dense predicted-up gap over actual: `+15.82 pp`
  - Pruned `25%`: `+12.82 pp`
  - Pruned `50%`: `+7.74 pp`
  - Pruned `75%`: `+17.62 pp`

Interpretation:

- Moderate pruning acted like regularization.
- The dense model and heavy-pruned model both over-predicted upward moves too aggressively.
- `50%` pruning kept enough capacity while suppressing the overconfident long bias that hurt the other variants.

Later evidence strengthens this conclusion:

- The later `1h` consecutive-path CE sweep on March 20, 2026 did **not** beat it.
- That later retrain reached only `49.53%` vs a `50.73%` baseline on the same `2026` hourly window.

Recommended wording:

`The hourly result is the clearest mechanistic finding in the study: moderate sparsification improved both directional accuracy and profitability, while dense and heavily pruned variants remained unprofitable.`

### 1-day

This is still the strongest period-specific deployment result in the repo, but it needs a more careful generalization claim.

Main daily winner:

- Model: `cons_only_lr3e6_last6_adapter`
- Accuracy: `62.50%` on `72` rows through `2026-03-13`
- Accuracy: `62.67%` on `75` rows through `2026-03-16`
- Baseline at March 16 window: `53.33%`
- Margin at March 16 window: `+9.33 pp`
- PnL through March 16: `+28,035.71 USDT`
- Compounded return through March 16: `+43.30%`

Why it beat the other daily models:

- It used a pure `consecutive_path_only` objective instead of mixing in a CE branch.
- It unfroze the last `6` layers instead of only the last `4`.
- It used an output adapter and a path-consistent rollout setup aligned with daily trajectory prediction.

The most important daily ablation result is this:

- `cons_only_lr3e6_last6_adapter`: `62.50%`, `+25,245.06 USDT`
- `cons_only_lr3e6_last4_adapter`: `45.83%`, `-270.70 USDT`
- `final_cons_hybrid_lr3e6_last4_adapter`: `47.22%`, `-10,123.34 USDT`

What this means:

- Last-`6` unfreezing mattered a lot.
- The hybrid daily objective was worse, not better.
- The hybrid models also showed a severe upward bias:
  - Hybrid predicted-up ratio: `69.44%`
  - Actual up ratio: `44.44%`

Important caveat from later evidence:

- When the same daily winner was refreshed to `2026-03-29`, its path-direction accuracy was still above baseline, but only barely:
  - Accuracy: `57.95%`
  - Baseline: `56.82%`
  - Margin: `+1.14 pp`
- The later April 1 daily generalization sweep improved on the late-2025 holdout, but then slipped back near or below baseline on the 2026 slice.
- The later context-magnitude daily experiments also did not beat baseline on the later March window.

Recommended wording:

`The daily path-consistent adapter produced the strongest saved deployment outcome in the repository, but later generalization sweeps indicate that the result should be framed as the best observed configuration among tested daily models rather than a solved or universally robust daily forecasting method.`

## Figures to include

The following figures were generated into this folder and are ready to use:

- `cross_timeframe_accuracy_vs_baseline.png`
  - Use this in the results section to show winner accuracy against baseline.
- `cross_timeframe_total_pnl.png`
  - Use this in the conclusion or deployment section to show the economic gap between horizons.
- `hourly_pruning_sweep.png`
  - This is the most important ablation figure. It shows accuracy, PnL, and directional bias for the hourly pruning ladder.
- `daily_model_screening.png`
  - Use this to justify why `last6 + consecutive only` is the daily winner.
- `daily_winner_extension.png`
  - Use this as the caveat figure: the daily winner stays strong through March 16, 2026, but its margin shrinks by March 29, 2026.

Supporting tables are in:

- `selected_winners_summary.csv`
- `hourly_pruning_sweep.csv`
- `daily_screening_summary.csv`
- `daily_winner_extension.csv`
- `later_experiments_summary.csv`
- `legacy_5min_baselines.csv`

## Website framing

You should not write the website as purely future work. The repo already contains a working prototype web app.

What is true right now:

- There is an existing Flask + Plotly web UI in `webui/`.
- The app already supports loading data, selecting models, running predictions, and plotting results.
- The current UI is a general Kronos demo, not yet a polished product around your selected SGP2 checkpoints.

Best report wording:

`In addition to the model experiments, the project includes a prototype Flask-based web interface for loading financial time-series data, running Kronos-family predictions, and visualizing forecast outputs. A natural next step is to integrate the selected 5-minute, 1-hour, and 1-day finetuned checkpoints into this interface with a horizon selector, archived benchmark charts, and model-card style explanations of each deployment regime.`

Good concrete future-work items for the website:

1. Add a horizon selector for `5min`, `1h`, and `1d`.
2. Expose the winning finetuned checkpoints directly instead of only generic base models.
3. Show the saved benchmark figures from this report next to live predictions.
4. Add an interval-specific warning panel:
   - `5min`: modest edge
   - `1h`: strongest clean short-horizon evidence
   - `1d`: strongest deployment result, but with later generalization caveats

## Literature comparison

Use the literature section to position the paper by contribution type, not by pretending the benchmarks are directly identical.

| Paper | Main contribution | How your paper differs |
| --- | --- | --- |
| Kronos, 2025/2026 | Financial-market foundation model for candlestick data and downstream financial tasks | Your paper is not a new foundation model. It is an interval-aware fine-tuning and deployment study inside the Kronos ecosystem. |
| Chronos, 2024 | Tokenized pretrained time-series transformers with strong zero-shot and transfer performance across 42 datasets | Your paper is narrower but more deployment-specific: same asset, same repo, but explicit comparison of adaptation mechanisms by timeframe. |
| Temporal Fusion Transformers, 2020/2021 | New interpretable architecture for multi-horizon forecasting | Your paper is not introducing a new architecture. It asks which fine-tuning recipe is best for each horizon once a pretrained backbone already exists. |
| Khaniki and Manthouri, 2024 | Crypto prediction with technical indicators plus Performer/BiLSTM on hourly and daily data | Your paper keeps the backbone family fixed and studies objective choice, pruning, and partial unfreezing instead of building a separate indicator-driven architecture. |
| Fischer and Krauss, 2017/2018 | LSTM-based direction prediction on equities with profitability analysis | Your paper extends this tradition into pretrained financial sequence models and compares deployment behavior across multiple horizons rather than one model on one market setup. |

Suggested paragraph:

`Compared with prior work, our contribution is not a new forecasting architecture but an adaptation study. Kronos and Chronos establish that pretrained sequence models can transfer across time-series tasks, while TFT and related models focus on new multi-horizon architectures. Our work instead asks a narrower practical question: once a pretrained financial backbone is available, which downstream fine-tuning recipe should be used at each forecast horizon? The answer from our BTCUSDT experiments is that the best adaptation mechanism changes with timeframe: native direction-head supervision is most competitive at 5 minutes, moderate pruning is most effective at 1 hour, and path-consistent autoregressive adaptation is strongest at 1 day.` 

## Sources for the literature section

- Kronos: https://arxiv.org/abs/2508.02739
- Chronos: https://arxiv.org/abs/2403.07815
- Temporal Fusion Transformers: https://arxiv.org/abs/1912.09363
- Enhancing Price Prediction in Cryptocurrency Using Transformer Neural Network and Technical Indicators: https://arxiv.org/abs/2403.03606
- Deep learning with long short-term memory networks for financial market predictions: https://www.econstor.eu/handle/10419/157808?locale=en

## Paste-ready final conclusion

`Across the saved BTCUSDT experiments in the repository, the strongest model clearly depends on the target horizon. At 5 minutes, a native direction-head fine-tune on the matching interval produced the best saved short-horizon result, but only with a modest edge. At 1 hour, a 50% pruned direction-head transfer model delivered the clearest improvement in the study, achieving the best accuracy and the only positive PnL in the hourly pruning ladder. At 1 day, a partially unfrozen last-6-layer consecutive-path adapter produced the strongest period-specific deployment result by a wide margin. However, later March and April sweeps indicate that daily generalization remains fragile, so the result should be interpreted as the best observed configuration among tested daily models rather than a universally stable solution. The broader contribution is therefore interval-aware adaptation: within the same pretrained Kronos ecosystem, native classification, sparse transfer, and path-consistent rollout each dominate in different operating regimes.` 
