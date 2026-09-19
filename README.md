# AI A股短中线交易辅助系统

本项目是个人研究、回测、风控和人工确认交易计划工具，不是自动下单机器人，也不承诺收益。

v0.1 目标：

- 过滤沪深主板 + 创业板可交易股票池
- 计算中线评分和短线择时信号
- 应用稍激进但有刹车的风控规则
- 生成 Markdown 日报、Excel 信号表和基础回测/烟雾测试摘要

默认不做：

- 自动下单
- 券商实盘账户接入
- 融资、杠杆、融券
- 科创板
- 黑箱深度学习买卖点

## 本地运行

安装开发依赖：

```bash
pip install -e ".[dev]"
```

运行测试：

```bash
pytest -v
```

### v0.2 真实公开数据运行

默认 provider 仍是 `demo`，用于离线测试：

```powershell
python -m a_share_ai.cli daily --config configs/default.yaml --date 2026-07-29 --provider demo
```

真实公开数据需要安装 AkShare 并显式选择 provider：

```powershell
pip install -e ".[real-data]"
```

开发并运行真实数据测试环境可使用 `pip install -e ".[dev,real-data]"`；核心自动测试本身仍完全离线，不要求安装 AkShare。


真实公开数据运行还依赖网络/API 可用性；若 AkShare 未安装、网络不可用或 provider API 失败，命令会明确报告 provider failure，或将数据质量标记为 `quality_status=blocked`；不会静默回退到 `demo`，也不会输出可买的真实数据候选。

```powershell
python -m a_share_ai.cli daily --config configs/default.yaml --date 2026-07-29 --provider akshare --current-drawdown -0.03 --current-holdings 2 --daily-new-buys 0
```

输出中的数据标识必须这样理解：

- `REAL PUBLIC MARKET DATA / 公开市场数据`：来自公开行情接口，仍可能延迟、缺字段或接口变化。
- `SYNTHETIC DEMO DATA / 合成演示数据`：合成演示数据，只用于离线演示和测试。
- `quality_status=blocked`：数据过期、未来数据、缺关键字段或 provider 失败，系统不会给出可买建议。

v0.2 只接受 `adjustment: qfq`（前复权）；配置为 `hfq` 或其他值会在加载配置或创建 provider 时明确失败。来源中晚于报告日的行会先从特征计算中删除，同时把整次数据质量标记为 `blocked`。

当前 AkShare 指数日线接口不提供成交额；指数标准 schema 中 `amount=0.0` 明确表示 unavailable，指数行情目前不参与个股候选、仓位或买卖判断。

#### v0.2.1 AkShare no-spot fallback

Some networks can open `https://www.eastmoney.com/` while blocking or resetting Eastmoney data hosts such as `82.push2.eastmoney.com`. v0.2.1 treats AkShare spot data as optional metadata enrichment. Code/name and listing dates come from AkShare list sources; 20-day turnover is derived from historical K-line `amount`.

For a small real-data smoke before attempting the full market:

```powershell
python scripts/smoke_akshare_limited.py --date 2026-07-30 --limit 5
```

This smoke writes to `outputs/smoke-akshare-limited/`. If it fails, read the `Data provider failed:` or `quality_status=blocked` reason; the system must not silently replace real public data with demo data.


#### v0.3 AkShare CSV cache

v0.3 can cache AkShare daily bars under `data/raw/akshare/daily/{code}.csv` and records per-symbol update status in `data/raw/akshare/manifest.csv`.

The cache is a reliability and speed aid, not a data-quality bypass:

- real public data never silently falls back to demo;
- stale cache can still make `quality_status=blocked`;
- remote failure without cache must not produce buyable candidates;
- `manifest.csv` should be checked when a run has missing or stale symbols.

Cache-aware limited smoke:

```powershell
python scripts/smoke_akshare_limited.py --date 2026-08-01 --limit 5 --cache-enabled --cache-dir data/raw/akshare-smoke
```

This mode verifies the v0.3 CSV cache and `manifest.csv` failure/status tracking. It remains `LIMITED OPERATIONAL SMOKE / NOT A RECOMMENDATION`; a blocked run is expected when public endpoints are unavailable.

#### v0.4 historical K-line fallback

v0.4 tries multiple public daily-history paths before giving up on a symbol. The primary path remains AkShare `stock_zh_a_hist`; when that fails, the provider tries the raw Eastmoney K-line fallback and AkShare Tencent history `stock_zh_a_hist_tx(symbol=code, adjust="qfq")` when available.

Fallback data must include real turnover `amount`. Sources without `amount` are rejected because liquidity gates depend on turnover.

If all history sources fail, the run remains blocked and no buyable recommendation is produced. Check `manifest.csv` for the per-symbol source failure details when cache mode is enabled.

#### v0.5 Real-data coverage smoke

v0.5 adds a bounded coverage diagnostic for real-public AkShare cache runs. Start with 20 symbols and a run-specific cache directory:

```powershell
python scripts/smoke_akshare_limited.py --date 2026-07-31 --limit 20 --cache-enabled --cache-dir data/raw/akshare-smoke-v05 --coverage-report
```

This writes a coverage report under `outputs/smoke-akshare-limited/coverage/` with cache-manifest status counts, success rate, latest cached date range, and grouped failure reasons. `--coverage-report` is a cache-manifest diagnostic: use it with `--cache-enabled` and a run-specific cache directory as shown. If cache is disabled but an existing cache directory is supplied, the report can summarize an older manifest and must not be treated as proof of the current non-cached run. Try larger limits such as 50 only after the 20-symbol run is stable.

The coverage smoke remains `LIMITED OPERATIONAL SMOKE / NOT A RECOMMENDATION`. It is for data-ingestion diagnostics only and must not be read as a trade recommendation.
本项目不自动下单、不接券商账户、不使用融资融券或杠杆，不承诺收益。

生成离线示例日报：

```bash
a-share-ai daily --config configs/default.yaml --date 2026-07-29
```
To supply the current portfolio risk state instead of the explicit offline demo default:

```bash
a-share-ai daily --config configs/default.yaml --date 2026-07-29 --current-drawdown -0.16 --current-holdings 2 --daily-new-buys 1
```

Any supplied field switches the report to a caller-supplied risk state; omitted risk fields safely default to zero. Reports record the exact effective values. `EXIT RULE STATUS: RESERVED / NOT ACTIVE`: stop-loss and trailing-profit settings are not executed by v0.1 candidate generation.


输出文件：

- `outputs/reports/2026-07-29_daily_report.md`
- `outputs/signals/2026-07-29_signals.xlsx`
- `outputs/backtest/backtest_summary.md`

## 风险边界

本工具只做研究辅助和人工确认交易计划，不自动下单、不接券商账户、不使用融资杠杆。

## 演示数据与输出标识

默认 CLI 使用截至 `2026-07-29` 的固定合成行情；其中股票代码和名称只用于演示表结构，不表示对应真实证券在该日期的真实行情或推荐。Markdown 日报、Excel 的“说明”工作表和烟雾测试摘要都会显示醒目的 `SYNTHETIC DEMO DATA / 合成演示数据` 标识与最新数据日期。

当合成演示数据的报告日期晚于 `2026-07-29` 时，系统仍可生成供检查的文件，但所有候选都会变为“仅观察”、建议仓位为 0，不生成可买建议。

## 风控实现范围

`run_daily_pipeline(..., risk_state=...)` 可以接收当前组合回撤、持仓数和当日新增买入数。候选按排名顺序逐一消耗每日新增买入额度和最大持仓额度，不会把同一个允许买入结论复制给全部候选。

未传入 `RiskState` 时，离线示例使用“空仓、当日未买入、无回撤”的默认演示状态，输出会明确标识该状态并未连接真实组合。

配置中的硬止损和移动止盈阈值仅为未来持仓管理模块预留；v0.1 的每日候选生成流程没有持仓成本/最高价状态，因此不会声称这些退出规则已经执行。

## 烟雾测试边界

`outputs/backtest/backtest_summary.md` 是同日报告链路、交易成本和指标输出的基础烟雾测试摘要，不是历史多日期策略回测，也不能作为策略有效性或未来收益的证据。

系统始终只生成研究辅助文件：不自动下单、不接券商账户、不使用融资、融券或杠杆。
