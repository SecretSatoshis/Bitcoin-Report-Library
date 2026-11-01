# BRK v0.13.0 integration audit — November 1, 2026

The official [v0.13.0 changelog](https://github.com/bitcoinresearchkit/mono/blob/v0.13.0/docs/CHANGELOG.md)
is dated **October 30, 2026**. The source change removing index aliases is
[9706ae08](https://github.com/bitcoinresearchkit/mono/commit/9706ae08), authored October 24,
2026. `day1` already existed; v0.13 removes the accepted `dateindex` alias. The tagged release
commit is `1a19c5e969761a0ce446483767da8c765e4aa895`.

The Report Library's last successful generation was October 30 at 06:14 UTC; the first
observed failure was October 31 at 00:25 UTC (October 30 evening in New York). This bounds
when the hosted change affected us; it does not establish the exact server deployment time.
The live `/version` and OpenAPI checked for this audit identify revision `4f961b7`.
The changelog is explicitly AI-generated, so mappings were checked against tagged source,
the live catalog, OpenAPI and real responses, rather than inferred from its prose alone.

## Changes affecting this integration

| Previously requested | Current handling |
| --- | --- |
| `index=dateindex` | Canonical `index=day1`, for both bulk on-chain data and OHLC. |
| `hash_rate` | Request `hashrate`; retain the published `hash_rate` column and H/s unit. The estimate also changed: the trailing window now weights each block by its own difficulty. |
| `supply` | Request `circulating_supply`; retain the BTC-valued `supply` column. |
| `addr_count` | Request `funded_address_count`. Funded means an address holding an unspent output, not a count of all addresses ever observed. |
| `active_addrs_average_24h` | Request `active_address_count_avg_24h`: trailing-24h arithmetic mean of per-block active addresses, sending or receiving. The legacy published key `daily_active_addresses_sending` stays compatible; its name must not be read as distinct daily senders. |
| Open-ended UTXO age supply/transfer cohorts | Sum the current disjoint `age_<band>_supply` and `age_<band>_transfer_volume_cumulative` bands, preserving each cutoff. |
| Open-ended address-balance counts | Sum the current disjoint `balance_<band>_address_count` bands. Historical “over” cutoffs mean **greater than or equal**, as the old source specified. Zero-balance addresses are excluded from the 1+ sat count. |
| `puell_multiple` | Removed from the server. Recompute BRK Studio's current formula: `365 * subsidy_sum_24h_usd / subsidy_sum_1y_usd`. This uses subsidy, not fees or total miner revenue. |
| `inflation_rate` | Identifier and percent unit remain, but BRK changed the definition to scheduled block subsidy × 52,560 / circulating supply. Previously it was trailing-year supply growth. We follow the new provider definition; that older quantity is available upstream as `supply_delta_1y_percent`. |

**Puell is a definition change as well as a removal.** v0.12.2 computed block USD subsidy
divided by its trailing-year per-block mean. v0.13's own Studio uses the rolling day/year
formula above. See the [old computation](https://github.com/bitcoinresearchkit/mono/blob/v0.12.2/crates/bitview_plugin_indicators/src/vecs/compute.rs)
and the [current Studio formula](https://github.com/bitcoinresearchkit/mono/blob/v0.13.0/showcases/studio/presets.js).
Undefined denominators and upstream nulls remain missing; they are not replaced by zero.

## Cohort definitions and numerical safeguards

The [current age ranges](https://github.com/bitcoinresearchkit/mono/blob/v0.13.0/crates/bitview_cohort/src/age_range.rs)
and the old [under-age](https://github.com/bitcoinresearchkit/mono/blob/v0.12.2/crates/bitview_cohort/src/under_age.rs)
and [over-amount](https://github.com/bitcoinresearchkit/mono/blob/v0.12.2/crates/bitview_cohort/src/over_amount.rs)
definitions establish half-open age ranges and inclusive lower balance cutoffs. Months use
30 days and years 365; 18 months is 365 + 180 days. Rebuilt groups include every split band,
including 4–5 months, 5–6 months, 6–7 years, 7–8 years, 10–12 years, 12–15 years and 15+ years.
A missing component makes its aggregate missing. Cumulative coin-transfer totals remain
cumulative; downstream calculations continue deriving flows from differences of daily totals.

BRK also changed some underlying computations: inflation, undefined ratios becoming null,
profitability boundaries, address type coverage (P2A removed from address totals), and price
handling for periods without blocks. Historical raw values may consequently differ after
BRK's recomputation. USD/BTC values, H/s hash rate, percentages and ratios are not rescaled
merely because their type or path changed. The daily-flow NVT and hash-price calculations
already run locally and therefore do not depend on the removed upstream indicators.

Hash rate now sums each block's difficulty-implied work rate in the trailing 24 hours and
divides by 144; the old estimate multiplied the window's block count by only the final
block's difficulty. Around adjustments this changes historical values: on October 29,
our previous release had about 1,010.41 EH/s and the new upstream series about 966.67 EH/s.
This is BRK's documented correction, not an H/s unit conversion. On that same date our
rebuilt under/over-one-year supplies agree with the previous release to publication precision,
their sum matches circulating supply, and the funded-address count is one lower after BRK's
address coverage change. The source close is unchanged.

## Verification

All 65 configured legacy columns resolve to **86 unique current inputs**. Every input was
checked in the live catalog for `day1` support and fetched successfully in bounded CSV requests.
The [official OpenAPI](https://bitview.space/openapi.json) continues to support our exact
`series`, `index`, `start` and `format` parameters; current CSV timestamps remain Unix seconds.
The existing OHLC routes, date arrays and half-open range metadata remain supported.

`tests/fixtures/brk-v0.13-contract.json` records the source URLs, observed revision, OpenAPI
hash and relevant type/index/description metadata. `brk-v0.13-sample.csv` holds three observed
rows for regression replay, including an unfinished current UTC day; production's report-date
cutoff excludes unfinished days. Tests check the request contract, cohort partitions, NaN
propagation, units, Puell formula and stable output columns. No CSV field names or chart
consumer contracts are renamed by this migration.
