<script>
  import { onMount } from 'svelte';
  import colors from './chart-colors.json';
  import historicalEvents from './chart-events.json';
  export let rows = [];
  export let candles = [];
  export let cases = [];
  export let reportDate = '';
  // The forecast year published with the case levels (price_outlook.csv outlook_year).
  export let outlookYear = '';
  let frame,
    frameReady = false,
    height = 920,
    payload = null,
    error = '';
  const metrics = [
    ['price_close', 'BTC Price', 'Bitcoin Price'],
    ['realized_price', 'Realized Price', 'Realized Price'],
    ['sth_realized_price', 'STH Realized Price', 'STH Realized Price'],
    ['realizedcap_multiple_3', '3x Realized Price', '3× Realized Price'],
    ['90_day_ma_price_close', '3-month MA', '3-month MA'],
    ['364_day_ma_price_close', '1-year MA', '1-year MA'],
    ['200_week_ma_price_close', '200-week MA', '200-week MA'],
  ];
  const iso = (value) => new Date(value).toISOString().slice(0, 10);
  const number = (value) =>
    value == null || !Number.isFinite(Number(value)) ? null : Number(value);
  function futureCalendar(start, end, interval = 'daily') {
    const result = [],
      date = new Date(start + 'T00:00:00Z');
    while (true) {
      if (interval === 'monthly') date.setUTCMonth(date.getUTCMonth() + 1, 1);
      else date.setUTCDate(date.getUTCDate() + (interval === 'weekly' ? 7 : 1));
      const label = iso(date);
      if (label > end) break;
      result.push(label);
    }
    return result;
  }
  function buildPayload() {
    // Inputs still loading: not an error, just nothing to draw yet.
    if (!rows?.length || !candles?.length || !cases?.length || !reportDate) return null;
    const history = rows.filter((r) => iso(r.date) <= reportDate),
      x = history.map((r) => iso(r.date));
    if (x.at(-1) !== reportDate)
      throw new Error(
        `price history ends ${x.at(-1) ?? 'before any data'}, not on the report date ${reportDate}`,
      );
    const year = String(outlookYear || reportDate.slice(0, 4)),
      end = year + '-12-31';
    const series = metrics.map(([id, key, name]) => ({
      id,
      name,
      axis: 'right',
      role: id === 'price_close' ? 'highlight' : 'normal',
      color: colors[id],
      lineWidth: id === 'price_close' ? 3 : 2,
      lineStyle: 'solid',
      opacity: 1,
      start: 0,
      values: history.map((r) => number(r[key])),
    }));
    // Keep the legend stable while hovering; rank models by report-date value.
    series.sort((a, b) => {
      if (a.id === 'price_close') return -1;
      if (b.id === 'price_close') return 1;
      return (b.values.at(-1) ?? -Infinity) - (a.values.at(-1) ?? -Infinity);
    });
    const events = [
      ...historicalEvents.flatMap((event) =>
        event.dates.map((date) => ({ date, name: event.name })),
      ),
      { date: year + '-01-01', name: year + ' Start' },
    ]
      .filter((e) => e.date >= x[0] && e.date <= reportDate)
      .sort((a, b) => a.date.localeCompare(b.date));
    const payload = {
      schemaVersion: 2,
      id: 'dashboard-price-outlook',
      family: 'timeseries',
      title: `Secret Satoshis ${year} Price Outlook`,
      description: 'Bitcoin price, on-chain valuation models and moving averages.',
      category: 'Price Outlook',
      source: 'Bitview',
      reportDate,
      coverage: `${x[0]}/${reportDate}`,
      axisKind: 'time',
      axes: { right: { unit: 'USD', label: 'Bitcoin Price (USD)', mode: 'linear' } },
      gridlines: false,
      defaultRange: '4Y',
      defaultPresentation: 'candles',
      defaultInterval: 'weekly',
      x: [...x, ...futureCalendar(x.at(-1), end)],
      series,
      events,
      unavailable: [],
      readingPoint: reportDate,
      note: 'Daily observations.',
      rangeEndDate: end,
      referenceLines: cases.map((c) => ({ name: c.name, price: Number(c.price), color: c.color })),
      candleViews: {},
    };
    const byDate = new Map(history.map((r, i) => [x[i], i]));
    for (const interval of ['daily', 'weekly', 'monthly']) {
      const source = candles.filter(
        (r) =>
          r.interval === interval &&
          iso(r.period_start) >= x[0] &&
          iso(r.observation_date) <= reportDate,
      );
      if (!source.length) continue;
      const periods = source.map((r) => ({
        time: iso(r.period_start),
        periodEnd: iso(r.period_end),
        observationDate: iso(r.observation_date),
        complete: r.complete === true || String(r.complete).toLowerCase() === 'true',
        open: Number(r.Open),
        high: Number(r.High),
        low: Number(r.Low),
        close: Number(r.Close),
      }));
      if (periods.at(-1).observationDate !== reportDate)
        throw new Error(
          `${interval} candles end ${periods.at(-1).observationDate}, not on the report date ${reportDate}`,
        );
      const dates = periods.map((c) => c.time),
        calendar = [...dates, ...futureCalendar(dates.at(-1), end, interval)];
      const selected = series.map((s) => ({
        ...s,
        values: periods.map((c) => s.values[byDate.get(c.observationDate)] ?? null),
      }));
      const periodEvents = events.flatMap((event) => {
        const candle = periods.find((c) => c.time <= event.date && c.observationDate >= event.date);
        return candle ? [{ ...event, date: candle.time, originalDate: event.date }] : [];
      });
      payload.candleViews[interval] = {
        x: calendar,
        series: selected,
        candles: periods,
        events: periodEvents,
        readingPoint: dates.at(-1),
        interval,
      };
      if (interval === 'daily') {
        payload.candleViews[interval].offset = x.indexOf(dates[0]);
        payload.candleViews[interval].ohlc = periods.map((c) => [c.open, c.high, c.low, c.close]);
      }
    }
    return payload;
  }
  // A prerendered iframe can load before Svelte hydrates and attaches on:load.
  onMount(() => {
    if (frame?.contentDocument?.readyState === 'complete') frameReady = true;
  });
  // Inconsistent inputs become a visible error instead of an endless loading spinner, and
  // the newsletter exporter fails on the same marker instead of waiting for a timeout.
  function computePayload() {
    try {
      return { payload: buildPayload(), error: '' };
    } catch (err) {
      return { payload: null, error: err.message };
    }
  }
  $: ({ payload, error } = (rows, candles, cases, reportDate, outlookYear, computePayload()));
  $: if (frameReady && payload)
    frame.contentWindow.postMessage({ type: 'ss-chart-init', payload }, window.location.origin);
  function receive(event) {
    if (event.source !== frame?.contentWindow || event.origin !== window.location.origin) return;
    const message = event.data;
    if (
      message?.type === 'ss-chart-size' &&
      message.id === 'dashboard-price-outlook' &&
      Number.isFinite(message.height) &&
      message.height >= 300 &&
      message.height <= 4000
    )
      height = message.height;
  }
</script>

<svelte:window on:message={receive} />
{#if error}
  <p class="price-outlook-error" role="alert" data-price-outlook-error>
    Price outlook unavailable: {error}.
  </p>
{/if}
<iframe
  bind:this={frame}
  on:load={() => (frameReady = true)}
  src="/shared-chart/frame.html"
  title="Interactive Bitcoin price outlook"
  class="price-outlook-frame"
  style:height="{height}px"
  data-price-outlook-frame
></iframe>

<style>
  .price-outlook-error {
    margin: 0 0 12px;
    padding: 10px 14px;
    border: 1px solid #ff3b30;
    color: #ff3b30;
    font-size: 13px;
  }
  .price-outlook-frame {
    display: block;
    width: 100%;
    border: 0;
    background: #08080c;
    min-height: 700px;
    color-scheme: dark;
  }
</style>
