// Seasonal return charts (MTD and YTD): year selection, Median/Average across the visible
// historical years, the colour ramp, ECharts series options and endpoint labels.
// Pure functions; pages/index.md wires them to its queries.

// Years hidden from the seasonal charts. 2017's magnitude compresses every other
// year into a flat band at the bottom of the plot. It is excluded from the chart
// only — it stays in the CSV, and the Median/Average lines are recomputed below
// over the visible historical years. The current year is plotted separately but
// excluded from those reference aggregates while it is still incomplete.
const HIDDEN_YEARS = ['2017'];

const _isYearCol = (c) => /^\d{4}$/.test(c);

// Off-white for Median, cypherpunk green for Average, Bitcoin-orange for current
// year. Those three are reserved, so the historical palette must avoid orange and
// green entirely or a past year reads as this year.
const _medianColor = '#e4e4ef'; // brand text (legible on dark)
const _averageColor = '#00FF88';
const _currentColor = '#F7931A';

// Historical years use one cool-blue recency ramp: the oldest visible year is
// darkest and the newest is brightest. This keeps every trajectory on the chart
// without suggesting that each year is a separate category. Exact year identity
// comes from the interactive legend and hover focus, so shade is not the only cue.
function _historicalShade(index, total) {
  const t = total <= 1 ? 1 : index / (total - 1);
  const saturation = Math.round(44 + t * 28);
  const lightness = Math.round(34 + t * 40);
  return `hsl(214, ${saturation}%, ${lightness}%)`;
}

export function yearCols(rows, xKey, { includeHidden = false } = {}) {
  if (!rows?.length) return [];
  return Object.keys(rows[0])
    .filter((c) => c !== xKey && _isYearCol(c))
    .filter((c) => includeHidden || !HIDDEN_YEARS.includes(c))
    .sort();
}

// The current year is the newest year column present in the data — including a
// hidden one, so the label stays honest even if the newest year were hidden.
export function currentYearFrom(rows, xKey) {
  const all = yearCols(rows, xKey, { includeHidden: true });
  return all.length ? all[all.length - 1] : '';
}

// Recompute Median/Average across the visible *historical* years. The newest year
// remains plotted, but is excluded from the aggregates while it is incomplete.
// The CSV ships precomputed columns covering every year including hidden/current
// ones, so those aggregate columns are deliberately ignored here.
export function withAggregates(rows, xKey) {
  const plottedYears = yearCols(rows, xKey);
  if (!rows?.length || !plottedYears.length) return [];
  const currentYear = currentYearFrom(rows, xKey);
  const historicalYears = plottedYears.filter((y) => y !== currentYear);
  return rows.map((r) => {
    const out = { [xKey]: r[xKey] };
    for (const y of plottedYears) out[y] = r[y];
    const vals = historicalYears
      .map((y) => r[y])
      .filter((v) => v != null && !isNaN(v))
      .map(Number)
      .sort((a, b) => a - b);
    if (vals.length) {
      const mid = Math.floor(vals.length / 2);
      out.Median = vals.length % 2 ? vals[mid] : (vals[mid - 1] + vals[mid]) / 2;
      out.Average = vals.reduce((s, v) => s + v, 0) / vals.length;
    } else {
      out.Median = null;
      out.Average = null;
    }
    return out;
  });
}

export function buildColorMap(years, currentYear) {
  // Colour follows the year, not its position in the list: the newest historical
  // years take the first palette slots, so a year keeps its hue across both charts
  // and does not repaint when the series count changes.
  const historical = years.filter((n) => _isYearCol(n) && n !== currentYear).sort();
  const shades = new Map(historical.map((yr, i) => [yr, _historicalShade(i, historical.length)]));
  return Object.fromEntries(
    years.map((name) => [
      name,
      name === 'Median'
        ? _medianColor
        : name === 'Average'
          ? _averageColor
          : name === currentYear
            ? _currentColor
            : (shades.get(name) ?? _medianColor),
    ]),
  );
}

// Per-series presentation: recent historical years gain a little weight and
// opacity, while current year + Median + Average remain the strongest references.
// Hover restores a historical line to full opacity and reveals its year at the
// endpoint; the scrollable legend still names every year and can toggle any line.
export function buildSeriesOptions(years, currentYear) {
  const historical = years.filter((name) => _isYearCol(name) && name !== currentYear).sort();
  const historicalRank = new Map(historical.map((yr, i) => [yr, i]));
  const legendData = [currentYear, ...historical.slice().reverse(), 'Median', 'Average'].filter(
    Boolean,
  );

  return {
    legend: {
      show: true,
      type: 'scroll',
      data: legendData,
      top: 4,
      left: 'center',
      right: 20,
      itemWidth: 18,
      itemHeight: 3,
      itemGap: 14,
      textStyle: {
        color: '#9090a8',
        fontFamily: 'JetBrains Mono',
        fontSize: 10,
      },
      pageIconColor: '#F7931A',
      pageIconInactiveColor: '#3a3a50',
      pageTextStyle: { color: '#9090a8' },
    },
    grid: { top: 62, containLabel: true },
    tooltip: { trigger: 'axis', confine: true },
    series: years.map((name) => {
      const wide = name === 'Median' || name === 'Average' || name === currentYear;
      const rank = historicalRank.get(name);
      const isHistorical = rank != null;
      const recency = rank == null || historical.length <= 1 ? 1 : rank / (historical.length - 1);
      const historicalOpacity = 0.76 + recency * 0.16;
      const historicalColor = isHistorical ? _historicalShade(rank, historical.length) : undefined;
      const lineStyle = {
        width: wide ? 2.5 : 1 + recency * 0.7,
        opacity: isHistorical ? historicalOpacity : 1,
      };
      if (name === 'Median') lineStyle.type = 'dashed';
      return {
        z: wide ? 3 : 1,
        lineStyle,
        triggerEvent: isHistorical ? 'line' : false,
        endLabel: isHistorical ? { show: false } : undefined,
        emphasis: {
          focus: 'series',
          lineStyle: { width: wide ? 3.5 : 2.5, opacity: 1 },
          endLabel: isHistorical
            ? {
                show: true,
                formatter: '{a}',
                color: historicalColor,
                backgroundColor: 'rgba(8, 8, 12, 0.9)',
                borderRadius: 3,
                padding: [3, 5],
                align: 'right',
                distance: 4,
                fontFamily: 'JetBrains Mono',
                fontSize: 10,
                fontWeight: 600,
              }
            : undefined,
        },
        blur: { lineStyle: { opacity: 0.12 } },
      };
    }),
  };
}

export function fmtUsd(n) {
  if (n == null || isNaN(n)) return '';
  return '$' + Math.round(Number(n)).toLocaleString();
}
export function buildLatestPoints(rows, xKey, currentYear) {
  if (!rows?.length || !currentYear) return { current: [], average: [] };
  // last row where current year col is not null (= today)
  let currentRow = null;
  for (let i = rows.length - 1; i >= 0; i--) {
    if (rows[i][currentYear] != null) {
      currentRow = rows[i];
      break;
    }
  }
  // last row of the full Average series (end of month / end of year)
  let endRow = null;
  for (let i = rows.length - 1; i >= 0; i--) {
    if (rows[i]['Average'] != null) {
      endRow = rows[i];
      break;
    }
  }
  return {
    current: currentRow
      ? [
          {
            x: currentRow[xKey],
            y: currentRow[currentYear],
            label: `${currentYear} · ${fmtUsd(currentRow[currentYear])}`,
          },
        ]
      : [],
    average: endRow
      ? [
          {
            x: endRow[xKey],
            y: endRow['Average'],
            label: `Average · ${fmtUsd(endRow['Average'])}`,
          },
        ]
      : [],
  };
}
