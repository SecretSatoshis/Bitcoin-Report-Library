// Seasonal return charts (MTD and YTD): year selection, Median/Average across the visible
// historical years, the colour ramp, ECharts series options and endpoint labels.
// Pure functions; pages/index.md wires them to its queries.

// Years left off the seasonal charts (still in the CSV). 2017's scale flattens every
// other year.
const HIDDEN_YEARS = ['2017'];

const _isYearCol = (c) => /^\d{4}$/.test(c);

// Reserved colours for Median, Average and the current year. The historical palette
// avoids orange and green so no past year reads as this year.
const _medianColor = '#e4e4ef';
const _averageColor = '#00FF88';
const _currentColor = '#F7931A';

// Past years share one blue ramp, oldest darkest; the legend and hover identify each year.
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

// The newest year column, counting hidden years.
export function currentYearFrom(rows, xKey) {
  const all = yearCols(rows, xKey, { includeHidden: true });
  return all.length ? all[all.length - 1] : '';
}

// Median and Average across the visible past years; the current year is plotted but
// left out of them.
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

// Current year, Median and Average are the strongest lines; recent past years are a
// little heavier than older ones. Hovering a line brings it forward and labels its year.
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
  // Latest row with a current-year value (the report date)
  let currentRow = null;
  for (let i = rows.length - 1; i >= 0; i--) {
    if (rows[i][currentYear] != null) {
      currentRow = rows[i];
      break;
    }
  }
  // Last row of the Average series (end of month or year)
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
