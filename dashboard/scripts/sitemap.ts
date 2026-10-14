// The template date records the last page-content change. New verified data
// releases advance it automatically; rebuilding the same release does not.
export function updateSitemap(
  template: string,
  releaseGeneratedAt: string,
): string {
  const contentDate = template.match(/<lastmod>(\d{4}-\d{2}-\d{2})<\/lastmod>/)?.[1];
  const releaseTime = Date.parse(releaseGeneratedAt);
  if (
    !contentDate ||
    !Number.isFinite(Date.parse(contentDate)) ||
    !Number.isFinite(releaseTime)
  )
    throw new Error("Invalid sitemap modification date");
  const releaseDate = new Date(releaseTime).toISOString().slice(0, 10);
  const lastmod = releaseDate > contentDate ? releaseDate : contentDate;
  return template.replace(
    `<lastmod>${contentDate}</lastmod>`,
    `<lastmod>${lastmod}</lastmod>`,
  );
}
