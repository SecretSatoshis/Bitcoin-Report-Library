// Only production pages load the shared tag; local previews do not collect visits.
if (location.protocol === 'https:' && location.hostname === 'dashboard.secretsatoshis.com') {
  const tag = document.createElement('script');
  tag.async = true;
  tag.src = 'https://secretsatoshis.com/js/analytics.js?v=1';
  document.head.appendChild(tag);
}
