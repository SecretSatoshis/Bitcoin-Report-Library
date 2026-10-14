// Only production pages load the shared tag; previews and file exports stay local.
if (location.protocol === 'https:' && location.hostname === 'charts.secretsatoshis.com') {
  const tag = document.createElement('script');
  tag.async = true;
  tag.src = 'https://secretsatoshis.com/js/analytics.js?v=1';
  document.head.appendChild(tag);
}
