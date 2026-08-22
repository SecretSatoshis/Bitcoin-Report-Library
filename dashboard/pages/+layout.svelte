<script>
	import '@evidence-dev/tailwind/fonts.css';
	import '../app.css';
	import { EvidenceDefaultLayout } from '@evidence-dev/core-components';
	export let data;

	let menuOpen = false;
	let navToggle;
	const copyrightYear = new Date().getUTCFullYear();

	function closeMenu() {
		menuOpen = false;
	}

	function handleKeydown(event) {
		if (event.key === 'Escape' && menuOpen) {
			closeMenu();
			navToggle?.focus();
		}
	}
</script>

<svelte:window on:keydown={handleKeydown} />

<div class="brand-shell" id="top">
	<a class="brand-skip-link" href="#bitcoin-snapshot">Skip to dashboard</a>

	<nav class="brand-site-nav" aria-label="Primary navigation">
		<a href="https://secretsatoshis.com/" class="brand-nav-logo" aria-label="Secret Satoshis home">
			<span class="brand-accent">//</span> <span>SECRET SATOSHIS</span>
		</a>
		<button
			bind:this={navToggle}
			class="brand-nav-toggle"
			type="button"
			aria-label={menuOpen ? 'Close menu' : 'Open menu'}
			aria-expanded={menuOpen}
			aria-controls="brandNavLinks"
			on:click={() => (menuOpen = !menuOpen)}
		>≡</button>
		<ul class:open={menuOpen} class="brand-nav-links" id="brandNavLinks">
			<li><a href="https://secretsatoshis.com/" on:click={closeMenu}>Home</a></li>
			<li><a href="https://newsletter.secretsatoshis.com/p/start-here" target="_blank" rel="noopener noreferrer" on:click={closeMenu}>Start Here</a></li>
			<li><a href="https://chatgpt.com/g/g-BZXtVdU6M-agent-21" target="_blank" rel="noopener noreferrer" on:click={closeMenu}>Agent 21</a></li>
			<li><a href="https://charts.secretsatoshis.com/" on:click={closeMenu}>Charts</a></li>
			<li><a href="https://dashboard.secretsatoshis.com/" aria-current="page" on:click={closeMenu}>Dashboard</a></li>
			<li><a href="https://newsletter.secretsatoshis.com/" target="_blank" rel="noopener noreferrer" on:click={closeMenu}>Newsletter</a></li>
		</ul>
	</nav>

	<div class="dashboard-evidence">
		<EvidenceDefaultLayout
			{data}
			title="Secret Satoshis"
			logo="/logo.png"
			homePageName="Dashboard"
			fullWidth={true}
			hideSidebar={true}
			hideTOC={true}
			builtWithEvidence={false}
		>
			<slot slot="content" />
		</EvidenceDefaultLayout>
	</div>

	<footer class="brand-footer brand-section-divider">
		<div class="brand-footer-inner">
			<div class="brand-footer-top">
				<div>
					<a href="https://secretsatoshis.com/" class="brand-footer-brand" aria-label="Secret Satoshis home">
						<span class="brand-accent">//</span> SECRET SATOSHIS
					</a>
					<p class="brand-footer-tagline">AI-native Bitcoin market intelligence</p>
				</div>
				<div>
					<h2 class="brand-footer-col-title">Platform</h2>
					<ul class="brand-footer-links">
						<li><a href="https://newsletter.secretsatoshis.com/p/start-here" target="_blank" rel="noopener noreferrer">Start Here</a></li>
						<li><a href="https://chatgpt.com/g/g-BZXtVdU6M-agent-21" target="_blank" rel="noopener noreferrer">Agent 21</a></li>
						<li><a href="https://newsletter.secretsatoshis.com/" target="_blank" rel="noopener noreferrer">Newsletter</a></li>
					</ul>
				</div>
				<div>
					<h2 class="brand-footer-col-title">Data</h2>
					<ul class="brand-footer-links">
						<li><a href="https://dashboard.secretsatoshis.com/">Market Dashboard</a></li>
						<li><a href="https://charts.secretsatoshis.com/">Chart Library</a></li>
						<li><a href="https://github.com/SecretSatoshis" target="_blank" rel="noopener noreferrer">GitHub</a></li>
					</ul>
				</div>
				<div>
					<h2 class="brand-footer-col-title">Connect</h2>
					<ul class="brand-footer-links">
						<li><a href="https://x.com/SecretSatoshis" target="_blank" rel="noopener noreferrer">𝕏 @SecretSatoshis</a></li>
						<li><a href="https://www.linkedin.com/company/secretsatoshis/" target="_blank" rel="noopener noreferrer">LinkedIn</a></li>
						<li><a href="https://treybrunson.com/" target="_blank" rel="noopener noreferrer">TreyBrunson.com</a></li>
					</ul>
				</div>
			</div>
			<div class="brand-footer-bottom">
				<span>© {copyrightYear} Secret Satoshis · Don't trust. Verify.</span>
			</div>
		</div>
	</footer>
</div>
