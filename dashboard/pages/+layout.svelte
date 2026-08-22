<script>
	import '@evidence-dev/tailwind/fonts.css';
	import '../app.css';
	import { EvidenceDefaultLayout } from '@evidence-dev/core-components';
	export let data;
</script>

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
