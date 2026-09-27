// Initialize only from the same-origin dashboard that owns this iframe.
let initialized=false;
addEventListener('message',event=>{
 if(initialized||event.source!==parent||event.origin!==location.origin||event.data?.type!=='ss-chart-init')return;
 const payload=event.data.payload;
 if(payload?.schemaVersion!==2||payload.id!=='dashboard-price-outlook')return;
 initialized=true;document.getElementById('chart-data').textContent=JSON.stringify(payload);
 const script=document.createElement('script');script.src="assets/renderer.9195cbfc58bb7407.js";document.body.append(script);
});
