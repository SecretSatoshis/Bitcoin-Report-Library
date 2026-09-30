// Initialize only from the same-origin dashboard that owns this iframe.
let initialized=false;
addEventListener('message',event=>{
 if(initialized||event.source!==parent||event.origin!==location.origin||event.data?.type!=='ss-chart-init')return;
 const payload=event.data.payload;
 if(payload?.schemaVersion!==2||payload.id!==document.body.dataset.chartId)return;
 initialized=true;document.getElementById('chart-data').textContent=JSON.stringify(payload);
 const script=document.createElement('script');script.src="assets/renderer.1660362d9b3f7396.js";
 const fail=error=>parent.postMessage({type:'ss-chart-error',id:payload.id,message:error.message||String(error)},location.origin);
 script.onerror=()=>fail(new Error('Shared renderer could not load'));
 script.onload=async()=>{try{
   const chart=await window.SecretSatoshisChart.ready;
   parent.postMessage({type:'ss-chart-ready',id:payload.id,reportDate:chart.payload.reportDate},location.origin);
 }catch(error){fail(error);}};
 document.body.append(script);
});
