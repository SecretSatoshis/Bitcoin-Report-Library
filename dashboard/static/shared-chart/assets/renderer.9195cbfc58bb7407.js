/* Shared renderer. Market calculations belong to the Python data adapters. */
(() => {
  'use strict';
  const THEME={bg:'#08080c',surface:'#0e0e16',border:'#2a2a42',text:'#e4e4ef',dim:'#9090a8',accent:'#F7931A',candleUp:'#78C99A',candleDown:'#E87878'};
  const $=id=>document.getElementById(id);
  // Independent of any series: hiding Bitcoin does not hide the historical events.
  class EventLines {
    constructor(events) {
      this.events = events;
      this.visible = true;
      this.views = [{zOrder:()=>'bottom', renderer:()=>({draw:target=>this.draw(target)})}];
    }
    attached({chart, requestUpdate}) { this.chart=chart; this.requestUpdate=requestUpdate; }
    detached() { this.chart=null; this.requestUpdate=null; }
    paneViews() { return this.views; }
    setVisible(value) { this.visible=value; this.requestUpdate?.(); }
    draw(target) {
      if (!this.visible || !this.chart) return;
      target.useMediaCoordinateSpace(({context:ctx, mediaSize}) => {
        let lastLabel = -Infinity;
        for (const event of this.events) {
          const x=this.chart.timeScale().timeToCoordinate(event.date);
          if (x===null || x<0 || x>mediaSize.width) continue;
          ctx.save();
          ctx.strokeStyle=THEME.border; ctx.lineWidth=1; ctx.setLineDash([3,5]);
          ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,mediaSize.height);ctx.stroke();
          // Keep every line; nearby labels remain available in the event list.
          if (x-lastLabel>22 && x<mediaSize.width-18) {
            ctx.translate(x+7,12);ctx.rotate(Math.PI/2);
            ctx.font='9px "JetBrains Mono"';ctx.fillStyle=THEME.dim;
            ctx.fillText(event.name,0,0);lastLabel=x;
          }
          ctx.restore();
        }
      });
    }
  }

  // Scenario levels sit behind data, with left-side labels away from latest price.
  class ReferenceLevels {
    constructor(levels,series){this.levels=levels;this.series=series;this.views=[{zOrder:()=> 'bottom',renderer:()=>({draw:target=>this.draw(target)})}];}
    paneViews(){return this.views;}
    draw(target){target.useMediaCoordinateSpace(({context:ctx,mediaSize})=>{
      for(const level of this.levels){const y=this.series.priceToCoordinate(level.price);if(y===null||y<0||y>mediaSize.height)continue;
        ctx.save();ctx.strokeStyle=level.color;ctx.globalAlpha=.5;ctx.lineWidth=1;ctx.setLineDash([4,5]);
        ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(mediaSize.width,y);ctx.stroke();
        ctx.globalAlpha=.85;ctx.fillStyle=level.color;ctx.font='10px "JetBrains Mono"';
        ctx.fillText(`${level.name} · ${format(level.price,'USD')}`,8,Math.max(12,y-6));ctx.restore();
      }
    });}
  }

  const lwc=window.LightweightCharts;
  const decimal=(value,digits=2)=>new Intl.NumberFormat('en-US',{maximumFractionDigits:digits,minimumFractionDigits:digits}).format(value);
  const compact=value=>new Intl.NumberFormat('en-US',{notation:'compact',maximumFractionDigits:1}).format(value);
  function format(value,unit,axis=false) {
    if(!Number.isFinite(value)) return '—';
    if(unit==='hashrate') {
      const [d,u]=[[1e18,'EH/s'],[1e15,'PH/s'],[1e12,'TH/s'],[1e9,'GH/s'],[1e6,'MH/s'],[1e3,'kH/s'],[1,'H/s']].find(([d])=>Math.abs(value)>=d)||[1,'H/s'];
      return `${decimal(value/d,axis?1:2)} ${u}`;
    }
    if(unit==='percent')return `${decimal(value,2)}%`;
    if(unit==='ratio')return decimal(value,Math.abs(value)<0.01?6:2);
    const n=axis&&Math.abs(value)>=1000?compact(value):decimal(value,unit==='count'?0:Math.abs(value)>0&&Math.abs(value)<0.01?8:2);
    if(unit.startsWith('USD'))return '$'+n+(axis||unit==='USD'?'':' /TH/s/day');
    return n+(axis||unit==='count'?'':` ${unit}`);
  }
  const dateString=t=>typeof t==='string'?t:typeof t==='number'?new Date(t*1000).toISOString().slice(0,10):`${t.year}-${String(t.month).padStart(2,'0')}-${String(t.day).padStart(2,'0')}`;
  const xLabel=(x,p)=>p.axisKind==='days'?`Day ${x}`:dateString(x);
  const rgba=(color,opacity)=>{
    if(opacity===1)return color;
    if(color.startsWith('hsl('))return color.replace('hsl(','hsla(').replace(')',`, ${opacity})`);
    return color;
  };
  function runs(series,payload,mode) {
    const result=[];let run=[];
    for(let i=0;i<series.values.length;i++){
      const value=series.values[i];
      if(Number.isFinite(value)&&(mode!=='log'||value>0))run.push({time:payload.x[series.start+i],value});
      else if(run.length){result.push(run);run=[];}
    }
    if(run.length)result.push(run);
    return result;
  }
  function createView(host,payload,state,stacked=false,onHover=()=>{},onRange=()=>{}) {
    const axes=Object.keys(payload.axes), configs=stacked&&axes.length>1?[['right'],['left']]:[axes];
    const charts=[],events=[],entries=payload.series.map(definition=>({definition,parts:[]}));
    let syncing=false;
    host.classList.toggle('stacked',configs.length>1);
    for(const visibleAxes of configs){
      const div=document.createElement('div');div.className='chart-pane';host.append(div);
      const options={autoSize:true,layout:{background:{type:lwc.ColorType.Solid,color:THEME.bg},textColor:THEME.dim,fontFamily:'JetBrains Mono',fontSize:10,attributionLogo:true},
        grid:{vertLines:{visible:false},horzLines:{visible:payload.gridlines!==false,color:'#1c1c2b'}},
        rightPriceScale:{visible:false},leftPriceScale:{visible:false},
        timeScale:{borderColor:THEME.border,minBarSpacing:0.02,rightOffset:5,lockVisibleTimeRangeOnResize:true},
        crosshair:{mode:lwc.CrosshairMode.Normal,vertLine:{color:'#69697c',labelBackgroundColor:'#292938'},horzLine:{color:'#69697c',labelBackgroundColor:'#292938'}},
        localization:{locale:'en-US',dateFormat:'yyyy-MM-dd',precision:0},handleScroll:{vertTouchDrag:false}};
      for(const id of visibleAxes){
        const actual=configs.length>1?'right':id;
        options[actual+'PriceScale']={visible:true,borderVisible:false,minimumWidth:configs.length>1?78:68,
          mode:state.modes[id]==='log'?1:0,scaleMargins:{top:0.15,bottom:0.08}};
      }
      const chart=(payload.axisKind==='days'?lwc.createOptionsChart:lwc.createChart)(div,options);charts.push(chart);
      // A whitespace calendar preserves actual elapsed spacing even across all-series gaps.
      const calendar=chart.addSeries(lwc.LineSeries,{visible:false,priceScaleId:'__calendar'});calendar.setData(payload.x.map(time=>({time})));
      for(const entry of entries){
        const s=entry.definition;if(!visibleAxes.includes(s.axis))continue;
        const scale=configs.length>1?'right':s.axis,unit=payload.axes[s.axis].unit;
        if(payload.candles && s.id==='price_close'){
          const api=chart.addSeries(lwc.CandlestickSeries,{upColor:THEME.candleUp,downColor:THEME.candleDown,wickUpColor:THEME.candleUp,wickDownColor:THEME.candleDown,borderVisible:false,
            priceScaleId:scale,visible:state.visible.has(s.id),priceLineVisible:false,lastValueVisible:false,
            priceFormat:{type:'custom',formatter:v=>format(v,unit,true),minMove:0.00000001}});
          api.setData(payload.candles.map(({time,open,high,low,close})=>({time,open,high,low,close})));
          entry.parts.push({api,chart});continue;
        }
        for(const data of runs(s,payload,state.modes[s.axis])){
          const api=chart.addSeries(lwc.LineSeries,{color:rgba(s.color,s.opacity),lineWidth:s.lineWidth,
            lineStyle:s.lineStyle==='dashed'?lwc.LineStyle.Dashed:lwc.LineStyle.Solid,
            priceScaleId:scale,visible:state.visible.has(s.id),priceLineVisible:false,lastValueVisible:false,
            pointMarkersVisible:data.length===1,pointMarkersRadius:2,crosshairMarkerRadius:3,
            priceFormat:{type:'custom',formatter:v=>format(v,unit,true),minMove:unit==='ratio'?0.000001:0.00000001}});
          api.setData(data);entry.parts.push({api,chart});
        }
      }
      if(payload.referenceLines?.length && visibleAxes.includes('right')){
        const levels=payload.referenceLines;
        const anchor=chart.addSeries(lwc.LineSeries,{priceScaleId:'right',color:'transparent',lineVisible:false,crosshairMarkerVisible:false,priceLineVisible:false,lastValueVisible:false,
          autoscaleInfoProvider:()=>({priceRange:{minValue:Math.min(...levels.map(l=>l.price)),maxValue:Math.max(...levels.map(l=>l.price))}})});
        anchor.setData([payload.x[0],payload.x.at(-1)].map(time=>({time,value:levels[0].price})));
        chart.panes()[0].attachPrimitive(new ReferenceLevels(levels,anchor));
      }
      const markers=new EventLines(payload.events);chart.panes()[0].attachPrimitive(markers);markers.setVisible(state.events);events.push(markers);
      if(configs.length>1){const label=document.createElement('div');label.className='pane-label';label.textContent=payload.axes[visibleAxes[0]].label;div.append(label);}
      chart.timeScale().subscribeVisibleTimeRangeChange(range=>{
        if(!range||syncing)return;syncing=true;
        for(const other of charts)if(other!==chart)other.timeScale().setVisibleRange(range);
        syncing=false;onRange(range);
      });
      chart.subscribeCrosshairMove(param=>{
        if(syncing)return;syncing=true;
        const inside=param.time!==undefined&&param.point&&param.point.x>=0&&param.point.y>=0;
        const time=inside?(payload.axisKind==='days'?param.time:dateString(param.time)):null;
        for(const other of charts)if(other!==chart){
          const candidate=entries.find(e=>state.visible.has(e.definition.id)&&e.parts.some(p=>p.chart===other)&&Number.isFinite(valueAt(e.definition,time,payload)));
          if(time!==null&&candidate){const part=candidate.parts.find(p=>p.chart===other);other.setCrosshairPosition(valueAt(candidate.definition,time,payload),time,part.api);}
          else other.clearCrosshairPosition();
        }
        syncing=false;onHover(time);
      });
    }
    return {charts,entries,events,range(){return charts[0].timeScale().getVisibleRange();},
      setRange(range){syncing=true;for(const chart of charts){chart.timeScale().setVisibleRange(range);chart.applyOptions({});}syncing=false;onRange(range);},
      fit(){charts[0].timeScale().fitContent();},
      visibility(id,visible){for(const e of entries)if(e.definition.id===id)for(const {api} of e.parts)api.applyOptions({visible});},
      destroy(){for(const chart of charts)chart.remove();host.replaceChildren();}};
  }
  function valueAt(series,time,payload){
    const index=payload.x.indexOf(time)-series.start;
    return index>=0&&index<series.values.length?series.values[index]:null;
  }
  let resolveReady,rejectReady;
  const ready=new Promise((resolve,reject)=>{resolveReady=resolve;rejectReady=reject;});
  window.SecretSatoshisChart={ready};
  async function initialize(){
    if(lwc.version()!=='5.2.1')throw new Error('Unexpected chart runtime');
    const source=JSON.parse($('chart-data').textContent);let payload=source;
    if(payload.schemaVersion!==2)throw new Error('Unsupported chart schema');
    await Promise.all([document.fonts.load('400 12px "JetBrains Mono"'),document.fonts.load('600 12px "JetBrains Mono"'),document.fonts.load('700 32px Syne')]);
    const state={modes:Object.fromEntries(Object.entries(payload.axes).map(([id,a])=>[id,a.mode])),visible:new Set(payload.series.map(s=>s.id)),events:true,presentation:'line',interval:'daily',requestedRange:null};
    const numeric=payload.axisKind==='days';let reading=payload.readingPoint,view,exporting=false;
    const mobile=matchMedia('(max-width:760px)'),embedded=document.body.classList.contains('embedded')||new URL(location.href).searchParams.get('embed')==='1';
    document.body.classList.toggle('embedded',embedded);
    const navToggle=$('navToggle'),navLinks=$('navLinks');
    if(navToggle&&navLinks){
    function closeNavigation(){navLinks.classList.remove('open');navToggle.setAttribute('aria-expanded','false');navToggle.setAttribute('aria-label','Open menu');}
    navToggle.onclick=()=>{const open=navToggle.getAttribute('aria-expanded')!=='true';navLinks.classList.toggle('open',open);navToggle.setAttribute('aria-expanded',String(open));navToggle.setAttribute('aria-label',open?'Close menu':'Open menu');};
    navLinks.addEventListener('click',e=>{if(e.target.closest('a'))closeNavigation();});
    document.addEventListener('keydown',e=>{if(e.key==='Escape'&&navToggle.getAttribute('aria-expanded')==='true'){closeNavigation();navToggle.focus();}});
    }
    function reportHeight(){if(embedded&&parent!==window)parent.postMessage({type:'ss-chart-size',id:payload.id,height:Math.ceil(document.documentElement.getBoundingClientRect().height)},location.protocol==='file:'?'*':location.origin);}
    function readingAt(time){
      reading=time===null?payload.readingPoint:time;
      $('reading-date').textContent=xLabel(reading,payload);
      $('reading-mode').textContent=time===null?(numeric?'CURRENT CYCLE DAY':'REPORT DATE'):(numeric?'CURSOR DAY':'CURSOR DATE');
      for(const s of payload.series)$('value-'+s.id).textContent=format(valueAt(s,reading,payload),payload.axes[s.axis].unit);
      const candle=payload.candles?.find(c=>c.time===reading),panel=$('candle-reading');panel.hidden=!candle;
      if(candle){
        const label=periodLabel(candle);$('reading-mode').textContent=label.toUpperCase();$('reading-date').textContent=candle.observationDate;
        panel.replaceChildren();const dates=document.createElement('div');dates.textContent=`${candle.time} — ${candle.observationDate}`;
        const values=document.createElement('div');values.className='ohlc-values';
        for(const [label,key] of [['Open','open'],['High','high'],['Low','low'],['Close','close']]){const item=document.createElement('span');item.textContent=`${label} ${format(candle[key],'USD')}`;values.append(item);}panel.append(dates,values);
      }

    }
    function rangeEnd(range){return payload.candles?.find(c=>c.time===dateString(range.to))?.observationDate||range.to;}
    function onRange(range){$('view-range').textContent=`${xLabel(range.from,payload)} — ${xLabel(rangeEnd(range),payload)}`;}
    function periodLabel(candle){return !candle.complete?(state.interval==='weekly'?'Week to date':'Month to date'):state.interval==='daily'?'Daily candle':state.interval==='weekly'?'Weekly candle':'Monthly candle';}
    function calendarRange(range){
      if(numeric)return range;
      const normalize=value=>typeof value==='string'?value:dateString(value);
      let from=normalize(range.from),to=normalize(range.to);
      if(payload.candles){
        const first=payload.candles.find(c=>c.observationDate>=from),last=payload.candles.findLast(c=>c.time<=to);
        from=first?.time||payload.x.at(-1);to=payload.rangeEndDate&&to>payload.candles.at(-1).observationDate?to:last?.time||payload.x[0];
      }
      from=from<payload.x[0]?payload.x[0]:from>payload.x.at(-1)?payload.x.at(-1):from;
      to=to>payload.x.at(-1)?payload.x.at(-1):to<payload.x[0]?payload.x[0]:to;
      return {from,to:to<from?from:to};
    }
    function rebuild(requested){const range=requested||view?.range();view?.destroy();view=createView($('chart'),payload,state,mobile.matches,readingAt,onRange);document.querySelector('.plot-wrap').classList.toggle('dual-mobile',mobile.matches&&Object.keys(payload.axes).length>1);if(range){const target=view,bounded=calendarRange(range);target.setRange(bounded);requestAnimationFrame(()=>{if(view===target)target.setRange(bounded);});}reportHeight();}
    function setPresentation(kind,interval=state.interval){
      if(!['line','candles'].includes(kind)||kind==='candles'&&!source.candleViews?.[interval])throw new Error('Candle presentation unavailable');
      const range=state.requestedRange||view.range();state.requestedRange=range;
      state.presentation=kind;state.interval=interval;payload=kind==='line'?source:{...source,...source.candleViews[interval]};
      if(kind==='candles'&&interval==='daily'){
        const prepared=source.candleViews.daily;
        payload.series=source.series.map(s=>({...s,start:0,values:Array.from({length:prepared.x.length},(_,i)=>s.values[prepared.offset+i-s.start]??null)}));
        payload.candles=prepared.ohlc.map(([open,high,low,close],i)=>({time:prepared.x[i],periodEnd:prepared.x[i],observationDate:prepared.x[i],complete:true,open,high,low,close}));
      }
      $('bitcoin-style').value=kind;$('candle-interval').value=interval;$('interval-control').hidden=kind!=='candles';
      $('observation-note').textContent=kind==='line'?source.note:interval==='daily'?'Daily candles.':`${interval==='weekly'?'Weekly':'Monthly'} candles and period-end observations. ${payload.candles.at(-1).complete?'Through':'Latest unfinished period through'} ${source.reportDate}.`;
      rebuild(range);readingAt(null);
    }
    $('bitcoin-controls').hidden=!source.candleViews||!Object.keys(source.candleViews).length;
    for(const option of $('candle-interval').options)option.disabled=!source.candleViews?.[option.value];
    $('bitcoin-style').onchange=()=>setPresentation($('bitcoin-style').value);
    $('candle-interval').onchange=()=>setPresentation('candles',$('candle-interval').value);
    function setVisible(s,visible){if(visible)state.visible.add(s.id);else state.visible.delete(s.id);view.visibility(s.id,visible);const b=$('series-'+s.id);b.setAttribute('aria-pressed',String(visible));b.querySelector('.eye').textContent=visible?'●':'○';b.setAttribute('aria-label',`${visible?'Hide':'Show'} ${s.name}`);}
    for(const s of payload.series){
      const row=document.createElement('div');row.className='indicator-row';
      const b=document.createElement('button');b.id='series-'+s.id;b.className='indicator';b.style.setProperty('--series-color',s.color);
      const swatch=document.createElement('span');swatch.className='swatch';const label=document.createElement('span');label.className='name';label.textContent=s.name;
      const val=document.createElement('strong');val.id='value-'+s.id;val.className='value';label.append(val);const eye=document.createElement('span');eye.className='eye';b.append(swatch,label,eye);
      const solo=document.createElement('button');solo.className='solo';solo.textContent='Only';solo.setAttribute('aria-label',`Show only ${s.name}`);
      solo.onclick=()=>{for(const other of payload.series)setVisible(other,other.id===s.id);};
      b.onclick=()=>{if(state.visible.has(s.id)&&state.visible.size===1){$('status').textContent='Keep at least one series visible.';return;}setVisible(s,!state.visible.has(s.id));};
      row.append(b,solo);$('legend').append(row);
    }
    rebuild();for(const s of payload.series)setVisible(s,true);readingAt(null);
    mobile.addEventListener('change',()=>rebuild());
    $('chart').addEventListener('mouseleave',()=>readingAt(null));
    const ranges=numeric?['365D','730D','CURRENT','ALL']:payload.family==='seasonal'?['ALL']:payload.defaultRange==='MTD'?['MTD']:payload.defaultRange==='YTD'?['YTD']:['YTD','1Y','4Y','10Y','ALL'];
    function selectRange(label){
      let range;
      if(label==='ALL')range={from:payload.x[0],to:payload.x.at(-1)};
      else if(numeric)range={from:0,to:Math.min(payload.x.at(-1),label==='CURRENT'?payload.readingPoint:parseInt(label))};
      else{const start=new Date(payload.reportDate+'T00:00:00Z');
        if(label==='MTD')start.setUTCDate(1);else if(label==='YTD')start.setUTCMonth(0,1);
        else{const month=start.getUTCMonth();start.setUTCFullYear(start.getUTCFullYear()-(label==='1Y'?1:label==='10Y'?10:4));if(start.getUTCMonth()!==month)start.setUTCDate(0);}
        range={from:[payload.x[0],start.toISOString().slice(0,10)].sort().at(-1),to:payload.rangeEndDate||payload.reportDate};}
      state.requestedRange=range;view.setRange(calendarRange(range));$('ranges').querySelectorAll('button').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.range===label)));
    }
    $('ranges').replaceChildren();for(const label of ranges){const b=document.createElement('button');b.dataset.range=label;b.textContent=label==='CURRENT'?'Current cycle':label;b.onclick=()=>selectRange(label);$('ranges').append(b);}
    function setScale(axis,mode){state.modes[axis]=mode;rebuild();$('scale-'+axis).value=mode;}
    $('scales').replaceChildren();for(const [id,axis] of Object.entries(payload.axes)){
      const label=document.createElement('label');label.className='axis-control';label.textContent=(id==='right'?'Right':'Left')+' ';
      const select=document.createElement('select');select.id='scale-'+id;select.setAttribute('aria-label',axis.label+' scale');
      for(const value of ['linear','log']){const option=document.createElement('option');option.value=value;option.textContent=value==='log'?'Log':'Linear';select.append(option);}select.value=axis.mode;
      select.onchange=()=>setScale(id,select.value);label.append(select);$('scales').append(label);
    }
    if(!payload.events.length){$('events').hidden=true;document.querySelector('.events-list').hidden=true;}
    for(const event of payload.events){const li=document.createElement('li');li.textContent=`${event.date} · ${event.name}`;$('event-list').append(li);}
    $('events').onclick=()=>{state.events=!state.events;for(const e of view.events)e.setVisible(state.events);$('events').setAttribute('aria-pressed',String(state.events));};
    $('show-all').onclick=()=>{for(const s of payload.series)setVisible(s,true);$('status').textContent='';};
    const retainedSeries=payload.series.find(s=>s.id==='price_close') || payload.series.find(s=>s.id.startsWith('price_close')) || payload.series.find(s=>s.role==='highlight') || payload.series[0];
    $('remove-all').title=`Keep only ${retainedSeries.name}`;
    $('remove-all').onclick=()=>{for(const s of payload.series)setVisible(s,s.id===retainedSeries.id);$('status').textContent='';};
    $('reset').onclick=()=>{state.events=true;state.modes=Object.fromEntries(Object.entries(source.axes).map(([k,a])=>[k,a.mode]));setPresentation(source.defaultPresentation||'line',source.defaultInterval||'daily');for(const s of payload.series)setVisible(s,true);for(const [id,a] of Object.entries(payload.axes))$('scale-'+id).value=a.mode;$('events').setAttribute('aria-pressed','true');selectRange(payload.defaultRange);readingAt(null);};
    const clearPreset=()=>{state.requestedRange=null;$('ranges').querySelectorAll('button').forEach(b=>b.setAttribute('aria-pressed','false'));};
    $('chart').addEventListener('wheel',clearPreset,{passive:true});$('chart').addEventListener('pointerdown',clearPreset);
    async function exportImage(download=true){
      if(exporting)throw new Error('Export already in progress');exporting=true;$('export').disabled=true;
      let host,exportView;
      const scenarios=payload.referenceLines||[],plotTop=scenarios.length?410:300,plotHeight=1155-plotTop;
      try{
        host=document.createElement('div');host.style.cssText=`position:fixed;left:-10000px;top:0;width:1690px;height:${plotHeight}px`;host.setAttribute('aria-hidden','true');document.body.append(host);
        exportView=createView(host,payload,state,false);
        await new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)));
        exportView.setRange(view.range());
        for(const c of exportView.charts)c.applyOptions({layout:{fontSize:14}});
        await new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)));
        const shot=exportView.charts[0].takeScreenshot(true,false),canvas=document.createElement('canvas');canvas.width=2400;canvas.height=1350;const ctx=canvas.getContext('2d');ctx.fillStyle=THEME.bg;ctx.fillRect(0,0,2400,1350);
        function text(value,x,y,size=20,color=THEME.text,font='JetBrains Mono',weight=400,maxWidth){ctx.font=`${weight} ${size}px "${font}"`;ctx.fillStyle=color;if(maxWidth)ctx.fillText(value,x,y,maxWidth);else ctx.fillText(value,x,y);}
        function rule(y,color=THEME.border){ctx.strokeStyle=color;ctx.lineWidth=1.5;ctx.beginPath();ctx.moveTo(64,y);ctx.lineTo(2336,y);ctx.stroke();}
        ctx.fillStyle=THEME.accent;ctx.fillRect(64,51,10,20);text('SECRET SATOSHIS',94,70,22,THEME.text,'JetBrains Mono',600);text(payload.category.toUpperCase(),1790,70,16,THEME.dim);rule(103);
        text(payload.title,64,178,50,THEME.text,'Syne',700,2240);const range=view.range();
        text(`${xLabel(range.from,payload)} — ${xLabel(rangeEnd(range),payload)}${payload.candles?'  /  '+state.interval.toUpperCase()+' CANDLES':''}  /  ${Object.entries(state.modes).map(([k,v])=>`${k}: ${v}`).join(' · ')}`,64,223,18,THEME.dim);
        text(`Data through ${payload.reportDate}`,1870,223,18,THEME.dim);
        if(scenarios.length){
          const width=2272/scenarios.length;
          scenarios.forEach((scenario,i)=>{const x=64+i*width;
            ctx.fillStyle=scenario.color;ctx.fillRect(x,279,3,64);
            text(scenario.name.toUpperCase(),x+20,299,16,THEME.dim);
            text('$'+decimal(scenario.price,0),x+20,336,28,THEME.text,'JetBrains Mono',600);
          });
        }
        rule(scenarios.length?367:255,THEME.accent);
        ctx.drawImage(shot,64,plotTop,1690,plotHeight);text(numeric?`AT DAY ${payload.readingPoint}`:'AT REPORT DATE',1800,plotTop,15,THEME.dim);
        const selected=payload.series.filter(s=>state.visible.has(s.id)),step=Math.min(100,(plotHeight-40)/selected.length),dense=selected.length>12;
        selected.forEach((s,i)=>{const y=plotTop+40+i*step;ctx.fillStyle=s.color;ctx.fillRect(1800,y-7,18,3);
          text(s.name,1830,y,dense?13:16,THEME.dim,'JetBrains Mono',400,dense?245:485);
          text(format(valueAt(s,payload.readingPoint,payload),payload.axes[s.axis].unit),dense?2090:1830,dense?y:y+30,dense?16:24,s.color,'JetBrains Mono',600,dense?230:485);});
        rule(1193);text(`${payload.source} · Bitcoin Report Library`,64,1230,17,THEME.dim);
        text(Object.values(payload.axes).map(a=>a.label).join(' · ')+(payload.candles?' · '+periodLabel(payload.candles.at(-1))+' through '+payload.candles.at(-1).observationDate:''),64,1260,15,THEME.dim,'JetBrains Mono',400,2150);
        text('SecretSatoshis.com',64,1310,18);text('TradingView Lightweight Charts™ · © 2025 TradingView, Inc. · tradingview.com',1130,1310,14,THEME.dim);
        const dataUrl=canvas.toDataURL('image/png');
        if(download){const a=document.createElement('a');a.href=dataUrl;a.download=`${payload.id}_${payload.reportDate}.png`;a.click();$('status').textContent='PNG exported.';}
        return dataUrl;
      }finally{exportView?.destroy();host?.remove();exporting=false;$('export').disabled=false;}
    }
    $('export').onclick=()=>exportImage().catch(e=>{$('status').textContent=`Export failed: ${e.message}`;});
    $('download-data').onclick=e=>{e.preventDefault();const url=URL.createObjectURL(new Blob([JSON.stringify(payload)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download=payload.id+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
    await new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)));
    selectRange(payload.defaultRange);
    if(source.defaultPresentation==='candles')setPresentation('candles',source.defaultInterval||'daily');
    $('loading').hidden=true;
    if(payload.unavailable.length)$('status').textContent=`Unavailable: ${payload.unavailable.join(', ')}`;
    const api={ready,get payload(){return payload;},exportImage,selectRange,setScale,setPresentation,get readingPoint(){return reading;},get view(){return view;},get state(){return state;}};
    window.SecretSatoshisChart=api;new ResizeObserver(reportHeight).observe(document.body);reportHeight();await new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)));resolveReady(api);
  }
  initialize().catch(error=>{$('loading').textContent=`Unable to load chart: ${error.message}`;$('status').textContent='Chart could not be rendered.';console.error(error);rejectReady(error);});
})();
