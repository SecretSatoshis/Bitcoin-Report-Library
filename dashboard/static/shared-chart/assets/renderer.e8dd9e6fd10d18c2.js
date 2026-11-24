/* Shared chart renderer. All market calculations happen in Python before the page is built. */
(() => {
  'use strict';
  const THEME={bg:'#08080c',surface:'#0e0e16',border:'#2a2a42',text:'#e4e4ef',dim:'#a4a4bc',eventLabel:'#b2b2c8',accent:'#F7931A',candleUp:'#78C99A',candleDown:'#E87878'};
  const TRUSTED_EMBED_ORIGINS=new Set([
    'https://secretsatoshis.com',
    'https://dashboard.secretsatoshis.com',
    'https://charts.secretsatoshis.com',
    'https://agent21.secretsatoshis.com',
  ]);
  const $=id=>document.getElementById(id);
  const domId=(prefix,id)=>prefix+String(id).replace(/[^A-Za-z0-9_-]/g,'_');
  function isTrustedOrigin(origin){
    if(!origin||origin==='null')return false;
    if(origin===location.origin||TRUSTED_EMBED_ORIGINS.has(origin))return true;
    try{const u=new URL(origin);return u.hostname==='localhost'||u.hostname==='127.0.0.1';}
    catch{return false;}
  }

  // Independent of any series: hiding Bitcoin does not hide the historical events.
  class EventLines {
    constructor(events, labels = true, labelOffset = 12) {
      this.events = events;
      this.labels = labels;
      this.labelOffset = labelOffset;
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
          // Background context: faint lines with high-contrast legible labels.
          ctx.globalAlpha=.65;ctx.strokeStyle=THEME.border; ctx.lineWidth=1; ctx.setLineDash([3,5]);
          ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,mediaSize.height);ctx.stroke();
          // Keep every line; nearby labels remain available in the event list.
          if (this.labels && x-lastLabel>22 && x<mediaSize.width-18) {
            ctx.translate(x+7,this.labelOffset);ctx.rotate(Math.PI/2);
            ctx.globalAlpha=.85;ctx.font='10px "JetBrains Mono"';ctx.fillStyle=THEME.eventLabel;
            ctx.fillText(event.name,0,0);lastLabel=x;
          }
          ctx.restore();
        }
      });
    }
  }

  // Annual cases span the plot; dated technical levels begin at their historical anchor.
  // The same primitive draws the live chart and exported images.
  class ReferenceLevels {
    constructor(levels,series,times=[]){
      this.levels=levels;this.series=series;this.supportResistance=true;
      this.starts=new Map(levels.filter(l=>l.startDate).map(level=>{
        const index=times.findIndex(time=>time>=level.startDate);
        // Interpolate within a weekly/monthly candle instead of moving its anchor to
        // the next observation. Dates preceding this chart's history clip at the edge.
        const logical=index===0?0:index<0?Infinity:index-1+
          (Date.parse(level.startDate)-Date.parse(times[index-1]))/(Date.parse(times[index])-Date.parse(times[index-1]));
        return [level,logical];
      }));
      this.views=[{zOrder:()=> 'bottom',renderer:()=>({draw:target=>this.draw(target)})}];
    }
    attached({chart,requestUpdate}){this.chart=chart;this.requestUpdate=requestUpdate;}
    detached(){this.chart=null;this.requestUpdate=null;}
    visibleLevels(){return this.levels.filter(level=>this.supportResistance||!['support','resistance'].includes(level.type));}
    setSupportResistance(visible){this.supportResistance=visible;this.requestUpdate?.();}
    paneViews(){return this.views;}
    draw(target){target.useMediaCoordinateSpace(({context:ctx,mediaSize})=>{
      for(const level of this.visibleLevels()){const y=this.series.priceToCoordinate(level.price);if(y===null||y<0||y>mediaSize.height)continue;
        const logical=this.starts.get(level);
        if(logical===Infinity)continue;
        // Map whole candle indexes, then interpolate their coordinates. Some renderer
        // versions return zero for a fractional logical index.
        const scale=this.chart?.timeScale(),index=Math.floor(logical);
        const left=logical===undefined?0:scale?.logicalToCoordinate(index);
        const next=logical===undefined||logical===index?left:scale?.logicalToCoordinate(index+1);
        const coordinate=left===null||left===undefined||next===null||next===undefined?null:
          logical===undefined?0:left+(next-left)*(logical-index);
        if(coordinate===null||coordinate===undefined||coordinate>=mediaSize.width)continue;
        const start=Math.max(0,coordinate),right=level.labelAlign==='right';
        // Keep the full right-aligned label readable even when the dated line is short.
        ctx.save();ctx.beginPath();ctx.rect(0,0,mediaSize.width,mediaSize.height);ctx.clip();
        ctx.strokeStyle=level.color;ctx.globalAlpha=.55;ctx.lineWidth=1;ctx.setLineDash(level.lineStyle==='solid'?[]:[4,5]);
        ctx.beginPath();ctx.moveTo(start,y);ctx.lineTo(mediaSize.width,y);ctx.stroke();
        ctx.globalAlpha=.9;ctx.fillStyle=level.color;ctx.font='10px "JetBrains Mono"';
        ctx.textAlign=right?'right':'left';
        const label=level.label||`${level.name} · ${readout(level.price,'USD')}`;
        ctx.fillText(label,right?mediaSize.width-8:start+8,Math.max(12,y-6));ctx.restore();
      }
    });}
  }

  // Prepared numeric thresholds or model curves; this primitive only maps them
  // to screen coordinates. Shared by live charts and the PNG compositor.
  class ValueBands {
    constructor(axis,payload,series,state){
      Object.assign(this,{axis,payload,series,state});
      this.views=[{zOrder:()=> 'bottom',renderer:()=>({draw:target=>this.draw(target)})}];
    }
    attached({chart,requestUpdate}){this.chart=chart;this.requestUpdate=requestUpdate;}
    detached(){this.chart=null;this.requestUpdate=null;}
    paneViews(){return this.views;}
    draw(target){
      if(!this.chart||this.axis.band_anchor&&!this.state.visible.has(this.axis.band_anchor))return;
      target.useMediaCoordinateSpace(({context:ctx,mediaSize:{width,height}})=>{
        const dynamic=this.axis.value_bands.some(b=>typeof b.upper==='string'||typeof b.lower==='string');
        const mode=this.state.modes[Object.keys(this.payload.axes).find(id=>this.payload.axes[id]===this.axis)];
        const definitions=new Map(this.payload.series.map(s=>[s.id,s]));
        const coordinate=(edge,i,unbounded)=>{
          if(edge===null)return unbounded;
          const s=definitions.get(edge),value=typeof edge==='number'?edge:s?.values[i-s.start];
          if(!Number.isFinite(value)||mode==='log'&&value<=0)return null;
          return this.series.priceToCoordinate(value);
        };
        const range=this.chart.timeScale().getVisibleLogicalRange();if(!range)return;
        const points=[];
        for(let i=Math.max(0,Math.floor(range.from)-1);i<=Math.min(this.payload.x.length-1,Math.ceil(range.to)+1);i++){
          const x=this.chart.timeScale().logicalToCoordinate(i);if(x!==null)points.push({x,i});
        }
        if(!points.length)return;
        ctx.save();ctx.beginPath();ctx.rect(0,0,width,height);ctx.clip();
        for(const [n,band] of this.axis.value_bands.entries()){
          let run=[];
          const paint=()=>{
            if(run.length>1){
              ctx.fillStyle=band.color;ctx.globalAlpha=(this.axis.band_fill??dynamic) ? 0.065 : 0;
              ctx.beginPath();ctx.moveTo(run[0].x,run[0].upper);
              for(const p of run.slice(1))ctx.lineTo(p.x,p.upper);
              for(const p of [...run].reverse())ctx.lineTo(p.x,p.lower);
              ctx.closePath();ctx.fill();
            }
            run=[];
          };
          for(const p of points){
            const lower=coordinate(band.lower,p.i,height),upper=coordinate(band.upper,p.i,0);
            if(lower===null||upper===null){paint();continue;}
            run.push({...p,lower,upper});
          }
          paint();
          // Constant boundaries belong to ratio panels; model boundaries are
          // normal selectable line series with prepared USD observations.
          if(typeof band.lower==='number'){
            const y=this.series.priceToCoordinate(band.lower);
            if(y!==null){ctx.globalAlpha=.55;ctx.strokeStyle=band.color;ctx.lineWidth=1;ctx.setLineDash([4,5]);ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(width,y);ctx.stroke();ctx.setLineDash([]);}
          }
          if(this.axis.band_labels===false)continue;
          let labelX=dynamic?width*(.22+.65*n/(this.axis.value_bands.length-1)):width-10;
          const p=points.reduce((a,b)=>Math.abs(b.x-labelX)<Math.abs(a.x-labelX)?b:a);
          let lower=coordinate(band.lower,p.i,height),upper=coordinate(band.upper,p.i,0);
          if(lower===null||upper===null||upper>height||lower<0)continue;
          lower=Math.min(height,lower);upper=Math.max(0,upper);
          let y=(lower+upper)/2;
          if(dynamic&&band.lower===null)y=Math.min(height-12,upper+16);
          if(dynamic&&band.upper===null)y=Math.max(14,lower-16);
          if(lower-upper<11)continue;
          ctx.globalAlpha=.92;ctx.font='9px "JetBrains Mono"';ctx.textAlign=dynamic?'center':'right';
          const textWidth=ctx.measureText(band.label).width;
          if(dynamic)labelX=Math.max(textWidth/2+5,Math.min(width-textWidth/2-5,labelX));
          const left=dynamic?labelX-textWidth/2:labelX-textWidth;
          ctx.fillStyle=THEME.bg;ctx.fillRect(left-4,y-9,textWidth+8,15);
          ctx.fillStyle=band.color;ctx.fillText(band.label,labelX,y+3);
        }
        ctx.restore();
      });
    }
  }

  // Draws a group of fund series as stacked bars or areas, from the same prepared
  // series the legend and export read. Positive and negative values stack separately.
  class FundStack {
    constructor(definitions,payload,state,kind){
      Object.assign(this,{definitions,payload,state,kind});
      this.views=[{zOrder:()=> 'bottom',renderer:()=>({draw:target=>this.draw(target)})}];
    }
    attached({chart,series,requestUpdate}){Object.assign(this,{chart,series,requestUpdate});}
    detached(){this.chart=null;this.requestUpdate=null;}
    paneViews(){return this.views;}
    visible(){return this.definitions.filter(s=>this.state.visible.has(s.id));}
    extent(first=0,last=this.payload.x.length-1){
      const selected=this.visible();if(!selected.length)return null;
      let min=0,max=0;
      for(let i=Math.max(0,Math.floor(first));i<=Math.min(last,this.payload.x.length-1);i++){
        let positive=0,negative=0;
        for(const s of selected){const v=s.values[i-s.start];if(!Number.isFinite(v))continue;if(v>=0)positive+=v;else negative+=v;}
        min=Math.min(min,negative);max=Math.max(max,positive);
      }
      return {minValue:min,maxValue:max===min?min+1:max};
    }
    refresh(){this.series?.applyOptions({});this.requestUpdate?.();}
    draw(target){
      if(!this.chart)return;
      const selected=this.visible(),range=this.chart.timeScale().getVisibleLogicalRange();
      if(!selected.length||!range)return;
      target.useMediaCoordinateSpace(({context:ctx,mediaSize:{width,height}})=>{
        ctx.save();ctx.beginPath();ctx.rect(0,0,width,height);ctx.clip();
        const first=Math.max(0,Math.floor(range.from)-1),last=Math.min(this.payload.x.length-1,Math.ceil(range.to)+1);
        const spacing=Math.abs(this.chart.timeScale().logicalToCoordinate(1)-this.chart.timeScale().logicalToCoordinate(0));
        if(this.kind==='stackedBar'){
          const barWidth=Math.max(.6,spacing*.78);
          for(let i=first;i<=last;i++){
            const x=this.chart.timeScale().logicalToCoordinate(i);let positive=0,negative=0;
            for(const s of selected){
              const v=s.values[i-s.start];if(!Number.isFinite(v)||v===0)continue;
              const lower=v>=0?positive:negative,upper=lower+v;
              if(v>=0)positive=upper;else negative=upper;
              const a=this.series.priceToCoordinate(lower),b=this.series.priceToCoordinate(upper);
              if(a===null||b===null)continue;
              ctx.fillStyle=s.color;ctx.globalAlpha=.88;ctx.fillRect(x-barWidth/2,Math.min(a,b),barWidth,Math.max(.6,Math.abs(a-b)));
            }
          }
        }else{
          const base=Array.from({length:last-first+1},()=>0);
          for(const s of selected){
            const layer=[];
            for(let i=first;i<=last;i++){
              const v=s.values[i-s.start],j=i-first,lower=base[j],upper=lower+(Number.isFinite(v)?v:0);
              base[j]=upper;
              const x=this.chart.timeScale().logicalToCoordinate(i),a=this.series.priceToCoordinate(lower),b=this.series.priceToCoordinate(upper);
              if(a!==null&&b!==null)layer.push({x,lower:a,upper:b});
            }
            if(layer.length<2)continue;
            ctx.fillStyle=s.color;ctx.globalAlpha=.68;ctx.beginPath();ctx.moveTo(layer[0].x,layer[0].upper);
            for(const point of layer.slice(1))ctx.lineTo(point.x,point.upper);
            for(const point of [...layer].reverse())ctx.lineTo(point.x,point.lower);
            ctx.closePath();ctx.fill();
          }
        }
        ctx.restore();
      });
    }
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
    if(unit==='ratio')return decimal(value,value!==0&&Math.abs(value)<0.01?6:2);
    const n=axis&&Math.abs(value)>=1000?compact(value):decimal(value,unit==='count'?0:Math.abs(value)>0&&Math.abs(value)<0.01?8:2);
    if(unit.startsWith('USD'))return (value<0?'-$':'$')+n.replace(/^-/,'')+(axis||unit==='USD'?'':' /TH/s/day');
    return n+(axis||unit==='count'?'':` ${unit}`);
  }
  // Series readings: whole units from 1,000 and compact units from a million. The
  // exact value stays in each reading's tooltip and in the chart data download.
  function readout(value,unit){
    if(!Number.isFinite(value)||['percent','ratio','hashrate'].includes(unit)||Math.abs(value)<1000)return format(value,unit);
    const n=Math.abs(value)>=1e6?new Intl.NumberFormat('en-US',{notation:'compact',maximumFractionDigits:2}).format(value):decimal(value,0);
    if(unit.startsWith('USD'))return (value<0?'-$':'$')+n.replace(/^-/,'')+(unit==='USD'?'':' /TH/s/day');
    return n+(unit==='count'?'':` ${unit}`);
  }
  const dateString=t=>typeof t==='string'?t:typeof t==='number'?new Date(t*1000).toISOString().slice(0,10):`${t.year}-${String(t.month).padStart(2,'0')}-${String(t.day).padStart(2,'0')}`;
  const xLabel=(x,p)=>p.axisKind==='days'?`${p.xAxisLabel||'Day'} ${x}`:dateString(x);
  const rgba=(color,opacity)=>{
    if(opacity===1)return color;
    if(/^#[0-9a-f]{6}$/i.test(color))return `rgba(${[1,3,5].map(i=>parseInt(color.slice(i,i+2),16)).join(',')}, ${opacity})`;
    return color;
  };
  function runs(series,payload,mode) {
    const result=[];let run=[];
    for(let i=0;i<series.values.length;i++){
      const value=series.values[i];
      if(series.breakOnYear&&i>0&&String(payload.x[series.start+i]).slice(0,4)!==String(payload.x[series.start+i-1]).slice(0,4)&&run.length){result.push(run);run=[];}
      if(Number.isFinite(value)&&(mode!=='log'||value>0))run.push({time:payload.x[series.start+i],value});
      else if(run.length){result.push(run);run=[];}
    }
    if(run.length)result.push(run);
    return result;
  }
  const MONTHS=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  function tickLabel(time,type){
    const [year,month]=dateString(time).split('-').map(Number);
    return type===lwc.TickMarkType.Year?String(year):type===lwc.TickMarkType.Month?MONTHS[month-1]:null;
  }

  // Reconstruct daily/weekly/monthly candleViews from the shared hashed candles.<hash>.json bundle.
  function buildCandleViewsFromBundle(bundle,source){
    const startDate=source.x[0],endDate=source.x.at(-1);
    const dateIndex=new Map(source.x.map((d,i)=>[d,i]));
    const views={};
    for(const interval of ['daily','weekly','monthly']){
      const raw=bundle[interval];
      if(!raw||!raw.x?.length)continue;
      const picked=[];
      for(let i=0;i<raw.x.length;i++){
        const time=raw.x[i],obs=interval==='daily'?time:raw.observationDate[i];
        if(time>=startDate&&obs<=endDate)picked.push(i);
      }
      if(!picked.length)continue;
      const x=picked.map(i=>raw.x[i]);
      const ohlc=picked.map(i=>raw.ohlc[i]);
      const candles=picked.map(i=>{
        const [open,high,low,close]=raw.ohlc[i];
        const time=raw.x[i];
        return {
          time,open,high,low,close,
          periodEnd:interval==='daily'?time:raw.periodEnd[i],
          observationDate:interval==='daily'?time:raw.observationDate[i],
          complete:interval==='daily'?true:Boolean(raw.complete[i]),
        };
      });
      const events=[];
      for(const event of (source.events||[])){
        const candle=candles.find(c=>c.time<=event.date&&event.date<=c.observationDate);
        if(candle)events.push({...event,originalDate:event.date,date:candle.time,name:event.name+' · '+event.date});
      }
      const view={x,events,readingPoint:x.at(-1),interval};
      if(interval==='daily'){
        const offset=dateIndex.get(x[0]);
        if(offset===undefined)continue;
        view.offset=offset;
        view.ohlc=ohlc;
      }else{
        view.candles=candles;
        view.series=source.series.map(s=>({
          ...s,
          start:0,
          values:candles.map(c=>{
            const idx=dateIndex.get(c.observationDate);
            if(idx===undefined)return null;
            const local=idx-s.start;
            return local>=0&&local<s.values.length?s.values[local]:null;
          }),
        }));
      }
      views[interval]=view;
    }
    return views;
  }

  function createView(host,payload,state,stacked=false,onHover=()=>{},onRange=()=>{}) {
    // A linear axis whose series never go negative and reach down near zero gets no
    // margin below its range, so it never shows negative tick labels.
    function startsNearZero(axis){
      let min=Infinity,max=-Infinity;
      for(const s of payload.series)if(s.axis===axis)for(const v of s.values)if(Number.isFinite(v)){min=Math.min(min,v);max=Math.max(max,v);}
      return min>=0&&max>0&&min<=max*.05;
    }
    const axes=Object.keys(payload.axes), panels=payload.panels;
    const configs=panels?panels.map(panel=>[panel.axis]):stacked&&axes.length>1?[['right'],['left']]:[axes];
    const charts=[],nodes=[],events=[],references=[],bands=[],stacks=[],entries=payload.series.map(definition=>({definition,parts:[]}));
    const edgeTimes=payload.x.length>1?[payload.x[0],payload.x.at(-1)]:payload.x.slice(0,1);
    let syncing=false,rangeRevision=0;
    const pendingRanges=new Map();
    const sameRange=(a,b)=>a&&b&&Math.abs(a.from-b.from)<1e-6&&Math.abs(a.to-b.to)<1e-6;
    function applyRange(chart,range){
      if(!pendingRanges.has(chart)&&sameRange(chart.timeScale().getVisibleLogicalRange(),range))return;
      pendingRanges.set(chart,range);
      chart.timeScale().setVisibleLogicalRange(range);
    }
    host.classList.toggle('stacked',configs.length>1);
    host.style.gridTemplateRows=panels?panels.map(panel=>`minmax(0, ${panel.weight}fr)`).join(' '):'';
    let alignmentFrame;
    function alignScales(){
      if(!panels||charts.length<2)return;
      cancelAnimationFrame(alignmentFrame);
      alignmentFrame=requestAnimationFrame(()=>{
        const width=Math.max(96,...charts.map(chart=>chart.priceScale('right').width()));
        for(const chart of charts)if(chart.priceScale('right').options().minimumWidth!==width)chart.priceScale('right').applyOptions({minimumWidth:width});
      });
    }
    for(const [panelIndex,visibleAxes] of configs.entries()){
      const div=document.createElement('div');div.className='chart-pane';host.append(div);nodes.push(div);
      const options={autoSize:true,layout:{background:{type:lwc.ColorType.Solid,color:THEME.bg},textColor:THEME.dim,fontFamily:'JetBrains Mono',fontSize:10,attributionLogo:false},
        grid:{vertLines:{visible:false},horzLines:{visible:false}},
        rightPriceScale:{visible:false},leftPriceScale:{visible:false},
        timeScale:{visible:!panels||panelIndex===configs.length-1,borderColor:THEME.border,minBarSpacing:0.02,rightOffset:5,lockVisibleTimeRangeOnResize:true},
        crosshair:{mode:lwc.CrosshairMode.Normal,vertLine:{color:'#69697c',labelBackgroundColor:'#292938'},horzLine:{color:'#69697c',labelBackgroundColor:'#292938'}},
        localization:{locale:'en-US',dateFormat:'yyyy-MM-dd',precision:0},handleScroll:{vertTouchDrag:false}};
      if(payload.axisKind!=='days')options.timeScale.tickMarkFormatter=tickLabel;
      for(const id of visibleAxes){
        const actual=panels||configs.length>1?'right':id;
        options[actual+'PriceScale']={visible:true,borderVisible:false,entireTextOnly:true,minimumWidth:panels?96:configs.length>1?78:68,
          mode:state.modes[id]==='log'?1:0,scaleMargins:{top:0.15,bottom:state.modes[id]!=='log'&&startsNearZero(id)?0:0.08}};
      }
      const chart=(payload.axisKind==='days'?lwc.createOptionsChart:lwc.createChart)(div,options);charts.push(chart);
      // A whitespace calendar preserves actual elapsed spacing even across all-series gaps.
      const calendar=chart.addSeries(lwc.LineSeries,{visible:false,priceScaleId:'__calendar'});calendar.setData(payload.x.map(time=>({time})));
      const groups=new Map();
      for(const s of payload.series){
        if(!visibleAxes.includes(s.axis)||!['stackedBar','stackedArea'].includes(s.render))continue;
        const key=s.axis+':'+s.stackGroup;
        if(!groups.has(key))groups.set(key,[]);groups.get(key).push(s);
      }
      for(const definitions of groups.values()){
        const s=definitions[0],scale=panels||configs.length>1?'right':s.axis;
        const overlay=new FundStack(definitions,payload,state,s.render);
        const anchor=chart.addSeries(lwc.LineSeries,{priceScaleId:scale,color:'transparent',lineVisible:false,crosshairMarkerVisible:false,priceLineVisible:false,lastValueVisible:false,
          priceFormat:{type:'custom',formatter:v=>format(v,payload.axes[s.axis].unit,true),minMove:0.00000001},
          autoscaleInfoProvider:()=>{const range=chart.timeScale().getVisibleLogicalRange();const extent=overlay.extent(range?.from,range?.to);return extent?{priceRange:extent}:null;}});
        anchor.setData(payload.x.map(time=>({time,value:0})));anchor.attachPrimitive(overlay);stacks.push(overlay);
        // Stacked holdings never go below zero, so the scale needs no room under it.
        if(s.render==='stackedArea')chart.priceScale(scale).applyOptions({scaleMargins:{top:0.15,bottom:0}});
      }
      // Two signed axes in one plot share the same zero coordinate. Extend
      // each range by the same negative/positive ratio without changing units.
      function zeroAlignedRange(axis){
        const range=chart.timeScale().getVisibleLogicalRange(),bounds={};
        for(const id of visibleAxes){
          let min=0,max=0;
          for(const s of payload.series.filter(s=>s.axis===id&&state.visible.has(s.id))){
            const first=Math.max(0,Math.floor(range?.from??0)-s.start),last=Math.min(s.values.length-1,Math.ceil(range?.to??payload.x.length-1)-s.start);
            for(let i=first;i<=last;i++){const v=s.values[i];if(Number.isFinite(v)){min=Math.min(min,v);max=Math.max(max,v);}}
          }
          bounds[id]={min,max};
        }
        const ratio=Math.max(0,...Object.values(bounds).map(b=>-b.min/(b.max||Math.abs(b.min)||1)));
        const b=bounds[axis],max=b.max||Math.abs(b.min)||1;
        return {minValue:-max*ratio,maxValue:max};
      }
      for(const entry of entries){
        const s=entry.definition;if(!visibleAxes.includes(s.axis))continue;
        const scale=panels||configs.length>1?'right':s.axis,unit=payload.axes[s.axis].unit;
        if(payload.candles && s.id==='price_close'){
          const api=chart.addSeries(lwc.CandlestickSeries,{upColor:THEME.candleUp,downColor:THEME.candleDown,wickUpColor:THEME.candleUp,wickDownColor:THEME.candleDown,borderVisible:false,
            priceScaleId:scale,visible:state.visible.has(s.id),priceLineVisible:false,lastValueVisible:false,
            priceFormat:{type:'custom',formatter:v=>format(v,unit,true),minMove:0.00000001}});
          api.setData(payload.candles.map(({time,open,high,low,close})=>({time,open,high,low,close})));
          entry.parts.push({api,chart});continue;
        }
        for(const data of runs(s,payload,state.modes[s.axis])){
          const stackedSeries=['stackedBar','stackedArea'].includes(s.render);
          const type=s.render==='bar'?lwc.HistogramSeries:s.render==='area'?lwc.BaselineSeries:lwc.LineSeries;
          const api=chart.addSeries(type,{color:rgba(s.color,s.opacity),lineWidth:s.lineWidth,
            ...(s.render==='area'?{baseValue:{type:'price',price:0},topLineColor:s.color,bottomLineColor:s.color,topFillColor1:rgba(s.color,.36),topFillColor2:rgba(s.color,.06),bottomFillColor1:rgba(s.color,.06),bottomFillColor2:rgba(s.color,.36)}:{}),
            ...(stackedSeries?{lineVisible:false,crosshairMarkerVisible:false,autoscaleInfoProvider:()=>null}:{}),
            ...(payload.alignZero&&configs.length===1&&visibleAxes.length>1?{autoscaleInfoProvider:()=>({priceRange:zeroAlignedRange(s.axis)})}:{}),
            lineStyle:s.lineStyle==='dashed'?lwc.LineStyle.Dashed:lwc.LineStyle.Solid,
            priceScaleId:scale,visible:state.visible.has(s.id),priceLineVisible:false,lastValueVisible:false,
            pointMarkersVisible:!stackedSeries&&(s.pointMarkers||data.length===1),pointMarkersRadius:2,crosshairMarkerRadius:3,
            priceFormat:{type:'custom',formatter:v=>format(v,unit,true),minMove:unit==='ratio'?0.000001:0.00000001}});
          api.setData(s.render==='bar'?data.map(p=>({...p,color:p.value>=0?(s.positiveColor||THEME.candleUp):(s.negativeColor||THEME.candleDown)})):data);entry.parts.push({api,chart});
        }
      }
      for(const id of visibleAxes){
        const lines=payload.axes[id].reference_lines;if(!lines?.length)continue;
        const scale=panels||configs.length>1?'right':id;
        const levels=lines.map(line=>({price:line.value,label:line.label,color:line.color||THEME.dim}));
        const anchor=chart.addSeries(lwc.LineSeries,{priceScaleId:scale,color:'transparent',lineVisible:false,crosshairMarkerVisible:false,priceLineVisible:false,lastValueVisible:false,
          autoscaleInfoProvider:()=>({priceRange:{minValue:Math.min(...levels.map(l=>l.price)),maxValue:Math.max(...levels.map(l=>l.price))}})});
        anchor.setData(edgeTimes.map(time=>({time,value:levels[0].price})));
        chart.panes()[0].attachPrimitive(new ReferenceLevels(levels,anchor,payload.x));
      }
      for(const id of visibleAxes){
        const axis=payload.axes[id];if(!axis.value_bands?.length)continue;
        const scale=panels||configs.length>1?'right':id;
        const bounds=axis.band_range?(state.modes[id]==='log'
          ?[...axis.band_range,...axis.value_bands.flatMap(b=>[b.lower,b.upper])].filter(v=>typeof v==='number'&&v>0)
          :axis.band_range):null;
        const anchor=chart.addSeries(lwc.LineSeries,{priceScaleId:scale,color:'transparent',lineVisible:false,crosshairMarkerVisible:false,priceLineVisible:false,lastValueVisible:false,
          ...(bounds?{autoscaleInfoProvider:()=>({priceRange:{minValue:Math.min(...bounds),maxValue:Math.max(...bounds)}})}:{autoscaleInfoProvider:()=>null})});
        anchor.setData(edgeTimes.map(time=>({time,value:bounds?Math.min(...bounds):1})));
        const overlay=new ValueBands(axis,payload,anchor,state);chart.panes()[0].attachPrimitive(overlay);bands.push(overlay);
      }
      if(payload.referenceLines?.length && visibleAxes.includes('right')){
        const levels=payload.referenceLines;
        let overlay;
        const anchor=chart.addSeries(lwc.LineSeries,{priceScaleId:'right',color:'transparent',lineVisible:false,crosshairMarkerVisible:false,priceLineVisible:false,lastValueVisible:false,
          autoscaleInfoProvider:()=>{
            const visible=overlay?.visibleLevels()||levels;
            return visible.length?{priceRange:{minValue:Math.min(...visible.map(l=>l.price)),maxValue:Math.max(...visible.map(l=>l.price))}}:null;
          }});
        anchor.setData(edgeTimes.map(time=>({time,value:levels[0].price})));
        overlay=new ReferenceLevels(levels,anchor,payload.x);overlay.setSupportResistance(state.supportResistance!==false);
        chart.panes()[0].attachPrimitive(overlay);references.push({overlay,anchor});
      }
      const markers=new EventLines(payload.events,!panels||panelIndex===0,panels?42:12);chart.panes()[0].attachPrimitive(markers);markers.setVisible(state.events);events.push(markers);
      if(panels||configs.length>1){const label=document.createElement('div');label.className='pane-label';label.textContent=panels?panels[panelIndex].label:payload.axes[visibleAxes[0]].label;div.append(label);}
      // Logical ranges retain fractional bars and empty space beyond the data.
      // Date ranges round to observations, so they drift during wheel/trackpad pans.
      chart.timeScale().subscribeVisibleLogicalRangeChange(range=>{
        if(!range)return;
        // Mirrored updates are delivered asynchronously during painting. They
        // must not become a new source and send a stale range back to the driver.
        const pending=pendingRanges.get(chart);
        if(pending){if(sameRange(pending,range))pendingRanges.delete(chart);return;}
        if(syncing)return;syncing=true;
        for(const other of charts)if(other!==chart){
          const current=other.timeScale().getVisibleLogicalRange();
          if(!sameRange(current,range))applyRange(other,range);
        }
        syncing=false;alignScales();
      });
      chart.timeScale().subscribeVisibleTimeRangeChange(range=>{if(range)onRange(range);});
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
    const alignmentObserver=panels?new ResizeObserver(alignScales):null;
    alignmentObserver?.observe(host);alignScales();
    return {charts,nodes,entries,events,stacks,get rangeRevision(){return rangeRevision;},range(){return charts[0].timeScale().getVisibleRange();},
      setRange(range){
        rangeRevision++;
        const scale=charts[0].timeScale(),from=scale.timeToIndex(range.from,true),to=scale.timeToIndex(range.to,true);
        // A range that starts at the first observation gets a little room before it, so the
        // first date label sits inside the axis. Later starts are left as they are, which
        // keeps a range unchanged when the chart is rebuilt.
        const logical={from:from<=0?from-Math.max(1,(to-from)*.02):from,to};
        syncing=true;for(const chart of charts){applyRange(chart,logical);chart.applyOptions({});}syncing=false;onRange(range);
      },
      fit(){charts[0].timeScale().fitContent();},
      supportResistance(visible){for(const {overlay,anchor} of references){overlay.setSupportResistance(visible);anchor.applyOptions({});}},
      visibility(id,visible){for(const e of entries)if(e.definition.id===id)for(const {api} of e.parts)api.applyOptions({visible});for(const overlay of bands)overlay.requestUpdate?.();for(const stack of stacks)stack.refresh();if(payload.alignZero)for(const entry of entries)for(const part of entry.parts)part.api.applyOptions({});alignScales();},
      destroy(){alignmentObserver?.disconnect();cancelAnimationFrame(alignmentFrame);for(const chart of charts)chart.remove();host.replaceChildren();}};
  }
  function valueAt(series,time,payload){
    const index=payload.x.indexOf(time)-series.start;
    return index>=0&&index<series.values.length?series.values[index]:null;
  }
  function orderedSeries(payload,time=payload.readingPoint){
    if(payload.seriesOrder){const rank=new Map(payload.seriesOrder.map((id,i)=>[id,i]));return payload.series.slice().sort((a,b)=>(rank.get(a.id)??Infinity)-(rank.get(b.id)??Infinity));}
    // Bitcoin (or the current year) first, then Median and Average, then the rest by value.
    const priority=s=>s.id==='price_close'?0:s.id.startsWith('price_close')||s.role==='highlight'?1:s.role==='median'?2:s.role==='mean'?3:4;
    const values=new Map(payload.series.map(s=>[s.id,valueAt(s,time,payload)]));
    const compare=(a,b)=>{
      const pinned=priority(a)-priority(b);if(pinned)return pinned;
      const av=values.get(a.id),bv=values.get(b.id),af=Number.isFinite(av),bf=Number.isFinite(bv);
      if(af!==bf)return af?-1:1;
      return af?bv-av:0;
    };
    // Sort inside each panel so prices are not ranked against multiples or counts.
    return payload.panels?payload.panels.flatMap(panel=>payload.series.filter(s=>s.axis===panel.axis).sort(compare)):[...payload.series].sort(compare);
  }
  let resolveReady,rejectReady;
  const ready=new Promise((resolve,reject)=>{resolveReady=resolve;rejectReady=reject;});
  window.SecretSatoshisChart={ready};
  async function initialize(){
    if(lwc.version()!=='5.2.1')throw new Error('Unexpected chart runtime');
    const source=JSON.parse($('chart-data').textContent);let payload=source;
    if(payload.schemaVersion!==2)throw new Error('Unsupported chart schema');
    await Promise.all([document.fonts.load('400 12px "JetBrains Mono"'),document.fonts.load('600 12px "JetBrains Mono"'),document.fonts.load('700 32px Syne')]);
    const defaultHidden=new Set(source.defaultHiddenSeries||[]);
    const state={modes:Object.fromEntries(Object.entries(payload.axes).map(([id,a])=>[id,a.mode])),visible:new Set(payload.series.filter(s=>!defaultHidden.has(s.id)).map(s=>s.id)),events:true,supportResistance:true,presentation:'line',interval:'daily',flowInterval:source.defaultFlowInterval||null,requestedRange:null,activePreset:payload.defaultRange};
    const numeric=payload.axisKind==='days';let reading=payload.readingPoint,view,exporting=false;
    const params=new URL(location.href).searchParams;
    const mobile=matchMedia('(max-width:760px)'),coarsePointer=matchMedia('(pointer: coarse)'),embedded=document.body.classList.contains('embedded')||params.get('embed')==='1';
    document.body.classList.toggle('embedded',embedded);
    // Capture mode is the still image for newsletters: no controls, every visible series listed.
    document.body.classList.toggle('capture',document.body.classList.contains('capture')||params.get('capture')==='1');
    // Phones start without event lines; the Events button turns them on.
    const defaultEvents=!mobile.matches;state.events=defaultEvents;
    document.body.classList.toggle('panel-layout',Boolean(payload.panels));

    const staleBanner=$('stale-banner');
    if(staleBanner&&payload.staleBanner){
      staleBanner.textContent=payload.staleBanner;
      staleBanner.hidden=false;
    }

    const hint=$('chart-hint');
    if(hint&&coarsePointer.matches){
      hint.textContent='Pinch to zoom · Drag to explore · Double-tap price axis to fit';
    }

    const toolbar=document.querySelector('.toolbar');
    function updateToolbarFade(){
      if(!toolbar)return;
      toolbar.classList.toggle('can-scroll-right',toolbar.scrollWidth-toolbar.scrollLeft-toolbar.clientWidth>6);
    }
    if(toolbar){
      toolbar.addEventListener('scroll',updateToolbarFade,{passive:true});
      window.addEventListener('resize',updateToolbarFade);
      updateToolbarFade();
    }

    const navToggle=$('navToggle'),navLinks=$('navLinks');
    if(navToggle&&navLinks){
      function closeNavigation(){navLinks.classList.remove('open');navToggle.setAttribute('aria-expanded','false');navToggle.setAttribute('aria-label','Open menu');}
      navToggle.onclick=()=>{const open=navToggle.getAttribute('aria-expanded')!=='true';navLinks.classList.toggle('open',open);navToggle.setAttribute('aria-expanded',String(open));navToggle.setAttribute('aria-label',open?'Close menu':'Open menu');};
      navLinks.addEventListener('click',e=>{if(e.target.closest('a'))closeNavigation();});
      document.addEventListener('keydown',e=>{if(e.key==='Escape'&&navToggle.getAttribute('aria-expanded')==='true'){closeNavigation();navToggle.focus();}});
    }

    let parentOrigin=location.origin;
    if(document.referrer){
      try{const refOrigin=new URL(document.referrer).origin;if(isTrustedOrigin(refOrigin))parentOrigin=refOrigin;}catch{}
    }
    let heightFrame=0,lastReportedHeight=0;
    function flushHeight(){
      if(!embedded||parent===window)return;
      const height=Math.ceil(document.documentElement.getBoundingClientRect().height);
      if(!Number.isFinite(height)||height===lastReportedHeight)return;
      lastReportedHeight=height;
      parent.postMessage({type:'ss-chart-size',id:payload.id,height},parentOrigin);
    }
    function reportHeight(){
      if(!embedded||parent===window)return;
      cancelAnimationFrame(heightFrame);
      heightFrame=requestAnimationFrame(flushHeight);
    }
    if(embedded&&parent!==window)window.addEventListener('message',event=>{
      if(event.source!==parent||!isTrustedOrigin(event.origin))return;
      const message=event.data;
      if(message?.type!=='ss-chart-viewport'||message.id!==payload.id||!Number.isFinite(message.height)||message.height<100||message.height>10000)return;
      parentOrigin=event.origin;
      const height=message.height+'px';
      if(document.body.style.getPropertyValue('--chart-viewport-height')===height)return;
      document.body.style.setProperty('--chart-viewport-height',height);
      reportHeight();
    });

    // Lazy loader for shared externalized candles.<hash>.json asset.
    let candlesPromise=null;
    async function ensureCandleViews(){
      if(source.candleViews)return source.candleViews;
      if(!source.candlesUrl)return null;
      if(!candlesPromise){
        candlesPromise=fetch(source.candlesUrl).then(response=>{
          if(!response.ok)throw new Error(`Unable to load candle data (${response.status})`);
          return response.json();
        }).then(bundle=>{
          source.candleViews=buildCandleViewsFromBundle(bundle,source);
          return source.candleViews;
        });
      }
      return candlesPromise;
    }

    // A fade at the bottom of the series list while more rows sit below the fold.
    function legendFade(){const legend=$('legend');legend.classList.toggle('more-below',legend.scrollHeight-legend.scrollTop-legend.clientHeight>4);}
    let legendFrame;
    function layoutLegend(){
      const body=document.body,legend=$('legend');
      if(!body.classList.contains('dashboard-legend')||body.classList.contains('capture'))return;
      // Measure a single column so a two-column legend that fits does not oscillate
      // back to one column. The final layout is applied before the browser paints.
      body.classList.remove('legend-two-columns');
      const count=legend.querySelectorAll('.indicator-row').length;
      if(!mobile.matches&&count>1&&legend.scrollHeight>legend.clientHeight+1)
        body.classList.add('legend-two-columns');
      legendFade();
    }
    function scheduleLegendLayout(){cancelAnimationFrame(legendFrame);legendFrame=requestAnimationFrame(layoutLegend);}
    window.addEventListener('resize',scheduleLegendLayout);
    function readingAt(time){
      reading=time===null?payload.readingPoint:time;
      $('reading-date').textContent=xLabel(reading,payload);
      $('reading-mode').textContent=time===null?(payload.family==='seasonal'?'REPORT DAY':numeric?'CURRENT CYCLE DAY':payload.dataDate?'LATEST DATA':'REPORT DATE'):(numeric?'CURSOR DAY':'CURSOR DATE');
      for(const s of payload.series){
        const value=valueAt(s,reading,payload),unit=payload.axes[s.axis].unit,cell=$(domId('value-',s.id));
        if(cell){cell.textContent=readout(value,unit);cell.title=format(value,unit);}
      }
      legendFade();
      const sorted=orderedSeries(payload,reading),order=sorted.map(s=>s.id).join('|');
      if(order!==legendOrder){
        for(const s of sorted)(legendGroups.get(s.axis)||$('legend')).append(legendRows.get(s.id));
        legendOrder=order;
      }
      const candle=payload.candles?.find(c=>c.time===reading),panel=$('candle-reading');
      // Keep the candle readout's space over calendar gaps. Hiding its box changes
      // legend capacity and can resize the plot underneath a stationary pointer.
      panel.hidden=!payload.candles;panel.style.visibility=candle?'':'hidden';
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
    // Column capacity belongs to chart/layout changes, not cursor updates. Otherwise
    // changing the plot width changes the hovered date and can trigger a resize loop.
    function rebuild(requested){
      const range=requested||view?.range();
      view?.destroy();
      view=createView($('chart'),payload,state,mobile.matches,readingAt,onRange);
      document.querySelector('.plot-wrap').classList.toggle('dual-mobile',mobile.matches&&Object.keys(payload.axes).length>1);
      if(range){
        const target=view,bounded=calendarRange(range);
        target.setRange(bounded);
        const revision=target.rangeRevision;
        requestAnimationFrame(()=>{if(view===target&&target.rangeRevision===revision)target.setRange(bounded);});
      }
      reportHeight();scheduleLegendLayout();updateToolbarFade();
    }
    // The payload is the default flow frequency; flowViews holds the others.
    const flowPayload=interval=>interval===source.defaultFlowInterval?source:{...source,...source.flowViews[interval]};
    function applyFlowPayload(interval){
      state.flowInterval=interval;payload=flowPayload(interval);
      $('observation-note').textContent=payload.note;
      document.querySelector('.plot-label').textContent=Object.entries(payload.axes).map(([id,a])=>`${id.toUpperCase()}: ${a.label}`).join(' · ');
      for(const s of payload.series){
        const btn=$(domId('series-',s.id));
        if(btn){
          const name=btn.querySelector('.name');name.firstChild.textContent=s.name;
          btn.setAttribute('aria-label',`${state.visible.has(s.id)?'Hide':'Show'} ${s.name}`);
        }
        legendRows.get(s.id)?.querySelector('.solo')?.setAttribute('aria-label',`Show only ${s.name}`);
      }
      for(const [id,a] of Object.entries(payload.axes))$('scale-'+id)?.setAttribute('aria-label',a.label+' scale');
      $('flow-intervals').querySelectorAll('button').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.interval===interval)));
      legendOrder='';
    }
    function setFlowInterval(interval){
      if(!source.flowIntervals?.includes(interval))throw new Error('Flow frequency unavailable');
      const range=state.requestedRange||view.range();
      const activeRange=$('ranges').querySelector('button[aria-pressed="true"]')?.dataset.range;
      applyFlowPayload(interval);
      rebuild(range);if(activeRange)selectRange(activeRange);readingAt(null);
    }
    const flowControls=$('flow-intervals');
    if(flowControls&&source.flowIntervals){
      flowControls.hidden=false;
      for(const interval of source.flowIntervals){
        const b=document.createElement('button');b.dataset.interval=interval;b.textContent=interval[0].toUpperCase()+interval.slice(1);
        b.setAttribute('aria-pressed',String(interval===state.flowInterval));b.onclick=()=>setFlowInterval(interval);flowControls.append(b);
      }
    }
    async function setPresentation(kind,interval=state.interval,{skipRebuild=false}={}){
      if(!['line','candles'].includes(kind))throw new Error('Candle presentation unavailable');
      if(kind==='candles'){
        await ensureCandleViews();
        if(!source.candleViews?.[interval])throw new Error('Candle presentation unavailable');
      }
      const range=state.requestedRange||view?.range();
      if(range)state.requestedRange=range;
      state.presentation=kind;state.interval=interval;
      payload=kind==='line'?(source.flowIntervals?flowPayload(state.flowInterval):source):{...source,...source.candleViews[interval]};
      if(kind==='candles'&&interval==='daily'){
        const prepared=source.candleViews.daily;
        payload.series=source.series.map(s=>({...s,start:0,values:Array.from({length:prepared.x.length},(_,i)=>s.values[prepared.offset+i-s.start]??null)}));
        payload.candles=prepared.ohlc.map(([open,high,low,close],i)=>({time:prepared.x[i],periodEnd:prepared.x[i],observationDate:prepared.x[i],complete:true,open,high,low,close}));
      }
      $('bitcoin-style').value=kind;$('candle-interval').value=interval;$('interval-control').hidden=kind!=='candles';
      $('observation-note').textContent=kind==='line'?payload.note:interval==='daily'?'Daily candles.':`${interval==='weekly'?'Weekly':'Monthly'} candles and period-end observations. ${payload.candles.at(-1).complete?'Through':'Latest unfinished period through'} ${source.reportDate}.`;
      if(!skipRebuild){rebuild(range);readingAt(null);}
    }
    const availableCandleIntervals=source.candleViews?Object.keys(source.candleViews):(source.candleIntervals||[]);
    $('bitcoin-controls').hidden=!availableCandleIntervals.length;
    for(const option of $('candle-interval').options)option.disabled=!availableCandleIntervals.includes(option.value);
    $('bitcoin-style').onchange=()=>setPresentation($('bitcoin-style').value).catch(e=>{$('status').textContent=`Candles unavailable: ${e.message}`;});
    $('candle-interval').onchange=()=>setPresentation('candles',$('candle-interval').value).catch(e=>{$('status').textContent=`Candles unavailable: ${e.message}`;});
    function setVisible(s,visible){
      s=payload.series.find(item=>item.id===s.id)||s;
      if(visible)state.visible.add(s.id);else state.visible.delete(s.id);
      view.visibility(s.id,visible);
      const b=$(domId('series-',s.id));
      if(b){b.setAttribute('aria-pressed',String(visible));b.querySelector('.eye').textContent=visible?'●':'○';b.setAttribute('aria-label',`${visible?'Hide':'Show'} ${s.name}`);}
    }
    const legendGroups=new Map(),legendRows=new Map();let legendOrder='';
    if(payload.panels)for(const panel of payload.panels){
      const section=document.createElement('section');section.className='legend-group';
      const heading=document.createElement('h3');heading.textContent=panel.label;section.append(heading);
      $('legend').append(section);legendGroups.set(panel.axis,section);
    }
    const legendSeries=orderedSeries(payload);
    for(const s of legendSeries){
      const row=document.createElement('div');row.className='indicator-row';
      const b=document.createElement('button');b.id=domId('series-',s.id);b.dataset.seriesId=s.id;b.className='indicator';b.style.setProperty('--series-color',s.color);
      const swatch=document.createElement('span');swatch.className='swatch';const label=document.createElement('span');label.className='name';label.textContent=s.name;
      const val=document.createElement('strong');val.id=domId('value-',s.id);val.className='value';label.append(val);const eye=document.createElement('span');eye.className='eye';b.append(swatch,label,eye);
      const solo=document.createElement('button');solo.className='solo';solo.textContent='Only';solo.setAttribute('aria-label',`Show only ${s.name}`);
      solo.onclick=()=>{for(const other of payload.series)setVisible(other,other.id===s.id);};
      b.onclick=()=>{if(state.visible.has(s.id)&&state.visible.size===1){$('status').textContent='Keep at least one series visible.';return;}setVisible(s,!state.visible.has(s.id));};
      row.append(b,solo);legendRows.set(s.id,row);(legendGroups.get(s.axis)||$('legend')).append(row);
    }
    $('legend').classList.toggle('many-series',payload.series.length>8);
    $('legend').addEventListener('scroll',legendFade,{passive:true});new ResizeObserver(legendFade).observe($('legend'));
    rebuild();for(const s of payload.series)setVisible(s,!defaultHidden.has(s.id));readingAt(null);
    mobile.addEventListener('change',()=>rebuild());
    $('chart').addEventListener('mouseleave',()=>readingAt(null));
    const ranges=payload.ranges|| (payload.family==='seasonal'?['ALL']:numeric?['365D','730D','CURRENT','ALL']:payload.defaultRange==='MTD'?['MTD']:payload.defaultRange==='YTD'?['YTD']:['YTD','1Y','4Y','10Y','ALL']);
    function selectRange(label){
      let range;
      if(label==='ALL')range={from:payload.x[0],to:payload.x.at(-1)};
      else if(numeric)range={from:0,to:Math.min(payload.x.at(-1),label==='CURRENT'?payload.readingPoint:parseInt(label))};
      else{const start=new Date((payload.rangeAnchorDate||payload.rangeEndDate||payload.reportDate)+'T00:00:00Z');
        if(label==='MTD')start.setUTCDate(1);else if(label==='YTD')start.setUTCMonth(0,1);
        else if(/^\d+M$/.test(label)){const day=start.getUTCDate();start.setUTCMonth(start.getUTCMonth()-parseInt(label));if(start.getUTCDate()!==day)start.setUTCDate(0);}
        else{const month=start.getUTCMonth();start.setUTCFullYear(start.getUTCFullYear()-(/^\d+Y$/.test(label)?parseInt(label):4));if(start.getUTCMonth()!==month)start.setUTCDate(0);}
        range={from:[payload.x[0],start.toISOString().slice(0,10)].sort().at(-1),to:payload.rangeEndDate||payload.reportDate};}
      state.activePreset=label;
      state.requestedRange=range;view.setRange(calendarRange(range));$('ranges').querySelectorAll('button').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.range===label)));
    }
    $('ranges').replaceChildren();for(const label of ranges){const b=document.createElement('button');b.dataset.range=label;b.textContent=label==='CURRENT'?'Current cycle':label;b.onclick=()=>selectRange(label);$('ranges').append(b);}
    function setScale(axis,mode){
      const allowed=payload.axes[axis]?.modes||['linear','log'];
      if(!allowed.includes(mode))return;
      state.modes[axis]=mode;rebuild();$('scale-'+axis).value=mode;
    }
    $('scales').replaceChildren();for(const [id,axis] of Object.entries(payload.axes)){
      const label=document.createElement('label');label.className='axis-control';label.textContent=(payload.panels?.find(p=>p.axis===id)?.control_label||(id==='right'?'Right':'Left'))+' ';
      const select=document.createElement('select');select.id='scale-'+id;select.setAttribute('aria-label',axis.label+' scale');
      for(const value of (axis.modes||['linear','log'])){const option=document.createElement('option');option.value=value;option.textContent=value==='log'?'Log':'Linear';select.append(option);}select.value=axis.mode;
      select.onchange=()=>setScale(id,select.value);label.append(select);$('scales').append(label);
    }
    $('events').setAttribute('aria-pressed',String(state.events));
    if(!payload.events.length){$('events').hidden=true;document.querySelector('.events-list').hidden=true;}
    for(const event of payload.events){const li=document.createElement('li');li.textContent=`${event.date} · ${event.name}`;$('event-list').append(li);}
    $('events').onclick=()=>{state.events=!state.events;for(const e of view.events)e.setVisible(state.events);$('events').setAttribute('aria-pressed',String(state.events));};
    $('support-resistance').hidden=!(source.referenceLines||[]).some(level=>['support','resistance'].includes(level.type));
    $('support-resistance').onclick=()=>{state.supportResistance=!state.supportResistance;view.supportResistance(state.supportResistance);$('support-resistance').setAttribute('aria-pressed',String(state.supportResistance));};
    $('show-all').onclick=()=>{for(const s of payload.series)setVisible(s,true);$('status').textContent='';};
    const retainedSeries=payload.series.find(s=>s.id==='price_close') || payload.series.find(s=>s.id.startsWith('price_close')) || payload.series.find(s=>s.role==='highlight') || payload.series[0];
    $('remove-all').title=`Keep only ${retainedSeries.name}`;
    $('remove-all').onclick=()=>{for(const s of payload.series)setVisible(s,s.id===retainedSeries.id);$('status').textContent='';};

    async function resetChart(){
      state.events=defaultEvents;
      state.supportResistance=true;
      state.modes=Object.fromEntries(Object.entries(source.axes).map(([k,a])=>[k,a.mode]));
      state.visible=new Set(source.series.filter(s=>!defaultHidden.has(s.id)).map(s=>s.id));
      state.requestedRange=null;
      if(source.flowIntervals){
        applyFlowPayload(source.defaultFlowInterval);
      }
      await setPresentation(source.defaultPresentation||'line',source.defaultInterval||'daily',{skipRebuild:true});
      for(const [id,a] of Object.entries(payload.axes)){
        const el=$('scale-'+id);if(el)el.value=a.mode;
      }
      $('events').setAttribute('aria-pressed',String(state.events));
      $('support-resistance').setAttribute('aria-pressed','true');
      rebuild();
      for(const s of payload.series)setVisible(s,!defaultHidden.has(s.id));
      for(const e of view.events)e.setVisible(state.events);
      selectRange(payload.defaultRange);
      readingAt(null);
      $('status').textContent='';
    }
    $('reset').onclick=()=>resetChart().catch(e=>{$('status').textContent=`Reset failed: ${e.message}`;});

    const clearPreset=()=>{state.activePreset=null;state.requestedRange=null;$('ranges').querySelectorAll('button').forEach(b=>b.setAttribute('aria-pressed','false'));};
    $('chart').addEventListener('wheel',clearPreset,{passive:true});$('chart').addEventListener('pointerdown',clearPreset);

    function csvCell(value){
      const text=value===null||value===undefined?'':String(value);
      return /[",\n\r]/.test(text)?`"${text.replace(/"/g,'""')}"`:text;
    }
    function exportCsv(){
      const activeSeries=orderedSeries(payload).filter(s=>state.visible.has(s.id));
      const header=[numeric?(payload.xAxisLabel||'day'):'date',...activeSeries.map(s=>s.name)];
      const lines=[header.map(csvCell).join(',')];
      for(let i=0;i<payload.x.length;i++){
        const row=[payload.x[i]];
        for(const s of activeSeries){
          const local=i-s.start;
          const v=local>=0&&local<s.values.length?s.values[local]:null;
          row.push(Number.isFinite(v)?v:'');
        }
        lines.push(row.map(csvCell).join(','));
      }
      const blob=new Blob([lines.join('\n')+'\n'],{type:'text/csv;charset=utf-8'});
      const url=URL.createObjectURL(blob);
      const a=document.createElement('a');
      a.href=url;a.download=`${payload.id}_${payload.dataDate||payload.reportDate}.csv`;a.click();
      setTimeout(()=>URL.revokeObjectURL(url),1000);
      $('status').textContent='CSV exported.';
    }
    if($('export-csv'))$('export-csv').onclick=exportCsv;

    function buildShareUrl(){
      const url=new URL(location.href);
      for(const key of ['embed','capture','range','style','interval','flow','scale_right','scale_left','events','sr','hidden']){
        url.searchParams.delete(key);
      }
      if(state.activePreset&&state.activePreset!==payload.defaultRange)url.searchParams.set('range',state.activePreset);
      if(state.presentation!==(source.defaultPresentation||'line'))url.searchParams.set('style',state.presentation);
      if(state.presentation==='candles'&&state.interval!==(source.defaultInterval||'daily'))url.searchParams.set('interval',state.interval);
      if(source.flowIntervals&&state.flowInterval!==source.defaultFlowInterval)url.searchParams.set('flow',state.flowInterval);
      for(const [id,axis] of Object.entries(source.axes)){
        if(state.modes[id]&&state.modes[id]!==axis.mode)url.searchParams.set('scale_'+id,state.modes[id]);
      }
      if(payload.events.length&&state.events!==defaultEvents)url.searchParams.set('events',state.events?'1':'0');
      if((source.referenceLines||[]).some(l=>['support','resistance'].includes(l.type))&&!state.supportResistance)url.searchParams.set('sr','0');
      const hidden=payload.series.filter(s=>!state.visible.has(s.id)).map(s=>s.id).sort();
      const defaultHiddenSorted=[...defaultHidden].filter(id=>payload.series.some(s=>s.id===id)).sort();
      if(hidden.join(',')!==defaultHiddenSorted.join(','))url.searchParams.set('hidden',hidden.join(','));
      return url.toString();
    }
    async function copyShareLink(){
      const shareUrl=buildShareUrl();
      try{
        await navigator.clipboard.writeText(shareUrl);
      }catch{
        const input=document.createElement('textarea');input.value=shareUrl;input.style.cssText='position:fixed;opacity:0';
        document.body.append(input);input.select();document.execCommand('copy');input.remove();
      }
      $('status').textContent='Link copied to clipboard.';
    }
    if($('copy-link'))$('copy-link').onclick=()=>copyShareLink().catch(e=>{$('status').textContent=`Copy failed: ${e.message}`;});

    async function exportImage(download=true){
      if(exporting)throw new Error('Export already in progress');exporting=true;$('export').disabled=true;
      let host,exportView;
      const scenarios=(payload.referenceLines||[]).filter(level=>!level.type||level.type==='case'),plotTop=scenarios.length?410:300,plotHeight=1155-plotTop;
      try{
        const range=view.range()||{from:payload.x[0],to:payload.x.at(-1)};
        host=document.createElement('div');host.style.cssText=`position:fixed;left:-10000px;top:0;width:1690px;height:${plotHeight}px`;host.setAttribute('aria-hidden','true');document.body.append(host);
        exportView=createView(host,payload,state,false);
        await new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)));
        exportView.setRange(range);
        for(const c of exportView.charts)c.applyOptions({layout:{fontSize:14}});
        await new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)));
        const canvas=document.createElement('canvas');canvas.width=2400;canvas.height=1350;const ctx=canvas.getContext('2d');ctx.fillStyle=THEME.bg;ctx.fillRect(0,0,2400,1350);
        function text(value,x,y,size=20,color=THEME.text,font='JetBrains Mono',weight=400,maxWidth){ctx.font=`${weight} ${size}px "${font}"`;ctx.fillStyle=color;if(maxWidth)ctx.fillText(value,x,y,maxWidth);else ctx.fillText(value,x,y);}
        function rule(y,color=THEME.border){ctx.strokeStyle=color;ctx.lineWidth=1.5;ctx.beginPath();ctx.moveTo(64,y);ctx.lineTo(2336,y);ctx.stroke();}
        ctx.fillStyle=THEME.accent;ctx.fillRect(64,51,10,20);text('SECRET SATOSHIS',94,70,22,THEME.text,'JetBrains Mono',600);text(payload.category.toUpperCase(),1790,70,16,THEME.dim);rule(103);
        text(payload.title,64,178,50,THEME.text,'Syne',700,2240);
        text(`${xLabel(range.from,payload)} — ${xLabel(rangeEnd(range),payload)}${payload.candles?'  /  '+state.interval.toUpperCase()+' CANDLES':''}${payload.flowInterval?'  /  '+payload.flowInterval.toUpperCase()+' FLOWS':''}  /  ${Object.entries(state.modes).map(([k,v])=>`${payload.panels?.find(p=>p.axis===k)?.control_label||k}: ${v}`).join(' · ')}`,64,223,18,THEME.dim);
        text(`Data through ${payload.dataDate||payload.reportDate}`,1870,223,18,THEME.dim);
        if(scenarios.length){
          const width=2272/scenarios.length;
          scenarios.forEach((scenario,i)=>{const x=64+i*width;
            ctx.fillStyle=scenario.color;ctx.fillRect(x,279,3,64);
            text(scenario.name.toUpperCase(),x+20,299,16,THEME.dim);
            text('$'+decimal(scenario.price,0),x+20,336,28,THEME.text,'JetBrains Mono',600);
          });
        }
        rule(scenarios.length?367:255,THEME.accent);
        const hostBounds=host.getBoundingClientRect();
        for(const [i,chart] of exportView.charts.entries()){
          const bounds=exportView.nodes[i].getBoundingClientRect(),top=plotTop+bounds.top-hostBounds.top;
          ctx.drawImage(chart.takeScreenshot(true,false),64,top,bounds.width,bounds.height);
          if(payload.panels){
            const label=payload.panels[i].label;
            ctx.font='600 16px "JetBrains Mono"';ctx.fillStyle=THEME.bg;
            ctx.fillRect(72,top+4,ctx.measureText(label).width+20,29);
            text(label,82,top+25,16,THEME.dim,'JetBrains Mono',600);
          }
        }
        text(numeric?`AT DAY ${payload.readingPoint}`:payload.dataDate?'AT LATEST DATA':'AT REPORT DATE',1800,plotTop,15,THEME.dim);
        const selected=orderedSeries(payload).filter(s=>state.visible.has(s.id));
        function legendEntry(s,y,dense){
          ctx.fillStyle=s.color;ctx.fillRect(1800,y-7,18,3);
          text(s.name,1830,y,dense?13:16,THEME.dim,'JetBrains Mono',400,dense?245:485);
          text(readout(valueAt(s,payload.readingPoint,payload),payload.axes[s.axis].unit),dense?2090:1830,dense?y:y+30,dense?16:24,s.color,'JetBrains Mono',600,dense?230:485);
        }
        if(payload.panels){
          for(const [i,panel] of payload.panels.entries()){
            const bounds=exportView.nodes[i].getBoundingClientRect(),top=plotTop+bounds.top-hostBounds.top;
            text(panel.control_label.toUpperCase(),1800,top+28,14,THEME.dim,'JetBrains Mono',600);
            const group=selected.filter(s=>s.axis===panel.axis);
            const available=Math.max(40,bounds.height-80);
            const spacing=Math.min(100,available/Math.max(1,group.length));
            const panelDense=group.length>1&&spacing<56;
            group.forEach((s,j)=>legendEntry(s,top+65+j*spacing,panelDense));
          }
        }else{
          const step=Math.min(100,(plotHeight-40)/Math.max(1,selected.length));
          const singleDense=selected.length>12||step<56;
          selected.forEach((s,i)=>legendEntry(s,plotTop+40+i*step,singleDense));
        }
        rule(1193);text(`${payload.source} · Bitcoin Report Library`,64,1230,17,THEME.dim);
        text(Object.values(payload.axes).map(a=>a.label).join(' · ')+(payload.candles?' · '+periodLabel(payload.candles.at(-1))+' through '+payload.candles.at(-1).observationDate:''),64,1260,15,THEME.dim,'JetBrains Mono',400,2150);
        text('SecretSatoshis.com',64,1310,18);text('TradingView Lightweight Charts™ · © 2025 TradingView, Inc. · tradingview.com',1130,1310,14,THEME.dim);
        const dataUrl=canvas.toDataURL('image/png');
        if(download){const a=document.createElement('a');a.href=dataUrl;a.download=`${payload.id}_${payload.dataDate||payload.reportDate}.png`;a.click();$('status').textContent='PNG exported.';}
        return dataUrl;
      }finally{exportView?.destroy();host?.remove();exporting=false;$('export').disabled=false;}
    }
    $('export').onclick=()=>exportImage().catch(e=>{$('status').textContent=`Export failed: ${e.message}`;});
    $('download-data').onclick=e=>{e.preventDefault();const url=URL.createObjectURL(new Blob([JSON.stringify(payload)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download=payload.id+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
    await new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)));

    // Apply initial defaults + optional URL view-state overrides.
    const urlFlow=params.get('flow');
    if(urlFlow&&source.flowIntervals?.includes(urlFlow)&&urlFlow!==state.flowInterval){
      applyFlowPayload(urlFlow);rebuild();
    }
    for(const [id,axis] of Object.entries(payload.axes)){
      const requestedScale=params.get('scale_'+id);
      if(requestedScale&&(axis.modes||['linear','log']).includes(requestedScale)&&requestedScale!==state.modes[id]){
        state.modes[id]=requestedScale;$('scale-'+id).value=requestedScale;rebuild();
      }
    }
    const urlEvents=params.get('events');
    if(urlEvents==='0'||urlEvents==='1'){
      state.events=urlEvents==='1';for(const e of view.events)e.setVisible(state.events);$('events').setAttribute('aria-pressed',String(state.events));
    }
    if(params.get('sr')==='0'){
      state.supportResistance=false;view.supportResistance(false);$('support-resistance').setAttribute('aria-pressed','false');
    }
    if(params.has('hidden')){
      const hiddenParam=new Set(params.get('hidden').split(',').filter(Boolean));
      if(payload.series.some(s=>!hiddenParam.has(s.id))){
        for(const s of payload.series)setVisible(s,!hiddenParam.has(s.id));
      }
    }
    const urlRange=params.get('range');
    selectRange(urlRange&&ranges.includes(urlRange)?urlRange:payload.defaultRange);
    const urlStyle=params.get('style'),urlInterval=params.get('interval');
    if(urlStyle==='candles'||(!urlStyle&&source.defaultPresentation==='candles')){
      const targetInterval=urlInterval&&availableCandleIntervals.includes(urlInterval)?urlInterval:(source.defaultInterval||'daily');
      await setPresentation('candles',targetInterval);
    }else if(urlStyle==='line'&&source.defaultPresentation==='candles'){
      await setPresentation('line');
    }

    $('loading').hidden=true;
    if(payload.unavailable.length)$('status').textContent=`Unavailable: ${payload.unavailable.join(', ')}`;
    function setCapture(on){document.body.classList.toggle('capture',Boolean(on));layoutLegend();legendFade();reportHeight();}
    const api={ready,get payload(){return payload;},exportImage,exportCsv,buildShareUrl,selectRange,setScale,setPresentation,setFlowInterval,setCapture,resetChart,get readingPoint(){return reading;},get view(){return view;},get state(){return state;}};
    window.SecretSatoshisChart=api;new ResizeObserver(reportHeight).observe(document.body);reportHeight();await new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)));resolveReady(api);
  }
  initialize().catch(error=>{$('loading').textContent=`Unable to load chart: ${error.message}`;$('status').textContent='Chart could not be rendered.';console.error(error);rejectReady(error);});
})();
