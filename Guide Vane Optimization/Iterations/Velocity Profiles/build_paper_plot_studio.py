"""Build a selectable, offline profile plotter from files beside this script.

Usage: python build_paper_plot_studio.py [--folder PATH] [--no-open]
The default folder is the folder containing this script. No input is modified.
"""

import argparse
import csv
import json
from pathlib import Path
import re
import webbrowser


NUMBER = re.compile(r"^[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?$")
XY_LINE = re.compile(r"^\s*([-+\d.eE]+)[\s,;]+([-+\d.eE]+)\s*$")


def number(value: str) -> bool:
    return bool(NUMBER.fullmatch(value.strip()))


def clean_pairs(rows):
    pairs = sorted((float(x), float(y)) for x, y in rows)
    unique = {}
    for x, y in pairs:
        unique.setdefault(x, []).append(y)
    return [[x, sum(ys) / len(ys)] for x, ys in sorted(unique.items())]


def read_profile(path: Path):
    """Return one selectable file with one or more numeric Y columns."""
    if path.stat().st_size > 12_000_000:
        return None
    raw = path.read_text(encoding="utf-8-sig", errors="ignore")
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not lines:
        return None

    # CSV exports may have several velocity columns. Keep one file row in the UI
    # and let the user choose which numeric column to plot.
    if path.suffix.lower() in {".csv", ".tsv"}:
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
        rows = list(csv.reader(lines, delimiter=delimiter))
        if not rows:
            return None
        first = [v.strip() for v in rows[0]]
        has_header = any(not number(v) for v in first)
        headers = first if has_header else [f"Column {i+1}" for i in range(len(first))]
        body = rows[1:] if has_header else rows
        columns = []
        for i in range(1, len(headers)):
            pairs = []
            for row in body:
                if len(row) > i and number(row[0]) and number(row[i]):
                    pairs.append((row[0], row[i]))
            pairs = clean_pairs(pairs)
            if len(pairs) >= 3:
                columns.append({"name": headers[i], "points": pairs})
        if columns:
            return {"name": path.name, "xName": headers[0], "columns": columns}

    # ANSYS Fluent XY files have a text preamble and two-column numeric lines.
    pairs = [m.groups() for line in lines if (m := XY_LINE.fullmatch(line))]
    pairs = clean_pairs(pairs)
    if len(pairs) >= 3:
        return {"name": path.name, "xName": "Position", "columns": [{"name": "Velocity", "points": pairs}]}
    return None


HTML = r'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Paper profile plotter</title><style>
:root{font-family:Segoe UI,Arial,sans-serif;color:#192432;background:#f2f4f7}*{box-sizing:border-box}body{margin:0}[hidden]{display:none!important}
main{max-width:1320px;margin:auto;padding:25px 22px 50px}h1{font-size:25px;margin:0 0 6px}h2{font-size:17px;margin:0 0 14px}
.sub,.note{color:#536171;font-size:13px;line-height:1.45}.sub{margin:0 0 18px}.card{background:white;border:1px solid #dce2e8;border-radius:10px;padding:18px;margin:0 0 16px;box-shadow:0 3px 15px #1825340b}
.controls{display:flex;gap:13px;flex-wrap:wrap;align-items:end}.field{display:flex;flex-direction:column;gap:5px;font-size:12px;font-weight:600}
input,select{font:inherit;border:1px solid #b9c5d0;border-radius:5px;padding:7px;background:white;color:#172331}
input[type=checkbox]{width:17px;height:17px;accent-color:#176394}input[type=color]{padding:0;width:42px;height:32px}
.field input[type=number]{width:96px}.field input[type=text]{width:210px}.tableWrap{overflow:auto;max-height:410px}
table{border-collapse:collapse;width:100%;font-size:13px}td,th{border-bottom:1px solid #e6eaf0;padding:7px 9px;text-align:left;white-space:nowrap}
th{position:sticky;top:0;background:#f4f7fa;z-index:1;color:#465567}.name{font-weight:600}.styleLabel{width:180px}.width{width:70px}
button{border:1px solid #aebdca;background:white;color:#193148;border-radius:6px;padding:8px 12px;font-weight:600;cursor:pointer}button:hover{background:#e9f2f9}
.actions{display:flex;gap:8px;flex-wrap:wrap}.heading{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap}
svg{width:100%;height:auto;display:block;border:1px solid #e4e8ec}#empty{padding:42px;color:#627386;text-align:center}
.tabs{display:flex;gap:8px;margin:14px 0 20px}.tabs button.active{background:#176394;color:white;border-color:#176394}
.sheetTable input{width:100%;min-width:112px;border:0;border-radius:0;background:transparent;padding:7px}.sheetTable th input{font-weight:700;min-width:135px}
.sheetTable td{padding:0}.sheetTable th:first-child,.sheetTable td:first-child{width:42px;text-align:right;color:#6a7886;background:#f5f7f9;padding:7px}
.sheetTable input:focus{outline:2px solid #176394;background:white}.sheetTable{min-width:430px}.sheetWrap{max-height:390px;overflow:auto;border:1px solid #dce2e8}
@media print{body{background:white}main{padding:0;max-width:none}.settings,.sub,.actions{display:none!important}.card{box-shadow:none;border:0;margin:0;padding:0}svg{border:0}}
</style></head><body><main><h1>Paper figure plotter</h1>
<nav class="tabs settings" aria-label="Plotting tools"><button id="profileTabButton" class="active" type="button">Velocity profiles</button><button id="generalTabButton" type="button">General figures</button></nav>
<div id="profileTab">
<p class="sub">Choose curves from the usable data files in this folder. Run the Python script again after adding or replacing files.</p>
<section class="card settings"><h2>Figure settings</h2><div class="controls">
<label class="field">Title<input id="title" type="text" value="Outlet velocity profiles"></label>
<label class="field">X-axis label<input id="xlabel" type="text" value="Outlet position, x (m)"></label>
<label class="field">Y-axis label<input id="ylabel" type="text" value="Velocity magnitude (m/s)"></label>
<label class="field">X min<input id="xmin" type="number" step="any" placeholder="Auto"></label>
<label class="field">X max<input id="xmax" type="number" step="any" placeholder="Auto"></label>
<label class="field">Y min<input id="ymin" type="number" step="any" placeholder="Auto"></label>
<label class="field">Y max<input id="ymax" type="number" step="any" placeholder="Auto"></label>
<label class="field">Width (in)<input id="width" type="number" min="3" max="16" step="0.1" value="7.2"></label>
<label class="field">Height (in)<input id="height" type="number" min="2" max="12" step="0.1" value="4.2"></label>
<label class="field">Font (pt)<input id="font" type="number" min="7" max="20" step="0.5" value="10"></label>
<label class="field">Grid<input id="grid" type="checkbox" checked></label>
<label class="field">Legend<input id="legend" type="checkbox" checked></label>
</div></section>
<section class="card settings"><div class="heading"><h2>Files in folder</h2><div class="actions"><button id="all">Select all</button><button id="none">Clear selection</button></div></div>
<div class="tableWrap"><table><thead><tr><th>Plot</th><th>File</th><th>Y column</th><th>Legend label</th><th>Colour</th><th>Width</th><th>Line</th></tr></thead><tbody id="rows"></tbody></table></div>
<p class="note" id="count"></p></section>
<section class="card"><div class="heading"><h2>Figure preview</h2><div class="actions"><button id="svgButton">Export SVG</button><button id="pngButton">Export PNG (300 dpi)</button><button id="printButton">Print / PDF</button></div></div>
<div id="empty" hidden>Select at least one file to draw the figure.</div><svg id="plot" viewBox="0 0 1200 700" role="img" aria-label="Selected profile curves"></svg></section>
<section class="card"><div class="heading"><h2>RMS deviation between curves</h2><div class="actions"><button id="rmsSvg">Export SVG</button><button id="rmsPng">Export PNG (300 dpi)</button><button id="rmsCsv">Export table CSV</button></div></div>
<div class="controls settings">
<label class="field">Comparison<select id="rmsMode"><option value="pair">Two chosen curves</option><option value="adjacent">Successive selected curves</option></select></label>
<label class="field">Curve A<select id="rmsA"></select></label>
<label class="field">Curve A Y column<select id="rmsAColumn"></select></label>
<label class="field">Curve B<select id="rmsB"></select></label>
<label class="field">Curve B Y column<select id="rmsBColumn"></select></label>
<label class="field">Plot value<select id="rmsMetric"><option value="absolute">RMS deviation (Y units)</option><option value="relative">Relative RMS deviation (%)</option></select></label>
<label class="field">Compare X min<input id="rmsXmin" type="number" step="any" placeholder="Common range"></label>
<label class="field">Compare X max<input id="rmsXmax" type="number" step="any" placeholder="Common range"></label>
<label class="field">Chart title<input id="rmsTitle" type="text" value="RMS deviation between curves"></label>
<label class="field">Horizontal axis<input id="rmsXlabel" type="text" value="Curve pair"></label>
<label class="field">Vertical axis<input id="rmsYlabel" type="text" value="RMS deviation (m/s)"></label>
<label class="field">Legend text<input id="rmsLegendText" type="text" value="RMS deviation"></label>
<label class="field">Legend<input id="rmsLegend" type="checkbox" checked></label>
<label class="field">Colour<input id="rmsColor" type="color" value="#176394"></label>
<label class="field">Line width<input id="rmsLineWidth" type="number" min="0.5" max="8" step="0.1" value="3"></label>
<label class="field">Line pattern<select id="rmsDash"><option value="solid">Solid</option><option value="dashed">Dashed</option><option value="dotted">Dotted</option></select></label>
<label class="field">Width (in)<input id="rmsWidth" type="number" min="3" max="16" step="0.1" value="7.2"></label>
<label class="field">Height (in)<input id="rmsHeight" type="number" min="2" max="12" step="0.1" value="3.6"></label>
<label class="field">Font (pt)<input id="rmsFont" type="number" min="7" max="20" step="0.5" value="10"></label>
</div><p class="note" id="rmsNote">RMS deviation is evaluated at 240 common positions within the overlapping X range using shape-preserving cubic interpolation.</p>
<svg id="rmsPlot" viewBox="0 0 1200 540" role="img" aria-label="RMS deviation between curves"></svg>
<p class="note" id="rmsStatus"></p><div class="tableWrap"><table><thead><tr><th>Curve A</th><th>Curve B</th><th>X interval</th><th>RMS deviation</th><th>Relative RMS deviation (%)</th></tr></thead><tbody id="rmsTable"></tbody></table></div></section>
</div><div id="generalTab" hidden>
<p class="sub">Enter data in the cells, paste a range from Excel, or import a CSV/TSV file. Choose any numeric X and Y columns to plot.</p>
<section class="card settings"><div class="heading"><h2>Data table</h2><div class="actions"><button id="addRow" type="button">Add row</button><button id="addColumn" type="button">Add column</button><button id="clearSheet" type="button">Clear cells</button><button id="downloadSheet" type="button">Download CSV</button><label class="field">Import CSV/TSV<input id="importSheet" type="file" accept=".csv,.tsv,.txt"></label></div></div>
<div class="sheetWrap"><table class="sheetTable"><thead id="sheetHead"></thead><tbody id="sheetBody"></tbody></table></div>
<p class="note">Click a cell and paste a block copied from a spreadsheet. Column names are editable.</p></section>
<section class="card settings"><h2>General figure settings</h2><div class="controls">
<label class="field">X column<select id="generalX"></select></label><label class="field">Y column<select id="generalY"></select></label>
<label class="field">Chart title<input id="generalTitle" type="text" value="Convergence with mesh refinement"></label>
<label class="field">Horizontal axis<input id="generalXlabel" type="text" value="Cell count"></label>
<label class="field">Vertical axis<input id="generalYlabel" type="text" value="Pressure loss"></label>
<label class="field">X min<input id="generalXmin" type="number" step="any" placeholder="Auto"></label>
<label class="field">X max<input id="generalXmax" type="number" step="any" placeholder="Auto"></label>
<label class="field">Y min<input id="generalYmin" type="number" step="any" placeholder="Auto"></label>
<label class="field">Y max<input id="generalYmax" type="number" step="any" placeholder="Auto"></label>
<label class="field">Legend text<input id="generalLegendText" type="text" value="Computed result"></label>
<label class="field">Chart type<select id="generalKind"><option value="line">Line and markers</option><option value="scatter">Markers only</option></select></label>
<label class="field">Sort by X<input id="generalSort" type="checkbox" checked></label>
<label class="field">Legend<input id="generalLegend" type="checkbox" checked></label>
<label class="field">Grid<input id="generalGrid" type="checkbox" checked></label>
<label class="field">Colour<input id="generalColor" type="color" value="#176394"></label>
<label class="field">Line width<input id="generalLineWidth" type="number" min="0.5" max="8" step="0.1" value="2.6"></label>
<label class="field">Width (in)<input id="generalWidth" type="number" min="3" max="16" step="0.1" value="7.2"></label>
<label class="field">Height (in)<input id="generalHeight" type="number" min="2" max="12" step="0.1" value="4.2"></label>
<label class="field">Font (pt)<input id="generalFont" type="number" min="7" max="20" step="0.5" value="10"></label>
</div></section>
<section class="card"><div class="heading"><h2>General figure preview</h2><div class="actions"><button id="generalSvg" type="button">Export SVG</button><button id="generalPng" type="button">Export PNG (300 dpi)</button><button id="generalPrint" type="button">Print / PDF</button></div></div>
<p id="generalStatus" class="note"></p><svg id="generalPlot" viewBox="0 0 1200 700" role="img" aria-label="General figure"></svg></section>
</div>
</main><script>
const FILES=__DATA__;
const NS='http://www.w3.org/2000/svg',palette=['#176394','#d65b36','#178465','#8a54a2','#b07a20','#58616d','#9b5574','#387f8d'];
const $=id=>document.getElementById(id);
const styles=FILES.map((f,i)=>({shown:true,column:Math.max(0,f.columns.findIndex(c=>/(?:new_|full_)fitted_curve|full_fitted_velocity/i.test(c.name))),label:f.name,color:palette[i%palette.length],width:i<2?2.5:1.8,dash:'solid'}));
function el(tag,attrs,parent){const e=document.createElementNS(NS,tag);for(const [k,v] of Object.entries(attrs))e.setAttribute(k,v);parent.appendChild(e);return e}
function txt(parent,x,y,value,attrs={}){const e=el('text',{x,y,...attrs},parent);e.textContent=value;return e}
function value(id,defaultValue){const n=Number($(id).value);return $(id).value.trim()&&Number.isFinite(n)?n:defaultValue}
function dash(v){return {solid:'',dashed:'9 5',dotted:'2 4'}[v]||''}
function addCell(row,child){const td=document.createElement('td');td.appendChild(child);row.appendChild(td)}
function rows(){const body=$('rows');body.replaceChildren();FILES.forEach((f,i)=>{const s=styles[i],tr=document.createElement('tr');
 const show=document.createElement('input');show.type='checkbox';show.checked=s.shown;show.onchange=()=>{s.shown=show.checked;draw()};addCell(tr,show);
 const name=document.createElement('span');name.className='name';name.textContent=f.name;addCell(tr,name);
 const col=document.createElement('select');f.columns.forEach((c,j)=>{const o=document.createElement('option');o.value=j;o.textContent=c.name;col.appendChild(o)});col.value=s.column;col.onchange=()=>{s.column=Number(col.value);draw()};addCell(tr,col);
 const label=document.createElement('input');label.className='styleLabel';label.value=s.label;label.oninput=()=>{s.label=label.value;draw()};addCell(tr,label);
 const color=document.createElement('input');color.type='color';color.value=s.color;color.oninput=()=>{s.color=color.value;draw()};addCell(tr,color);
 const width=document.createElement('input');width.type='number';width.className='width';width.min='0.5';width.max='8';width.step='0.1';width.value=s.width;width.oninput=()=>{s.width=Math.max(.5,Number(width.value)||1);draw()};addCell(tr,width);
 const pattern=document.createElement('select');for(const d of ['solid','dashed','dotted']){const o=document.createElement('option');o.value=d;o.textContent=d;pattern.appendChild(o)}pattern.value=s.dash;pattern.onchange=()=>{s.dash=pattern.value;draw()};addCell(tr,pattern);
 body.appendChild(tr)});$('count').textContent=`${FILES.length} usable data file${FILES.length===1?'':'s'} found.`}
function draw(){const chosen=FILES.map((f,i)=>[f,styles[i]]).filter(([,s])=>s.shown),svg=$('plot');svg.replaceChildren();$('empty').hidden=chosen.length>0;svg.hidden=chosen.length===0;if(!chosen.length)return;
 const W=1200,H=Math.round(1200*value('height',4.2)/value('width',7.2));svg.setAttribute('viewBox',`0 0 ${W} ${H}`);
 const legend=$('legend').checked,font=value('font',10)*1.65,m={l:94,r:25,t:64,b:legend?90+Math.ceil(chosen.length/4)*22:73};
 const all=chosen.flatMap(([f,s])=>f.columns[s.column].points),xs=all.map(p=>p[0]),ys=all.map(p=>p[1]);
 const ax=Math.min(...xs),bx=Math.max(...xs),ay=Math.min(...ys),by=Math.max(...ys),pad=(by-ay||1)*.06;
 let xmin=value('xmin',ax),xmax=value('xmax',bx),ymin=value('ymin',ay-pad),ymax=value('ymax',by+pad);if(!(xmax>xmin&&ymax>ymin))return;
 const xp=x=>m.l+(x-xmin)/(xmax-xmin)*(W-m.l-m.r),yp=y=>H-m.b-(y-ymin)/(ymax-ymin)*(H-m.t-m.b);
 el('rect',{x:0,y:0,width:W,height:H,fill:'white'},svg);const defs=el('defs',{},svg),clip=el('clipPath',{id:'plotClip'},defs);el('rect',{x:m.l,y:m.t,width:W-m.l-m.r,height:H-m.t-m.b},clip);
 for(let i=0;i<=5;i++){const x=xmin+i*(xmax-xmin)/5,px=xp(x),y=ymin+i*(ymax-ymin)/5,py=yp(y);if($('grid').checked){el('line',{x1:px,y1:m.t,x2:px,y2:H-m.b,stroke:'#e3e7ec'},svg);el('line',{x1:m.l,y1:py,x2:W-m.r,y2:py,stroke:'#e3e7ec'},svg)}
 txt(svg,px,H-m.b+23,x.toFixed(2),{'text-anchor':'middle',fill:'#313943','font-size':font*.8,'font-family':'Georgia,serif'});txt(svg,m.l-12,py+5,y.toFixed(2),{'text-anchor':'end',fill:'#313943','font-size':font*.8,'font-family':'Georgia,serif'})}
 el('line',{x1:m.l,y1:H-m.b,x2:W-m.r,y2:H-m.b,stroke:'#111','stroke-width':1.5},svg);el('line',{x1:m.l,y1:m.t,x2:m.l,y2:H-m.b,stroke:'#111','stroke-width':1.5},svg);
 const title=$('title').value.trim();if(title)txt(svg,W/2,32,title,{'text-anchor':'middle',fill:'#151b20','font-size':font*1.16,'font-family':'Georgia,serif','font-weight':'bold'});
 txt(svg,W/2,H-(legend?52:13),$('xlabel').value,{'text-anchor':'middle',fill:'#151b20','font-size':font,'font-family':'Georgia,serif'});
 const ytext=txt(svg,22,(m.t+H-m.b)/2,$('ylabel').value,{'text-anchor':'middle',fill:'#151b20','font-size':font,'font-family':'Georgia,serif'});ytext.setAttribute('transform',`rotate(-90 22 ${(m.t+H-m.b)/2})`);
 const lines=el('g',{'clip-path':'url(#plotClip)'},svg);chosen.forEach(([f,s])=>{const points=f.columns[s.column].points.map(p=>`${xp(p[0]).toFixed(2)},${yp(p[1]).toFixed(2)}`).join(' ');el('polyline',{points,fill:'none',stroke:s.color,'stroke-width':s.width,'stroke-dasharray':dash(s.dash),'stroke-linejoin':'round','stroke-linecap':'round'},lines)});
 if(legend){const cols=Math.min(4,chosen.length),dx=(W-m.l-m.r)/cols;chosen.forEach(([,s],i)=>{const x=m.l+(i%cols)*dx,y=H-29-Math.floor((chosen.length-1)/cols)*22+Math.floor(i/cols)*22;el('line',{x1:x,y1:y-4,x2:x+27,y2:y-4,stroke:s.color,'stroke-width':Math.max(s.width,2),'stroke-dasharray':dash(s.dash)},svg);txt(svg,x+34,y,s.label,{fill:'#26313d','font-size':font*.74,'font-family':'Georgia,serif'})})}}
function pchip(points){
 const n=points.length,h=[],d=[],m=new Array(n);for(let i=0;i<n-1;i++){h.push(points[i+1][0]-points[i][0]);d.push((points[i+1][1]-points[i][1])/h[i])}
 function end(a,b,da,db){let v=((2*a+b)*da-a*db)/(a+b);if(Math.sign(v)!==Math.sign(da))v=0;else if(Math.sign(da)!==Math.sign(db)&&Math.abs(v)>3*Math.abs(da))v=3*da;return v}
 m[0]=end(h[0],h[1],d[0],d[1]);m[n-1]=end(h[n-2],h[n-3],d[n-2],d[n-3]);
 for(let i=1;i<n-1;i++){if(d[i-1]===0||d[i]===0||Math.sign(d[i-1])!==Math.sign(d[i]))m[i]=0;else{const w1=2*h[i]+h[i-1],w2=h[i]+2*h[i-1];m[i]=(w1+w2)/(w1/d[i-1]+w2/d[i])}}
 return x=>{let lo=0,hi=n-1;while(hi-lo>1){const mid=(lo+hi)>>1;if(points[mid][0]<=x)lo=mid;else hi=mid}const t=(x-points[lo][0])/h[lo],t2=t*t,t3=t2*t;return (2*t3-3*t2+1)*points[lo][1]+(t3-2*t2+t)*h[lo]*m[lo]+(-2*t3+3*t2)*points[lo+1][1]+(t3-t2)*h[lo]*m[lo+1]}
}
let rmsResults=[];
function rmsCurve(which){const fileIndex=Number($('rms'+which).value),column=Number($('rms'+which+'Column').value),file=FILES[fileIndex];return {key:`${fileIndex}:${column}`,name:file.name+(file.columns.length>1?` — ${file.columns[column].name}`:''),points:file.columns[column].points}}
function rmsColumnOptions(which){const file=FILES[Number($('rms'+which).value)],select=$('rms'+which+'Column');select.replaceChildren();if(!file)return;file.columns.forEach((c,i)=>{const option=document.createElement('option');option.value=i;option.textContent=c.name;select.appendChild(option)});select.value=String(styles[Number($('rms'+which).value)].column);drawRms()}
function rmsCurveOptions(){for(const which of ['A','B']){const select=$('rms'+which);select.replaceChildren();FILES.forEach((file,i)=>{const option=document.createElement('option');option.value=i;option.textContent=file.name;select.appendChild(option)})} $('rmsA').value='0';$('rmsB').value=String(Math.min(1,FILES.length-1));rmsColumnOptions('A');rmsColumnOptions('B')}
function drawRms(){const svg=$('rmsPlot'),table=$('rmsTable');svg.replaceChildren();table.replaceChildren();rmsResults=[];const mode=$('rmsMode').value;
 const iter=o=>{const m=o.name.match(/iteration[-_ ]?(\d+)/i);return m?Number(m[1]):Infinity};
 let comparisons=[];
 if(mode==='pair'){if(FILES.length<2){$('rmsStatus').textContent='Add a second curve file to compare.';svg.hidden=true;return}const a=rmsCurve('A'),b=rmsCurve('B');comparisons=[{a,b,label:iter(a)<Infinity&&iter(b)<Infinity?`${iter(a)}→${iter(b)}`:'A → B'}]}
 else{let selected=FILES.map((file,i)=>({key:`${i}:${styles[i].column}`,name:file.name,points:file.columns[styles[i].column].points,shown:styles[i].shown})).filter(o=>o.shown);selected.sort((a,b)=>iter(a)-iter(b)||a.name.localeCompare(b.name));comparisons=selected.slice(1).map((b,i)=>({a:selected[i],b,label:iter(selected[i])<Infinity&&iter(b)<Infinity?`${iter(selected[i])}→${iter(b)}`:`${i+1}→${i+2}`}))}
 if(!comparisons.length){$('rmsStatus').textContent='Select at least two curves to calculate RMS deviation.';svg.hidden=true;return}svg.hidden=false;
 const all=[...new Map(comparisons.flatMap(c=>[c.a,c.b]).map(o=>[o.key,o])).values()],xmin=Math.max(...all.map(o=>o.points[0][0])),xmax=Math.min(...all.map(o=>o.points.at(-1)[0]));
 const lo=value('rmsXmin',xmin),hi=value('rmsXmax',xmax);if(!(hi>lo&&lo>=xmin&&hi<=xmax)){$('rmsStatus').textContent=`Choose an X interval inside the shared range ${xmin.toFixed(3)}–${xmax.toFixed(3)}.`;svg.hidden=true;return}
 const curves=new Map();for(const o of all)curves.set(o.key,pchip(o.points));
 rmsResults=comparisons.map(c=>{let sum=0,base=0;for(let j=0;j<240;j++){const x=lo+(hi-lo)*j/239,a=curves.get(c.a.key)(x),b=curves.get(c.b.key)(x);sum+=(b-a)**2;base+=a*a}const absolute=Math.sqrt(sum/240),relative=base>0?100*Math.sqrt(sum/base):NaN;return {a:c.a.name,b:c.b.name,label:c.label,lo,hi,absolute,relative}});
 const metric=$('rmsMetric').value,values=rmsResults.map(r=>metric==='relative'?r.relative:r.absolute);
 rmsResults.forEach(r=>{const tr=document.createElement('tr');for(const v of [r.a,r.b,`${lo.toFixed(4)}–${hi.toFixed(4)}`,r.absolute.toFixed(9),Number.isFinite(r.relative)?r.relative.toFixed(9):'undefined (zero RMS for A)']){const td=document.createElement('td');td.textContent=v;tr.appendChild(td)}table.appendChild(tr)});
 $('rmsStatus').textContent=`${rmsResults.length} comparison${rmsResults.length===1?'':'s'} calculated from 240 common X positions.`;
 if(values.some(v=>!Number.isFinite(v))){$('rmsStatus').textContent='Relative RMS deviation is undefined when curve A is zero over the comparison interval.';svg.hidden=true;return}
 const W=1200,H=Math.round(1200*value('rmsHeight',3.6)/value('rmsWidth',7.2)),showLegend=$('rmsLegend').checked;
 svg.setAttribute('viewBox',`0 0 ${W} ${H}`);
 const font=value('rmsFont',10)*1.65,color=$('rmsColor').value,lineWidth=value('rmsLineWidth',3),lineDash=dash($('rmsDash').value);
 const m={l:92,r:30,t:65,b:showLegend?105:76},top=Math.max(...values)*1.2||1,xp=i=>m.l+(i+.5)*(W-m.l-m.r)/values.length,yp=y=>H-m.b-y/top*(H-m.t-m.b);
 el('rect',{x:0,y:0,width:W,height:H,fill:'white'},svg);
 for(let j=0;j<=4;j++){const y=j*top/4,py=yp(y);el('line',{x1:m.l,y1:py,x2:W-m.r,y2:py,stroke:'#e3e7ec'},svg);txt(svg,m.l-12,py+5,y.toFixed(2),{'text-anchor':'end',fill:'#333','font-size':font*.95,'font-family':'Georgia,serif'})}
 el('line',{x1:m.l,y1:H-m.b,x2:W-m.r,y2:H-m.b,stroke:'#111','stroke-width':1.5},svg);el('line',{x1:m.l,y1:m.t,x2:m.l,y2:H-m.b,stroke:'#111','stroke-width':1.5},svg);
 const pts=values.map((v,i)=>`${xp(i)},${yp(v)}`).join(' ');el('polyline',{points:pts,fill:'none',stroke:color,'stroke-width':lineWidth,'stroke-dasharray':lineDash},svg);
 values.forEach((v,i)=>{const x=xp(i),y=yp(v);el('circle',{cx:x,cy:y,r:5,fill:color},svg);txt(svg,x,y-12,v.toFixed(3),{'text-anchor':'middle',fill:'#1b394b','font-size':font*.95,'font-family':'Georgia,serif'});txt(svg,x,H-m.b+27,comparisons[i].label,{'text-anchor':'middle',fill:'#333','font-size':font*.95,'font-family':'Georgia,serif'})});
 const title=$('rmsTitle').value.trim();if(title)txt(svg,W/2,32,title,{'text-anchor':'middle',fill:'#151b20','font-size':font*1.2,'font-family':'Georgia,serif','font-weight':'bold'});
 txt(svg,W/2,H-(showLegend?42:12),$('rmsXlabel').value,{'text-anchor':'middle',fill:'#111','font-size':font,'font-family':'Georgia,serif'});const yt=txt(svg,21,H/2,$('rmsYlabel').value,{'text-anchor':'middle',fill:'#111','font-size':font,'font-family':'Georgia,serif'});yt.setAttribute('transform',`rotate(-90 21 ${H/2})`);
 if(showLegend){const lx=W/2-90,ly=H-13;el('line',{x1:lx,y1:ly-4,x2:lx+34,y2:ly-4,stroke:color,'stroke-width':lineWidth,'stroke-dasharray':lineDash},svg);el('circle',{cx:lx+17,cy:ly-4,r:4,fill:color},svg);txt(svg,lx+43,ly,$('rmsLegendText').value,{fill:'#26313d','font-size':font*.9,'font-family':'Georgia,serif'})}
 $('rmsNote').textContent=`RMS deviation of curve B from curve A on 240 common positions, ${lo.toFixed(3)}–${hi.toFixed(3)}, using shape-preserving cubic interpolation. Relative deviation is normalized by the RMS magnitude of curve A.`+(mode==='adjacent'?' Successive mode uses the curves checked in “Files in folder”.':'');
}
function svgSource(id='plot'){const s=$(id).cloneNode(true);s.setAttribute('xmlns',NS);s.setAttribute('width',value(id==='rmsPlot'?'rmsWidth':'width',7.2)+'in');s.setAttribute('height',value(id==='rmsPlot'?'rmsHeight':'height',id==='rmsPlot'?3.6:4.2)+'in');return new XMLSerializer().serializeToString(s)}
function save(blob,name){const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}
$('svgButton').onclick=()=>{if(!$('plot').hidden)save(new Blob([svgSource()],{type:'image/svg+xml'}),'profile_figure.svg')};
$('pngButton').onclick=()=>{if($('plot').hidden)return;const img=new Image(),url=URL.createObjectURL(new Blob([svgSource()],{type:'image/svg+xml'}));img.onload=()=>{const c=document.createElement('canvas');c.width=Math.round(value('width',7.2)*300);c.height=Math.round(value('height',4.2)*300);const ctx=c.getContext('2d');ctx.fillStyle='white';ctx.fillRect(0,0,c.width,c.height);ctx.drawImage(img,0,0,c.width,c.height);c.toBlob(b=>{if(b)save(b,'profile_figure.png')},'image/png');URL.revokeObjectURL(url)};img.src=url};
function saveRmsPng(){if($('rmsPlot').hidden)return;const img=new Image(),url=URL.createObjectURL(new Blob([svgSource('rmsPlot')],{type:'image/svg+xml'}));img.onload=()=>{const c=document.createElement('canvas');c.width=Math.round(value('rmsWidth',7.2)*300);c.height=Math.round(value('rmsHeight',3.6)*300);const ctx=c.getContext('2d');ctx.fillStyle='white';ctx.fillRect(0,0,c.width,c.height);ctx.drawImage(img,0,0,c.width,c.height);c.toBlob(b=>{if(b)save(b,'profile_rmse.png')},'image/png');URL.revokeObjectURL(url)};img.src=url}
$('rmsSvg').onclick=()=>{if(!$('rmsPlot').hidden)save(new Blob([svgSource('rmsPlot')],{type:'image/svg+xml'}),'profile_rmse.svg')};$('rmsPng').onclick=saveRmsPng;
$('rmsCsv').onclick=()=>{if(!rmsResults.length)return;const head=['Curve A','Curve B','X min','X max','RMS deviation (Y units)','Relative RMS deviation (%)'];const data=[head,...rmsResults.map(r=>[r.a,r.b,r.lo,r.hi,r.absolute,r.relative])];const csv=data.map(row=>row.map(v=>'"'+String(v).replaceAll('"','""')+'"').join(',')).join('\r\n');save(new Blob([csv],{type:'text/csv'}),'rms_deviation_table.csv')};
$('printButton').onclick=()=>window.print();$('all').onclick=()=>{styles.forEach(s=>s.shown=true);rows();draw()};$('none').onclick=()=>{styles.forEach(s=>s.shown=false);rows();draw()};
for(const id of ['title','xlabel','ylabel','xmin','xmax','ymin','ymax','width','height','font','grid','legend'])$(id).addEventListener('input',draw);
const drawProfiles=draw;draw=function(){drawProfiles();drawRms()};
$('rmsMode').onchange=drawRms;$('rmsA').onchange=()=>rmsColumnOptions('A');$('rmsB').onchange=()=>rmsColumnOptions('B');$('rmsAColumn').onchange=drawRms;$('rmsBColumn').onchange=drawRms;
$('rmsMetric').onchange=()=>{$('rmsYlabel').value=$('rmsMetric').value==='relative'?'Relative RMS deviation (%)':'RMS deviation (m/s)';$('rmsLegendText').value=$('rmsMetric').value==='relative'?'Relative RMS deviation':'RMS deviation';drawRms()};
$('rmsXmin').oninput=drawRms;$('rmsXmax').oninput=drawRms;
for(const id of ['rmsTitle','rmsXlabel','rmsYlabel','rmsLegendText','rmsLegend','rmsColor','rmsLineWidth','rmsDash','rmsWidth','rmsHeight','rmsFont'])$(id).addEventListener('input',drawRms);
rmsCurveOptions();rows();draw();
let sheetHeaders=['Cell count','Pressure loss','RMS change'];
let sheet=Array.from({length:8},()=>sheetHeaders.map(() => ''));
function ensureSize(rowCount,columnCount){while(sheetHeaders.length<columnCount){sheetHeaders.push(`Column ${sheetHeaders.length+1}`);sheet.forEach(row=>row.push(''))}while(sheet.length<rowCount)sheet.push(sheetHeaders.map(() => ''))}
function refreshGeneralSelectors(){const x=$('generalX'),y=$('generalY'),oldX=Number(x.value)||0,oldY=Number(y.value)||1;x.replaceChildren();y.replaceChildren();sheetHeaders.forEach((name,i)=>{for(const select of [x,y]){const o=document.createElement('option');o.value=i;o.textContent=name;select.appendChild(o)}});x.value=String(Math.min(oldX,sheetHeaders.length-1));y.value=String(Math.min(oldY,sheetHeaders.length-1))}
function renderSheet(){const head=$('sheetHead'),body=$('sheetBody');head.replaceChildren();body.replaceChildren();const hr=document.createElement('tr'),blank=document.createElement('th');blank.textContent='#';hr.appendChild(blank);
 sheetHeaders.forEach((name,c)=>{const th=document.createElement('th'),input=document.createElement('input');input.value=name;input.setAttribute('aria-label',`Column ${c+1} name`);input.oninput=()=>{sheetHeaders[c]=input.value;refreshGeneralSelectors();drawGeneral()};th.appendChild(input);hr.appendChild(th)});head.appendChild(hr);
 sheet.forEach((values,r)=>{const tr=document.createElement('tr'),index=document.createElement('td');index.textContent=String(r+1);tr.appendChild(index);values.forEach((v,c)=>{const td=document.createElement('td'),input=document.createElement('input');input.value=v;input.setAttribute('aria-label',`Row ${r+1}, column ${c+1}`);input.oninput=()=>{sheet[r][c]=input.value;drawGeneral()};input.onpaste=ev=>{const text=ev.clipboardData.getData('text');if(!/[\t\r\n]/.test(text))return;ev.preventDefault();const block=text.trimEnd().split(/\r?\n/).map(line=>line.split('\t'));const width=Math.max(...block.map(row=>row.length));ensureSize(r+block.length,c+width);block.forEach((row,ri)=>row.forEach((cell,ci)=>{sheet[r+ri][c+ci]=cell}));renderSheet();drawGeneral()};td.appendChild(input);tr.appendChild(td)});body.appendChild(tr)});
 refreshGeneralSelectors();drawGeneral()}
function parseDelimited(text,delimiter){const rows=[],row=[];let field='',quoted=false;for(let i=0;i<text.length;i++){const ch=text[i];if(ch==='"'){if(quoted&&text[i+1]==='"'){field+='"';i++}else quoted=!quoted}else if(ch===delimiter&&!quoted){row.push(field);field=''}else if((ch==='\n'||ch==='\r')&&!quoted){if(ch==='\r'&&text[i+1]==='\n')i++;row.push(field);if(row.some(v=>v.trim()!==''))rows.push(row.splice(0));else row.length=0;field=''}else field+=ch}row.push(field);if(row.some(v=>v.trim()!==''))rows.push(row);return rows}
function generalTick(v){const a=Math.abs(v);if(a>=1e6)return (v/1e6).toFixed(2)+'M';if(a>=1e4)return (v/1e3).toFixed(1)+'k';if(a>=1e3)return (v/1e3).toFixed(2)+'k';return Number(v.toFixed(a>=10?1:3)).toString()}
function drawGeneral(){const svg=$('generalPlot');svg.replaceChildren();const xi=Number($('generalX').value),yi=Number($('generalY').value);if(!sheetHeaders.length||xi===yi){$('generalStatus').textContent='Choose different X and Y columns.';svg.hidden=true;return}
 let points=sheet.map(row=>[row[xi],row[yi]]).filter(pair=>pair.every(v=>v!==undefined&&String(v).trim()!==''&&Number.isFinite(Number(v)))).map(pair=>pair.map(Number));
 if(points.length<2){$('generalStatus').textContent='Enter or import at least two rows with numeric X and Y values.';svg.hidden=true;return}svg.hidden=false;
 if($('generalSort').checked)points.sort((a,b)=>a[0]-b[0]);$('generalStatus').textContent=`Plotting ${points.length} numeric rows from “${sheetHeaders[xi]}” and “${sheetHeaders[yi]}”.`;
 const W=1200,H=Math.round(1200*value('generalHeight',4.2)/value('generalWidth',7.2)),legend=$('generalLegend').checked,font=value('generalFont',10)*1.65,m={l:96,r:25,t:64,b:legend?104:73};svg.setAttribute('viewBox',`0 0 ${W} ${H}`);
 const xs=points.map(p=>p[0]),ys=points.map(p=>p[1]),minX=Math.min(...xs),maxX=Math.max(...xs),minY=Math.min(...ys),maxY=Math.max(...ys),xPad=(maxX-minX||1)*.04,yPad=(maxY-minY||1)*.08;
 const xmin=value('generalXmin',minX-xPad),xmax=value('generalXmax',maxX+xPad),ymin=value('generalYmin',minY-yPad),ymax=value('generalYmax',maxY+yPad);
 if(!(xmax>xmin&&ymax>ymin)){$('generalStatus').textContent='Axis maximum must exceed minimum.';svg.hidden=true;return}
 const xp=x=>m.l+(x-xmin)/(xmax-xmin)*(W-m.l-m.r),yp=y=>H-m.b-(y-ymin)/(ymax-ymin)*(H-m.t-m.b),color=$('generalColor').value;
 el('rect',{x:0,y:0,width:W,height:H,fill:'white'},svg);const defs=el('defs',{},svg),clip=el('clipPath',{id:'generalClip'},defs);el('rect',{x:m.l,y:m.t,width:W-m.l-m.r,height:H-m.t-m.b},clip);
 for(let j=0;j<=5;j++){const x=xmin+j*(xmax-xmin)/5,y=ymin+j*(ymax-ymin)/5,px=xp(x),py=yp(y);if($('generalGrid').checked){el('line',{x1:px,y1:m.t,x2:px,y2:H-m.b,stroke:'#e3e7ec'},svg);el('line',{x1:m.l,y1:py,x2:W-m.r,y2:py,stroke:'#e3e7ec'},svg)}txt(svg,px,H-m.b+23,generalTick(x),{'text-anchor':'middle',fill:'#313943','font-size':font*.8,'font-family':'Georgia,serif'});txt(svg,m.l-12,py+5,generalTick(y),{'text-anchor':'end',fill:'#313943','font-size':font*.8,'font-family':'Georgia,serif'})}
 el('line',{x1:m.l,y1:H-m.b,x2:W-m.r,y2:H-m.b,stroke:'#111','stroke-width':1.5},svg);el('line',{x1:m.l,y1:m.t,x2:m.l,y2:H-m.b,stroke:'#111','stroke-width':1.5},svg);
 const title=$('generalTitle').value.trim();if(title)txt(svg,W/2,32,title,{'text-anchor':'middle',fill:'#151b20','font-size':font*1.16,'font-family':'Georgia,serif','font-weight':'bold'});
 txt(svg,W/2,H-(legend?43:12),$('generalXlabel').value,{'text-anchor':'middle',fill:'#151b20','font-size':font,'font-family':'Georgia,serif'});
 const yt=txt(svg,22,(m.t+H-m.b)/2,$('generalYlabel').value,{'text-anchor':'middle',fill:'#151b20','font-size':font,'font-family':'Georgia,serif'});yt.setAttribute('transform',`rotate(-90 22 ${(m.t+H-m.b)/2})`);
 const plot=el('g',{'clip-path':'url(#generalClip)'},svg);if($('generalKind').value==='line')el('polyline',{points:points.map(p=>`${xp(p[0]).toFixed(2)},${yp(p[1]).toFixed(2)}`).join(' '),fill:'none',stroke:color,'stroke-width':value('generalLineWidth',2.6),'stroke-linecap':'round','stroke-linejoin':'round'},plot);
 points.forEach(p=>el('circle',{cx:xp(p[0]),cy:yp(p[1]),r:4.5,fill:color},plot));
 if(legend){const lx=W/2-95,ly=H-13;el('line',{x1:lx,y1:ly-4,x2:lx+34,y2:ly-4,stroke:color,'stroke-width':value('generalLineWidth',2.6)},svg);el('circle',{cx:lx+17,cy:ly-4,r:4,fill:color},svg);txt(svg,lx+43,ly,$('generalLegendText').value,{fill:'#26313d','font-size':font*.9,'font-family':'Georgia,serif'})}}
function generalSvgSource(){const s=$('generalPlot').cloneNode(true);s.setAttribute('xmlns',NS);s.setAttribute('width',value('generalWidth',7.2)+'in');s.setAttribute('height',value('generalHeight',4.2)+'in');return new XMLSerializer().serializeToString(s)}
function generalPng(){if($('generalPlot').hidden)return;const img=new Image(),url=URL.createObjectURL(new Blob([generalSvgSource()],{type:'image/svg+xml'}));img.onload=()=>{const c=document.createElement('canvas');c.width=Math.round(value('generalWidth',7.2)*300);c.height=Math.round(value('generalHeight',4.2)*300);const ctx=c.getContext('2d');ctx.fillStyle='white';ctx.fillRect(0,0,c.width,c.height);ctx.drawImage(img,0,0,c.width,c.height);c.toBlob(b=>{if(b)save(b,'general_figure.png')},'image/png');URL.revokeObjectURL(url)};img.src=url}
function showTab(which){$('profileTab').hidden=which!=='profile';$('generalTab').hidden=which!=='general';$('profileTabButton').classList.toggle('active',which==='profile');$('generalTabButton').classList.toggle('active',which==='general')}
$('profileTabButton').onclick=()=>showTab('profile');$('generalTabButton').onclick=()=>showTab('general');
$('addRow').onclick=()=>{ensureSize(sheet.length+1,sheetHeaders.length);renderSheet()};$('addColumn').onclick=()=>{ensureSize(sheet.length,sheetHeaders.length+1);renderSheet()};
$('clearSheet').onclick=()=>{sheet=sheet.map(row=>row.map(() => ''));renderSheet()};
$('generalX').onchange=()=>{$('generalXlabel').value=sheetHeaders[Number($('generalX').value)];drawGeneral()};$('generalY').onchange=()=>{$('generalYlabel').value=sheetHeaders[Number($('generalY').value)];drawGeneral()};
for(const id of ['generalTitle','generalXlabel','generalYlabel','generalXmin','generalXmax','generalYmin','generalYmax','generalLegendText','generalKind','generalSort','generalLegend','generalGrid','generalColor','generalLineWidth','generalWidth','generalHeight','generalFont'])$(id).addEventListener('input',drawGeneral);
$('generalSvg').onclick=()=>{if(!$('generalPlot').hidden)save(new Blob([generalSvgSource()],{type:'image/svg+xml'}),'general_figure.svg')};$('generalPng').onclick=generalPng;$('generalPrint').onclick=()=>window.print();
$('downloadSheet').onclick=()=>{const table=[sheetHeaders,...sheet],csv=table.map(row=>row.map(v=>'"'+String(v).replaceAll('"','""')+'"').join(',')).join('\r\n');save(new Blob([csv],{type:'text/csv'}),'figure_data.csv')};
$('importSheet').onchange=async ev=>{const file=ev.target.files[0];if(!file)return;const text=await file.text(),delimiter=file.name.toLowerCase().endsWith('.tsv')||(!file.name.toLowerCase().endsWith('.csv')&&text.includes('\t'))?'\t':',';const rows=parseDelimited(text,delimiter);if(!rows.length||rows[0].length<2){alert('This file needs at least two columns.');return}const first=rows[0],hasHeader=first.some(v=>v.trim()===''||!Number.isFinite(Number(v)));sheetHeaders=hasHeader?first.map((v,i)=>v.trim()||`Column ${i+1}`):first.map((_,i)=>`Column ${i+1}`);const body=hasHeader?rows.slice(1):rows;sheet=body.map(row=>sheetHeaders.map((_,i)=>row[i]??''));if(!sheet.length)sheet=[sheetHeaders.map(() => '')];$('generalX').value='0';$('generalY').value='1';renderSheet();$('generalXlabel').value=sheetHeaders[0];$('generalYlabel').value=sheetHeaders[1];drawGeneral()};
renderSheet();showTab('profile');
</script></body></html>'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()
    folder = args.folder.resolve()
    if not folder.is_dir():
        parser.error(f"Folder not found: {folder}")
    files = []
    for path in sorted(folder.iterdir(), key=lambda p: p.name.casefold()):
        if not path.is_file() or path.suffix.lower() in {".py", ".html", ".png", ".pdf", ".jpg", ".svg", ".json"}:
            continue
        try:
            profile = read_profile(path)
        except (OSError, ValueError, csv.Error):
            profile = None
        if profile:
            files.append(profile)
    if not files:
        raise SystemExit(f"No usable numeric profile files found in {folder}")
    output = folder / "paper_plot_studio.html"
    output.write_text(HTML.replace("__DATA__", json.dumps(files, separators=(",", ":"))), encoding="utf-8")
    print(f"Listed {len(files)} files in {output}")
    if not args.no_open:
        webbrowser.open(output.as_uri())


if __name__ == "__main__":
    main()
