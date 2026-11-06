"""Build an offline, standalone browser viewer from files in this folder."""
from pathlib import Path
import csv
import json
import re

ROOT = Path(__file__).resolve().parent

def xy(path):
    pairs = re.findall(r"^\s*([-+\d.eE]+)[,\s]+([-+\d.eE]+)\s*$",
                       path.read_text(encoding="utf-8"), re.M)
    return sorted([[float(a), float(b)] for a, b in pairs])

profiles = {f"Iteration {i}": xy(next(ROOT.glob(f"iteration-{i}-*")))
            for i in range(1, 8)}
paper = xy(ROOT / "aiac_fig24_naca0009_pdf_points.csv")
fit_rows = list(csv.DictReader((ROOT / "iteration7_local_dip_fit_diagnostic.csv").open()))
fitted = [[float(r["x_m"]), float(r["new_fitted_curve_m_per_s"])] for r in fit_rows]
metrics = list(csv.DictReader((ROOT / "outlet_velocity_profile_rmse.csv").open()))
data = {"profiles": profiles, "paper": paper, "fitted": fitted,
        "rmse": {f"Iteration {r['iteration']}": {
            "paper": float(r["rmse_published_reference_m_per_s"]),
            "fitted": float(r["rmse_iteration7_fitted_diagnostic_m_per_s"])}
            for r in metrics}}

html = r'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Outlet velocity profile explorer</title>
<style>
:root{font-family:Inter,Segoe UI,Arial,sans-serif;color:#172338;background:#f2f5f8}
*{box-sizing:border-box}body{margin:0}.wrap{max-width:1440px;margin:auto;padding:26px 24px 50px}
h1{font-size:27px;margin:0 0 6px}h2{font-size:18px;margin:0 0 12px}.sub{color:#5c687b;margin:0 0 24px}
.card{background:white;border:1px solid #d9e0e7;border-radius:14px;box-shadow:0 5px 24px #24344f0b;padding:20px;margin-bottom:18px}
.controls{display:flex;gap:18px;flex-wrap:wrap;align-items:end}.group{display:flex;flex-direction:column;gap:6px}
label{font-size:13px;font-weight:600}select,input[type=number],input[type=file]{border:1px solid #b9c6d5;border-radius:7px;padding:8px;background:white}
button{border:1px solid #aab9ca;border-radius:7px;background:#fff;padding:9px 12px;cursor:pointer;font-weight:600;color:#213149}
button:hover{background:#eaf2fc}.primary{background:#116caa;color:white;border-color:#116caa}.primary:hover{background:#095786}
.checks{display:flex;flex-wrap:wrap;gap:5px 16px;margin-top:18px}.checks label{display:inline-flex;gap:5px;align-items:center;font-weight:500}
.note{font-size:13px;color:#56657b;line-height:1.5;margin:10px 0 0}.warn{border-left:4px solid #ad357e;background:#fcf5fa;padding:10px 14px;border-radius:5px}
.chartHead{display:flex;justify-content:space-between;gap:14px;align-items:center;flex-wrap:wrap}.actions{display:flex;gap:8px;flex-wrap:wrap}
svg{width:100%;height:auto;display:block}.chart{position:relative}.tooltip{position:absolute;pointer-events:none;background:#172338;color:white;padding:8px 11px;border-radius:7px;font-size:12px;white-space:pre-line;display:none;box-shadow:0 3px 12px #0003}
.grid{stroke:#e1e6ec;stroke-width:1}.axis{stroke:#68778b;stroke-width:1.2}.tick{fill:#526176;font-size:12px}.axislabel{fill:#25364e;font-size:13px;font-weight:600}
.legend{display:flex;gap:8px 18px;flex-wrap:wrap;font-size:12px;color:#34445b;margin-top:8px}.key{display:inline-block;width:22px;height:0;border-top:3px solid;vertical-align:middle;margin-right:5px}
table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:right;padding:9px 12px;border-bottom:1px solid #e6ebf0}th:first-child,td:first-child{text-align:left}th{color:#526176;background:#f5f8fb}tr.highlight{font-weight:700;background:#edf6fc}
@media print{body{background:white}.wrap{max-width:none;padding:0}.controlsCard,.actions,.note,.sub{display:none!important}.card{border:0;box-shadow:none;break-inside:avoid;margin:0 0 14px;padding:6px}h1{font-size:18px}h2{font-size:15px}svg{max-height:43vh}}
</style></head><body><main class="wrap"><h1>Outlet velocity profile explorer</h1>
<p class="sub">Five-vane NACA 0009 · Fluent iterations 1–7 · offline browser tool</p>
<section class="card controlsCard"><div class="controls">
<div class="group"><label for="ref">Comparison curve</label><select id="ref"><option value="fitted">Iteration 7 fitted curve (diagnostic)</option><option value="paper">Published AIAC Figure 24</option></select></div>
<div class="group"><label for="xmin">X minimum (m)</label><input id="xmin" type="number" step="0.1" value="2.85"></div>
<div class="group"><label for="xmax">X maximum (m)</label><input id="xmax" type="number" step="0.1" value="10.75"></div>
<div class="group"><label for="ymin">Velocity minimum (m/s)</label><input id="ymin" type="number" step="0.5" value="44.5"></div>
<div class="group"><label for="ymax">Velocity maximum (m/s)</label><input id="ymax" type="number" step="0.5" value="54.5"></div>
<div class="group"><label for="upload">Add Fluent XY or two-column CSV</label><input id="upload" type="file" accept=".csv,.txt,*"></div>
</div><div id="checks" class="checks"></div>
<p class="note warn">The default curve was fitted using iteration 7. Its RMSE measures fit to that run. Select “Published AIAC Figure 24” for comparison with the independent paper figure.</p></section>
<section class="card"><div class="chartHead"><h2>Outlet velocity profile</h2><div class="actions"><button onclick="downloadSVG('profile','velocity_profile.svg')">SVG</button><button onclick="downloadPNG('profile','velocity_profile.png')">PNG</button><button onclick="window.print()">Print / PDF</button></div></div><div class="chart"><svg id="profile" viewBox="0 0 1200 480" role="img" aria-label="Velocity profile chart"></svg><div id="tip" class="tooltip"></div></div><div id="legend" class="legend"></div></section>
<section class="card"><div class="chartHead"><h2>RMSE against selected curve</h2><div class="actions"><button onclick="downloadSVG('errors','velocity_rmse.svg')">SVG</button><button onclick="downloadPNG('errors','velocity_rmse.png')">PNG</button></div></div><svg id="errors" viewBox="0 0 1200 360" role="img" aria-label="RMSE chart"></svg><p class="note">RMSE uses 240 common x positions from 2.85 to 10.75 m. Supplied iteration scores match the Python calculation; uploaded curves use linear interpolation.</p></section>
<section class="card"><h2>Numerical results</h2><table id="scores"></table></section>
</main><script>const DATA=__DATA__;
const palette={'Iteration 1':'#a6abb3','Iteration 2':'#949aa3','Iteration 3':'#858b94','Iteration 4':'#747b84','Iteration 5':'#626972','Iteration 6':'#009E73','Iteration 7':'#0072B2'};
let visible=new Set(Object.keys(DATA.profiles));let custom=0;const NS='http://www.w3.org/2000/svg';
function el(tag,attrs={},parent){let e=document.createElementNS(NS,tag);for(const [k,v] of Object.entries(attrs))e.setAttribute(k,v);parent.appendChild(e);return e}
function txt(s,x,y,str,attrs={}){let e=el('text',{x,y,...attrs},s);e.textContent=str;return e}
function interp(a,x){if(x<=a[0][0])return a[0][1];if(x>=a[a.length-1][0])return a[a.length-1][1];let lo=0,hi=a.length-1;while(hi-lo>1){let m=(lo+hi)>>1;if(a[m][0]<x)lo=m;else hi=m}let t=(x-a[lo][0])/(a[hi][0]-a[lo][0]);return a[lo][1]*(1-t)+a[hi][1]*t}
function rmse(name,ref){if(DATA.rmse[name])return DATA.rmse[name][ref];let a=DATA.profiles[name],b=DATA[ref],sum=0;for(let i=0;i<240;i++){let x=2.85+i*(10.75-2.85)/239,d=interp(a,x)-interp(b,x);sum+=d*d}return Math.sqrt(sum/240)}
function readRange(){let v=id=>+document.getElementById(id).value;let r={xmin:v('xmin'),xmax:v('xmax'),ymin:v('ymin'),ymax:v('ymax')};return r.xmax>r.xmin&&r.ymax>r.ymin?r:{xmin:2.85,xmax:10.75,ymin:44.5,ymax:54.5}}
function drawAxes(s,W,H,m,r){let xp=x=>m.l+(x-r.xmin)/(r.xmax-r.xmin)*(W-m.l-m.r),yp=y=>H-m.b-(y-r.ymin)/(r.ymax-r.ymin)*(H-m.t-m.b);for(let j=0;j<=8;j++){let x=r.xmin+j*(r.xmax-r.xmin)/8,px=xp(x);el('line',{x1:px,y1:m.t,x2:px,y2:H-m.b,class:'grid'},s);txt(s,px,H-m.b+19,x.toFixed(1),{'text-anchor':'middle',class:'tick'})}for(let j=0;j<=5;j++){let y=r.ymin+j*(r.ymax-r.ymin)/5,py=yp(y);el('line',{x1:m.l,y1:py,x2:W-m.r,y2:py,class:'grid'},s);txt(s,m.l-10,py+4,y.toFixed(1),{'text-anchor':'end',class:'tick'})}el('line',{x1:m.l,y1:H-m.b,x2:W-m.r,y2:H-m.b,class:'axis'},s);el('line',{x1:m.l,y1:m.t,x2:m.l,y2:H-m.b,class:'axis'},s);txt(s,W/2,H-8,'Outlet position, x (m)',{'text-anchor':'middle',class:'axislabel'});let t=txt(s,17,H/2,'Velocity magnitude (m/s)',{'text-anchor':'middle',class:'axislabel'});t.setAttribute('transform',`rotate(-90 17 ${H/2})`);return {xp,yp}}
function drawProfile(){let s=document.getElementById('profile');s.replaceChildren();let r=readRange(),W=1200,H=480,m={l:72,r:18,t:22,b:55};let {xp,yp}=drawAxes(s,W,H,m,r);let ref=document.getElementById('ref').value;let curves=[[ref==='paper'?'Published AIAC reference':'Iteration 7 fitted curve',DATA[ref],ref==='paper'?'#111111':'#ad357e',3.2,ref==='paper'?'':'8 5']];for(let [n,a] of Object.entries(DATA.profiles))if(visible.has(n))curves.push([n,a,palette[n]||'#9467bd',n==='Iteration 6'||n==='Iteration 7'?2.8:1.3,'']);for(let [name,a,color,width,dash] of curves){let points=a.filter(p=>p[0]>=r.xmin&&p[0]<=r.xmax).map(p=>`${xp(p[0]).toFixed(1)},${yp(p[1]).toFixed(1)}`).join(' ');el('polyline',{points,fill:'none',stroke:color,'stroke-width':width,'stroke-dasharray':dash,'stroke-linejoin':'round','stroke-linecap':'round',opacity:width<2?.68:1},s)}let legend=document.getElementById('legend');legend.innerHTML=curves.map(c=>`<span><i class="key" style="border-color:${c[2]}"></i>${c[0]}</span>`).join('');let overlay=el('rect',{x:m.l,y:m.t,width:W-m.l-m.r,height:H-m.t-m.b,fill:'transparent'},s);overlay.addEventListener('mousemove',ev=>{let p=s.createSVGPoint();p.x=ev.clientX;p.y=ev.clientY;let q=p.matrixTransform(s.getScreenCTM().inverse()),x=r.xmin+(q.x-m.l)/(W-m.l-m.r)*(r.xmax-r.xmin);let tip=document.getElementById('tip');tip.textContent=`x = ${x.toFixed(3)} m\n`+curves.map(c=>`${c[0]}: ${interp(c[1],x).toFixed(2)} m/s`).join('\n');tip.style.display='block';tip.style.left=Math.min(ev.offsetX+16,s.clientWidth-200)+'px';tip.style.top=Math.max(4,ev.offsetY-8)+'px'});overlay.addEventListener('mouseleave',()=>document.getElementById('tip').style.display='none')}
function drawRMSE(){let s=document.getElementById('errors');s.replaceChildren();let ref=document.getElementById('ref').value,names=Object.keys(DATA.profiles),W=1200,H=360,m={l:72,r:18,t:24,b:56};let max=Math.max(...names.map(n=>rmse(n,ref)))*1.16;for(let j=0;j<=4;j++){let y=j*max/4,py=H-m.b-y/max*(H-m.t-m.b);el('line',{x1:m.l,y1:py,x2:W-m.r,y2:py,class:'grid'},s);txt(s,m.l-10,py+4,y.toFixed(1),{'text-anchor':'end',class:'tick'})}names.forEach((n,i)=>{let val=rmse(n,ref),space=(W-m.l-m.r)/names.length,bw=space*.52,x=m.l+i*space+(space-bw)/2,y=H-m.b-val/max*(H-m.t-m.b);el('rect',{x,y,width:bw,height:H-m.b-y,rx:4,fill:palette[n]||'#9467bd',opacity:i<5?.7:1},s);txt(s,x+bw/2,y-8,val.toFixed(3),{'text-anchor':'middle',class:'tick'});txt(s,x+bw/2,H-m.b+20,n.replace('Iteration ','Iter '),{'text-anchor':'middle',class:'tick'})});el('line',{x1:m.l,y1:H-m.b,x2:W-m.r,y2:H-m.b,class:'axis'},s);let t=txt(s,18,H/2,'RMSE (m/s)',{'text-anchor':'middle',class:'axislabel'});t.setAttribute('transform',`rotate(-90 18 ${H/2})`)}
function drawTable(){let names=Object.keys(DATA.profiles),ref=document.getElementById('ref').value;document.getElementById('scores').innerHTML='<thead><tr><th>Curve</th><th>RMSE vs published AIAC (m/s)</th><th>RMSE vs iteration 7 fit (m/s)</th></tr></thead><tbody>'+names.map(n=>`<tr class="${n==='Iteration 6'||n==='Iteration 7'?'highlight':''}"><td>${n}</td><td>${rmse(n,'paper').toFixed(3)}</td><td>${rmse(n,'fitted').toFixed(3)}</td></tr>`).join('')+'</tbody>'}
function render(){drawProfile();drawRMSE();drawTable()}
function controls(){let checks=document.getElementById('checks');checks.innerHTML=Object.keys(DATA.profiles).map(n=>`<label><input type="checkbox" data-name="${n}" ${visible.has(n)?'checked':''}>${n}</label>`).join('');checks.querySelectorAll('input').forEach(e=>e.onchange=()=>{e.checked?visible.add(e.dataset.name):visible.delete(e.dataset.name);render()})}
for(let id of ['ref','xmin','xmax','ymin','ymax'])document.getElementById(id).addEventListener('change',render);
document.getElementById('upload').addEventListener('change',async ev=>{let f=ev.target.files[0];if(!f)return;let pairs=(await f.text()).split(/\r?\n/).map(line=>line.trim().match(/^([-+\d.eE]+)[,\s]+([-+\d.eE]+)$/)).filter(Boolean).map(m=>[+m[1],+m[2]]).sort((a,b)=>a[0]-b[0]);if(pairs.length<3){alert('Could not find at least three XY pairs.');return}let name='Custom '+(++custom);DATA.profiles[name]=pairs;palette[name]='#9467bd';visible.add(name);controls();render()});
function svgText(id){let s=document.getElementById(id).cloneNode(true);s.setAttribute('xmlns',NS);s.setAttribute('width','1200');s.setAttribute('height',id==='profile'?'480':'360');let style=document.createElementNS(NS,'style');style.textContent='.grid{stroke:#e1e6ec;stroke-width:1}.axis{stroke:#68778b;stroke-width:1.2}.tick{fill:#526176;font:12px Arial}.axislabel{fill:#25364e;font:600 13px Arial}';s.insertBefore(style,s.firstChild);return new XMLSerializer().serializeToString(s)}
function save(blob,name){let a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}
function downloadSVG(id,name){save(new Blob([svgText(id)],{type:'image/svg+xml'}),name)}
function downloadPNG(id,name){let img=new Image(),url=URL.createObjectURL(new Blob([svgText(id)],{type:'image/svg+xml'}));img.onload=()=>{let c=document.createElement('canvas');c.width=2400;c.height=id==='profile'?960:720;let ctx=c.getContext('2d');ctx.fillStyle='white';ctx.fillRect(0,0,c.width,c.height);ctx.drawImage(img,0,0,c.width,c.height);c.toBlob(b=>save(b,name));URL.revokeObjectURL(url)};img.src=url}
controls();render();</script></body></html>'''
(ROOT / "outlet_velocity_explorer.html").write_text(
    html.replace("__DATA__", json.dumps(data, separators=(",", ":"))), encoding="utf-8")
print(ROOT / "outlet_velocity_explorer.html")
