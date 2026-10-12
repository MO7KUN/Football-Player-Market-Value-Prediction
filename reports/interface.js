'use strict';
const data = JSON.parse(document.getElementById('player-data').textContent);
const el = id => document.getElementById(id);
const leagues = {GB1:'Premier League',ES1:'La Liga',L1:'Bundesliga',IT1:'Serie A',FR1:'Ligue 1'};
const controls = ['q','position','league','profile'];
const pageSize = 20;
let page = 1, key = 'market_value_in_eur', direction = -1, filtered = [];
const money = value => new Intl.NumberFormat('en-GB',{minimumFractionDigits:2,maximumFractionDigits:2}).format(value / 1e6);
for (const value of [...new Set(data.map(row => row.position))].sort()) el('position').add(new Option(value,value));
for (const value of [...new Set(data.map(row => row.league))].sort()) el('league').add(new Option(leagues[value] || value,value));
for (const value of [...new Set(data.map(row => row.cluster))].sort((a,b)=>a-b)) el('profile').add(new Option('Profile '+value,value));
function cell(tr, text, className) {const td=document.createElement('td');td.textContent=text;if(className)td.className=className;tr.append(td);return td;}
function renderTable(){
  const pages=Math.max(1,Math.ceil(filtered.length/pageSize));page=Math.max(1,Math.min(page,pages));
  const body=el('rows');body.replaceChildren();
  for(const row of filtered.slice((page-1)*pageSize,page*pageSize)){
    const tr=document.createElement('tr');const name=cell(tr,'');const avatar=document.createElement('span');avatar.className='avatar';avatar.setAttribute('aria-hidden','true');avatar.textContent=row.name.split(' ').map(x=>x[0]).slice(0,2).join('');name.append(avatar,document.createTextNode(row.name));
    const pos=cell(tr,'');const tag=document.createElement('span');tag.className='tag';tag.textContent=row.position;pos.append(tag);
    cell(tr,leagues[row.league]||row.league);cell(tr,Number(row.age).toFixed(1));cell(tr,money(row.market_value_in_eur),'value');cell(tr,money(row.prediction_eur));cell(tr,'Profile '+row.cluster);body.append(tr);
  }
  if(!filtered.length){const tr=document.createElement('tr');const td=cell(tr,'No players match these filters. Try another name or reset the filters.','empty');td.colSpan=7;body.append(tr);}
  el('count').textContent=filtered.length.toLocaleString('en-GB')+' matching players';
  el('page-info').textContent=filtered.length ? `Showing ${(page-1)*pageSize+1}–${Math.min(page*pageSize,filtered.length)} of ${filtered.length.toLocaleString('en-GB')} players`:'No matching players';
  el('page-number').textContent=`${page} / ${pages}`;el('prev').disabled=page===1;el('next').disabled=page===pages;
  document.querySelectorAll('th[data-key]').forEach(th=>th.setAttribute('aria-sort',th.dataset.key===key?(direction===1?'ascending':'descending'):'none'));
}
const NS='http://www.w3.org/2000/svg';
function node(tag,attrs={},text){const n=document.createElementNS(NS,tag);for(const [k,v] of Object.entries(attrs))n.setAttribute(k,String(v));if(text!==undefined)n.textContent=text;return n;}
function svg(id,label,w=520,h=280){const s=node('svg',{viewBox:`0 0 ${w} ${h}`,role:'img','aria-label':label});s.append(node('title',{},label));el(id).replaceChildren(s);return s;}
function empty(id){const p=document.createElement('p');p.className='empty';p.textContent='No matching players to display.';el(id).replaceChildren(p);}
function charts(){
  if(!filtered.length){empty('valuation-chart');empty('league-chart');return;}
  const s=svg('valuation-chart','Observed and predicted player values, logarithmic axes in millions of euros');
  const max=Math.max(1,...filtered.flatMap(r=>[r.market_value_in_eur/1e6,r.prediction_eur/1e6]));
  const maxLog=Math.log10(max+1)*1.06,x=v=>55+Math.log10(Math.max(0,v)+1)/maxLog*440,y=v=>235-Math.log10(Math.max(0,v)+1)/maxLog*210;
  for(const t of [0,1,5,10,25,50,100,200]){if(t>max*1.05)continue;s.append(node('line',{x1:55,x2:495,y1:y(t),y2:y(t),stroke:'#edf0f2'}));s.append(node('text',{x:45,y:y(t)+3,'text-anchor':'end'},t));s.append(node('text',{x:x(t),y:252,'text-anchor':'middle'},t));}
  s.append(node('line',{x1:55,y1:235,x2:x(max),y2:y(max),stroke:'#8fa4a8','stroke-dasharray':'4 4'}));
  for(const r of filtered){const c=node('circle',{cx:x(r.market_value_in_eur/1e6),cy:y(r.prediction_eur/1e6),r:3,fill:'#32866a',opacity:.4});c.append(node('title',{},`${r.name}: observed €${money(r.market_value_in_eur)}m, predicted €${money(r.prediction_eur)}m`));s.append(c);}
  s.append(node('text',{x:270,y:277,'text-anchor':'middle'},'Observed market value (€m)'));s.append(node('text',{x:12,y:134,transform:'rotate(-90 12 134)','text-anchor':'middle'},'Predicted value (€m)'));
  const groups=Object.entries(leagues).map(([code,name])=>{const a=filtered.filter(r=>r.league===code).map(r=>r.market_value_in_eur/1e6).sort((a,b)=>a-b);return {name,n:a.length,value:a.length?(a[Math.floor((a.length-1)/2)]+a[Math.floor(a.length/2)])/2:0};}).filter(r=>r.n).sort((a,b)=>b.value-a.value);
  const b=svg('league-chart','Median observed market value by league in millions of euros');const maxV=Math.max(1,...groups.map(g=>g.value));
  groups.forEach((g,i)=>{const yy=20+i*48;b.append(node('text',{x:0,y:yy+11},g.name));b.append(node('text',{x:510,y:yy+11,'text-anchor':'end'},`€${g.value.toFixed(2)}m · ${g.n} players`));b.append(node('rect',{x:0,y:yy+20,width:520,height:7,rx:3.5,fill:'#eff3ed'}));b.append(node('rect',{x:0,y:yy+20,width:Math.max(2,g.value/maxV*520),height:7,rx:3.5,fill:i===0?'#7eaa5e':'#c0d6a9'}));});
}
function filter(){const q=el('q').value.trim().toLocaleLowerCase();filtered=data.filter(r=>(!q||r.name.toLocaleLowerCase().includes(q))&&(!el('position').value||r.position===el('position').value)&&(!el('league').value||r.league===el('league').value)&&(el('profile').value===''||String(r.cluster)===el('profile').value));sort();page=1;renderTable();charts();}
function sort(){filtered.sort((a,b)=>direction*(typeof a[key]==='string'?a[key].localeCompare(b[key]):a[key]-b[key]));}
controls.forEach(id=>el(id).addEventListener('input',filter));
el('reset').addEventListener('click',()=>{controls.forEach(id=>el(id).value='');filter();});
document.querySelectorAll('th[data-key] button').forEach(button=>button.addEventListener('click',()=>{const selected=button.parentElement.dataset.key;direction=key===selected?-direction:1;key=selected;sort();page=1;renderTable();}));
el('prev').addEventListener('click',()=>{page--;renderTable();});el('next').addEventListener('click',()=>{page++;renderTable();});
el('export').addEventListener('click',()=>{const keys=['name','position','league','age','market_value_in_eur','prediction_eur','cluster'];const quote=value=>{let v=String(value??'');if(/^[=+@\-]/.test(v))v="'"+v;return '"'+v.replaceAll('"','""')+'"';};const csv=[keys.map(quote).join(','),...filtered.map(r=>keys.map(k=>quote(r[k])).join(','))].join('\r\n');const url=URL.createObjectURL(new Blob(['\uFEFF'+csv],{type:'text/csv;charset=utf-8'}));const a=document.createElement('a');a.href=url;a.download='touchline-historical-player-values.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
const navLinks=[...document.querySelectorAll('.sidebar nav a')];if('IntersectionObserver' in window){const observer=new IntersectionObserver(entries=>{for(const entry of entries){if(!entry.isIntersecting)continue;navLinks.forEach(link=>{const active=link.hash==='#'+entry.target.id;link.classList.toggle('active',active);if(active)link.setAttribute('aria-current','location');else link.removeAttribute('aria-current');});}},{rootMargin:'-10% 0px -55% 0px'});navLinks.forEach(link=>{const section=document.querySelector(link.hash);if(section)observer.observe(section);});}
filter();
