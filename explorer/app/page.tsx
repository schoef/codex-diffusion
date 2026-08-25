"use client";

import { useEffect, useMemo, useState } from "react";

type Family = {
  id: string; name: string; short: string; support: string; variance: string;
  a: number; b: number; c: number; lo: number; hi: number; color: string;
};

const families: Family[] = [
  { id: "gaussian", name: "Gaussian", short: "G", support: "ℝ", variance: "σ²", a: 1, b: 0, c: 0, lo: -4, hi: 4, color: "#30d8c2" },
  { id: "poisson", name: "Poisson", short: "P", support: "ℕ₀", variance: "m", a: 0, b: 1, c: 0, lo: 0, hi: 10, color: "#ffb454" },
  { id: "gamma", name: "Gamma", short: "Γ", support: "ℝ₊", variance: "m² / p", a: 0, b: 0, c: .5, lo: 0, hi: 10, color: "#f46ca5" },
  { id: "binomial", name: "Binomial", short: "B", support: "{0,…,n}", variance: "m(1−m/n)", a: 0, b: 1, c: -.1, lo: 0, hi: 10, color: "#a78bfa" },
  { id: "neg-binomial", name: "Negative binomial", short: "NB", support: "ℕ₀", variance: "m + m²/r", a: 0, b: 1, c: .2, lo: 0, hi: 12, color: "#60a5fa" },
  { id: "hyperbolic", name: "Hyperbolic secant", short: "HS", support: "ℝ", variance: "1 + m²", a: 1, b: 0, c: 1, lo: -5, hi: 5, color: "#fb7185" },
];

function rng(seed: number) { let s = seed >>> 0; return () => ((s = Math.imul(1664525, s) + 1013904223 >>> 0) / 4294967296); }
function normal(r: () => number) { const u = Math.max(r(), 1e-9), v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
function clamp(x:number, lo:number, hi:number){ return Math.max(lo, Math.min(hi, x)); }
function labTime(u:number,log:boolean,tMax:number){return log?.01*(Math.exp(u*Math.log(1+tMax/.01))-1):tMax*u}
function labPosition(t:number,log:boolean,tMax:number){return log?Math.log(1+Math.max(0,t)/.01)/Math.log(1+tMax/.01):Math.max(0,t)/tMax}
function variance(f:Family, x:number){ return Math.max(.002, f.a + f.b*x + f.c*x*x); }

function referenceStep(f:Family,x:number,mu:number,dt:number,kappa:number,r:()=>number){
  const w=Math.exp(-kappa*dt);
  if(f.id==="gaussian")return mu+w*(x-mu)+Math.sqrt(1-w*w)*normal(r);
  if(f.id==="poisson")return binomialSample(Math.max(0,Math.round(x)),w,r)+poissonSample(mu*(1-w),r);
  if(f.id==="gamma"){const shape=2,scale=mu/shape,M=poissonSample(w*x/(scale*(1-w)),r);return gammaSample(shape+M,scale*(1-w),r)}
  if(f.id==="binomial"){const N=10,p=mu/N,xi=clamp(Math.round(x),0,N);return binomialSample(xi,p+(1-p)*w,r)+binomialSample(N-xi,p*(1-w),r)}
  const shape=5,c=mu/(shape+mu),a=w*(1-c)/(1-c*w),ct=c*(1-w)/(1-c*w),B=binomialSample(Math.max(0,Math.round(x)),a,r);return B+nbSample(shape+B,ct,r);
}

function simulate(f: Family, mu:number, kappa:number, horizon:number, seed:number){
  if(f.id==="hyperbolic")return {paths:[] as number[][],finals:[] as number[]};
  const steps=140, dt=horizon/steps, n=260, rr=rng(seed); const paths:number[][]=[]; const finals:number[]=[];
  for(let j=0;j<n;j++){
    let x=f.id==="gaussian"?mu+2:f.id==="gamma"?mu+3:Math.round(mu+4);if(f.id==="binomial")x=Math.min(10,x);const p=[x];
    for(let i=1;i<=steps;i++){
      x=referenceStep(f,x,mu,dt,kappa,rr);p.push(x);
    }
    if(j<14) paths.push(p); finals.push(x);
  }
  return {paths,finals};
}

function DynamicsEquation({f}:{f:Family}){
  if(f.id==="hyperbolic")return <div className="equation no-process"><b>No Markov reference process</b><small>The spectral kernel takes negative values.</small></div>;
  if(f.id==="gaussian"||f.id==="gamma")return <div className="equation canonical-eq">dX<sub>t</sub> = κ(μ − X<sub>t</sub>)dt + √<span className="radicand">κ[2V(μ)+V′(μ)(X<sub>t</sub>−μ)]</span> dW<sub>t</sub></div>;
  return <div className="equation canonical-eq">λ<sub>±</sub>(x) = <span className="frac">κ⁄2</span> [2V(μ)+V′(μ)(x−μ) ± (μ−x)]</div>;
}

function PathChart({data,f,horizon}:{data:number[][];f:Family;horizon:number}){
  const W=760,H=330,left=52,right=20,top=24,bottom=48; const ymin=f.lo, ymax=f.hi; const x=(i:number)=>left+i/(data[0].length-1)*(W-left-right); const y=(v:number)=>H-bottom-(v-ymin)/(ymax-ymin)*(H-top-bottom);
  return <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${f.name} diffusion sample paths`}>
    <defs><clipPath id="path-plot"><rect x={left} y={top} width={W-left-right} height={H-top-bottom}/></clipPath></defs>
    {[0,.25,.5,.75,1].map(q=><g key={q}><line x1={left} x2={W-right} y1={top+q*(H-top-bottom)} y2={top+q*(H-top-bottom)} className="gridline"/><text x={left-9} y={top+q*(H-top-bottom)+4} textAnchor="end">{(ymax-q*(ymax-ymin)).toFixed(1)}</text></g>)}
    {[0,.25,.5,.75,1].map(q=><text key={q} x={left+q*(W-left-right)} y={H-bottom+18} textAnchor="middle">{(q*horizon).toFixed(q===0?0:1)}</text>)}
    <g clipPath="url(#path-plot)">{data.map((p,j)=><polyline key={j} points={p.map((v,i)=>`${x(i)},${y(v)}`).join(" ")} fill="none" stroke={f.color} strokeWidth={j===0?2.6:1.15} opacity={j===0?1:.34}/>)}</g>
    <line x1={left} x2={W-right} y1={H-bottom} y2={H-bottom} className="axis"/><line x1={left} x2={left} y1={top} y2={H-bottom} className="axis"/>
    <text className="axis-title" x={(left+W-right)/2} y={H-5} textAnchor="middle">process time t</text><text className="axis-title" transform={`translate(13 ${(top+H-bottom)/2}) rotate(-90)`} textAnchor="middle">state Xₜ</text>
  </svg>
}

function Histogram({values,f}:{values:number[];f:Family}){
  const W=540,H=280,left=58,right=18,top=22,bottom=50,bins=24,lo=Math.min(...values),hi=Math.max(...values),span=Math.max(.01,hi-lo); const counts=Array(bins).fill(0); values.forEach(v=>counts[Math.min(bins-1,Math.floor((v-lo)/span*bins))]++); const mx=Math.max(...counts),maxProb=mx/values.length,isDiscrete=["poisson","binomial","neg-binomial"].includes(f.id);
  return <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Endpoint ensemble histogram">
    {[0,.5,1].map(q=><g key={q}><line x1={left} x2={W-right} y1={H-bottom-q*(H-top-bottom)} y2={H-bottom-q*(H-top-bottom)} className="gridline"/><text x={left-8} y={H-bottom-q*(H-top-bottom)+4} textAnchor="end">{(q*maxProb).toFixed(2)}</text></g>)}
    {counts.map((v,i)=>{const bw=(W-left-right)/bins,h=v/mx*(H-top-bottom);return <rect key={i} x={left+i*bw+1} y={H-bottom-h} width={Math.max(1,bw-2)} height={h} fill={f.color} opacity={.78}/>})}
    <line x1={left} x2={W-right} y1={H-bottom} y2={H-bottom} className="axis"/><line x1={left} x2={left} y1={top} y2={H-bottom} className="axis"/><text x={left} y={H-bottom+18}>{lo.toFixed(1)}</text><text x={W-right} y={H-bottom+18} textAnchor="end">{hi.toFixed(1)}</text>
    <text className="axis-title" x={(left+W-right)/2} y={H-5} textAnchor="middle">state x</text><text className="axis-title" transform={`translate(13 ${(top+H-bottom)/2}) rotate(-90)`} textAnchor="middle">{isDiscrete?"probability mass":"probability per bin"}</text>
  </svg>
}

function Slider({label,value,min,max,step,onChange,format=(v)=>v.toFixed(2)}:{label:string;value:number;min:number;max:number;step:number;onChange:(v:number)=>void;format?:(v:number)=>string}){
  return <label className="control"><span><b>{label}</b><output>{format(value)}</output></span><input type="range" value={value} min={min} max={max} step={step} onChange={e=>onChange(+e.target.value)}/></label>
}

type LabFamily = {id:string;name:string;baseline:string;process:string;lo:number;hi:number;discrete:boolean;color:string;kernel:string;mean:number;v:number;vp:number};
type LabParams = {base1:number;base2:number;dataCenter:number;dataSpread:number;sampleN:number;densityN:number};
const labFamilies:LabFamily[]=[
  {id:"gaussian",name:"Gaussian",baseline:"N(0, 1)",process:"Ornstein–Uhlenbeck",lo:-4,hi:5,discrete:false,color:"#24bfae",kernel:"Xₜ = w·x + √(1−w²)·ξ,   ξ ∼ N(0,1)",mean:0,v:1,vp:0},
  {id:"poisson",name:"Poisson",baseline:"Pois(4)",process:"immigration–death",lo:0,hi:16,discrete:true,color:"#e6922e",kernel:"Xₜ = Bin(x,w) + Pois(4·(1−w))",mean:4,v:4,vp:1},
  {id:"gamma",name:"Gamma",baseline:"Ga(3, 1)",process:"CIR / squared Bessel",lo:0,hi:14,discrete:false,color:"#de5b91",kernel:"M ∼ Pois(w·x/(1−w));  Xₜ|M ∼ Ga(3+M, 1−w)",mean:3,v:3,vp:2},
  {id:"binomial",name:"Binomial",baseline:"Bin(20, .35)",process:"Ehrenfest",lo:0,hi:20,discrete:true,color:"#8b6fe8",kernel:"Xₜ = Bin(x, .35+.65w) + Bin(20−x, .35(1−w))",mean:7,v:4.55,vp:.3},
  {id:"neg-binomial",name:"Neg. binomial",baseline:"NB(4, .45)",process:"birth–death + immigration",lo:0,hi:22,discrete:true,color:"#468fe2",kernel:"B ∼ Bin(x,aₜ);  Xₜ=B+NB(4+B,cₜ)",mean:3.2727,v:5.9504,vp:2.6364},
];
const defaultLabParams:Record<string,LabParams>={
  gaussian:{base1:0,base2:1,dataCenter:2.4,dataSpread:1.6,sampleN:2400,densityN:16000},
  poisson:{base1:4,base2:1,dataCenter:10,dataSpread:4,sampleN:2400,densityN:16000},
  gamma:{base1:3,base2:1,dataCenter:8,dataSpread:3.5,sampleN:2400,densityN:16000},
  binomial:{base1:20,base2:.35,dataCenter:17,dataSpread:6,sampleN:2400,densityN:16000},
  "neg-binomial":{base1:4,base2:.45,dataCenter:11,dataSpread:5,sampleN:2400,densityN:16000},
};

function configuredFamily(base:LabFamily,p:LabParams):LabFamily{
  const centre=p.dataCenter,spread=Math.max(.1,p.dataSpread);
  if(base.id==="gaussian"){
    const mu=p.base1,sigma=Math.max(.05,p.base2),lo=Math.floor(Math.min(mu-4.5*sigma,centre-2.6*spread)),hi=Math.ceil(Math.max(mu+4.5*sigma,centre+2.6*spread));
    return {...base,baseline:`N(${mu.toFixed(1)}, ${sigma.toFixed(1)}²)`,mean:mu,v:sigma*sigma,vp:0,lo,hi,kernel:`Xₜ = μ + w(x−μ) + σ√(1−w²)ξ,   ξ ∼ N(0,1)`};
  }
  if(base.id==="poisson"){
    const mu=Math.max(.05,p.base1),hi=Math.ceil(Math.max(mu+5*Math.sqrt(mu),centre+2.6*spread));
    return {...base,baseline:`Pois(${mu.toFixed(1)})`,mean:mu,v:mu,vp:1,lo:0,hi,kernel:`Xₜ = Bin(x,w) + Pois(${mu.toFixed(2)}·(1−w))`};
  }
  if(base.id==="gamma"){
    const shape=Math.max(.15,p.base1),scale=Math.max(.05,p.base2),mu=shape*scale,sd=Math.sqrt(shape)*scale,hi=Math.ceil(Math.max(mu+5*sd,centre+2.6*spread));
    return {...base,baseline:`Ga(${shape.toFixed(1)}, ${scale.toFixed(1)})`,mean:mu,v:shape*scale*scale,vp:2*scale,lo:0,hi,kernel:`M ∼ Pois(wx/[${scale.toFixed(2)}(1−w)]);  Xₜ|M ∼ Ga(${shape.toFixed(2)}+M, ${scale.toFixed(2)}(1−w))`};
  }
  if(base.id==="binomial"){
    const trials=Math.round(clamp(p.base1,2,80)),prob=clamp(p.base2,.02,.98),mu=trials*prob;
    return {...base,baseline:`Bin(${trials}, ${prob.toFixed(2)})`,mean:mu,v:trials*prob*(1-prob),vp:1-2*prob,lo:0,hi:trials,kernel:`Xₜ = Bin(x, p+(1−p)w) + Bin(${trials}−x, p(1−w)),   p=${prob.toFixed(2)}`};
  }
  const shape=Math.max(.15,p.base1),c=clamp(p.base2,.02,.94),mu=shape*c/(1-c),v=shape*c/((1-c)*(1-c)),hi=Math.ceil(Math.max(mu+5*Math.sqrt(v),centre+2.6*spread));
  return {...base,baseline:`NB(${shape.toFixed(1)}, ${c.toFixed(2)})`,mean:mu,v,vp:(1+c)/(1-c),lo:0,hi,kernel:`B ∼ Bin(x,aₜ);  Xₜ=B+NB(${shape.toFixed(2)}+B,cₜ),   c=${c.toFixed(2)}`};
}

function logGamma(z:number){const c=[676.5203681218851,-1259.1392167224028,771.32342877765313,-176.6150291621406,12.507343278686905,-.13857109526572012,9.984369578019572e-6,1.5056327351493116e-7];if(z<.5)return Math.log(Math.PI)-Math.log(Math.sin(Math.PI*z))-logGamma(1-z);z-=1;let x=.9999999999998099;for(let i=0;i<c.length;i++)x+=c[i]/(z+i+1);const t=z+c.length-.5;return .5*Math.log(2*Math.PI)+(z+.5)*Math.log(t)-t+Math.log(x)}
function poissonSample(lambda:number,r:()=>number){if(lambda<=0)return 0;if(lambda<30){const L=Math.exp(-lambda);let k=0,p=1;do{k++;p*=r()}while(p>L);return k-1}const sl=Math.sqrt(lambda),b=.931+2.53*sl,a=-.059+.02483*b,ia=1.1239+1.1328/(b-3.4),vr=.9277-3.6224/(b-2),ll=Math.log(lambda);for(;;){const u=r()-.5,v=r(),us=.5-Math.abs(u),k=Math.floor((2*a/us+b)*u+lambda+.43);if(us>=.07&&v<=vr)return k;if(k<0||(us<.013&&v>us))continue;if(Math.log(v*ia/(a/(us*us)+b))<=-lambda+k*ll-logGamma(k+1))return k}}
function gammaSample(shape:number,scale:number,r:()=>number):number{if(shape<1)return gammaSample(shape+1,scale,r)*Math.pow(r(),1/shape);const d=shape-1/3,c=1/Math.sqrt(9*d);for(;;){let x=normal(r),v=1+c*x;if(v<=0)continue;v=v*v*v;const u=r();if(u<1-.0331*x*x*x*x||Math.log(u)<.5*x*x+d*(1-v+Math.log(v)))return scale*d*v}}
function binomialSample(n:number,p:number,r:()=>number){let k=0;for(let i=0;i<n;i++)if(r()<p)k++;return k}
function nbSample(shape:number,c:number,r:()=>number){return poissonSample(gammaSample(shape,c/(1-c),r),r)}

function projectData(id:string,x:number,p:LabParams){
  if(id==="gaussian")return x;
  if(id==="gamma")return Math.max(.001,x);
  if(id==="binomial")return clamp(Math.round(x),0,Math.round(clamp(p.base1,2,80)));
  return Math.max(0,Math.round(x));
}

function initialSample(id:string,law:string,p:LabParams,r:()=>number){
  const centre=p.dataCenter,spread=Math.max(.1,p.dataSpread);
  if(law==="comb"){const u=r(),offset=u<.25?-spread:u<.75?0:spread;return projectData(id,centre+offset,p)}
  if(law==="outliers"){const x=r()<.92?centre+normal(r)*spread*.08:centre+(r()<.5?-2.4:2.4)*spread;return projectData(id,x,p)}
  if(law==="skewed"){const x=centre+spread*(-Math.log(Math.max(1e-9,1-r()))-1);return projectData(id,x,p)}
  if(law==="edges"){const x=centre+spread*Math.cos(Math.PI*r());return projectData(id,x,p)}
  if(law==="delta")return id==="binomial"?clamp(Math.round(centre),0,Math.round(p.base1)):id==="gaussian"||id==="gamma"?Math.max(id==="gamma"?.001:-Infinity,centre):Math.max(0,Math.round(centre));
  if(id==="gaussian"){if(law==="shifted")return centre+normal(r)*Math.max(.08,spread*.3);if(law==="mixture")return centre+(r()<.5?-spread:spread)+normal(r)*spread*.09;return centre-spread+2*spread*r()}
  if(id==="poisson"){if(law==="shifted")return poissonSample(Math.max(.05,centre),r);if(law==="mixture")return Math.max(0,Math.round(centre+(r()<.5?-spread:spread)));return Math.max(0,Math.round(centre-spread+2*spread*r()))}
  if(id==="gamma"){if(law==="shifted")return gammaSample(Math.max(.15,p.base1),Math.max(.05,centre/Math.max(.15,p.base1)),r);if(law==="mixture")return Math.max(.01,centre+(r()<.5?-spread:spread));return Math.max(.01,centre-spread+2*spread*r())}
  if(id==="binomial"){const n=Math.round(clamp(p.base1,2,80));if(law==="shifted")return binomialSample(n,clamp(centre/n,.01,.99),r);if(law==="mixture")return clamp(Math.round(centre+(r()<.5?-spread:spread)),0,n);return clamp(Math.round(centre-spread+2*spread*r()),0,n)}
  if(law==="shifted"){const shape=Math.max(.15,p.base1),c=clamp(centre/(shape+centre),.01,.94);return nbSample(shape,c,r)}if(law==="mixture")return Math.max(0,Math.round(centre+(r()<.5?-spread:spread)));return Math.max(0,Math.round(centre-spread+2*spread*r()))
}
function baselineSample(id:string,p:LabParams,r:()=>number){if(id==="gaussian")return p.base1+Math.max(.05,p.base2)*normal(r);if(id==="poisson")return poissonSample(Math.max(.05,p.base1),r);if(id==="gamma")return gammaSample(Math.max(.15,p.base1),Math.max(.05,p.base2),r);if(id==="binomial")return binomialSample(Math.round(clamp(p.base1,2,80)),clamp(p.base2,.02,.98),r);return nbSample(Math.max(.15,p.base1),clamp(p.base2,.02,.94),r)}
function kernelSample(id:string,x:number,t:number,p:LabParams,r:()=>number){const w=Math.exp(-t);if(t<1e-7)return x;if(id==="gaussian"){const mu=p.base1,sigma=Math.max(.05,p.base2);return mu+w*(x-mu)+sigma*Math.sqrt(1-w*w)*normal(r)}if(id==="poisson")return binomialSample(Math.max(0,Math.round(x)),w,r)+poissonSample(Math.max(.05,p.base1)*(1-w),r);if(id==="gamma"){const shape=Math.max(.15,p.base1),scale=Math.max(.05,p.base2),M=poissonSample(w*Math.max(0,x)/(scale*(1-w)),r);return gammaSample(shape+M,scale*(1-w),r)}if(id==="binomial"){const n=Math.round(clamp(p.base1,2,80)),prob=clamp(p.base2,.02,.98),xi=clamp(Math.round(x),0,n);return binomialSample(xi,prob+(1-prob)*w,r)+binomialSample(n-xi,prob*(1-w),r)}const shape=Math.max(.15,p.base1),c=clamp(p.base2,.02,.94),a=w*(1-c)/(1-c*w),ct=c*(1-w)/(1-c*w),B=binomialSample(Math.max(0,Math.round(x)),a,r);return B+nbSample(shape+B,ct,r)}

function hist(values:number[],lo:number,hi:number,bins:number,discrete:boolean){const n=discrete?Math.round(hi-lo)+1:bins,out=Array(n).fill(0),span=hi-lo;values.forEach(v=>{const i=discrete?Math.round(v-lo):Math.floor((v-lo)/span*n);if(i>=0&&i<n)out[i]++});return out.map(x=>x/values.length)}
function LawChart({initial,current,baseline,f}:{initial:number[];current:number[];baseline:number[];f:LabFamily}){const W=760,H=330,left=58,right=18,top=22,bottom=50,n=f.discrete?Math.round(f.hi-f.lo)+1:42,a=hist(initial,f.lo,f.hi,n,f.discrete),b=hist(current,f.lo,f.hi,n,f.discrete),q=hist(baseline,f.lo,f.hi,n,f.discrete),mx=Math.max(...a,...b,...q,.01),bw=(W-left-right)/n;return <svg className="law-chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Initial law relaxing to the baseline">
  {[0,.5,1].map(v=><g key={v}><line x1={left} x2={W-right} y1={H-bottom-v*(H-top-bottom)} y2={H-bottom-v*(H-top-bottom)} className="gridline"/><text x={left-8} y={H-bottom-v*(H-top-bottom)+4} textAnchor="end">{(v*mx).toFixed(2)}</text></g>)}
  {q.map((v,i)=>{const h=v/mx*(H-top-bottom);return <rect key={`q${i}`} x={left+i*bw+1} y={H-bottom-h} width={Math.max(1,bw-2)} height={h} fill="#aeb2ac" opacity=".48"/>})}
  {b.map((v,i)=>{const h=v/mx*(H-top-bottom);return <rect key={`b${i}`} x={left+i*bw+2} y={H-bottom-h} width={Math.max(1,bw-4)} height={h} fill={f.color} opacity=".82"/>})}
  {a.map((v,i)=>{const h=v/mx*(H-top-bottom);return <line key={`a${i}`} x1={left+i*bw+1} x2={left+(i+1)*bw-1} y1={H-bottom-h} y2={H-bottom-h} stroke="#17251f" strokeWidth="2"/>})}
  <line x1={left} x2={W-right} y1={H-bottom} y2={H-bottom} className="axis"/><line x1={left} x2={left} y1={top} y2={H-bottom} className="axis"/><text x={left} y={H-bottom+18}>{f.lo}</text><text x={W-right} y={H-bottom+18} textAnchor="end">{f.hi}</text>
  <text className="axis-title" x={(left+W-right)/2} y={H-5} textAnchor="middle">state x</text><text className="axis-title" transform={`translate(13 ${(top+H-bottom)/2}) rotate(-90)`} textAnchor="middle">{f.discrete?"probability mass":"probability per bin"}</text>
 </svg>}

function TrajectoryChart({paths,density,f,m0,v0,t,logAxis,tMax}:{paths:number[][];density:number[][];f:LabFamily;m0:number;v0:number;t:number;logAxis:boolean;tMax:number}){
  const W=760,H=300,left=52,right=18,top=22,bottom=48,steps=paths[0].length-1,xu=(u:number)=>left+u*(W-left-right),xt=(s:number)=>xu(labPosition(s,logAxis,tMax)),y=(z:number)=>H-bottom-(z-f.lo)/(f.hi-f.lo)*(H-top-bottom),mom=(s:number)=>{const w=Math.exp(-s),mean=f.mean+w*(m0-f.mean),vari=Math.max(0,w*w*v0+f.v*(1-w*w)+f.vp*(m0-f.mean)*w*(1-w));return{mean,sd:Math.sqrt(vari)}};
  const grid=Array.from({length:steps+1},(_,i)=>labTime(i/steps,logAxis,tMax)),meanLine=grid.map((s,i)=>`${xu(i/steps)},${y(mom(s).mean)}`).join(" "),idx=Math.min(steps,Math.round(labPosition(t,logAxis,tMax)*steps)),dCols=density.length,dRows=density[0]?.length||1,dMax=Math.max(.0001,...density.flat()),cellW=(W-left-right)/dCols,cellH=(H-top-bottom)/dRows,logIntensity=(value:number)=>value<=0?0:clamp((Math.log10(value/dMax)+3)/3,0,1),ticks=logAxis?[0,.01,.03,.1,.3,1,3,10].filter(s=>s<tMax).concat(tMax):[0,.25,.5,.75,1].map(q=>q*tMax),tickLabel=(s:number)=>s===0?"0":s<.1?s.toFixed(2):s<1?s.toFixed(1):Number.isInteger(s)?s.toFixed(0):s.toFixed(1);
  return <svg className="trajectory-chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Random trajectories following the analytic expectation">
    <defs><clipPath id="trajectory-clip"><rect x={left} y={top} width={W-left-right} height={H-top-bottom}/></clipPath></defs>
    {[0,.5,1].map(q=><g key={q}><line x1={left} x2={W-right} y1={top+q*(H-top-bottom)} y2={top+q*(H-top-bottom)} className="gridline"/><text x={left-8} y={top+q*(H-top-bottom)+4} textAnchor="end">{(f.hi-q*(f.hi-f.lo)).toFixed(1)}</text></g>)}
    {ticks.map(s=><text key={s} x={xt(s)} y={H-bottom+18} textAnchor="middle">{tickLabel(s)}</text>)}
    <g clipPath="url(#trajectory-clip)">{density.map((col,i)=>col.map((value,j)=>{const level=logIntensity(value);return <rect key={`${i}-${j}`} x={left+i*cellW} y={H-bottom-(j+1)*cellH} width={cellW+.35} height={cellH+.35} fill={f.color} opacity={level===0?0:.02+.9*level}/>}))}{paths.map((p,j)=><polyline key={j} points={p.map((z,i)=>`${xu(i/steps)},${y(z)}`).join(" ")} fill="none" stroke="#203e34" strokeWidth="1" opacity=".31"/>)}<polyline points={meanLine} fill="none" stroke="#101d18" strokeWidth="3"/><line x1={xt(t)} x2={xt(t)} y1={top} y2={H-bottom} stroke="#14241e" strokeDasharray="3 3"/>{paths.slice(0,10).map((p,j)=><circle key={j} cx={xt(t)} cy={y(p[idx])} r="2.4" fill="#10251d"/>)}</g>
    <line x1={left} x2={W-right} y1={H-bottom} y2={H-bottom} className="axis"/><line x1={left} x2={left} y1={top} y2={H-bottom} className="axis"/><text className="axis-title" x={(left+W-right)/2} y={H-5} textAnchor="middle">process time t {logAxis?"(log-spaced)":"(linear)"}</text><text className="axis-title" transform={`translate(13 ${(top+H-bottom)/2}) rotate(-90)`} textAnchor="middle">state Xₜ</text>
  </svg>
}

type EvidencePoint={x:number;reference:number;correction:number;total:number;q:number};
function smoothSeries(values:number[],passes=2){let out=values.slice();for(let pass=0;pass<passes;pass++)out=out.map((_,i)=>{let sum=0,weight=0;[-2,-1,0,1,2].forEach((d,k)=>{const j=clamp(i+d,0,out.length-1),w=[1,4,6,4,1][k];sum+=w*out[j];weight+=w});return sum/weight});return out}
function buildEvidenceFlow(current:number[],baseline:number[],f:LabFamily,p:LabParams):EvidencePoint[]{
  const n=f.discrete?Math.round(f.hi-f.lo)+1:64,pc0=hist(current,f.lo,f.hi,n,f.discrete),pb0=hist(baseline,f.lo,f.hi,n,f.discrete),pc=f.discrete?pc0:smoothSeries(pc0,3),pb=f.discrete?pb0:smoothSeries(pb0,3),alpha=f.discrete?.35/Math.max(1,current.length):1e-7;
  const q=pc.map((v,i)=>(v+alpha)/(pb[i]+alpha)),xs=Array.from({length:n},(_,i)=>f.discrete?f.lo+i:f.lo+(i+.5)*(f.hi-f.lo)/n),logq=q.map(v=>clamp(Math.log(Math.max(1e-9,v)),-8,8));
  return xs.map((x,i)=>{
    const reference=f.mean-x;
    if(!f.discrete){const i0=Math.max(0,i-1),i1=Math.min(n-1,i+1),score=(logq[i1]-logq[i0])/Math.max(1e-9,xs[i1]-xs[i0]),a=f.id==="gaussian"?2*Math.max(.05,p.base2)**2:2*Math.max(.05,p.base2)*Math.max(0,x),correction=a*score;return{x,reference,correction,total:reference+correction,q:q[i]}}
    let up=0,down=0;
    if(f.id==="poisson"){up=Math.max(.05,p.base1);down=x}
    else if(f.id==="binomial"){const N=Math.round(clamp(p.base1,2,80)),prob=clamp(p.base2,.02,.98);up=prob*Math.max(0,N-x);down=(1-prob)*Math.max(0,x)}
    else {const shape=Math.max(.15,p.base1),c=clamp(p.base2,.02,.94);up=c*(shape+x)/(1-c);down=x/(1-c)}
    const upRatio=i<n-1?q[i+1]/Math.max(1e-9,q[i]):1,downRatio=i>0?q[i-1]/Math.max(1e-9,q[i]):1,total=up*upRatio-down*downRatio;return{x,reference,correction:total-reference,total,q:q[i]}
  })
}

function EvidenceFlowChart({points,f,probe,onProbe}:{points:EvidencePoint[];f:LabFamily;probe:number;onProbe:(x:number)=>void}){
  const W=1080,H=310,left=72,right=26,top=25,bottom=56,span=Math.max(1,f.hi-f.lo),cap=Math.max(4,span),x=(v:number)=>left+(v-f.lo)/span*(W-left-right),y=(v:number)=>H-bottom-(clamp(v,-cap,cap)+cap)/(2*cap)*(H-top-bottom),nearest=points.reduce((best,p)=>Math.abs(p.x-probe)<Math.abs(best.x-probe)?p:best,points[0]),line=(key:"reference"|"correction"|"total")=>points.map(p=>`${x(p.x)},${y(p[key])}`).join(" "),ticks=[-1,-.5,0,.5,1],xTicks=[0,.25,.5,.75,1];
  return <div className="evidence-visual">
    <svg className="evidence-chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Reverse evidence-flow decomposition for the ${f.name} family at the current noising time`} onPointerDown={e=>{const box=e.currentTarget.getBoundingClientRect(),px=(e.clientX-box.left)/box.width*W;onProbe(clamp(f.lo+(px-left)/(W-left-right)*span,f.lo,f.hi))}}>
      <defs><clipPath id="evidence-clip"><rect x={left} y={top} width={W-left-right} height={H-top-bottom}/></clipPath></defs>
      {ticks.map(v=><g key={v}><line x1={left} x2={W-right} y1={y(v*cap)} y2={y(v*cap)} className={v===0?"zero-line":"gridline"}/><text x={left-10} y={y(v*cap)+4} textAnchor="end">{(v*cap).toFixed(cap<10?1:0)}</text></g>)}
      {xTicks.map(v=><text key={v} x={x(f.lo+v*span)} y={H-bottom+19} textAnchor="middle">{(f.lo+v*span).toFixed(f.discrete?0:1)}</text>)}
      <g clipPath="url(#evidence-clip)"><polyline points={line("reference")} className="evidence-reference"/><polyline points={line("correction")} className="evidence-correction" style={{stroke:f.color}}/><polyline points={line("total")} className="evidence-total"/><line x1={x(nearest.x)} x2={x(nearest.x)} y1={top} y2={H-bottom} className="probe-line"/>{(["reference","correction","total"] as const).map((key,i)=><circle key={key} cx={x(nearest.x)} cy={y(nearest[key])} r={i===2?4:3} className={`probe-dot ${key}`} style={key==="correction"?{fill:f.color}:undefined}/>)}</g>
      <line x1={left} x2={W-right} y1={H-bottom} y2={H-bottom} className="axis"/><line x1={left} x2={left} y1={top} y2={H-bottom} className="axis"/><text className="axis-title" x={(left+W-right)/2} y={H-6} textAnchor="middle">state x</text><text className="axis-title" transform={`translate(16 ${(top+H-bottom)/2}) rotate(-90)`} textAnchor="middle">local drift  dx/dτ  [state / reverse-time]</text>
    </svg>
    <label className="probe-control"><span>Inspect state x <output>{nearest.x.toFixed(f.discrete?0:2)}</output></span><input type="range" min={f.lo} max={f.hi} step={f.discrete?1:span/200} value={nearest.x} onChange={e=>onProbe(+e.target.value)}/></label>
    <div className="evidence-values" aria-label="Local reverse-flow decomposition"><span>reference <b>{nearest.reference.toFixed(2)}</b></span><i>+</i><span>evidence correction <b>{nearest.correction.toFixed(2)}</b></span><i>=</i><span>reverse total <b>{nearest.total.toFixed(2)}</b></span><span className="q-value">q<sub>t</sub>(x) ≈ <b>{nearest.q.toFixed(2)}</b></span></div>
  </div>
}

function NumberField({label,value,min,max,step,onChange}:{label:string;value:number;min:number;max:number;step:number;onChange:(v:number)=>void}){
  return <label className="param-field"><span>{label}</span><input type="number" value={value} min={min} max={max} step={step} onChange={e=>onChange(clamp(+e.target.value,min,max))}/></label>
}
const dataLawNames:Record<string,string>={delta:"δ-data",shifted:"shifted family",mixture:"balanced bimodal",uniform:"uniform block",comb:"three-point comb",outliers:"sparse contamination",skewed:"one-sided tail",edges:"edge-loaded interval"};

function NoisingLab(){
  const [id,setId]=useState("gaussian"),[law,setLaw]=useState("delta"),[u,setU]=useState(0),[logTime,setLogTime]=useState(true),[axisLog,setAxisLog]=useState(false),[tMax,setTMax]=useState(5),[playing,setPlaying]=useState(false),[seed,setSeed]=useState(31),[probe,setProbe]=useState(0),[paramSets,setParamSets]=useState<Record<string,LabParams>>(defaultLabParams);
  const p=paramSets[id],baseFamily=labFamilies.find(x=>x.id===id)!,f=configuredFamily(baseFamily,p),setParam=(key:keyof LabParams,value:number)=>setParamSets(all=>({...all,[id]:{...all[id],[key]:value}}));
  const timeFromPosition=(v:number,log=logTime)=>labTime(v,log,tMax);
  const t=timeFromPosition(u);
  useEffect(()=>{if(!playing)return;const timer=setInterval(()=>setU(v=>{if(v>=1){setPlaying(false);return 1}return Math.min(1,v+.005)}),40);return()=>clearInterval(timer)},[playing]);
  useEffect(()=>{setPlaying(false);setU(0)},[id,law]);
  useEffect(()=>{setProbe(f.mean)},[id,p.base1,p.base2]);
  const laws=useMemo(()=>{const n=Math.round(p.sampleN),ri=rng(seed),rk=rng(seed+991),rb=rng(seed+1777),initial=Array.from({length:n},()=>initialSample(id,law,p,ri));return{initial,current:initial.map(x=>kernelSample(id,x,t,p,rk)),baseline:Array.from({length:n},()=>baselineSample(id,p,rb))}},[id,law,t,seed,p]);
  const trajectories=useMemo(()=>{const n=16,steps=120,times=Array.from({length:steps+1},(_,i)=>labTime(i/steps,axisLog,tMax)),r=rng(seed+3331),out:number[][]=[];for(let j=0;j<n;j++){let x=initialSample(id,law,p,r),path=[x];for(let i=1;i<=steps;i++){x=kernelSample(id,x,times[i]-times[i-1],p,r);path.push(x)}out.push(path)}return out},[id,law,seed,p,axisLog,tMax]);
  const density=useMemo(()=>{const cols=64,bins=f.discrete?Math.round(f.hi-f.lo)+1:48,n=Math.round(p.densityN);return Array.from({length:cols},(_,i)=>{const s=labTime(i/(cols-1),axisLog,tMax),ri=rng(seed+7103+i*97),rk=rng(seed+9109+i*131),values=Array.from({length:n},()=>kernelSample(id,initialSample(id,law,p,ri),s,p,rk));return hist(values,f.lo,f.hi,bins,f.discrete)})},[id,law,seed,p,f.lo,f.hi,f.discrete,axisLog,tMax]);
  const evidence=useMemo(()=>buildEvidenceFlow(laws.current,laws.baseline,f,p),[laws.current,laws.baseline,f,p]),m0=laws.initial.reduce((s,x)=>s+x,0)/laws.initial.length,v0=laws.initial.reduce((s,x)=>s+(x-m0)*(x-m0),0)/laws.initial.length,currentHist=hist(laws.current,f.lo,f.hi,f.discrete?1:42,f.discrete),baselineHist=hist(laws.baseline,f.lo,f.hi,f.discrete?1:42,f.discrete),tv=.5*currentHist.reduce((s,x,i)=>s+Math.abs(x-baselineHist[i]),0),w=Math.exp(-t),mt=f.mean+w*(m0-f.mean),vt=Math.max(0,w*w*v0+f.v*(1-w*w)+f.vp*(m0-f.mean)*w*(1-w));
  return <section className="noising-lab">
    <div className="lab-grid">
      <aside className="panel lab-controls"><div className="panel-title"><span>05</span><h2>Experiment</h2></div>
        <label className="select-control"><span>Baseline family</span><select value={id} onChange={e=>setId(e.target.value)}>{labFamilies.map(x=><option key={x.id} value={x.id}>{x.name}</option>)}</select></label>
        <label className="select-control"><span>Initial data law</span><select value={law} onChange={e=>setLaw(e.target.value)}><option value="delta">Delta spike</option><option value="shifted">Shifted family member</option><option value="mixture">Balanced bimodal</option><option value="uniform">Uniform block</option><option value="comb">Three-point comb</option><option value="outliers">Sparse contamination</option><option value="skewed">One-sided exponential tail</option><option value="edges">Edge-loaded interval</option></select></label>
        <div className="process-tag"><small>REFERENCE PROCESS</small><b>{f.process}</b></div>
        <details className="parameter-drawer">
          <summary>Family &amp; data parameters</summary>
          <div className="parameter-body">
            <small className="parameter-group">BASELINE LAW</small>
            {id==="gaussian"&&<><NumberField label="Mean μ" value={p.base1} min={-8} max={8} step={.1} onChange={v=>setParam("base1",v)}/><NumberField label="Standard deviation σ" value={p.base2} min={.1} max={5} step={.1} onChange={v=>setParam("base2",v)}/></>}
            {id==="poisson"&&<NumberField label="Mean λ" value={p.base1} min={.1} max={30} step={.1} onChange={v=>setParam("base1",v)}/>} 
            {id==="gamma"&&<><NumberField label="Shape r" value={p.base1} min={.2} max={20} step={.1} onChange={v=>setParam("base1",v)}/><NumberField label="Scale θ" value={p.base2} min={.1} max={5} step={.1} onChange={v=>setParam("base2",v)}/></>}
            {id==="binomial"&&<><NumberField label="Trials n" value={p.base1} min={2} max={80} step={1} onChange={v=>setParam("base1",Math.round(v))}/><NumberField label="Success probability p" value={p.base2} min={.02} max={.98} step={.01} onChange={v=>setParam("base2",v)}/></>}
            {id==="neg-binomial"&&<><NumberField label="Shape r" value={p.base1} min={.2} max={20} step={.1} onChange={v=>setParam("base1",v)}/><NumberField label="Probability c" value={p.base2} min={.02} max={.85} step={.01} onChange={v=>setParam("base2",v)}/></>}
            <small className="parameter-group">DATA LAW</small>
            <NumberField label="Centre" value={p.dataCenter} min={id==="gaussian"?-12:0} max={id==="binomial"?Math.round(p.base1):60} step={id==="gaussian"?.1:1} onChange={v=>setParam("dataCenter",v)}/>
            <NumberField label="Width / separation" value={p.dataSpread} min={.1} max={id==="binomial"?Math.max(1,Math.round(p.base1)/2):20} step={.1} onChange={v=>setParam("dataSpread",v)}/>
            <small className="parameter-group">MONTE CARLO</small>
            <NumberField label="Law instances N" value={p.sampleN} min={400} max={5000} step={200} onChange={v=>setParam("sampleN",Math.round(v))}/>
            <NumberField label="Density events / slice" value={p.densityN} min={4000} max={40000} step={2000} onChange={v=>setParam("densityN",Math.round(v))}/>
          </div>
        </details>
        <div className="time-config-row"><label><span>Maximum time t<sub>max</sub></span><input type="number" value={tMax} min={1} max={12} step={.5} onChange={e=>{setPlaying(false);setTMax(clamp(+e.target.value,1,12))}}/></label><div><span>Animation spacing</span><div className="clock-switch" role="group" aria-label="Animation time spacing"><button className={!logTime?"active":""} onClick={()=>{const now=t;setLogTime(false);setU(labPosition(now,false,tMax))}}>Linear</button><button className={logTime?"active":""} onClick={()=>{const now=t;setLogTime(true);setU(labPosition(now,true,tMax))}}>Log</button></div></div></div>
        <Slider label="Noising time  t" value={u} min={0} max={1} step={.001} onChange={v=>{setPlaying(false);setU(v)}} format={v=>timeFromPosition(v).toFixed(3)}/>
        <div className="transport"><button className="play" onClick={()=>{if(u>=1)setU(0);setPlaying(x=>!x)}}>{playing?"Ⅱ  Pause":"▶  Animate"}</button><button onClick={()=>{setPlaying(false);setU(0)}}>↤ Reset</button><button aria-label="Resample" onClick={()=>setSeed(x=>x+1)}>↻</button></div>
        <div className="legend"><span><i className="initial-key"/>initial data</span><span><i style={{background:f.color}}/>law at t</span><span><i className="base-key"/>baseline</span></div>
      </aside>
      <div className="panel lab-display">
        <div className="chart-head"><h2 className="compact-title"><span>Law evolution:</span> {dataLawNames[law]} → {f.baseline}</h2><div className="time-readout"><small>TIME</small><b>{t.toFixed(2)}</b></div></div>
        <LawChart {...laws} f={f}/>
        <div className="info-row"><div><small>EMPIRICAL TV TO BASELINE</small><b>{tv.toFixed(3)}</b></div><div><small>FIRST-MODE MEMORY</small><b>{(100*w).toFixed(1)}%</b></div><div><small>EXACT CONTRACTION</small><b>w = e<sup>−t</sup></b></div></div>
        <div className="grade-flow"><span>OPS memory</span>{[1,2,3,4,5].map(n=><div key={n}><small>n={n}</small><i><b style={{width:`${100*Math.exp(-n*t)}%`,background:f.color}}/></i><output>{Math.exp(-n*t).toFixed(2)}</output></div>)}</div>
        <div className="trajectory-block"><div className="trajectory-head"><div><p className="eyebrow">EXPECTATION VS REALISATIONS</p><h3>Random paths follow a deterministic moment flow</h3></div><div className="trajectory-tools"><div className="axis-toggle"><span>Time axis</span><div className="clock-switch" role="group" aria-label="Lower chart time-axis scale"><button className={!axisLog?"active":""} onClick={()=>setAxisLog(false)}>Linear</button><button className={axisLog?"active":""} onClick={()=>setAxisLog(true)}>Log</button></div></div><div className="moment-now"><span>E[X<sub>t</sub>] <b>{mt.toFixed(2)}</b></span><span>SD[X<sub>t</sub>] <b>{Math.sqrt(vt).toFixed(2)}</b></span></div></div></div><TrajectoryChart paths={trajectories} density={density} f={f} m0={m0} v0={v0} t={t} logAxis={axisLog} tMax={tMax}/><div className="trajectory-legend"><span><i className="mean-key"/> analytic expectation</span><span><i className="path-key"/> random realisations</span></div><div className="density-legend" style={{"--density-color":f.color} as React.CSSProperties}><span>10<sup>−3</sup> × max</span><i/><span>max</span><b>log<sub>10</sub> relative density p<sub>t</sub>(x)</b></div></div>
      </div>
    </div>
    <details className="evidence-drawer"><summary><span>Section 13</span> Reverse evidence flow</summary><div className="evidence-body">
      <div className="evidence-head"><div><p className="eyebrow">LOCAL DOOB-TRANSFORM DECOMPOSITION</p><h3>Reference motion + learned evidence = reverse motion</h3></div><div className="evidence-time"><small>NOISING TIME</small><b>t = {t.toFixed(3)}</b></div></div>
      <div className="evidence-formula">{f.discrete?<span>Ĵ<sub>t</sub>(x,y) = J(x,y) · q<sub>t</sub>(y) / q<sub>t</sub>(x)</span>:<span>b̂<sub>t</sub>(x) = b(x) + a(x) ∂<sub>x</sub> log q<sub>t</sub>(x)</span>}<small>{f.discrete?"The chart shows the resulting net jump drift Ĵ₊−Ĵ₋.":"The chart separates the baseline drift from the amplitude-score correction."}</small></div>
      <div className="evidence-legend"><span><i className="ref-flow"/>reference drift</span><span><i className="correction-flow" style={{borderColor:f.color}}/>evidence correction</span><span><i className="total-flow"/>reverse total</span></div>
      <EvidenceFlowChart points={evidence} f={f} probe={probe} onProbe={setProbe}/>
      <p className="evidence-note">q<sub>t</sub> = p<sub>t</sub>/p<sub>ref</sub> is estimated from the displayed Monte Carlo ensemble. The vertical scale is linear and fixed to ±{Math.max(4,f.hi-f.lo).toFixed(0)} state units per unit reverse-time.{!f.discrete&&t<.01?" At t=0 a continuous delta law is singular; the smoothed finite-sample score is only indicative until t>0.":""}</p>
    </div></details>
    <details className="scientific-notes"><summary>Analytic moments &amp; exact transition kernel</summary><div><p><b>Mean:</b> E[X<sub>t</sub>] = μ + w(E[X<sub>0</sub>] − μ)</p><p><b>Variance:</b> Var[X<sub>t</sub>] = w²Var[X<sub>0</sub>] + V(μ)(1−w²) + V′(μ)(E[X<sub>0</sub>]−μ)w(1−w)</p><code>{f.kernel}</code></div></details>
  </section>
}

export default function Home(){
  const [fid,setFid]=useState("gaussian"),[theta,setTheta]=useState(1),[kappa,setKappa]=useState(1.15),[time,setTime]=useState(3),[seed,setSeed]=useState(7),[tab,setTab]=useState<"paths"|"ensemble">("paths");
  const f=families.find(x=>x.id===fid)!; const safeTheta=clamp(theta,f.lo+.2,f.hi-.2);
  const sim=useMemo(()=>simulate(f,safeTheta,kappa,time,seed),[f,safeTheta,kappa,time,seed]),hasProcess=f.id!=="hyperbolic";
  const mean=hasProcess?sim.finals.reduce((a,b)=>a+b,0)/sim.finals.length:NaN; const vari=hasProcess?sim.finals.reduce((a,b)=>a+(b-mean)**2,0)/sim.finals.length:NaN;
  return <main>
    <header><div className="brand"><span className="mark">∂</span><span>STOCHASTIC LAB / 01</span></div><div className="status"><i/> LIVE SIMULATION</div></header>
    <section className="hero">
      <div><p className="eyebrow">NATURAL EXPONENTIAL FAMILIES</p><h1>NEF–QVF <em>Diffusion Explorer</em></h1></div>
    </section>
    <section className="family-strip" aria-label="Distribution family">
      {families.map(x=><button className={x.id===fid?"active":""} style={{"--accent":x.color} as React.CSSProperties} key={x.id} onClick={()=>{setFid(x.id);setTheta(clamp(theta,x.lo+.5,x.hi-.5));}}><span>{x.short}</span><div><b>{x.name}</b><small>{x.support}</small></div></button>)}
    </section>
    <section className="workspace">
      <aside className="panel controls"><div className="panel-title"><span>01</span><h2>Process controls</h2></div>
        <DynamicsEquation f={f}/>
        <Slider label="Baseline mean  μ" value={safeTheta} min={f.lo+.1} max={f.hi-.1} step={.1} onChange={setTheta}/>
        <Slider label="Mean reversion  κ" value={kappa} min={.05} max={3} step={.05} onChange={setKappa}/>
        <Slider label="Time horizon  T" value={time} min={.5} max={8} step={.25} onChange={setTime} format={v=>`${v.toFixed(2)} s`}/>
        <button className="reroll" disabled={!hasProcess} onClick={()=>setSeed(s=>s+1)}>↻ &nbsp; Resample trajectories</button>
        <p className="note">Exact transition steps · stationary NEF baseline · support preserving</p>
      </aside>
      <div className="main-panel panel">
        <div className="chart-head"><h2 className="compact-title"><span>Current family:</span> {f.name} {f.id==="gaussian"||f.id==="gamma"?"diffusion":f.id==="hyperbolic"?"exception":"birth–death process"}</h2>{hasProcess&&<div className="tabs"><button className={tab==="paths"?"active":""} onClick={()=>setTab("paths")}>Sample paths</button><button className={tab==="ensemble"?"active":""} onClick={()=>setTab("ensemble")}>Endpoint law</button></div>}</div>
        {hasProcess?(tab==="paths"?<PathChart data={sim.paths} f={f} horizon={time}/>:<Histogram values={sim.finals} f={f}/>):<div className="ghs-empty"><b>Positivity fails.</b><p>The OPS spectral kernel exists, but it takes negative values and therefore cannot be a transition probability. GHS remains part of the NEF–QVF algebra, not the five-family Markov animation.</p></div>} 
        <div className="metrics"><div><small>ENSEMBLE MEAN</small><b>{hasProcess?mean.toFixed(3):"—"}</b></div><div><small>ENSEMBLE VARIANCE</small><b>{hasProcess?vari.toFixed(3):"—"}</b></div><div><small>BASELINE V(μ)</small><b>{variance(f,safeTheta).toFixed(3)}</b></div><div><small>TRAJECTORIES</small><b>{hasProcess?"260":"0"}</b></div></div>
      </div>
    </section>
    <NoisingLab/>
    <footer><span>NEF–QVF / INTERACTIVE NOTEBOOK</span><span>Adjust · compare · infer</span></footer>
  </main>
}
