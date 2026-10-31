"""Reproducible scalar nonlinear filtering benchmark.

KF is a deliberately linearized fixed-model baseline, not an optimal KF for
this nonlinear system. AGCR-PF is a transparent illustrative implementation,
not a verified reproduction of any previously unpublished user algorithm.
"""
import argparse
import csv
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def f(x, k):
    return 0.55*x + 18*x/(1+x*x) + 7*np.cos(0.7*k)

def df(x):
    return 0.55 + 18*(1-x*x)/(1+x*x)**2

def h(x):
    return x*x/15

def dh(x):
    return 2*x/15

def systematic(w, rng):
    n=len(w); p=(rng.random()+np.arange(n))/n
    return np.searchsorted(np.cumsum(w),p)

def logweights(y, particles, r):
    return -0.5*(y-h(particles))**2/r

def normalize(logw):
    a=np.exp(logw-np.max(logw)); return a/a.sum()

def run_filter(name, ys, q, r, n, seed):
    rng=np.random.default_rng(seed)
    x=0.; P=5.; particles=rng.normal(0,np.sqrt(P),n)
    w=np.full(n,1/n); estimates=[]; esses=[]
    for k,y in enumerate(ys):
        if name=='KF':
            # Constant linear surrogate: x[k]=0.55*x[k-1]+7cos(.7k)+w;
            # y[k]=x[k]/15+v. It is intentionally misspecified.
            x=0.55*x+7*np.cos(.7*k); P=0.55**2*P+q
            H=1/15; S=H*H*P+r; K=P*H/S
            x=x+K*(y-H*x); P=max((1-K*H)*P,1e-10)
        elif name=='EKF':
            A=df(x); x=f(x,k); P=A*A*P+q
            H=dh(x); S=H*H*P+r; K=P*H/S
            x=x+K*(y-h(x)); P=max((1-K*H)*P,1e-10)
        elif name=='UKF':
            alpha=0.6; beta=2.; kappa=0.; L=1
            lam=alpha**2*(L+kappa)-L; scale=L+lam
            wm=np.array([lam/scale,1/(2*scale),1/(2*scale)])
            wc=wm.copy(); wc[0]+=1-alpha**2+beta
            def sigmas(m,c):
                d=np.sqrt(max(scale*c,1e-12));return np.array([m,m+d,m-d])
            z=f(sigmas(x,P),k); xp=np.dot(wm,z)
            Pp=np.dot(wc,(z-xp)**2)+q
            zp=h(sigmas(xp,Pp)); yp=np.dot(wm,zp)
            S=np.dot(wc,(zp-yp)**2)+r
            C=np.dot(wc,(sigmas(xp,Pp)-xp)*(zp-yp))
            K=C/max(S,1e-12);x=xp+K*(y-yp);P=max(Pp-K*K*S,1e-10)
        else:
            # Bootstrap PF uses transition proposal. AGCR uses a mixture
            # of Gaussian and Cauchy moves after a regular resample step.
            particles=f(particles,k)+rng.normal(0,np.sqrt(q),n)
            w=normalize(np.log(w+1e-300)+logweights(y,particles,r))
            ess=1/np.sum(w*w); esses.append(ess)
            if name=='AGCR-PF':
                # Likelihood-informed resample-move MH step. Mixture proposals
                # are symmetric; acceptance uses likelihood only, because
                # the target is the discrete predictive mixture posterior.
                # Here a Gaussian kernel approximation of the predictive prior
                # makes the continuous move target explicit.
                idx=systematic(w,rng); base=particles.copy(); particles=particles[idx]
                # Mixture prior density under predictive particles
                bandwidth=max(0.35,1.06*np.std(base)*n**(-0.2))
                # Limit quadratic cost via a fixed subset of predictive support.
                support=base[rng.choice(n,size=min(n,100),replace=False)]
                def logtarget(z):
                    a=-0.5*((z[:,None]-support[None,:])/bandwidth)**2
                    m=a.max(axis=1)
                    prior=m+np.log(np.mean(np.exp(a-m[:,None]),axis=1))
                    return prior+logweights(y,z,r)
                scale_move=max(0.1,0.35*np.std(particles))
                is_cauchy=rng.random(n)<0.2
                step=rng.normal(0,scale_move,n)
                step[is_cauchy]=np.clip(rng.standard_cauchy(is_cauchy.sum()),-8,8)*scale_move
                candidate=particles+step
                accepted=np.log(rng.random(n))<np.minimum(0,logtarget(candidate)-logtarget(particles))
                particles[accepted]=candidate[accepted]
                w=np.full(n,1/n)
            elif ess<n/2:
                particles=particles[systematic(w,rng)];w=np.full(n,1/n)
            x=np.dot(w,particles)
        estimates.append(x)
    return np.array(estimates), np.mean(esses) if esses else np.nan

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--runs',type=int,default=30)
    ap.add_argument('--steps',type=int,default=60)
    ap.add_argument('--particles',type=int,default=200)
    ap.add_argument('--seed',type=int,default=2026)
    ap.add_argument('--output',default='results');a=ap.parse_args()
    if min(a.runs,a.steps,a.particles)<1: raise ValueError('Counts must be positive')
    q=1.;r=0.01;names=['KF','EKF','UKF','BPF','AGCR-PF']
    records=[];sample=None
    for run in range(a.runs):
        rng=np.random.default_rng(a.seed+run)
        truth=np.empty(a.steps);ys=np.empty(a.steps);prev=0.
        for k in range(a.steps):
            prev=f(prev,k)+rng.normal(0,np.sqrt(q));truth[k]=prev
            ys[k]=h(prev)+rng.normal(0,np.sqrt(r))
        for j,name in enumerate(names):
            t=time.perf_counter()
            pred,ess=run_filter(name,ys,q,r,a.particles,a.seed+100000+run*11+j)
            elapsed=time.perf_counter()-t
            records.append(dict(run=run,method=name,rmse=np.sqrt(np.mean((pred-truth)**2)),seconds=elapsed,mean_ess=ess))
            if run==0:
                if sample is None:sample={'truth':truth,'measurement':ys}
                sample[name]=pred
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    with (out/'per_run.csv').open('w',newline='') as fp:
        wr=csv.DictWriter(fp,fieldnames=list(records[0]));wr.writeheader();wr.writerows(records)
    print('Model: x[k]=0.55x[k-1]+18x[k-1]/(1+x[k-1]^2)+7cos(0.7k)+w, y=x^2/15+v')
    print(f'q={q}, r={r}, runs={a.runs}, steps={a.steps}, particles={a.particles}, seed={a.seed}')
    print(f'{"Method":<12} {"RMSE mean":>12} {"RMSE std":>12} {"Time(s)":>12} {"Mean ESS":>12}')
    for name in names:
        rows=[d for d in records if d['method']==name]
        rm=np.array([d['rmse'] for d in rows]);tm=np.array([d['seconds'] for d in rows]);ess=np.array([d['mean_ess'] for d in rows])
        print(f'{name:<12} {rm.mean():12.4f} {rm.std():12.4f} {tm.mean():12.4f} {np.nanmean(ess) if name in names[3:] else float("nan"):12.2f}')
    plt.figure(figsize=(11,5));plt.plot(sample['truth'],label='True state',color='black',linewidth=2)
    for name in names:plt.plot(sample[name],label=name,alpha=.8)
    plt.xlabel('Step');plt.ylabel('State');plt.legend(ncol=3);plt.tight_layout();plt.savefig(out/'trajectories.png',dpi=170);plt.close()
    plt.figure(figsize=(9,5))
    data=[[d['rmse'] for d in records if d['method']==name] for name in names]
    plt.boxplot(data,tick_labels=names);plt.ylabel('Run RMSE');plt.tight_layout();plt.savefig(out/'rmse_boxplot.png',dpi=170);plt.close()
if __name__=='__main__':main()
