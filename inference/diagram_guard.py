"""Small appearance classifier used only to defer uncertain gore blocks to review."""
import math
import numpy as np
from PIL import Image
from image_preprocess import normalize
FEATURE_VERSION=1
def is_monochrome(path):
    x=np.asarray(normalize(path).resize((128,128),Image.Resampling.BICUBIC),dtype=np.float64)/255.
    chroma=x.max(2)-x.min(2)
    return bool(chroma.mean()<.02 and np.quantile(chroma,.99)<.05)
def features(path):
    im=normalize(path).resize((128,128),Image.Resampling.BICUBIC);x=np.asarray(im,dtype=np.float64)/255.;flat=x.reshape(-1,3)
    q=np.minimum((flat*15).astype(int),15);codes=q[:,0]*256+q[:,1]*16+q[:,2];hist=np.bincount(codes,minlength=4096).astype(float)/len(flat);freq=np.sort(hist)[::-1]
    gray=flat@np.array([.299,.587,.114]);gh=np.bincount(np.minimum((gray*63).astype(int),63),minlength=64)/len(gray)
    red=(flat[:,0]>.25)&(flat[:,0]>flat[:,1]*1.25)&(flat[:,0]>flat[:,2]*1.25)
    sat=flat.max(1)-flat.min(1);dx=np.abs(x[:,1:]-x[:,:-1]);dy=np.abs(x[1:]-x[:-1]);redx=flat[red]
    def entropy(h):
        a=h[h>0];return float(-(a*np.log(a)).sum())
    rh=np.bincount(codes[red],minlength=4096).astype(float);rh=rh/max(1.,rh.sum())
    values=[entropy(hist)/math.log(4096),freq[0],sum(freq[:2]),sum(freq[:4]),float((hist>1/len(flat)).sum()/4096),entropy(gh)/math.log(64),
            float((flat.min(1)>.85).mean()),float((flat.max(1)<.15).mean()),float(red.mean()),float(sat.mean()),float(sat.std()),
            *flat.mean(0).tolist(),*flat.std(0).tolist(),float(dx.mean()),float(dy.mean()),float(dx.std()),float(dy.std()),
            float(redx.std(0).mean()) if len(redx) else 0.,float(rh.max()),entropy(rh)/math.log(4096)]
    assert len(values)==24 and all(math.isfinite(v) for v in values)
    return values
def score(path,model):
    assert model['feature_version']==FEATURE_VERSION
    x=features(path);z=sum((v-m)/s*w for v,m,s,w in zip(x,model['mean'],model['scale'],model['coef']))+model['intercept']
    assert len(x)==len(model['coef']) and math.isfinite(z)
    return 1/(1+math.exp(-max(-60.,min(60.,z))))
