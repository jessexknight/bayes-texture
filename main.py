import numpy as np
import matplotlib.pyplot as pp
from itertools import chain
from numpy import array as arr
from skimage.transform import rescale
from scipy import misc

def imgsplit(img,nn):
  # split an image into a 2D list-of-list of sub-arrays (technically 4D array)
  mm = [np.linspace(0,img.shape[i],nn[i]+1)[1:-1].astype(np.int) for i in [0,1]]
  return arr([[ss for ss in np.split(s  ,mm[1],axis=1)]
                  for s  in np.split(img,mm[0],axis=0)])

def imgunsplit(imgarr):
  # reconstruct an image from a 2D list-of-list of sub-arrays
  return arr(np.concatenate([np.concatenate([ss for ss in s],axis=1)for s in imgarr],axis=0))

def vec2block(v,size):
  # this magic function reshapes a vector of objects to a list of blocks
  # having the shape size.
  # e.g. vec2block([1,2,3,4,5,6,7,8,...],[2,2]) returns:
  # [[[1,2],[3,4]],  [[5,6],[7,8]], ...]
  def reshape(v,size):
    w = size[1]
    return [v[i:j] for i,j in zip(range(0,len(v)+1,w),range(w,len(v)+1,w))]
  N  = np.prod(size)
  return [reshape(v[i:j],size) for i,j in zip(range(0,len(v),N),range(N,len(v)+1,N))]

def extractallfeats(x):
  # extract all features from this array of blocks
  fx = arr([[extractfeats(ss) for ss in s] for s in x])
  fa = fx.reshape(-1,fx.shape[-1])
  fu = np.mean(fa,axis=0)
  fc = np.cov(fa,rowvar=False)
  return fa,fu,fc,fx

def extractfeats(x):
  # extract 10 features from this [NxN ]block
  f = []
  f.extend([np.mean(x)])   # mean
  f.extend([np.std(x)])    # std
  f.extend(feat_hifreq(x)) # fft features
  f.extend(feat_acorr(x))  # autocorrelation features
  return f

def feat_hifreq(x):
  # 4 fft features: mean magnitude of the fft in hexadecants 1,3,4,8
  xfft = np.fft.fft2(x)
  sfft = imgsplit(xfft,[4,4])
  fvx = [np.mean(abs(sfft[0][0])),
         np.mean(abs(sfft[0][2])),
         np.mean(abs(sfft[0][3])),
         np.mean(abs(sfft[2][3]))]
  return fvx

def feat_acorr(x):
  # 4 autocorrelation features: tau = [1,4], vertical and horizontal
  fvx = []
  for t in [1,4]:
    fvx.extend([np.sqrt(np.mean(np.multiply(x[:,0:-t],x[:,t:])))])
    fvx.extend([np.sqrt(np.mean(np.multiply(x[0:-t,:],x[t:,:])))])
  return fvx

def bayes_classify(x,u,c):
  # perform bayes classification on an array of feature vectors x
  # for classes defined by the list of means u and covariances c
  D = np.zeros(len(u))
  Y = np.zeros(len(x))
  for p in range(len(x)):
    for y in range(len(u)):
      D[y] = np.dot((x[p]-u[y]),np.dot(np.linalg.pinv(c[y]),np.transpose(x[p]-u[y])))
    Y[p] = np.argmin(D)
  return Y

def colorover(Ig,Ic,cmap,alpha=0.5):
  # add a color overlay (Ic) to a grayscale image (Ig)
  # IG = arr(pp.get_cmap('gray')(Ig),dtype=np.float)
  IG = Ig
  IC = arr(pp.get_cmap(cmap)  (Ic),dtype=np.float)
  return (1.0-alpha)*IG + (alpha)*IC

# load the image and split into [4x4] patches
I = arr(pp.imread('img/brodatz.png'),dtype=np.float)
S = imgsplit(I,[4,4])
# split patches into [16x16] blocks and extract features from each
X = [[[]]*4]*4;
fav=[]; fuv=[]; fcv=[];
for i,s in enumerate(S):
  for j,ss in enumerate(s):
    X[i][j] = imgsplit(ss,[8,8])
    fai,fui,fci,_ = extractallfeats(X[i][j])
    fav.extend(fai) # list of [64x10] - 10 features, 64 blocks in this patch
    fuv.append(fui) # list of [10,]   - mean of 10 features for this patch
    fcv.append(fci) # list of [10x10] - covariance matrix for this patch
# perform bayes classification
y = bayes_classify(fav,fuv,fcv)
# reconstruct the label image from the label vector
Y = imgunsplit(vec2block(vec2block(y.tolist(),[8,8]),[4,4])[0])
Z = rescale(Y,16,order=0,mode='reflect')
C = colorover(I,Z/16,'hsv',alpha=0.35)
# show the results
f, ax = pp.subplots(1,3,figsize=(8,3))
ax[0].imshow(I,cmap=pp.get_cmap('gray'))
ax[1].imshow(C)
ax[2].imshow(Z,cmap=pp.get_cmap('hsv'))
f.tight_layout()
pp.savefig('img/result-3')
pp.close()
misc.imsave('img/result.png',C)
