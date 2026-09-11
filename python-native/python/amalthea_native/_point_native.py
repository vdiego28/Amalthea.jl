"""Owned physical-spectrum batches using existing portable Rust kernels."""
import numpy as np
from . import _native


class _NativePoints:
    def __init__(self,model):
        grid=model.grid
        self.n,self.components=model.n,model.npol
        kerr=sum(response.kerr for response in model.responses)/(1. if grid.is_real else .75)
        kernels=[response.h for response in model.responses if response.h is not None]
        kernel=np.sum(kernels,axis=0).tolist() if kernels else None
        self.handle=_native.ModalPoints(len(grid.t),self.components,grid.is_real,
            grid.towin.tolist(),model.pre.tolist(),kerr,grid.to[1]-grid.to[0],kernel)

    def __call__(self,spectrum):
        supplied=np.asarray(spectrum)
        if supplied.ndim!=3 or supplied.shape[1:]!=(self.n,self.components):
            raise ValueError('native point spectra must have point/frequency/component axes')
        fields=np.array(supplied,dtype=complex,copy=True).transpose(0,2,1).ravel()
        result=self.handle.evaluate(fields,len(supplied))
        return np.asarray(result,dtype=complex).reshape(len(supplied),self.components,self.n).transpose(0,2,1).copy()
