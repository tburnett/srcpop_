
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import optimize # type: ignore

var_labels = dict(sqrt_d = '$\sqrt{d}$',
              log_epeak = '$\log_{10}(E_p)$',
              diffuse = '$D$')



def get_deltas(self, norms, unID):
    pi = self.projection_info(unID)
    deltas = dict()
    for var_name in self.names: 
    
        df = pi[var_name]
        x = df.x
        unID = df.unID
    
        var_norm = self.size[var_name]/self.N
        total = np.zeros(self.N)    
        
        for cls_name in self.class_names:
            y = norms[cls_name]*var_norm* df[cls_name]
            total+=y
        deltas[var_name] = total-unID    
    return deltas

def show_diffs(self, norms, unID, fig=None, palette=None):
    """
    """
    if fig is None:
        fig, ax = plt.subplots(figsize=(8,6))
    else:
        ax = fig.subplot_mosaic('..AAA.')['A']
    deltas = get_deltas(self, norms, unID)
    for var_name, m, color in zip(self.names, 'oDv', palette):
        ax.plot(-deltas[var_name], m, ls='--' , ms=8, 
                label=var_labels[var_name], color=color)
    ax.legend(loc='upper left', bbox_to_anchor=(1., 1),fontsize='small')
    ax.axhline(0, color='0.5')
    ax.set(xlabel='bin number', ylabel='unID data - model');
    return fig

def slice_loglike(self, x, norms, cls_name, *, var_name, idx,):  
    """Evaluate log likelihood

    norms: dict with input norms
    x : dependent variable, number of counts for cls_name
    var_name 
    """
    n = norms.copy()
    n[cls_name] = x
    pi = self.projection_info(unID)
    df = pi[var_name]
    N = df.unID[idx].sum() 
    
    var_norm = self.size[var_name]/self.N
    
    model = np.zeros(self.N)        
    for cls_name in self.class_names:
        y = n[cls_name]*var_norm* df[cls_name]
        model+=y
    mu = model[idx].sum()#.round(3) 
    return N * np.log(mu) - mu

def fit_plots(self, norms, unID, palette):
    fig = plt.figure(layout='constrained', figsize=(12, 8))
    subfig1, subfig2= fig.subfigures(nrows=2,  hspace=0.1, height_ratios=(3,2))
    data_model(self,  norms, unID, subfig1, palette)
    show_diffs(self,  norms, unID, subfig2, palette)
    fig.text(0.2, 0.4, 'Model contents\n'+'\n'.join(str(pd.Series(norms)).split('\n')[:-1]),
             ha='right',va='top')
    return fig;


class Fitter:
    
    def __init__(self, fs, unID,  fit_slice): 
        """fs: FeatureSpace object
        """
        self.fs = fs
        self.unID = unID
        self.fit_slice=fit_slice
        self.proj_info = fs.projection_info(unID)

    def fit_function(self,norms):
        """Return a dict with the three slice likelihood functions
        """
        
        def slice_loglike( x, cls_name, *, var_name, idx,):  
            """Evaluate log likelihood
        
            norms: dict with input norms
            x : dependent variable, number of counts for cls_name
            var_name 
            """
            n = norms.copy()
            n[cls_name] = x
            fs = self.fs
            df = self.proj_info[var_name]

            N = df.unID[idx].sum() 
            
            var_norm = fs.size[var_name]/fs.N
            
            model = np.zeros(fs.N)        
            for cls_name in fs.class_names:
                y = n[cls_name]*var_norm* df[cls_name]
                model+=y
            mu = model[idx].sum()#.round(3) 
            return N * np.log(mu) - mu
 
        dct =  dict(
            blazar= lambda x: slice_loglike(x, 'blazar', **self.fit_slice['blazar']),
            psr   = lambda x: slice_loglike(x, 'psr',    **self.fit_slice['psr']),
            msp  =  lambda x: slice_loglike(x, 'msp',    **self.fit_slice['msp']),
            )
        return dct
        # f = fitter.fit_function(xnorm)
        # return  -np.sum([f[n]( xnorm[n] ) for n in self.classes])  
        
       
    def fit_3d(self, norms):
        """Return 3-d optimization object
        """

        fitter = self
        class F3:
            def __init__(self, norms):
                """
                * fitter: the Fitter object
                implements fit_function
                * norms: a dict-like object with class names and initial values for fit
                """
                self.classes = norms.keys()
                self.norms = dict(norms) # convert Series if necessary
                
            def __call__(self, x):
                """x : 3-d array, order as in norms.keys()
                return negative of log likelihood
                """
                # make a new dict
                xnorm = dict( (n,v) for n,v in zip(self.classes, x ))
                # get dict of functions set for x
                f = fitter.fit_function(xnorm)
                return  -np.sum([f[n]( xnorm[n] ) for n in self.classes])  
                    
            def maximize(self,):
                """Maximize the log likelihood
                set `opt` in outer class with miminize result
                return DF with fit values, diagonal uncertainties
                """
                x0 = list(self.norms.values())
                fitter.opt = opt =   optimize.minimize(self, x0)
                fitval = pd.Series(dict( (k,v) for k,v in zip(self.classes, opt.x.round(1))),name='fit')
                cov = opt.hess_inv
                fitunc = pd.Series(dict( (k,v) for k,v in
                    zip(self.classes,np.sqrt( cov.diagonal()).round(1))),name='unc')
                return pd.DataFrame([fitval, fitunc])
            
        return F3( norms)
    
    def plot_fit_range(self, fig, palette, dark_mode=True):#, fit_slice):
        """Add shaded areas to plots showing fits.
        """
        fit_slice = self.fit_slice
        fs = self.fs
        
        fs_df = pd.DataFrame(fit_slice).T
        fs_df.index.name='cls'
        fs_df['cls']=fs_df.index
        fs_df = fs_df.set_index('var_name')
        # show(fs_df)
        for ax in fig.axes:
            var_name = ax.get_label()
            fsi = fs_df.loc[var_name]
            t =  fs.bins[var_name][fsi.idx]
            cls = fsi.cls
            cidx = list(fs.class_names).index(cls)
            ax.axvspan(t[0], t[-1], alpha=0.3 if dark_mode else 0.1, color=palette[cidx])

    @classmethod
    def main(cls, fs, unID, norms=dict(psr=520, msp=150, blazar=823)):
        fit_slice = dict(
            blazar = dict(var_name ='sqrt_d',  idx=slice(0,7,1),),
            psr   = dict(var_name ='diffuse',  idx=slice(19,-1,1)),
            msp = dict(var_name ='log_epeak',  idx=slice(11,15,1)),
            ) 
        fs_df = pd.DataFrame(fit_slice).T; fs_df.index.name='cls'
        myf   = cls(fs, unID, fit_slice)
        f3    = myf.fit_3d( norms = norms  )
        fit_df = f3.maximize()

        opt = myf.opt
        if not opt.success:
            print(f'Warning: fit failure {opt.message}', file=sys.stderr)
        cov = opt.hess_inv
        sigs = np.sqrt(cov.diagonal())
        corr = cov / np.outer(sigs,sigs)

        return dict(fit_df=fit_df,  opt=opt,
                    cov=cov,
                    sigs=sigs, corr=corr,
                    fitter=myf)
    
