from scipy import stats
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


class Gaussian_kde(stats.gaussian_kde):
    """
    """
    def __init__(self, data: pd.DataFrame, 
                 cols=None, 
                 **kwargs):
        df = data if cols is None else data.loc[:,cols] 
        self.columns = df.columns.values
        self.limits = df.describe().loc[('min','max'),:]
        super().__init__(df.to_numpy().T, **kwargs)

    def __repr__(self):
        return f'Gaussian_kde with columns {list(self.columns)}, {self.n} data points'

    def __call__(self, df:pd.DataFrame):
        """Override the __call__ to apply to a DataFrame, which must have columns with same name as
        used to generate this object.
        """
        assert np.all(np.isin( self.columns, df.columns)), \
            f'The DataFrame does not have the columns {list(self.columns)}'
        return  self.evaluate(df.loc[:,self.columns].to_numpy().T) 
    
    def pdf(self, df):
        """ For convenience"""
        return self.evaluate(df.loc[:,self.columns].T) 
    
    def make_grid(self, *, nbins=25, limits=None):
        """Create a grid of values with nbins bins in each dimension.
        if limits specified, must be a dict with keys=column names. Otherwise uses observed 
        variable limits.

        Returns dataframe
        """
        def make_bins(a,b,nbins):
            # centers for nbins bins from a to b
            d = (b-a)/nbins
            return np.linspace(a+d/2, b-d/2, nbins) 
        

        assert len(self.columns==2), 'Implemented only for 2-D'

        # extract from limits
        lims = self.limits if limits is None else limits
        self.grid = [make_bins(*lims[x],nbins) for x in self.columns]    
        xg, yg = self.grid
        gdf = pd.DataFrame( [[self.evaluate((x,y)) for x in xg ] for y in yg], 
                        index=yg, columns=xg).astype(float)
        return gdf
    
    def contourplot(self, ax=None, limits=None, nbins=25,  **kwargs):
        """Make a contour plot
        
        """
        fig, ax = plt.subplots(figsize=(6,6)) if ax is None else (ax.figure, ax)
        gdf = self.make_grid(limits=limits, nbins=nbins)
        ax.contour(gdf.columns, gdf.index, gdf, **kwargs)
        return fig
    
    @property
    def extent(self):
        """For imshow"""
        return self.limits.values.T.ravel() 
    
        
    @classmethod
    def example(cls, n=2000):
        """ The 2-D example from the reference"""
        def meas(n=2000):
            m1 = np.random.normal(size=n)
            m2 = np.random.normal(scale=1/2, size=n)
            return pd.DataFrame.from_dict(dict(x=m1-m2, y=m1+m2, ))
        return cls(meas(n), 'x y'.split())

Gaussian_kde.__doc__ = f"""\
Adapt stats.gaussian_kde to use a DataFrame as input

The example 
```
self = Gaussian_kde.example()
import matplotlib.pyplot as plt
X,Y = np.mgrid[-4:4:100j, -4:4:100j]
positions = pd.DataFrame.from_dict(dict(x=X.ravel(),y=Y.ravel()))
Z = np.reshape(self(positions).T, X.shape)
fig, ax = plt.subplots(figsize=(5,5))
ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r, extent=self.extent, )
plt.show()
```

"""

class FeatureSpace(dict):
    """ class FeatureSpace:
        Manage binning properties and evaluation in the KDE features space
    """

    default_limits =  dict( 
        diffuse=(-1, 2), 
        sqrt_d=(0.,np.sqrt(2)),
        log_epeak=np.log10([0.15, 4]),
       )
    
    def __init__(self, limits=None,  N=25):
        # set up axes of the grid and define it
        if limits is None: limits = self.default_limits
        self['names'] = names = list(limits.keys())#names
        self['limits'] = limits 
        self['bins'] = bins =dict( (var, np.linspace(*self['limits'][var],num=N+1 ).round(3)) for var in names)
        cvals = lambda b: (b[1:]+b[:-1])/2
        self['centers'] = dict((nm, cvals(v).round(3)) for nm,v in bins.items()) 
        delta = lambda name: self['limits'][name][1]-self['limits'][name][0]
        self['size'] = dict((name, delta(name)) for name in names)
        self['N']=N
        self['volume']=  np.prod( list(self['size'].values())) 

        self.__dict__.update(self)

    def evaluate_kdes(self, kdes):
        """ Use a set of KDE functions to populate a grid
        """
        self.grid = pd.DataFrame(dict( (name, mg.flatten()) for name,mg in
                       zip(self.names, np.meshgrid(*self.centers.values()))))

        # Evaluate KDE on the grid, measure integrals, renormalize"
        self.class_names = kdes.keys()
        for comp in  self.class_names:
            grid_kde =  self.grid[comp+'_kde'] = kdes[comp](self.grid)

        # calculate normalization, save factors
        norms = grid_kde.sum() * self.volume/self.N**len(self.names)
        grid_kde /= norms
        assert np.any(~pd.isna(grid_kde)), 'Found a Nan on the grid!'


    def __repr__(self):
        return f'class {self.__class__.__name__}:\n'+'\n'.join( [f'{key}: {value}' for key, value in self.items()])

    def generate_kde(self, df, bw_method=None):
        """ Return a KDE using the DataFrame df
        """
        return Gaussian_kde(df, self.names, bw_method=bw_method)

    def train(self, df_dict,  bw_method=None):
        # make dicts with DF for class data sets and unID, then KDE for each
        # from collections import OrderedDict
        # grps = data.groupby('subset')
        # df_dict = dict( (name, grps.get_group(name)) for name in class_names )
        
        kdes = dict( (name, self.generate_kde(df, bw_method=bw_method))
                        for name, df in df_dict.items() )  
        self.evaluate_kdes(kdes)#, kdes.keys())
        
    # ------------These assume evaluate_kdes has been called to add columns -------------
    
    def projection(self, varname):
        """ Return DataFrame with index varname's value and columns for class components,
        basically an integral over the other variables
        """        
        t = self.grid.copy().groupby(varname).sum()*(self.volume/self.size[varname])/self.N**2
        return t.iloc[:, -3:] #3=components? len(self.names):]
    
    def projection_info(self, unID):
        """Return a dict in variable names of projection dataframes
        Each has a column "x" for the bin center, "unID" for the unID histogram, 
        and columns for the projections of each component

        """
        df_dict={}
        for var in self.names: 
            d = {}
            d['x'] = self.centers[var]
            d['unID'] =  np.histogram(unID[var], self.bins[var])[0] 
            df = self.projection(var)
            for y in  df.columns:
                d[y[:-4]] = df[y].values # strip off _kde
            df_dict[var] = pd.DataFrame(d)
        return df_dict
        
    def projection_dict(self, var_name, cls_name):
        """For sns.lineplot"""
        td = self.projection(var_name)
        return dict(x=td.index, y=td[cls_name+'_kde'])
        
    def  normalize(self, var, norm:dict):
        """ return df with normalized counts
        """
        td = self.projection(var)
        norm = pd.Series(norm)
        coldata = td.loc[:, [n+'_kde' for n in norm.index]].to_numpy() * norm.values *self.size[var]/self.N
        df = pd.DataFrame(coldata ,  index=td.index, columns=norm.index)
        df['sum'] = df.sum(axis=1)
        return df
    
    # ---------------------- Following make plots  --------------------

    def projection_check(self, df_dict, palette):
        """Density plots for each of the features comparing the training data with projections of 
        the KDE fits.

        """
        labels = dict(sqrt_d = r'$\sqrt{d}$',
                log_epeak = r'$\log_{10}(E_p)$',
                diffuse = '$D$')
        fig = plt.figure(figsize=(12,7), layout='constrained')
        fig.set_facecolor('k' if self.dark_mode else 'w') # why here?
        axx = fig.subplots(ncols=len(self.limits) ,nrows=3,  sharex='col',
                            gridspec_kw=dict(hspace=0.1, wspace=0.1)
        )    
        for (i, (cls_name, df)), color in zip(enumerate(df_dict.items()), palette):
            
            for j, var_name in enumerate( self.names ):
                ax = axx[i,j]
                for spine in 'top right left'.split():
                    ax.spines[spine].set_visible(False)
                ax.spines['bottom'].set_color('k')
                sns.histplot(ax=ax,  x=df[var_name], bins=self.bins[var_name], 
                    element='step', stat='density', color='0.4' if self.dark_mode else '0.6', 
                    edgecolor='w' if self.dark_mode else '0.8')
        
                sns.lineplot( ax=ax, **self.projection_dict(var_name, cls_name), 
                            color=color, ls='-', marker='', lw=2)
                ax.set( xlim=self.limits[var_name], ylabel='', yticks=[], 
                    facecolor='k' if self.dark_mode else 'w',
                    xlabel= labels[var_name])
                                
            axx[i,0].set_ylabel(cls_name+f'\n{len(df)}', rotation='horizontal', ha='center')

        return fig
        
    def component_comparison(self, unID, norm, palette):
        """Histograms of unID data for the KDE feature variables, compared with an estimate
        of the class contents. Each component was normalized to the total shown in the legend.
        """
        fig, axx =plt.subplots(ncols=3, figsize=(12,4), sharey=True, 
                            gridspec_kw=dict(wspace=0.1))
        
        norm_sum = np.sum(list(norm.values()))
        for var,ax in zip(self.names, axx):
            
            df = self.normalize( var, norm)   
            x= df.index
            for y,color in zip(df.columns, palette+['white']):
                sns.lineplot(df,ax=ax, x=x, y=y,  color=color, 
                            label=f'{norm[y] if y!="sum" else norm_sum} {y}', 
                            lw=2 if y=='sum' else 1,     legend=None)
            sns.histplot(unID, bins=self.bins[var], ax=ax, x=var, element='step', color='0.2', edgecolor='w', 
                        label=f'{len(unID)} unID') 
            ax.set(ylabel='counts / bin')
                        
        ax.legend(fontsize=12, bbox_to_anchor=(0.9,1.15), loc='upper left')
        ax.set(yticks=np.arange(0,151,50))
        return fig
    
    def data_model(self, norms, unID, fig=None, palette=None):
        
        var_labels = dict(sqrt_d = r'$\sqrt{d}$',
              log_epeak = r'$\log_{10}(E_p)$',
              diffuse = '$D$')

        if fig is None:
            fig= plt.figure(figsize=(12,5), layout="constrained")
        axd = fig.subplot_mosaic([self.names], sharey=True)
        pi = self.projection_info(unID)
        xtick_dict=dict(sqrt_d=np.arange(0,1.5,0.5), 
                        log_epeak=np.arange(-1, 0.6, 0.5), 
                        diffuse=np.arange(-1,2.1,1))

        for var_name in self.names:
            ax = axd[var_name]    
            df =pi[var_name]
            x = df.x
            ax.errorbar(x=x, y=df.unID, xerr=self.size[var_name]/2/self.N, 
                        yerr=np.sqrt(df.unID), fmt='.', label='unID')
            ax.set(xlabel=var_labels[var_name], xticks=xtick_dict[var_name],
                ylim=(0,None))
            var_norm = self.size[var_name]/self.N
            total = np.zeros(self.N)    
            
            for cls_name, color in zip(self.class_names, palette):
                y = norms[cls_name]*var_norm* df[cls_name]
                total+=y
                ax.plot(x, y,   color=color, label=cls_name)
                
            ax.plot(x,total, color='w' if self.dark_mode else 'k', label='sum', lw=4)
            if var_name=='sqrt_d':
                ax.legend(loc='upper right', fontsize='small',frameon=False,
                        ncols=2,  bbox_to_anchor=(1, 1.4))
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

        axd['diffuse'].set( ylabel='Counts / bin', yticks=np.arange(0,201,50));
        return fig
    
    def get_deltas(self, norms, unID):
        """Return deltaa
            norms - a dict of fit
            unID  = 
        """
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
    
    def plot_residuals(self, fit_df, unID):
    
        df_deltas = pd.DataFrame(self.get_deltas( fit_df.T.fit, unID))
        fig= plt.figure(figsize=(12,4), layout="constrained")
        axd = fig.subplot_mosaic([self.names], sharey=True)
        grange = dict(diffuse=(-0.25,1.25), sqrt_d=(0.4,1.414), log_epeak=(-0.8,0))
        for var_name in self.names:
            ax = axd[var_name]
            ax.plot(self.centers[var_name], -df_deltas[var_name], '.');
            ax.set(xlabel=var_name)
            ax.axhline(0, color='0.2', ls='--')
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.axvspan(*grange[var_name], color='lightyellow', alpha=0.2)
        axd['diffuse'].set(yticks=np.arange(0,101,50));
        return fig
    
    @classmethod
    def runit(cls, data, dark_mode, palette, limits=None):
        """Perform the KDE analysis
        
        """
        from pylib.ipynb_docgen import show, show_fig
        from pylib.gevatar_fits import Fitter
        
        self = fs= cls(N=25,
                limits=limits, #dict( 
                #     diffuse=(-1, 2),
                #     sqrt_d=(0.,np.sqrt(2)),
                #     log_epeak=np.log10([0.15, 10]),
                # )
        )


        self.palette = palette
        self.dark_mode = dark_mode
        show(f"""### Evaluate KDE's on a 3-D grid<br>
        * N = {self.N} for components & unID<br>
        * Uses the scipy  [gaussian_kde](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html) "bw_method"=
                {(bw_method:=0.25)}""")

        grps = data.groupby('subset')
        df_dict = dict( (name, grps.get_group(name)) for name in 'blazar msp psr'.split())
        self.train( df_dict, bw_method)

        show(f"""### Projections""")
        #     show("""### Residuals from fit""")
        show_fig(self.projection_check,  df_dict,  palette, save_to='figures/projections.png') 

        # fitting(self, data[data.subset=='unID'])
        show(f"""### 3-D optimization""")
        unID = data[data.subset=='unID']
        fit_dict = Fitter.main(fs, unID)
        show('__Fit__')
        show(fit_dict['fit_df'].T)
        show( rf"""sum: {(s:=(fit_dict['opt'].x).sum()):.1f} $\pm$ {np.sqrt(fit_dict['cov'].sum()):.1f}""")
        show('Correlation matrix:'); show(fit_dict["corr"].round(2))
        show(fr"""$\rightarrow$ Implied number of gevatars:
            {len(unID)}-{s:.1f}={len(unID)-s:.1f} $\pm$ {np.sqrt(fit_dict['cov'].sum()):.1f} """)
        
        fit_df = fit_dict['fit_df']
        show("""### Fit result plot""")
        def fit_results():
            """Fit results
            """

            fig = fs.data_model(fit_df.T.fit,  unID, None, fs.palette);
            fit_dict['fitter'].plot_fit_range( fig, fs.palette, fs.dark_mode )
            return fig
        show_fig(fit_results, save_to='figures/fit_result.png')
            
        show("""### Residuals from fit""")
        show_fig( fs.plot_residuals, fit_df, unID , save_to='figures/residuals.png')

        return fs





# def fitting(fs, unID):
#     from pylib.gevatar_fits import Fitter
#     from pylib.ipynb_docgen import show
#     # unID = data[data.subset=='unID']

#     show(f"""### 3-D optimization""")
#     fit_dict = Fitter.main(fs, unID)
#     show('__Fit__')
#     show(fit_dict['fit_df'].T)
#     show( rf"""sum: {(s:=(fit_dict['opt'].x).sum()):.1f} $\pm$ {np.sqrt(fit_dict['cov'].sum()):.1f}""")
#     show('Correlation matrix:'); show(fit_dict["corr"].round(2))
#     show(fr"""$\rightarrow$ Implied number of gevatars:
#         {len(unID)}-{s:.1f}={len(unID)-s:.1f} $\pm$ {np.sqrt(fit_dict['cov'].sum()):.1f} """)
    
#     show("""### Fit result plot""")
#     fit_df = fit_dict['fit_df']
#     fig = fs.data_model(fit_df.T.fit,  unID, None, fs.palette);
#     fit_dict['fitter'].plot_fit_range( fig, fs.palette, fs.dark_mode )
#     show(fig)
        
#     show("""### Residuals from fit""")
#     show( fs.plot_residuals(fit_df, unID ))
