import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pylib.kde import FeatureSpace, Gaussian_kde
from pylib.tools import (set_theme, epeak_kw, update_legend, fpeak_kw)
from pylib.ipynb_docgen import (show, show_fig, show_date)

def add_classification_prob(df, filename='files/dr4_2_class_classification.csv'):
    """
    Extract the pulsar classification probability from the supervised classification and add
    """
    q = pd.read_csv(filename, index_col=0)
    df['p_pulsar'] = q.p_pulsar
    
def apply_diffuse(df, nc=2, dataset='dr4'):
    df3 = pd.read_csv(f'files/{dataset}_{nc}_class_classification.csv', index_col=0)
    df['diffuse'] = df3.diffuse

def apply_kde(df, features):

    for name, sdf in df.groupby('subset'):
        try:
            gde = Gaussian_kde(sdf,  features)
        except Exception as msg:
            print(msg, file=sys.stderr)
        u = gde(df)
        df[name+'_kde'] = u
    return df

def kde_setup(kde_vars = 'sqrt_d log_epeak diffuse'.split(), 
            cut = '0.12<Ep<10 & variability<30' ,
            title='Unsupervised KDE analysis'  ):
    

    show(f"""<font size="+3"> {title}</font>""")
    show_date()
    show(f"""# Data setup
    * Load source data from `{(filename:='files/dr4_2_class_3_features.csv')}'""")
    df = pd.read_csv( filename,index_col=0)
    df['sqrt_d'] = np.sqrt(df.d.clip(0,2))
    df['log_epeak'] = np.log10(df.Ep)
    df['log_fpeak'] = np.log10(df.Fp)

    def make_group(df):

        def groupit(s):
            if s.association in 'psr msp unID'.split(): return s.association
            if s.association in  'bll fsrq'.split(): return 'blazar'
            if s.association in 'bcu unk'.split(): return 'bcu_unk'
            return np.nan

        df['subset'] = df.apply(groupit, axis=1)
    
    def make_aclass(df):
        """ Return the super classification group name for each source
        """
        gtbl = dict (
            pulsar = 'msp psr'.split(),
            blazar = 'fsrq bll'.split(),
            bcu_unk = 'bcu unk'.split(),
            egal  = 'agn gal sey nlsy1 sbg ssrq css rdg'.split(),
            Gal ='bin glc hmb lmb pwn sfr snr spp'.split(), # leave off gc, nov
            unID   = [''],
            )
        inv = dict()
        for gname, cls_list in gtbl.items():
            for cls in cls_list:
                inv[cls]=gname
        df['association_class']=  df.association.apply(lambda s: inv.get(s,s))
    
    make_group(df) 
    make_aclass(df)
    
    show(f'### Data selection cut: "{cut}"')
    dfc = df.query(cut).copy()
    all = pd.Series(df.groupby('association_class').size(), name='total')
    sel = pd.Series(dfc.groupby('association_class').size(), name='selected')
    pct = pd.Series((100*sel/all).round(0).astype(int), name='%')
    classes = 'blazar pulsar Gal egal bcu_unk unID'.split()
    t =pd.DataFrame([all,sel,pct])[classes]; 

    show(t)  

    apply_diffuse(dfc)
    
    add_classification_prob(df)
    
    show(f"""# Create KDE functions instead of ML training

    * Features: {', '.join(kde_vars)} 

    """)
    kde_probs = apply_kde( dfc, kde_vars)

    df.to_csv((filename:='files/kde_data.csv'))
    show(f'saved KDE setup to `{filename}`')
    return dfc 

class StudyML:
    """Class to hold the ML functions and data
    """

    def __init__(self, 
                 title = 'Study ML results with galacticity ',
                 ml_filename='files/dr4_2_class_3_features.csv', 
                 kde_vars='sqrt_d log_epeak diffuse'.split(), 
                 cut = '0.12<Ep<10 & variability<30' ):
        
        self.df = df = pd.read_csv(ml_filename, index_col=0)
        self.kde_vars = kde_vars
        self.cut = cut
        show(f"""<font size="+3"> {title}</font>""")
        show_date()
        show(f"""# Data setup
        * Load ML analysis results from `{ml_filename}`""")

        df['sqrt_d'] = np.sqrt(df.d.clip(0,2))
        df['log_epeak'] = np.log10(df.Ep)
        df['log_fpeak'] = np.log10(df.Fp)
        # df['diffuse'] = df.diffuse.clip(0,2)

        def make_group(df):

            def groupit(s):
                if s.association in 'psr msp unID'.split(): return s.association
                if s.association in 'bll fsrq'.split(): return 'blazar'
                if s.association in 'bcu unk'.split(): return 'bcu_unk'
                return np.nan

            df['subset'] = df.apply(groupit, axis=1)
        make_group(df) 
        show(f"""* Create `subset` with simple grouping of associations """)
        show(pd.Series(self.df.groupby('subset').size(), name=''))
        

        def make_aclass(df):
            """ Return the super classification group name for each source
            """
            gtbl = dict (
                pulsar = 'msp psr'.split(),
                blazar = 'fsrq bll'.split(),
                bcu_unk = 'bcu unk'.split(),
                egal  = 'agn gal sey nlsy1 sbg ssrq css rdg'.split(),
                Gal ='bin glc hmb lmb pwn sfr snr spp'.split(), # leave off gc, nov
                unID   = [''],
                )
            inv = dict()
            for gname, cls_list in gtbl.items():
                for cls in cls_list:
                    inv[cls]=gname
            df['association_class']=  df.association.apply(lambda s: inv.get(s,s))
    
        make_aclass(df)
        show(f"""* Create `association_class` with super classification """)
        show(pd.Series(self.df.groupby('association_class').size(), name=''))
            
        def apply_kde(df, features):

            for name, sdf in df.groupby('subset'):
                try:
                    gde = Gaussian_kde(sdf,  features)
                except Exception as msg:
                    print(msg, file=sys.stderr)
                u = gde(df)
                df[name+'_kde'] = u
            return df
        
        self.kde_probs = apply_kde(df, kde_vars)
        self.kde_probs.to_csv('files/kde_data.csv')
        show(f'* saved KDE setup to `files/kde_data.csv`')

def axis_kw(axis, d):
    return dict( (axis+k, v) for k,v in d.items() )

def d_kw(axis='x'):
    return axis_kw(axis, 
                   dict(label='$d$', scale='log', ticks=[0.2, 1, 2], ticklabels='0.2 1 2'.split(), lim=(0.2,2.05))
    )

def ep_kw(axis='x'):
    d = dict(label='$E_p$ (GeV)', scale='log', ticks=[0.1,1,10], ticklabels='0.1 1 10'.split(),
             lim=(0.1,10))
    return dict( (axis+k, v) for k,v in d.items() )

def G_kw(axis='x'):
    return axis_kw(axis,
                dict(label='$G$', lim=(-1,2), ticks=np.arange(-1,2.1,1))
                )

def multi_d(data, ):
    r"""
    Matrix of scatter plots of $d_{unc}$ vs $d$ for each population, by row, with left column for $Ep<1 $ GeV.
    The dashed red lines denote $ 1\sigma$ boundaries for the range 0 (no curvature) to 4/3 (the limit for curvataure radiation).
    """

    def d_dist(df, name, cut, ax=None, code=''):

        fig, ax = plt.subplots(figsize=(8,5)) if ax is None else (ax.figure, ax)
        sns.scatterplot(df.query(cut),ax=ax, x='d', y='d_unc',s=20,
                    #    hue=np.log10(df.significance), palette='coolwarm', hue_norm=(0.7,2.01),
                    hue='diffuse', palette='coolwarm',
                        legend=False)
        ax.plot([0,2], [0,2], '--r'); 
        ax.plot([4/3,2], [0,2/3], '--r')
        ax.set(ylim=(0,1),xlim=(0,2), xticks=np.arange(0,2.01,0.5), )
        if code=='B': ax.set(yticklabels=[], ylabel='')

    def xlabel(ax,text):
        ax.text(0.8,0.5, text, ha='center', va='center', fontsize='large')
    def ylabel(ax, text):
        ax.text(0.5,0.5, text, ha='center', va='center', fontsize='large', rotation='vertical')
        
    fig, axd =plt.subplot_mosaic(
       [ ['.', '.', 't',  't',],
         ['s', '.', 'a',  'b',],
         ['s', 'P', 'PA', 'PB'],
         ['s', 'B', 'BA', 'BB'],
         ['s', 'U', 'UA', 'UB'], ], 
        width_ratios = [0.1, 0.25, 3, 3],
        height_ratios = [0.1, 0.2, 3,3,3],
        figsize=(15,15), sharex=True, #sharey=True,
        layout='constrained',
        )
    g = data.groupby('association_class')
    
    for k, ax in axd.items():
        if len(k)==2:
            t,c = k
            name = dict(B='blazar', P='pulsar', U='unID')[t]
            df = g.get_group(name)
            cut = dict(A='Ep<1', B='Ep>1')[c]
            d_dist(df, name, cut, ax=ax, code=c)
        else:
            ax.set_axis_off() 
            match k:
                case 'a': xlabel(ax, 'Ep<1')
                case 'b': xlabel(ax, 'Ep>1')
                case 'B': ylabel(ax, 'Blazar')
                case 'P': ylabel(ax, 'Pulsar')
                case 'U': ylabel(ax, 'unID')
                case '_': pass
    return fig

def multi_spectra(data):
    """
    Left column: spectral shape parameter scatter plot. Center column: scatter plot
    of the relative spectral shape uncertainties. This and the left plot have palette
    representing the log significance. Right column: distribution of the log significance.
    """
    def ylabel(ax, text):
        ax.text(0.5,0.5, text, ha='center', va='center', fontsize='large', rotation='vertical')
    def xlabel(ax,text):
        ax.text(0.5,0.5, text, ha='center', va='center', fontsize='large')
    def unc_scatter(df, ax):
        sns.scatterplot(df,ax=ax, x=(df.Ep_unc/df.Ep).clip(0,1), s=30,
                        y=(df.d_unc/df.d).clip(0,1), 
                       hue=np.log10(df.significance), palette='coolwarm', legend=False)
        ax.set( xlabel='Ep_unc/Ep', ylabel='d_unc/d' , ylim=(0,1.05), xlim=(0,1.05))

    def sighist(df, ax, lim=(0.7,2.01)):
        hkw= dict(element='step', fill=False, bins=np.arange(*lim,0.1),stat='density' )
        sns.kdeplot(df, ax=ax,x=np.log10(df.significance).clip(*lim), )#**hkw);
        ax.set(xlabel = 'log10(significance)', xlim=lim, ylabel='', yticks=[])
    
    huefunc, leg_title = lambda df:np.log10(df.significance),'log(sig)'
    huefunc, leg_title = lambda df: df.diffuse, 'G'
    

    def d_vs_Ep(df, ax=None,  ):
        fig, ax = plt.subplots(figsize=(8,6)) if ax is None else (ax.figure, ax)
        sns.scatterplot(df, ax=ax, x=df.Ep, y=df.d.clip(0.05,2), s=50, edgecolor='none', 
            hue=huefunc(df),   palette='coolwarm',)
        ax.set( **d_kw('y'), **ep_kw('x'),)
        ax.legend(fontsize=12, loc='center right', title=leg_title)
        ax.tick_params(which="both", bottom=True, left=True)
        return fig

    fig, axd =plt.subplot_mosaic(
       [ 
        ['.', 'a',  'b',  'c'],
        ['P', 'Pa', 'Pb', 'Pc'],
        ['B', 'Ba', 'Bb', 'Bc'],
        ['U', 'Ua', 'Ub', 'Uc'], ], 
        width_ratios = [0.5, 4, 3, 1.5],
        height_ratios = [ 0.5, 3,3,3],
        figsize=(15,15), #sharex='col', #sharey=True,
        # gridspec_kw=dict(wspace=0.4),
        layout='constrained',
        )
    g = data.groupby('association_class')
    names = dict(B='blazar', P='pulsar', U='unID')
    
    for k, ax in axd.items():
        if len(k)==2:
            t,c = k
            df = g.get_group(names[t])
            match c:    
                case 'a': d_vs_Ep(df, ax)
                case 'b': unc_scatter(df, ax)
                case 'c': sighist(df, ax)  
            if t != 'U': # suppress x labels except bottom plots
                ax.set(xlabel='', xticklabels=[])                  
        else:
            ax.set_axis_off() 
            match k:
                case 'U' | 'P'| 'B':
                    ylabel(ax,names[k]+f' ({len(g.get_group(names[k]))})')
                case 'a':  xlabel(ax, 'Spectral Shape')
                case 'b':  xlabel(ax, 'Resolutions')
                case 'c':  xlabel(ax, 'Significance')

    return fig

class GlogLike:
    r"""## GlogLike: unbinned likelihood

    $x$:   independent variable: list of coefficients to apply to KDE<br>
    model: N X 3 scaled kde values.<br>
    $\mu$: predicted probability. <br>
    $ \log(L)=\sum[ \log(\mu) - \mu] $
    """
    
    def __init__(self, unid):
        # last 3 columns are KDE values
        self.kdata = unid.iloc[:,-3:].copy()
        
    def __call__(self, x):
        """ return -log like """
        mu = (self.kdata * x).sum(axis=1).clip(1e-4,None)
        return -np.sum(np.log(mu)-mu)
        
    def maximize(self, x0 = [0.74, 0.41, 1.63]):
        from scipy import optimize
        opt = optimize.minimize(self, x0, method=None,
                               bounds= [(0,None),(0,None),(0,None)])
        return opt

class GalAnalysis:
    """Manage Galacticity fits
    """

    G05 = np.arange(-0.875, 2.01, 0.05) # bin centers for 2 per bin
    hist_binsize = 0.1
    G10 = np.arange(-0.85, 2, 0.1)
    Gbins =np.arange(-0.9,2.01, 0.1) 
    
    def __init__(self,  unid,  train_data):

        self.kde_dict = self.train(train_data)

        # an equivalent one with the functions
        self.fun_dict = dict((name, kde.evaluate) for name,kde in self.kde_dict.items())
        self.setup_unid(unid)

    def train(self, train_data, bw_method=0.3):
        from pylib.kde import Gaussian_kde
        class GalKDE(Gaussian_kde):
            def evaluate(self,x):
                return super().evaluate(x)+super().evaluate(4-x)
        # Setup KDE for blazar, msp, psr
        return dict( (name, GalKDE(df, cols=['diffuse'], bw_method=bw_method),)
                for name, df in train_data.groupby('subset') if name!='unID')

    
    def add_gev_fun(self, mean, rms):
        """Add the gevatar function and apply to all data
        """
        from scipy.stats import norm
        gev_fun = self.fun_dict['gev'] = norm(mean, rms).pdf
        

    def setup_unid(self, unid):
        """## Apply to unID
        Add G-kde value columns to the unid subset"""
        self.unid = unid.copy() 
        unid_g = self.unid.diffuse # G values for unID
        for name, gkde in self.kde_dict.items():
            self.unid[name] = gkde.evaluate(unid_g)  
        # bins =np.arange(-0.9,2.01, 0.1) #binning for unid
        self.unid_hist = np.histogram(self.unid.diffuse, self.Gbins)[0]
        
    def __repr__(self):
        return str(self.kde_dict)
        
    def make_grid(self, Amp=np.ones(4),  ):
        return pd.DataFrame.from_dict(dict((name, fn(self.G05)* A ) 
                    for (name, fn), A in zip(self.fun_dict.items(), Amp)))

    def plot_kde(self, palette):
        fig, ax = plt.subplots(figsize=(6,4))
        x = np.arange(-0.9, 2.01, 0.05)
        for (name, gkde), color in zip(self.kde_dict.items(), palette):
            ax.plot(x, gkde.evaluate(x), label=name, color=color)
        ax.legend(fontsize=12);
        ax.set(**G_kw(), ylabel='Density');
        return fig

    def add_gev_fun(self, mean=0.58, rms=0.48):
        """Add the gevatar function
        """
        self.gvals=[mean, rms]
        from scipy.stats import norm
        self.fun_dict['gev'] =f = norm(mean, rms).pdf
        # apply to unid
        self.unid['gev'] = f(self.unid.diffuse.values)

    def plot_funcs(self):
        """Plot the fit functions
        """
        fig, ax = plt.subplots(figsize=(6,4))
        x = np.arange(-0.9, 2.01, 0.05)
        for (name, f), color in zip(self.fun_dict.items(), palette+['blue']):
            ax.plot(x, f(x), label=name, color=color)
        ax.legend(fontsize=12)
        return fig

    def initial_optimization(self, x0=[0.74, 1.0, 1.63]):
        """ Perform simple optimization
        """
        gll = GlogLike(self.unid) #.query('diffuse<0 | diffuse>1.3')
        return gll.maximize( x0)
        
    def make_gfit(self, *, x0= [90, 415, 369, 374], gvals=None, fit_to=None):
        """

        """
        from scipy import optimize

        def chisq( x):
            # this evaluate the function on a 0.05 grid, then averages to 0.10
            tdf = self.make_grid( Amp=x/10) 
            t = tdf.sum(axis=1).to_numpy().reshape(-1,2).sum(axis=1)/2
            return np.sum((t-target_hist)**2/t)
            
        npar = len(x0)
        
        if gvals is not None: # change Gaussian parameters
            self.add_gev_fun(*gvals)
            
        data = self.unid.diffuse if fit_to is None else fit_to.diffuse
        target_hist = np.histogram(data, self.Gbins)[0]
        assert len(data)>50, f'target histogram too small: {len(data)}<50'
        opt = optimize.minimize(chisq, x0,
                        bounds= optimize.Bounds([0.1]*npar, [1e3]*npar )
                            )
        return opt

    def plot_fit(self, x, *, to_hist=None, label=''):
        if to_hist is None: to_hist=self.unid
        tdf = self.make_grid(Amp=np.array(x)/10)
        G05 = np.arange(-0.875, 2.01, 0.05)
        fig, ax = plt.subplots(figsize=(8,5))
        for label, c in tdf.items():
            ax.plot(G05, c, '-', label=label+f' ({np.sum(c)/2:.0f})');
        ax.plot(G05, tdf.sum(axis=1), '-', label='Sum', color='w', lw=2);
        ax.hist(to_hist.diffuse, bins=np.arange(-0.9,2.01, 0.1), histtype='step',
            label='unid' if to_hist is None else label  )
        ax.legend(fontsize=12, loc='upper right')
        ax.set(**G_kw(), ylabel='counts / 0.1 bin')
        return fig

class GalFunctions(dict):
    """Implement galacticity functions
    """  
    # 0.05 and 0.1 grids from -0.9 to 2
    hist_binsize = 0.1
    G05 = np.arange(-0.875, 2.01, 0.05)
    G10 = np.arange(-0.85, 2, 0.1)
    Gbins =np.arange(-0.9,2.01, 0.1) 
    # palette for iso, msp psr gevatar
    palette = '0.5 cyan cornflowerblue magenta'.split()
    # default gevatar skewnorm parameters 
    gevpars=dict(loc=0.754, scale=0.341, a=0.03)
    
    def __init__(self, data, *,
            bw_method = 0.3, 
            gpars=None,
            ):
        """
        data-- for pulsar training
        """
        from pylib.kde import Gaussian_kde
        from pylib.diffuse import IsotropicDiffuse 
        
        class GalKDE(Gaussian_kde):
            # even out at G=2
            def evaluate(self,x):
                return super().evaluate(x)+super().evaluate(4-x)
        
        # first, the blazar expectation               
        self['isotropic'] = IsotropicDiffuse()

        # Setup KDE for blazar, msp, psr
        self.update(  dict( (name, GalKDE(df, cols=['diffuse'], bw_method=bw_method).evaluate,)
                for name, df in data.groupby('subset') if name in 'psr msp'.split() ))

        def gevfun( gevpars):
            #define gevatar function as skewed norm
            from scipy.stats import skewnorm
            return skewnorm(**gevpars).pdf
        self['gevatar'] = gevfun(gpars if gpars is not None else self.gevpars)

        # evaluate all on the 0.05 grid
        self.func_grid = pd.DataFrame.from_dict(dict((name, fn(self.G05) ) 
                        for name, fn in self.items()))
        self.func_grid.index = self.G05
        self.data = data
        
    def plot_unID_pred(self, pred='pulsar', *, ax=None, df=None, 
        adjust=[365,280], # total msp, psr source counts
        text_kw=dict(x=0.7, y=30, s='?', color='r', ha='center', fontsize=60),
        ):
        """
        """
        assert pred in 'pulsar blazar'.split()
        kde_table = self.func_grid.copy()['msp psr'.split()]*adjust*self.hist_binsize
        data = self.data if df is None else df
        unid_pred = data.query(f'source_type=="unID-{pred}"')

        fig, ax = plt.subplots(figsize=(6,5)) if ax is None else (ax.figures, ax)
        
        sns.lineplot(kde_table, ax=ax, palette=self.palette[1:3], dashes=False )    
        
        ax.set(ylim=(0,80), ylabel='Counts / bin', yticks=np.arange(0,76,25),
            xlabel='$G$',xlim=(-1,2), xticks=np.arange(-1,2.01,1) )
        
        leg = ax.legend(title='Adjusted KDE', title_fontsize=14)
        for tobj in leg.get_texts():
            text = tobj.get_text()
            n = adjust[0] if text=='msp' else adjust[1]
            tobj.set_text( text + f' ({n})' )
        
        update_legend(ax, kde_table, hue='msp psr'.split())
        axt = ax.twinx()

        sns.histplot(unid_pred, ax=axt, x='G', element='step', alpha=0.15,
                    color='yellow', bins=self.Gbins, label=f'unID-{pred}\n({len(unid_pred)})')
        axt.set(ylim = ax.get_ylim(), yticks=[], ylabel='')
        axt.legend(loc='upper right', frameon=False)
        
        if text_kw: ax.text(**text_kw) #0.7, 30, '?', color='r', ha='center', fontsize=60)
        sns.move_legend(ax,loc='upper left', fontsize=14, frameon=False, )
        return fig


    
    def plot_fit(self, tdata, # df or df.G values 
                x0, # component values
                *, name='unID', palette=None, ax=None, **kwargs): 
        

        if palette is None: palette = self.palette
        if isinstance(tdata, pd.DataFrame): tdata=tdata.G
        
        # multiply evaluated functions by component values
        tdf = self.func_grid * x0 * self.hist_binsize
    
        fig, ax = plt.subplots(figsize=(8,6)) if ax is None else (ax.figure, ax)

        # plot each component
        for (label, y), color in zip( tdf.items(), palette):
            ax.plot(tdf.index, y, '-', color=color, lw=2, label=label+f' ({np.sum(y)/2:.0f})')
            
        # sum of components
        ax.plot(tdf.index, tdf.sum(axis=1), '-', label='Sum', color='w', lw=4)
        totmax = max(tdf.sum(axis=1))

        # data histogram
        sns.histplot(ax=ax, x=tdata, bins=self.Gbins, element='step', alpha=0.1, color='yellow',
            label=f'{name} ({len(tdata)})'  )
        
        ax_kw = dict(xlabel='$G$', xticks=np.arange(-1,2.1, 1), xlim=(-1,2),
            ylabel='Counts / 0.1 bin', ylim=(0,1.4*totmax), yticks=np.arange(0,totmax,50),)
        ax_kw.update(kwargs)
        ax.set(**ax_kw)
        ax.legend(fontsize=12, frameon=False, ncols=2)
        return fig

    def chisq_fit(self, target_data, x0=[300]*4):
        """
        """
            
        from scipy import optimize
        if isinstance( target_data, pd.DataFrame):
            target_data = target_data.G
        target_hist = np.histogram(target_data, self.Gbins)[0]
    
        def chisq( x):
            # x is dict, numpy array, or list. 
            tdf = self.func_grid*self.hist_binsize * x
            t = tdf.sum(axis=1).to_numpy().reshape(-1,2).sum(axis=1)/2
            return np.sum((t-target_hist)**2/t)
        
        npar = len(self)
        if type(x0) == dict:
            x0 = x0.values()

        opt = optimize.minimize(chisq, x0 , 
                                 bounds= optimize.Bounds([0.1]*npar, [1e3]*npar )
                                )
        if opt.success:
            opt.x = dict(zip(self.keys(), opt.x))
        return opt

    def fit_values(self, data, cut=''):
        """ For the dataset, return array of predicted counts from chisq fit
        """
        norms = (pd.Series( self.chisq_fit( 
                                            data.query(cut) if cut else data
                                        ).x
                        ) 
                ).values
        return norms

    def binned_likelihood(self, target_data):
        """Return a class to perform a binned likelihood fit to list of G values
        """
        if isinstance( target_data, pd.DataFrame):
            target_data = target_data.G
        h, _ = np.histogram(target_data, self.Gbins)
        N = sum(h)
        fgrid = self.func_grid

        class LogLike:
            """Implement binned multinomial likelihood
            """
        
            def __init__(self):
                self.N = N

            def model(self, x):
                # array of model predictions for the histogram
                tdf = fgrid*0.1 * np.append( x, N-np.sum(x)) # expand vars to 4 with 
                return tdf.sum(axis=1).to_numpy().reshape(-1,2).sum(axis=1)/2 #average 2 values per bin
                        
            def __call__(self, x):

                # returns negative of log likelihood for minimization
                # tdf = fgrid*0.1 * np.append( x, N-np.sum(x)) # expand vars to 4 with 
                # m =  tdf.sum(axis=1).to_numpy().reshape(-1,2).sum(axis=1)/2 #average 2 values per bin
                return -np.sum( h * np.log(self.model(x)))
        
            def maximize(self, x0=np.array([150]*3)):
                from scipy import optimize
                return optimize.minimize(self, x0, 
                                        bounds= optimize.Bounds([0.1]*3, [1e3]*3 ))
        return LogLike()

    def integral(self,  range=(-1,2), norms=np.ones(4) ):
        """ Return dict of function integrals for given range
        """
        from scipy.integrate import quad
        return dict( (name, quad(f, *range, epsabs=1e-3)[0]*norm)
                            for (name, f), norm in  zip(self.items(),norms))
    
    def fitter(self, data, name, 
               # default palette for isotrop, msp, psr, gevatar
               palette='0.5 cyan deepskyblue magenta'.split() 
               ):
        """ Return a Fitter object, which will have first performed fit (chisq now)
        """
        galfunc = self
     
        class Fitter:
        
            def __init__(self):
                self.opt = galfunc.chisq_fit(data.G) #,x0=[300]*4) # x0=dict(msp=268, psr=419, gevatar=479,isotropic=392))
                if not self.opt.success:
                    print(f'Warning: fit failed: {opt.message}', file=sys.stderr)

            def show_fit(self, round=1, names=None):
                # return show_fit(self.opt, round,names)
                # def show_fit(opt, round=1, names=None):
                opt = self.opt
                if opt is None:
                    show('No fit to display')
                    return
                if not opt.success:
                    print(f'Warning: {opt.message}', file=sys.stderr)

                if type(opt.x)==dict: # 'assumed that opt.x was converted to a dict'
                    names, fit_pars = opt.x.keys(), np.array(list(opt.x.values()))
                else:
                    fit_pars = opt.x
                    names = [f'p{i}' for i in range(len(fit_pars))] if names is None else names

                show(f"""Function value: {opt.fun:.1f}""")
                
                cov = opt.get('hess_inv', None)    
                if cov is not None:
                    if hasattr(cov,  'todense'):
                        cov = cov.todense()
                    if len(opt.x)==1:
                        sigs = np.sqrt(cov[0,0])
                        corr = None
                    else: 
                        sigs = np.array(np.sqrt(cov.diagonal()))
                        corr = cov / np.outer(sigs,sigs)    
                else: show('no covariance')
                if round>0:
                    a = pd.Series(fit_pars.round(round), index=names, name='fit')
                    b = pd.Series(sigs.round(round), index=names, name='+/-')
                else:
                    a = pd.Series(fit_pars, index=names, name='fit').astype(int)
                    b = pd.Series(sigs,    index=names, name='+/-').astype(int)
                show( pd.DataFrame([a,b]) )
                if corr is not None:
                    show(f"""Correlations (%):""")
                    show((100*corr).astype(int))

            def plot_fit(self, ax=None, palette=palette, **kwargs):
                return   galfunc.plot_fit(data.G, self.opt.x, name=name, ax=ax, palette=palette, **kwargs)

            def pie(self, ax=None, palette=palette,  **kwargs):
                """ Make a pie chart"""
                def get_cov(opt):
                    cov = opt.get('hess_inv', None)    
                    if cov is not None:
                        if hasattr(cov,  'todense'):
                            cov = cov.todense()
                        if len(opt.x)==1:
                            sigs = np.sqrt(cov[0,0])
                            corr = None
                        else: 
                            sigs = np.array(np.sqrt(cov.diagonal()))
                            corr = cov / np.outer(sigs,sigs)   
                    return sigs, corr
                
                fig, ax = plt.subplots(figsize=(5,5)) if ax is None else (ax.figure, ax)
                x = self.opt.x
                xerr,_ = get_cov(self.opt)
                ax.pie(list(x.values()), colors=palette,
                    labels=[ f'{name}'+'\n'+f'{val:.0f}'+'±'+f'{sig:.0f}' for ((name,val), sig) in zip(x.items(),xerr)],
                    textprops=dict(fontsize=12),
                    **kwargs)
                return fig
                
        return Fitter()
    
    def fit_and_plot(self, data, name, fig=None, **kwargs):
    
        f = self.fitter(data, name)
        if fig is None:
            fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(12, 5), width_ratios=(5,4))
        else:
            (ax1,ax2) = fig.subplots(ncols=2, width_ratios=(5,4))
        f.plot_fit(ax=ax1, **kwargs)
        f.pie(ax=ax2,labeldistance=1.2 , radius=0.8, explode = (0,0,0, 0.1), startangle=90)
        
        return fig
    
    def fit_and_pie(self, data, name, ax=None, radius=0.8):
        fig, ax = plt.subplots(figsize=(4,4)) if ax is None else (ax.figure, ax)
        f = self.fitter(data, name)
        f.pie(ax=ax,labeldistance=1.2 , radius=radius, explode = (0,0,0, 0.1), startangle=90)
        return fig
    
    def multiplot(self, data, q, title, *, fig=None, **kwargs):
    
        from pylib.skymaps import AITfigure
        from astropy.coordinates import SkyCoord
        df = data.query(q) if q else data
        if fig is None:
            fig = plt.figure(layout='constrained', figsize=(12,10))
        fig.suptitle( title+'\n'+q)
        ufig, lfig = fig.subfigures(nrows=2, height_ratios=(2,3), hspace=0) #, wspace=0.05, width_ratios=(2,3))
        # upper subfigures
        self.fit_and_plot(  df, 'unID', fig=ufig, **kwargs)
        
        # lower subfigures
        lfig, rfig = lfig.subfigures(ncols=2, wspace=0.05, width_ratios=(2,3))
        ax1=lfig.add_axes(111)
        sns.scatterplot(df, ax=ax1, x='log_epeak', y='d', s=20, color='cyan',
                    )
        
        ax1.set(**epeak_kw(), ylim=(0,2), ylabel='$d$', yticks=np.arange(0,2.01, 0.5), aspect=1);
        
        (AITfigure(fig=rfig, )
            .scatter(SkyCoord(df.glon, df.glat, unit='deg', frame='galactic'),
                    s=8, color='cyan',)
        )
        return fig

    def semiplot(self, data, q, title, *, radius=0.8, fig=None):
        """Upper plot: the pie; lower plot: scatter plot of d vs Ep
        """
        
        df = data.query(q) if q else data
        if fig is None:
            fig = plt.figure(layout='constrained', figsize=(6,9))
        fig.suptitle( title+'\n'+q)
        lfig, ufig = fig.subfigures(nrows=2, height_ratios=(2,2), hspace=0) #, wspace=0.05, width_ratios=(2,3))
        
        # upper subfigure
        f = self.fitter(df, '')
        ax2 = ufig.subplots()
        f.pie(ax=ax2, labeldistance=1.2 , radius=radius, explode = (0,0,0, 0.1), startangle=90)
        
        # lower subfigure
        ax1 = lfig.subplots()
        sns.scatterplot(df, ax=ax1, x='log_epeak', y='d', s=20, color='cyan',)        
        ax1.set(**epeak_kw(), ylim=(0,2), ylabel='$d$', yticks=np.arange(0,2.01, 0.5), aspect=1);
        
        return fig


def show_fit(opt, round=1, names=None):
    if opt is None:
        show('No fit to display')
        return
    if not opt.success:
        print(f'Warning: {opt.message}', file=sys.stderr)

    if type(opt.x)==dict: # 'assumed that opt.x was converted to a dict'
        names, fit_pars = opt.x.keys(), np.array(list(opt.x.values()))
    else:
        fit_pars = opt.x
        names = [f'p{i}' for i in range(len(fit_pars))] if names is None else names

    show(f"""Function value: {opt.fun:.1f}""")
    
    cov = opt.get('hess_inv', None)    
    if cov is not None:
        if hasattr(cov,  'todense'):
            cov = cov.todense()
        if len(opt.x)==1:
            sigs = np.sqrt(cov[0,0])
            corr = None
        else: 
            sigs = np.array(np.sqrt(cov.diagonal()))
            corr = cov / np.outer(sigs,sigs)    
    else: show('no covariance')
    if round>0:
        a = pd.Series(fit_pars.round(round), index=names, name='fit')
        b = pd.Series(sigs.round(round), index=names, name='+/-')
    else:
        a = pd.Series(fit_pars, index=names, name='fit').astype(int)
        b = pd.Series(sigs,    index=names, name='+/-').astype(int)
    show( pd.DataFrame([a,b]) )
    if corr is not None:
        show(f"""Correlations:""")
        show(corr.round(2))

class KDEuv(dict):
    kde_binsize=0.025
    unid_binsize=0.1
    norm = 1/kde_binsize**2
    x_grid, y_grid =(np.arange(-1,1,kde_binsize), np.arange(-1,0.5,kde_binsize))
    unid_bins = np.arange(-1,1.01,unid_binsize)
    trainers=['pulsar', 'blazar']

    class GKDE_prime(Gaussian_kde):
        # reflect at 1
        def evaluate(self,p):
            return super().evaluate(p) +super().evaluate([p[0],2-p[1]])
    
    def __init__(self, data, title='', transform=False, bw_method=None):
    
        # initial selection
        self.dfuv = dfuv = data.copy()
        self.title = title

        # centered log_epeak and log_d
        xp, yp = self.dfuv.log_epeak-0.085, dfuv.log_d+0.12
        xp, yp = self.dfuv.log_epeak, dfuv.log_d
        if transform :
            dfuv['u'] =  xp - 0.6 * yp 
            dfuv['v'] =  1.5 * yp    
        else: # nope
            dfuv['u'] = xp
            dfuv['v'] = yp 
        self.unid = self.dfuv.query('association=="unID"')
        
        # create 2-D KDE with perhaps rescaled, adjusted variables 
        self.update( 
            # dict( (key, Gaussian_kde(dfuv.groupby('trainer').get_group(key),  ['v','u'],
            #                          bw_method=bw_method)  ) for key in self.trainers)
            dict( (key, self.GKDE_prime(dfuv.groupby('trainer').get_group(key),  ['v','u'],
                                     bw_method=bw_method)  ) for key in self.trainers)
        )

        # evaluate the KDE functions on a 2-D grid
        self.grids ={}
        xg, yg = self.x_grid, self.y_grid
        for tr in self.trainers:
            fp = self[tr]
            self.grids[tr] = pd.DataFrame(
                [[fp.evaluate((y,x)) for x in xg] for y in yg], index=yg, columns=xg).astype(float)
         
    def plot_kde_contours(self,  ax,
                          cmaps ='Reds Greens'.split(), alpha=0.5 ):
        
        x,y = self.x_grid, self.y_grid
        for (key,grid), cmap in zip(self.grids.items(), cmaps):
            ax.contour(x,y, grid, cmap=cmap,  alpha=alpha);

    def plot_unid(self, ax1, ax2, cut='', **kwargs):
        # scatter and hist
        unid = self.unid.query(cut) if cut else self.unid
        sns.scatterplot(unid, #.query('Ep_unc/Ep<0.5'),
                ax=ax1, x='u', y='v', 
                s=20, edgecolor='none', color='orange',
                alpha=0.4,legend=False);
        
        sns.histplot(unid, bins=self.unid_bins,
                     ax=ax2, x='u', element='step', fill=False, color='orange', **kwargs)
        ax2.set(ylabel=f'Counts / {self.unid_binsize} bin', yticks=np.arange(0,151,50),)

    def scat_plot(self, norms ):
        fig, (ax1,ax2) = plt.subplots(nrows=2, figsize=(8,11), sharex=True, height_ratios=(3,2))
        ax1.set(xlim=(-1,1), ylim=(-1, 0.35), 
                yticks=np.arange(-1,0.5,0.5), ylabel='log($d$)',
                xticks=np.arange(-1,1.01,0.5),)
        self.plot_kde_contours(ax1, cmaps='Reds Greys'.split())
        
        self.plot_unid(ax1, ax2,(q:=''), label=f'unID ({len(self.unid)})')
        x = self.x_grid
        total = np.zeros(len(x))
        for train, norm in zip(self.keys(), norms):
            g = self.grids[train]
            # integral over curvature, per unid bins
            y = g.sum(axis=0)*self.kde_binsize * norm * self.unid_binsize
            ax2.plot(x,y, label=train+f' ({norm})')
            total += y
        ax2.plot(x, total, lw=4, color='w', ls='--', label=f'Total ({sum(norms)})')
        ax2.set(**epeak_kw(), )
        ax2.legend(fontsize=12)
        
        fig.suptitle(self.title+' '+q)
        return fig

### ---------- Execution ---------------
dark_mode = set_theme(sys.argv)
palette =['cyan', 'magenta', 'yellow'] if dark_mode else 'green red blue'.split()

# data = kde_setup(cut = '0.15<Ep<10 & variability<25' )
# fs = FeatureSpace.runit(data, dark_mode, palette)
