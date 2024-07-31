import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pylib.kde import FeatureSpace, Gaussian_kde
# from pylib.ml_fit  import doc
from pylib.tools import set_theme, epeak_kw, diffuse_kw , update_legend
from pylib.ipynb_docgen import show, show_fig, show_date

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
            include_bcu=True, # nc=2,
            cut = '0.15<Ep<10 & variability<30',
            title='Unsupervised KDE analysis'  ):
    
    # self = doc(nc=nc, np=1, kde=True,)
    # df = self.df
    show(f"""# {title}""")
    show_date()
    show(f"""* Load source data from `{(filename:='files/dr4_2_class_2_features.csv')}'""")
    df = pd.read_csv( filename,index_col=0)
    df['sqrt_d'] = np.sqrt(df.d.clip(0,2))
    df['log_epeak'] = np.log10(df.Ep)

    def make_group(df):

        def groupit(s):
            if s.association in 'unID bcu'.split(): return 'unID' # special
            if s.association in 'psr msp'.split(): return s.association
            if s.association in  'bll fsrq'.split(): return 'blazar'
            # if ~ pd.isna(s.trainer): return s.trainer
            return np.nan

        df['subset'] = df.apply(groupit, axis=1)
    
    def make_aclass(df):
        """ Return the super classification group name for each source
        """
        gtbl = dict (
            pulsar = 'msp psr'.split(),
            blazar = 'fsrq bll'.split(),
            egal  = 'agn gal sey nlsy1 sbg ssrq css rdg'.split(),
            Gal ='gc bin glc hmb lmb nov pwn sfr snr spp'.split(),
            unID   = ['', 'unk', 'bcu'],
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
    classes = 'blazar pulsar Gal egal unID'.split()
    t =pd.DataFrame([all,sel,pct])[classes]; 

    show(t)    

    apply_diffuse(dfc)
    
    add_classification_prob(df)
    
    show(f"""## Create KDE functions instead of ML training

    * Features: {', '.join(kde_vars)} 
    
    Apply to unIDs {'+ bcus' if include_bcu else ''}
    """)
    apply_kde( dfc, kde_vars)

    df.to_csv((filename:='files/kde_data.csv'))
    show(f'saved KDE setup to `{filename}`')
    return dfc

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
    """
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
    of the relative spectral shape uncertainties. This and the left plot have colors
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

### ---------- Execution ---------------
dark_mode = set_theme(sys.argv)
palette =['cyan', 'magenta', 'yellow'] if dark_mode else 'green red blue'.split()

# data = kde_setup(cut = '0.15<Ep<10 & variability<25' )
# fs = FeatureSpace.runit(data, dark_mode, palette)
