"""
Support for secondary plots with results of the ML study

"""
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns   
from pylib.tools import epeak_kw, fpeak_kw, set_theme, show_date #, diffuse_kw, var_kw
from pylib.ipynb_docgen import show, show_fig, capture_show, capture_hide

dark_mode = set_theme(sys.argv)
palette =(['cyan', 'magenta', 'yellow'] if dark_mode else 'green red blue'.split())
title = sys.argv[-1] if 'title' in sys.argv else None

def gbar(ax, orientation='vertical', label='Galacticity $G$',
         ticks=np.arange(-0.5, 1.51, 0.5), norm=(-0.5,1.5), **kwargs):
    from matplotlib import colors
    cbar = plt.colorbar(
        plt.cm.ScalarMappable(
            cmap=sns.color_palette('coolwarm', as_cmap= True), 
            norm=colors.Normalize(*norm), **kwargs
            ),
        cax=ax, orientation=orientation, label=label,
        )
    cbar.set_label(label)
    cbar.set_ticks(ticks)    
    return cbar
def fpeak_kw(axis='x'):
    return {axis+'label':r'Peak flux $F_p\ \ \mathrm{ (eV\ cm^{-2}\ s^{-1})}$', 
            axis+'ticks': [-2, -1 ,0 ,1],
            axis+'ticklabels': '$10^{-2}$ 0.1 1 10'.split(),
            axis+'lim': (-2,1.),
            }
def d_kw(axis='x'):
    return {axis+'label': 'Curvature $d$',
            axis+'lim': (0,2.05),
            axis+'ticks': np.arange(0,2.1,0.5)
           }
def pulsar_kw(axis='y'):
    return {axis+'label': 'Pulsar probability',
            axis+'lim': (-0.02, 1.02), 
            axis+'ticks': np.arange(0,1.1, 0.2),
            }



class FileAnalysis:

    def __init__(self,filename ='files/dr4_2_class_classification.csv', 
                    query='0.1<Ep<10 & variability<25 & Fp<10'):
        
        show(f"""# {title}""")
        show_date()
        data = self.data = pd.read_csv(filename, index_col=0).query(query)
        data.diffuse = data.diffuse.clip(-0.5,1.5)
        show(f"""* Loaded `{filename}`, selected "{query}" """)
        
        def make_group(df):
            def groupit(s):
                if s.association in 'psr msp unID'.split(): return s.association
                if s.association in 'bll fsrq'.split(): return 'blazar'
                return np.nan
            df['subset'] = df.apply(groupit, axis=1)
        make_group(data)
        show(pd.Series(data.groupby('subset').size(), name='Sources'))
        data['log_epeak'] = np.log10(data.Ep)
        data['log_fpeak'] = np.log10(data.Fp)

        self.unid  = self.data.query('subset=="unID"')
        self.assoc = self.data.query('subset=="psr" | subset=="msp" | subset=="blazar"')

    
    def multi_pulsar_vs_x(self, x):
        """
        """
        assert x in 'log_fpeak d log_epeak'.split()
        
        def pulsar_vs_x(data, ax, title='', palette='coolwarm'): 
            assert(len(data)>0), f'No data for {title}'
            kw = dict( ax=ax, y='p_pulsar', x=x, s=30, legend=False )
            sns.scatterplot(data,  hue='diffuse', hue_norm=(-0.5, 1.5),  
                            edgecolor='none',   palette=palette,    **kw, );
            ax.set(title=title,  **pulsar_kw(),
                ** dict(d=d_kw(), log_epeak=epeak_kw(), log_fpeak=fpeak_kw())[x]  )

        fig, axd = plt.subplot_mosaic( [
                        [ 'Associated', 'unID', 'cbar'],              
                        ], width_ratios=(25,25,1),                                
                        figsize=(15,6),  layout='constrained',
                        gridspec_kw=dict(wspace=0.1))
        for label, ax in axd.items():
            if label=='cbar':
                gbar(ax)                               
            else:
                pulsar_vs_x( self.unid if label=='unID'  else self.assoc, 
                        ax=ax, title=label);
                if label=='unID':
                    ax.set(yticks=[], ylabel='')
        return fig

    def d_vs_ep(self, unid_cut='0.15<p_pulsar<0.85', hue='p_pulsar',hue_norm=None):
        fig, axd = plt.subplot_mosaic(
                    'AAUU;.CC.',   height_ratios=[20,1],
                    gridspec_kw=dict(hspace=0.2),
                    figsize=(15,8), sharey=False, layout='constrained')
        if hue_norm is None:
            hue_norm = (0,1) if hue=='p_pulsar' else (-0.5,1.5)
        scat_kw=dict( y='d', x='log_epeak', s=60, edgecolor='none', legend=False,
                            hue=hue, hue_norm=hue_norm, 
                    palette=sns.color_palette('coolwarm', as_cmap=True))
        for key, ax in axd.items():
            if key=='U':
                unid = self.unid.query(unid_cut) if unid_cut else self.unid
                sns.scatterplot(unid, ax=ax, **scat_kw)
                ax.set(**epeak_kw('x'), **d_kw('y'), 
                       title=f'unID' + (f'({unid_cut})' if unid_cut else ''), )        
            elif key=='A':
                sns.scatterplot( self.assoc, ax=ax, **scat_kw, style='subset' )
                ax.set(**epeak_kw('x'), **d_kw('y'), title='Associated')
            else:
                gbar(ax, orientation='horizontal',
                    label='$P_{pulsar}$' if hue=='p_pulsar' else 'Galacticity',
                    norm=hue_norm,  ticks=hue_norm)
        return fig    
 
    # def multiple_pulsar_vs_ep(self, data=None, palette='coolwarm'):
    #     """Scatter plots of the ML pulsar probability $P_{pulsar}$ vs $Ep$ for the data subsets
    #     shown. 
    #     The color scale is $G$, the measure of the Galactic correlation.

    #     """
    #     # from matplotlib import colors
    #     data = self.data if data is None else data
            
    #     fig = plt.figure(figsize=(15,6), layout="constrained",)
    #     axd = fig.subplot_mosaic([
    #                             ['psr',   'msp',  '.'   ],
    #                             ['psr',   'msp',  'cbar'],
    #                             ['psr',   'msp',  'cbar'],
    #                             ['blazar','unID', 'cbar'],
    #                             ['blazar','unID', 'cbar'],
    #                             ['blazar','unID', '.'],
    #                           #  ['bcu',   'unk',  'cbar'],
    #                           #  ['bcu',   'unk',  '.'   ], 
    #                               ],
    #                             width_ratios=[20,20,1],
    #                        gridspec_kw=dict(bottom=0.1,left=0.1 )
    #                         )
            
    #     def select_data(name):
    #         if name in 'psr msp blazar unID'.split():
    #             return data[data.subset==name]
    #         else:
    #             return data[data.class1==name]
                
    #     for label, ax in axd.items():
    #         if label=='cbar':
    #             cbar = gbar(ax)
    #         else:    
    #             self.pulsar_vs_ep(select_data(label),  ax, label, palette, no_label=True)
    #             ax.set(ylabel=' ')
    #             # instead of sharex...
    #             if label in 'psr msp'.split(): ax.set(xlabel='', xticklabels=[])
    #             if label in 'msp unID' .split(): ax.set(ylabel='', yticklabels=[])
    #     # apply axis labels (can't offset the plot??)
    #     fig.text(0.5, 0, '$E_p$ (GeV)', ha='center', va='bottom')
    #     fig.text(0, 0.5, '$P_{pulsar}$', rotation='vertical', ha='left', va='center')

    #     return fig


