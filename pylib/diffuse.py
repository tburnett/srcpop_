"""
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import healpy
import seaborn as sns

from pylib.skymaps import HPmap
from pylib.tools import  update_legend, set_theme

# dark_mode = set_theme(sys.argv)

class Diffuse:
    
    def __init__(self, filename='gll_iem_v07_hpx.fits', field=11, energy=1000):
        import os
        self.energy = energy
        self.fits_file = os.path.expandvars(f'$FERMI/diffuse/{filename}')
        self.diffuse_hpm = HPmap.from_FITS(self.fits_file, field=field, name='',)
        self.unit = r'$\rm{eV\ cm^{-2}\ s^{-1}\ deg^{-2}}$' #self.diffuse_hpm.unit
        print(f"""* Load diffuse file,  `{self.fits_file}`\n  unit={self.unit}\n select energy= {energy} MeV""")
 
        # convert units of the HEALPix array in the HPmap guy
        dmap = (np.log10(self.diffuse_hpm.map) 
                + 2*np.log10(energy) +6   # multiply by E^2, convert from MeV to eV
                - 2*np.log10(180/np.pi) # convert from sr to deg^2
                ) 
        self.diffuse_hpm.map = dmap
        self.diffuse_hpm.unit = self.unit
        self.dark_mode = plt.rcParams['figure.facecolor']=='black'
    
        # mask for the ridge selection
        glon,glat = healpy.pix2ang(self.diffuse_hpm.nside, range(len(dmap)), lonlat=True)
        glon[glon>180]-=360
        self.ridge_mask= (np.abs(glat)<2) & (np.abs(glon)<45)
        
    def check_hpm(self): # for superclass to invoke 
        pass 
        
    def eflux_plot(self, glon=0):
        """The Galactic diffuse energy flux. 
        """
        from astropy.io import fits
        with fits.open(self.fits_file) as hdus:
            data = hdus[1].data
            energies = hdus[2].data.field(0)

        fig, ax = plt.subplots(figsize=(6,6))
        ax.set(xlabel='Energy (GeV)',
            ylabel=r'Energy flux (eV s-1 cm-2 deg-2)')#\ ($\rm{MeV \s^{-1}\ cm^{-2}\ sr^{-1}}$)') 
        for b in (-90,-30, -2 ,0,2,30,90):
            pix = healpy.ang2pix(512, glon,b, lonlat=True)
            ax.loglog(energies/1e3, energies**2*data[pix]*1e6/3282.80635, label=f'{b}');
        ax.legend(title='b', fontsize=12);
        ax.set_title(f'Diffuse energy flux at $l$={glon}')
        return fig
    
    def get_values_at(self, *args):
        """Return log10 values in units of eV s-1 cm-2 deg-2
        """
        if len(args)==1:
            sdir = args[0]
            return self.diffuse_hpm(sdir)
        raise  Exception(f'Unexpected call {args}')
        
    def plot_limits(self, ax, n=50, **kwargs):
        self.check_hpm()
    
        def ecdf(x, n=None, xlim=None):
            """Return empirical cumulative distribution function
            Parameters
            - x input array
            - n -- number of x, y values to return for sampling mode
            - xlim -- x limits: use min,max of x if NOne
            returns:
                x, y
            """
            xs = np.sort(x)
            ys = np.arange(1, len(xs)+1)/float(len(xs))
            if n is None:
                return xs, ys
            # a,b = xlim if xlim is not None else x.min
            if xlim is None:  a,b = x.min(), x.max()
            else: a,b = xlim # xfull.min(), xfull.max()# if xlim is not None else xlim
            xrange = np.arange(a, b, (b-a)/n);
            idx = np.searchsorted(xs[:-1], xrange); 
            yrange = ys[idx]
            return xrange, yrange
        
        dfm= self.diffuse_hpm.map
        x1, y1 = ecdf(dfm, n, )
        dfmx= self.diffuse_hpm.map[self.ridge_mask]
        x2, y2 = ecdf(dfmx, n, xlim=(dfm.min(),dfm.max()))   
        
        kw = dict(color='0.5', alpha=0.6)
        kw.update(kwargs)
        ax.fill_between(x1, y1, y2, **kw );
        
    def fluxticks(self, x, ):
        ticks =  np.arange(0,2.1,1).astype(int)
        # energy = 900 if not hasattr(self, 'energy') else self.energy
        # unit = '' if not hasattr(self, 'unit') else f'({self.unit})'
        return {x+'ticks':ticks,
                x+'ticklabels' : [f'$10^{{{x}}}$' for x in ticks],
                x+'label': f'Diffuse energy flux'}# at {energy:.0f} MeV {unit}' }

    def plot_diffuse_flux(self,   figsize=(8,8),  df=None, unid=False,  **kwargs):
        """Distributions of diffuse energy flux background values for the source types shown.
        Upper plot: histogram with overlaid KDE curves. Lower plot: corresponding cumulative empirical distribution functions.
        The gray area shows the range between an isotropic distribution and a uniform sampling of the Galactic ridge.
        """
        data = getattr(self, 'df', None) if df is None else df 
        
        if unid:
            hkw = dict(hue='source_type',
                    hue_order = ['unid-'+ name for name in self.target_names],
                    palette=self.palette)    
        else:
            hkw = self.hue_kw.copy()
            hkw.pop('edgecolor', '')
            hkw.update(kwargs)
        x='diffuse'

        fig, (ax1,ax2) = plt.subplots(nrows=2, figsize=figsize, sharex=True,
                                    gridspec_kw=dict(hspace=0.1))

        sns.histplot(data, ax=ax1, bins=25, kde=True, element='step', x=x,  **hkw, )
        update_legend(ax1, data, hkw['hue'], fontsize=12)
        ax1.set(**self.fluxticks('x') )
        
        sns.ecdfplot(data, ax=ax2, x=x, legend=False,  **hkw)
        self.plot_limits( ax2, color='0.3' if self.dark_mode else '0.8')
        ax2.set(**self.fluxticks('x') )
        return fig
    
    def diffuse_vs_ep(self, df=None, hue_kw=None):
        """Diffuse vs peak energy for pulsar-like sources. The lines are KDE contours for psr and msp sources."""

        data=df if df is not None else self.df
        if hue_kw is None: hue_kw = self.hue_kw
        x,y = 'log_epeak diffuse'.split()
        g = sns.JointGrid(height=12, ratio=4 )
        ax = g.ax_joint
        size_kw = dict(size='log TS', sizes=(20,200) ) if not hasattr(self,'size_kw') else self.size_kw
        sns.scatterplot(data, ax=ax, x=x, y=y, **hue_kw, **size_kw);
        axis_kw= lambda a, label, v: {f'{a}label':label,f'{a}ticks':np.log10(v), f'{a}ticklabels':v }
        
        ax.set(**axis_kw('x','$E_p$ (GeV)', [0.1, 0.25,0.5,1,2,4]),xlim=np.log10((0.1,6)), 
            **self.fluxticks('y')
            )

        hkw = dict(element='step', kde=True, bins=25, **hue_kw, legend=False)
        sns.histplot(data, y=y, ax=g.ax_marg_y, **hkw)
        sns.histplot(data, x=x, ax=g.ax_marg_x, **hkw)
        update_legend(ax, data, hue=hue_kw['hue'],  fontsize=12,   loc='lower left')
        return g.fig     
