"""
See https://fermi.gsfc.nasa.gov/ssc/data/access/lat/14yr_catalog/
for files.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import healpy
# from astropy_healpix import healpy
import seaborn as sns
from astropy.coordinates import SkyCoord
from pylib.skymaps import HPmap, AITfigure
from pylib.tools import  update_legend, set_theme

dark_mode = set_theme(sys.argv)

filename_dict = dict(
    v07='gll_iem_v07_hpx.fits',  # hp projection of standard
    v13 = 'gll_iem_uw1216_v13.fits', # DR4 smoothed version
    )

version = 'v13'
class Diffuse:
    
    def __init__(self, 
                    filename=filename_dict[version],           
                field=11, energy=1000):
        """field is the index into the 28 long energy array, close to 1000
        """
        import os
        from astropy.io import fits
        from astropy.wcs import WCS
        from astropy.coordinates import SkyCoord

        self.energy = energy
        self.fits_file = os.path.expandvars(f'$FERMI/diffuse/{filename}')
        with fits.open(self.fits_file) as hdus:
            if hdus[1].name=='ENERGIES':
                # it is an image files. to use wcs must fake a 2-d image map
                hdr = dict((key,val) for (key,val) in hdus[0].header.items() if key[-1]!='3' and key!='COMMENT')
                hdr['NAXIS'] = 2  # need to fool WCS into thinking only two axes
                wcs = WCS(hdr)
                self.nside=256 #use this
                coords = healpy.pix2ang(self.nside, np.arange(12*self.nside**2), lonlat=True)
                xpix, ypix = wcs.world_to_pixel(SkyCoord(*coords, unit='deg', frame='galactic'))
                hpm = HPmap(hdus[0].data.T[(xpix+0.5).astype(int), (ypix+0.5).astype(int), field], name='')
            else:
                # assume healpix
                hpm = HPmap.from_FITS(self.fits_file, field=field, name='',)
                self.nside=hdus['SKYMAP'].header['NSIDE']
            self.energies = np.array(hdus['ENERGIES'].data).astype('float')

        self.diffuse_hpm = hpm #HPmap.from_FITS(self.fits_file, field=field, name='',)
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
            energies = hdus['ENERGIES'].data.field(0)

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
        return g.figure     


class IsotropicDiffuse:
    """Implement isotropic (extra-galactic) function of distribution values
    """
    def __init__(self, reload=False, binsize=0.05,
            cache_file=f'files/isotropic_diffuse_{version}.pkl'):

        from pathlib import Path
        import pandas as pd
        if Path(cache_file).exists() and not reload:
            self.cache = pd.read_pickle(cache_file)
        else:
            diff = Diffuse()
            gbins=np.arange(-1,2.01,binsize)
            gx = 0.5*(gbins[1:]+gbins[:-1])
            ghist, _ = np.histogram(diff.diffuse_hpm.map , gbins )
            gy = ghist/sum(ghist)/binsize
            self.cache = pd.DataFrame.from_dict(dict(gx=gx, gy=gy))
            self.cache.to_pickle(cache_file)
            print(f'IsotropicDiffuse: Wrote cache file {cache_file}')
            
    def __call__(self,x):
        return np.interp(x, self.cache.gx, self.cache.gy)


class DiffuseSED:
    
    unit_factor = 1e6/3282.80635  # convert from erg/sr to eV/deg^2s

    def __init__(self, filename=filename_dict[version],  
                 ):
        import os
        from astropy.io import fits
        from astropy.wcs import WCS
        
        with fits.open(os.path.expandvars(f'$FERMI/diffuse/')+filename)  as hdus: 
            ## TODO: allow for healpix format
            # need to fool WCS into thinking only two axes
            hdr = dict((key,val) for (key,val) in hdus[0].header.items() if key[-1]!='3' and key!='COMMENT')
            hdr['NAXIS'] = 2  
            self.wcs = WCS(hdr)
            self.grid = np.array(hdus[0].data)
            self.energies = np.array(hdus['ENERGIES'].data).astype(float)
            
    def __call__(self, skycoord):
        " return flux array for pixel at skycoord"
        xpix,ypix = self.wcs.world_to_pixel( skycoord )
        return self.grid[:, int(ypix),int(xpix)]

    def sed(self, skycoord):
        """ Return a SED for the point
        input is log(e), returns log(flux)
        """
        loge = np.log(self.energies)
        df = self
        energies = self.energies
        
        class SED:
 
            def __init__(self):
                from scipy.interpolate import CubicSpline
                self.cs = CubicSpline(loge, np.log(  df(skycoord) * energies**2 ))
                
            def __call__(self, x, nu=0):
                return self.cs(x,nu) #np.interp(x, self.xp, self.yp)

            def max(self, lims=(4,9)):
                # position of max
                from scipy.optimize import brentq
                try:
                    return brentq(lambda x: self(x,1), *lims)
                except:
                    return lims[0]
   
            def curvature(self): 
                """curvature (negative of appox 2nd derivative)
                """
                mx = self.max()
                u, d = self([mx-1, mx+1])
                return round(2*self(mx) -u -d , 2)
  
        return SED()   
    
    def eflux_plot(self, *, l=0, ax=None):
        """
        """

        fig, ax = plt.subplots(figsize=(6,6)) if ax is None else (ax.figure, ax)
        
        # normalization factor to units below
        norm = 1e6/3282.80635 
        ee = np.logspace(2,5, 40)
        txt_kw = dict( backgroundcolor='k' if dark_mode else 'w',
                    va='center', fontsize=14)
        
        for b in (-90,-30, -2 ,0,2,30,90):
            f = self.sed( SkyCoord(l,b, unit='deg', frame='galactic') ) 
            # convert from log space, adjust units
            fp = lambda e :  np.exp(f( np.log(e))) * norm
            if b==0:
                ax.text(10, fp(1e4), '$b=0$', ha='center', **txt_kw)
            else: 
                ax.text( 10,  fp(1e4), f'{b}', ha=('right' if b<0 else 'left'), **txt_kw)
            ax.loglog(ee/1e3, fp(ee), '-', label=f'{b}')
            
        ax.set(xlabel='Energy (GeV)', xticks=np.logspace(-1,2,4), xticklabels='0.1 1 10 100'.split(),
            ylim=(1e-2, 3e2),  ylabel=r'Energy flux ($\mathrm{eV\ s^{-1}\ cm^{-2}\ deg^{-2}}$)');
        return fig

    @classmethod
    def ait_plot(cls, nside=128, energy=1e3, fig=None, **kwargs):
        # takes a minute with nside 64

        self = cls()
        lb = healpy.pix2ang(nside, range(12*nside**2), lonlat=True)
        sc = SkyCoord(*lb, unit='deg', frame='galactic')
        loge = np.log(energy)
        z = np.array([self.sed(t)(loge) for t in sc] )

        return (AITfigure(fig, **kwargs)
            .imshow(np.array(z)/2.303 + np.log10(self.unit_factor), cmap='jet')
            .colorbar(shrink=0.8, label=r'$\mathrm{\log_{10}(Flux)}$')
            ).figure  
