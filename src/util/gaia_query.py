"""

Test how to get data and plot from GAIA

"""
import matplotlib.pyplot as plt
import astropy.units as u 
import astropy.coordinates as coord
from astroquery.vizier import Vizier 

def gaia_query(ra_deg, dec_deg, rad_deg, maxmag=18, 
               maxsources=10000): 
    """
    Query Gaia DR1 @ VizieR using astroquery.vizier
    :param ra_deg: RA in degrees
    :param dec_deg: Declination in degrees
    :param rad_deg: field radius in degrees
    :param maxmag: upper limit G magnitude (optional)
    :param maxsources: maximum number of sources
    :return: astropy.table object
    """
    vquery = Vizier(columns=['Source', 'RA_ICRS', 'DE_ICRS', 
                             'Gmag',"Teff","RPlx"], 
                    column_filters={"Gmag": 
                                    ("<%f" % maxmag)}, 
                    row_limit = maxsources) 
 
    field = coord.SkyCoord(ra=ra_deg, dec=dec_deg, 
                           unit=(u.deg, u.deg), 
                           frame='icrs')
    
    
    
    return vquery.query_region(field, 
                               width=("%fd" % rad_deg), 
                               catalog="I/345/gaia2")[0] 



def plot_gaia(ra,dec,data,ax,TMin=1000,TMax=11000):
    
    

    dmax,dmin = max(data["Gmag"]),min(data["Gmag"])
    
    for i in data:
        vis = 0.1+(dmax - i["Gmag"])/(dmax-dmin)*0.8
        siz = 3 +(dmax - i["Gmag"])/(dmax-dmin)*15
        """
        #vis = 0.1+(dmax - i["Gmag"])/(dmax-dmin)*0.8
        c = "w"
        if str(i["Teff"]) == "--":
            vis = 0.1
        elif float(i["Teff"]) > TMax:
            vis = 1
            c = "b"
        elif float(i["Teff"]) < TMin:
            vis = 0.1
            c = "r"
        else:
            vis = 0.1+(TMax - i["Teff"])/(TMax-TMin)*0.8
            c = "k"
        if str(i["Teff"]) != "--":
            print(float(i["Teff"]),TMax,float(i["Teff"]) > TMax)
        else:
            continue
        
        

        plt.plot(i["RA_ICRS"],i["DE_ICRS"],"o",markersize = siz+1, color="r",alpha = 1)
        plt.plot(i["RA_ICRS"],i["DE_ICRS"],"o",markersize = siz, color="w",alpha = 1)
        
        plt.plot(i["RA_ICRS"],i["DE_ICRS"],"o",markersize = siz, color=c,alpha = vis)
        """
        ax.plot(i["RA_ICRS"],i["DE_ICRS"],"o",markersize = siz+1, color="r",alpha = 1)
        ax.plot(i["RA_ICRS"],i["DE_ICRS"],"o",markersize = siz, color="w",alpha = 1)
        
        ax.plot(i["RA_ICRS"],i["DE_ICRS"],"o",markersize = siz, color="k",alpha = 1-vis)
        

    ax.plot(ra,dec, "*",markersize = 5,color="r",alpha=0.8)
    ax.plot(ra+0.008,dec+0.008, "*",markersize = 0.1,color="k",alpha=0.1)
    ax.plot(ra-0.008,dec-0.008, "*",markersize = 0.1,color="k",alpha=0.1)
    




if __name__ == "__main__":
    
    
    
    ra = 84.893
    dec = -25.148
    
    data = gaia_query(ra,dec,0.01)
    
    print(data)
    
    ax = plt.subplot()
    plot_gaia(ra,dec,data,ax,TMin=1000,TMax=11000)
    
    
    
    """
    ra = 7.19516
    dec = -67.86237
    """
    """
    ra = 317.63150
    dec = -27.18311
    
    ax = plt.subplot()
    
    plot_gaia(ra,dec,ax,0.1,4000.,10000.)

    plt.show()
    """