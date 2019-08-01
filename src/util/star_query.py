# -*- coding: utf-8 -*-
"""

There is a potential synergy between Gaia and Exofop

Redo the functions from preseach so that we can get what we need


DR2 columns
    ['solution_id', 'designation', 'source_id', 'random_index', 'ref_epoch', 
    'ra', 'ra_error', 'dec', 'dec_error', 'parallax', 'parallax_error', 'parallax_over_error', 
    'pmra', 'pmra_error', 'pmdec', 'pmdec_error', 'ra_dec_corr', 'ra_parallax_corr', 
    'ra_pmra_corr', 'ra_pmdec_corr', 'dec_parallax_corr', 'dec_pmra_corr', 
    'dec_pmdec_corr', 'parallax_pmra_corr', 'parallax_pmdec_corr', 'pmra_pmdec_corr', 
    'astrometric_n_obs_al', 'astrometric_n_obs_ac', 'astrometric_n_good_obs_al', 
    'astrometric_n_bad_obs_al', 'astrometric_gof_al', 'astrometric_chi2_al', 
    'astrometric_excess_noise', 'astrometric_excess_noise_sig', 'astrometric_params_solved', 
    'astrometric_primary_flag', 'astrometric_weight_al', 'astrometric_pseudo_colour', 
    'astrometric_pseudo_colour_error', 'mean_varpi_factor_al', 'astrometric_matched_observations', 
    'visibility_periods_used', 'astrometric_sigma5d_max', 'frame_rotator_object_type', 
    'matched_observations', 'duplicated_source', 'phot_g_n_obs', 'phot_g_mean_flux', 
    'phot_g_mean_flux_error', 'phot_g_mean_flux_over_error', 'phot_g_mean_mag', 
    'phot_bp_n_obs', 'phot_bp_mean_flux', 'phot_bp_mean_flux_error', 
    'phot_bp_mean_flux_over_error', 'phot_bp_mean_mag', 'phot_rp_n_obs', 'phot_rp_mean_flux', 
    'phot_rp_mean_flux_error', 'phot_rp_mean_flux_over_error', 'phot_rp_mean_mag', 
    'phot_bp_rp_excess_factor', 'phot_proc_mode', 'bp_rp', 'bp_g', 'g_rp', 'radial_velocity', 
    'radial_velocity_error', 'rv_nb_transits', 'rv_template_teff', 'rv_template_logg', 
    'rv_template_fe_h', 'phot_variable_flag', 'l', 'b', 'ecl_lon', 'ecl_lat', 'priam_flags', 
    'teff_val', 'teff_percentile_lower', 'teff_percentile_upper', 'a_g_val', 
    'a_g_percentile_lower', 'a_g_percentile_upper', 'e_bp_min_rp_val', 
    'e_bp_min_rp_percentile_lower', 'e_bp_min_rp_percentile_upper', 'flame_flags', 
    'radius_val', 'radius_percentile_lower', 'radius_percentile_upper', 'lum_val', 
    'lum_percentile_lower', 'lum_percentile_upper', 'datalink_url', 'epoch_photometry_url', 
    'source_id_2', 'r_est', 'r_lo', 'r_hi', 'r_len', 'result_flag', 'modality_flag', 'r']


"""

from __future__ import print_function, division

import sys
import mechanize
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

br = mechanize.Browser()
br.set_handle_robots(False)   # ignore robots
#br.set_handle_refresh(False)  # can sometimes hang without this

Temp = sys.stdout
sys.stdout = None 
import astropy.units as u
from astropy.table import Table
try:
    from astroquery.gaia import Gaia
except:
    pass
from astroquery.vizier import Vizier 
from astropy.coordinates import SkyCoord, ICRS
# mute import text
sys.stdout = Temp

import warnings
warnings.filterwarnings("ignore")



def get_exofop_nearby(TIC, verbose=False):

    nearby_url = "https://exofop.ipac.caltech.edu/tess/nearbytarget.php?id=%s"%(TIC)
    response = br.open(nearby_url)
    result = response.read()
    soup = BeautifulSoup(result, 'html.parser')
    
    nearby_table = soup.find_all("table",{"class":"table table-condensed table-hover table-bordered wid90"})[0]
    df = pd.read_html(str(nearby_table),header=0)[0]
    
    df = df.rename(index=str, columns={"Separation (deg)": 'Separation (")',
                                       "Position Angle (deg E. of N.)":"(deg E. of N.)"})
    
    df['Separation (")']*=3600
    
    
    
    def convert(time_str_list,isra=False):
        
        for i,time_str in enumerate(time_str_list):
            h, m, s = time_str.split(':')
            if isra:
                time_str_list[i] = (float(h)*3600+float(m)*60 + float(s))/3600/24*360
            else:
                if float(h) >= 0:
                    time_str_list[i] = (float(h)*3600+float(m)*60 + float(s))/3600
                else:
                    time_str_list[i] = -(abs(float(h)*3600)+float(m)*60 + float(s))/3600
        return time_str_list
    
    
        
    df["RA"] = convert(df["RA"],True)
    df["Dec"] = convert(df["Dec"])
    
    nearby_star_info = df.to_string()
                       
    if verbose:
        print(nearby_star_info)

    return df
    


def get_exofop_info(TIC,verbose=False):
    
    url = "https://exofop.ipac.caltech.edu/tess/target.php?id=%s"%(TIC)
    response = br.open(url)
    result = response.read()
    soup = BeautifulSoup(result, 'html.parser')
    
    tds = soup.find_all("td",valign="top")
    
    left_panel = tds[0]
    main_panel = tds[1]
    
    tables = main_panel.find_all("table")
    
    Basic_Info = tables[0]
    
    Basic_Info_rows = Basic_Info.find_all("tr")
    
    Basic_Info_Title  = Basic_Info_rows[0]
    Basic_Info_Header = Basic_Info_rows[1]
    Basic_Info_Values = Basic_Info_rows[2]
    
    Header_Texts = Basic_Info_Header.find_all("th")
    Values_Texts = Basic_Info_Values.find_all("td")
    
    
    Basic_Information = {}
    # There's a bug in the proper motion.??
    
    if verbose:
        print("Basic Information:")
    
    for i in range(len(Header_Texts)):
        
        
        try:
            val = Values_Texts[i].text.replace('\xa0'," ").replace("\n","/").strip()
        except:
            val = Values_Texts[i].text.replace("\n","/").strip()
        
        if verbose:
            print("    ",Header_Texts[i].text,val)
        
        Basic_Information[Header_Texts[i].text] = val
    
    star_info = soup.find_all("table",{"class":"tfop striped"})
    try:
        star_params = []
        for i in star_info:
            if "Stellar Parameters" in i.text:
                star_params = i
        
        star_params_rows = star_params.find_all("tr")
        
        star_params_Title  = star_params_rows[0]
        star_params_Header = star_params_rows[1]
        star_params_Values = star_params_rows[2]
        
        star_Header_Texts = star_params_Header.find_all("th")
        star_Values_Texts = star_params_Values.find_all("td")
        
        Stellar_Information = {}
        
        # There's a bug in the proper motion.??
        
        if verbose:
            print("Stellar Information:")
        
        for i in range(len(star_Header_Texts)):
           
            try:
                val = star_Values_Texts[i].text.replace("\xa0"," ").replace("\n","/").strip()
            except:
                val = star_Values_Texts[i].text.replace("\n","/").strip()
            
            #if val != "":
            #    if verbose:
            #        print("    ",star_Header_Texts[i].text,val)
            
            #    print("    ",star_Header_Texts[i].text,val)
            Stellar_Information[star_Header_Texts[i].text] = val   
     
    except:
        Stellar_Information = {}
    
        Stellar_Information["Teff K"] = None
        Stellar_Information["log(g)"] = None
        Stellar_Information["Radius R_Sun"] = None
        Stellar_Information["Mass M_Sun"] = None
        Stellar_Information["Density g/cm3"] = None
        Stellar_Information["Luminosity L_Sun"] = None
        Stellar_Information["Distance pc"] = None

    
        
        
        if verbose:
            print("No Stellar Information")
    
        
    try:
        Magnitude_Information = {}
        mag_params = []
        for i in star_info:
            if "Magnitudes" in i.text:
                mag_params = i
        mag_params_rows = mag_params.find_all("tr")
        
        mag_params_Title  = mag_params_rows[0]
        mag_params_Header = mag_params_rows[1]
        
        for mag in mag_params_rows[2:]:
            mag_ = mag.text.split()
            if "WISE" in mag_:
                try:
                    errorbar = str(float("%.2e"%(float(mag_[5]))))
                    Magnitude_Information[" ".join(mag_[0:2])] = "".join([mag_[3],mag_[4],errorbar])
                except:
                    Magnitude_Information[" ".join(mag_[0:2])] = mag_[3]
            else:
                try:
                    errorbar = str(float("%.2e"%(float(mag_[3]))))
                    Magnitude_Information[mag_[0]] = "".join([mag_[1],mag_[2],errorbar])
                except:
                    Magnitude_Information[mag_[0]] = mag_[1]
            
    except:
        Magnitude_Information = {}
    
    
    return Basic_Information,Stellar_Information,Magnitude_Information




def gaia_query(ra, dec, radius=5):
    '''Return an ADQL query string for Gaia DR2 + geometric distances 
        from Bailer-Jones et al. 2018 '''
    
    radius = radius*u.arcsec

    coords = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='icrs')    
    query_string = '''SELECT *, DISTANCE(POINT('ICRS',g.ra, g.dec), 
                    POINT('ICRS', %s, %s)) as r
                    FROM gaiadr2.gaia_source AS g, external.gaiadr2_geometric_distance AS d
                    WHERE g.source_id = d.source_id AND CONTAINS(POINT('ICRS',g.ra, g.dec), 
                    CIRCLE('ICRS',%15.10f, %15.10f,%15.10f))=1 ORDER BY r ASC''' % \
        (ra, dec, ra, dec, radius.to(u.degree).value) 
    job = Gaia.launch_job(query_string, verbose=False)
    
    #print(job.get_results())
    return job.get_results()

def get_star_info(tic_id, ra=-1, dec=-1, radius=5):
    """
    function to get all the useful information we need for the star
    radius is in arcsecond
    """
    
    star_info = {}
    
    basic,star,magnitude = get_exofop_info(TIC=tic_id)
    
    exofop_name = basic["Star Name & Aliases"]
    
    
    if ra == -1 or dec == -1:
        radec = basic["RA/Dec (J2000)"].split("/")
        
        temp = []
        for i in radec:
            if i == "":
                continue
            if ":" in i:
                continue
            try:
                if "°".decode("utf-8") in i:
                    i = i.replace("°".decode("utf-8"),"")
            except:
                if "°" in i:
                    i = i.replace("°","")
            temp.append(float(i))
        ra,dec = temp
    
    
    try:
        a = gaia_query(ra,dec,radius)[0]
        gaia_pmra           = a["pmra"]
        gaia_pmdec          = a["pmdec"] 
        gaia_parallex       = a["parallax"]
        gaia_plex_err       = a["parallax_error"]
        gaia_temperature    = a["teff_val"]
        gaia_radius         = a["radius_val"]
        gaia_rmag           = a["phot_rp_mean_mag"]
        gaia_distance       = 1000./gaia_parallex
    
    except:
        gaia_pmra           = 0
        gaia_pmdec          = 0
        gaia_parallex       = 0
        gaia_plex_err       = 0
        gaia_temperature    = 0
        gaia_radius         = 0
        gaia_rmag           = 0
        gaia_distance       = 0
    
    
    exofop_name = basic["Star Name & Aliases"]
    
    star_temp = star["Teff K"]
    star_logg = star["log(g)"]
    star_radi = star["Radius R_Sun"]
    star_mass = star["Mass M_Sun"]
    star_denc = star["Density g/cm3"]
    star_lumi = star["Luminosity L_Sun"]
    star_dist = star["Distance pc"]
    
    
    star_info["Name"] = exofop_name
    
    
    if star_radi:
        star_info["Radius"] = star_radi
    elif "-" not in str(gaia_radius):
        star_info["Radius"] = gaia_radius
    else:
        star_info["Radius"] = 0
 
    if star_temp:
        star_info["Temperature"] = star_temp
    elif "-" not in str(gaia_temperature):
        star_info["Temperature"] = gaia_temperature
    else:
        star_info["Temperature"] = 0
        
    if star_mass:
        star_info["Mass"] = star_mass
    else:
        star_info["Mass"] = 0
    
    if magnitude != {}:
        star_info["Tmag"] = magnitude["TESS"]
    else:
        star_info["Tmag"] = 0
        
        
    
    star_info["Pmra"]       = gaia_pmra  #mas/yr
    star_info["Pmdec"]      = gaia_pmdec #mas/yr
    star_info["Parallax"]   = gaia_parallex #mas
    star_info["Distance"]   = gaia_distance
    star_info["Rmag"]       = gaia_rmag
    star_info["Ra"]         = ra
    star_info["Dec"]        = dec
    
    
    
     
    return star_info
    



if __name__ == "__main__":
    
    
    tic_id=65448527
    ra = 317.63150
    dec = -27.18311
    star_info = get_star_info(tic_id, ra, dec)
    
    
    #tic_id = 21535482
    star_info = get_star_info(tic_id)
    
    for key in star_info.keys():
        print(key,star_info[key])

    get_exofop_nearby(tic_id,True)















