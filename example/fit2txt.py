# -*- encoding: utf-8 -*-
"""

convert the data into csv format that 

"""
import os,sys,glob
DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(DIR, '../..'))

from src.util.common_task import convert_fits_to_txt



#convert_fits_to_txt(4,1028594,"../../input/Sector4/tess2018292075959-s0004-0000000001028594-0124-s_lc.fits.gz")

#for TIC in [168847194,44743153,260160482,176831592,153708460,309146836,121078334,261136000,65347864,65448527]:
#    convert_fits_to_txt(4,TIC)

"""
for TIC in [469979014,471013508]:
    convert_fits_to_txt(1,TIC)
for TIC in [50380257,160081043]:
    convert_fits_to_txt(2,TIC)

for TIC in [146539195,77350951]:
    convert_fits_to_txt(5,TIC,savepath="output5")
for TIC in [25133286,60585499,153096543,272551828,445146149,
            471013582,96918621]:
    convert_fits_to_txt(6,TIC,savepath="output6")


#for TIC in [178366477,64260391]:
#    convert_fits_to_txt(7,TIC,savepath="output7")

for TIC in [141494666,356069146]:
    convert_fits_to_txt(8,TIC,savepath="output_saul")
for TIC in [141494666,356069146]:
    convert_fits_to_txt(9,TIC,savepath="output_saul")  
"""
"""
for TIC in [9560142, 25133286, 379121408]:
    convert_fits_to_txt(10,TIC,savepath="output_saul2")

"""
"""
for TIC in glob.glob("very_interesting/*.png"):
    #for TIC in [9560142, 25133286, 379121408]:
    print(TIC.split("/")[-1].split("_")[2],end=",")
"""
for TIC in [360205899,141978095,399421408,
            360148823,150166721,107548305,
            120472041,272551828,75934024,359892714,159864583,149160359,179966966]:
    convert_fits_to_txt(11,TIC,savepath="output_saul_11")




 
    
    