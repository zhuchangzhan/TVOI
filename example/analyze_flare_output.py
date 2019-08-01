"""
Tasks related to flares

1. download all flares (done)
2. compare flares with old  (hmmmmm)
    a.move the difference to seperate folder for manual inspection
    b.compare the number of flares caught
    
    Sector 1: 
    new, old: 867 968
    same:721, old_only:146,new_old:247
    
    Sector 2: 
    new, old: 884 969
    same:736, old_only:148,new_old:233
    
3. injection test
4. optimize

"""
import os

import glob
from shutil import copyfile

sector = 2


oldpath = "/Users/azariven/Dropbox (MIT)/TESS Data Report/Sector%s/old/output_sec%s_flare/*.png"%(sector,sector)
old_TIC = []
old_paths = glob.glob(oldpath)
for path in old_paths:
    old_TIC.append(path.split("_")[-2])
    

newpath = "/Users/azariven/New_Workspace/TESS/output/Sector%s/new_flares/*.png"%sector
new_TIC = []
new_paths = glob.glob(newpath)
for path in new_paths:
    new_TIC.append(path.split("_")[-2])

print(len(set(old_TIC)),len(set(new_TIC)))

overlap = set(old_TIC) & set(new_TIC)
old = set(old_TIC) - set(new_TIC)
new = set(new_TIC) - set(old_TIC)

print(len(overlap),len(old),len(new))


if not os.path.isdir("sector_%s_old"%sector):
    os.makedirs("sector_%s_old"%sector)  
if not os.path.isdir("sector_%s_new"%sector):
    os.makedirs("sector_%s_new"%sector)  
if not os.path.isdir("sector_%s_overlap"%sector):
    os.makedirs("sector_%s_overlap"%sector)


for path in old_paths:
    old_TIC_ID = path.split("_")[-2]
    filename = path.split("/")[-1]
    if old_TIC_ID in old:
        copyfile(path,"sector_%s_old/%s"%(sector,filename))
        
for path in new_paths:
    new_TIC_ID = path.split("_")[-2]
    filename = path.split("/")[-1]
    if new_TIC_ID in new:
        copyfile(path,"sector_%s_new/%s"%(sector,filename))

for path in new_paths:
    new_TIC_ID = path.split("_")[-2]
    filename = path.split("/")[-1]
    if new_TIC_ID in overlap:
        copyfile(path,"sector_%s_overlap/%s"%(sector,filename))


print("done")