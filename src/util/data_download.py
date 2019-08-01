"""

Figure out various ways to download data

"""

def download_interesting(TIC_IDs,filepath="/pdo/spoc-data/sector-09/light-curve/"):
    
    center = ",".join(["*%s*"%x for x in TIC_IDs])
    
    text = "scp zzhan@pdo:%s\{%s\} ."%(filepath,center)
    
    return text 


if __name__ == "__main__":
    print(download_interesting([279955276],"/pdo/spoc-data/sector-09/light-curve/"))