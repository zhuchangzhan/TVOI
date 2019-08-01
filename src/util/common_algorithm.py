import numpy as np



def nearest_multiple(value,multiple,scale=1):
    """
    Return the nearest multiple of the value
    for nearest_multiple(129, 2, 1), the output is 128, or 2**7
    
    """
    
    orig = multiple
    comp_value = value*scale
    
    while True:
        
        if multiple > comp_value:
            break
        
        multiple*=orig
    
    if multiple/comp_value < comp_value/(multiple/orig):
        return multiple
    else:
        return multiple//orig

def pad_array_index(low,high,segment_length,reverse=False):
    """
    make it so that it's good for the bin to calculate local mean?
    """
    
    remainder = (segment_length-(high-low)%segment_length)
    if not reverse:
        return high + remainder
    else:
        return low - remainder



def segment_array(array=[],N=0,value=0):
    """
    segment the array into N segments. 
    append the last array with value to match same length
    """
    
    array = np.concatenate([array,np.ones(N-len(array)%N)*value])
    
    return array.reshape((-1,N))




