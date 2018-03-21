def get_scales(input_array, nscales, kernel):
    scales = []
    lowpass = input_array.astype('float32')
    for i in range(nscales):
        bandpass,lowpass = iterscale(lowpass,kernel,i)
        scales.append(bandpass)
    
    scales.append(lowpass)
    return scales

def get_bandpass(input_array, scale1, scale2, kernel, add_lowpass=False):
    lowpass = input_array.astype('float32')
    for i in range(scale1+1):
        output,lowpass = iterscale(lowpass,kernel,i)

    for i in range(scale1+1, scale2+1):
        bandpass,lowpass = iterscale(lowpass,kernel,i)
        output += bandpass
            
    if add_lowpass:
        output += lowpass

    return output

def get_lowpass(input_array, n_discarded, kernel):
    lowpass = input_array.astype('float32')
    for i in range(n_discarded):
        bandpass,lowpass = iterscale(lowpass,kernel,i)

    return lowpass

#Edge aware
def get_scales_ea(input_array, nscales, kernel, sigmar, alpha):
    scales = []
    max_input = int(input_array.max())
    sigmar = sigmar * max_input
    lowpass = input_array.astype('float32')
    for i in range(nscales):
        dmap = get_dmap(i,nscales,sigmar,alpha,max_input)
        bandpass,lowpass = iterscale_ea(lowpass,kernel,dmap,i)
        scales.append(bandpass)
    
    scales.append(lowpass)
    return scales

def get_bandpass_ea(input_array, scale1, scale2, kernel, sigmar, alpha, add_lowpass=False):
    max_input = int(input_array.max())
    sigmar = sigmar * max_input
    lowpass = input_array.astype('float32')
    for i in range(scale1+1):
        dmap = get_dmap(i,nscales,sigmar,alpha,max_input)
        output,lowpass = iterscale_ea(lowpass,kernel,dmap,i)

    for i in range(scale1+1, scale2+1):
        dmap = get_dmap(i,nscales,sigmar,alpha,max_input)
        bandpass,lowpass = iterscale_ea(lowpass,kernel,dmap,i)
        output += bandpass
            
    if add_lowpass:
        output += lowpass

    return output

def get_lowpass_ea(input_array, n_discarded, kernel, sigmar, alpha):
    max_input = int(input_array.max())
    sigmar = sigmar * max_input
    lowpass = input_array.astype('float32')
    for i in range(n_discarded):
        dmap = get_dmap(i,nscales,sigmar,alpha,max_input)
        bandpass,lowpass = iterscale_ea(lowpass,kernel,dmap,i)

    return lowpass

