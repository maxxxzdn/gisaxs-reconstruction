import numpy as np
from json import loads

from dataset.sample import normalize


def compute_qz():
    angle_conv=np.arctan(50e-6/1.277)*180/np.pi
    y_vec2=(np.arange(1024)-(1024-611))*angle_conv+0.64
    y_vec_lin2=y_vec2[::-1]
    return 2*np.pi/(1.4073e-10)*(np.sin(0.64*np.pi/180)+np.sin(y_vec_lin2*np.pi/180))*1e-9

def minmax(x):
    a = x.min()
    b = x.max()
    return (x - a)/(b-a)

def compile_exp(x_exp, qz = None):
    if qz is None:
        qz = compute_qz()    
    a = (x_exp[341:560,200:230], x_exp[660:800,200:230])         
    
    f_left = (1/3*(qz[range(341,560)]-0.99)**-1.4)**-1
    f_right = (1.*(0.99-qz[range(660,800)])**-1.1)**-1
    
    profile_exp = (minmax(np.mean(a[0], 1)*f_left),
                   minmax(np.mean(a[1], 1)*f_right))
    
    return profile_exp

def get_lisa_fit():
    with open('dataset/config.json') as f:
        config = loads(f.read())
    param_names = ['delta', 'beta', 'thickness', 'roughness', 'hurst', 'corrlen']

    params_normalized = []
    for layer_id in range(1,13):
        for p_idx in range(6):
            param = config[param_names[p_idx]+'_default'][config['sample_layers'][layer_id]]
            params_normalized.append(normalize(param, config, layer_id, param_names[p_idx]))
    return np.array(params_normalized).reshape(12,6)