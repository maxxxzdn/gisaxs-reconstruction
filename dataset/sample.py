from random import uniform
from warnings import warn
from numpy import array, zeros, save
import bornagain as ba

from .simulation import get_simulation

class Sample():
    def __init__(self, config, mode, n_layers = 12):
        self.config = config
        self.mode = mode
        self.p_names = ['delta', 'beta', 'thickness', 'roughness', 'hurst', 'corrlen']
        self.multilayer, self.params, self.names = create_multilayer(config, mode, n_layers)
        self.params_norm = self.normalize_params()
        self.simres = self.simulate()
        
    def simulate(self):
        sample = self.multilayer
        simulation = get_simulation(detector=None)
        simulation.setSample(sample)
        simulation.runSimulation()
        return simulation.result().array()
    
    def normalize_params(self):
        y = array(self.params).reshape(-1,6)
        params_norm = zeros(y.shape)
        for i in range(len(y)):
            params_norm[i] = array([normalize(p, self.config, i+1, self.p_names[ind])  for ind, p in enumerate(y[i])])
        return params_norm.reshape(-1)
    
    def save(self, path, data_id):
        x = self.simres
        y = array(self.params).reshape(-1)
        y_norm = self.params_norm
        save(path + '/data_' + str(data_id), (x,y,y_norm))
    
def get_material(config, layer_id, mode):
    name = config['sample_layers'][layer_id]
    mode = mode_check(name, mode)
    if mode == 'default':
        delta = config['delta_default'][name]
        beta = config['beta_default'][name]
    elif mode == 'range':  
        delta = uniform(*config['delta_range'][name])
        beta = uniform(config['beta_range'][name][0], min(delta, config['beta_range'][name][1]))
    if name not in ['Air', 'SiO2', 'Substrate']:
        assert delta > beta, f'delta: {delta} < beta: {beta}'
    return ba.HomogeneousMaterial(name, delta, beta), delta, beta

def get_thickness(config, layer_id, mode):
    name = config['sample_layers'][layer_id]
    mode = mode_check(name, mode)
    if mode == 'default':
        thickness = config['thickness_default'][name]
    elif mode == 'range':  
        thickness = uniform(*config['thickness_range'][name])
    return thickness

def get_layer(config, layer_id, mode):
    material, delta, beta = get_material(config, layer_id, mode)
    if layer_id == 0 or layer_id == 14:
        return ba.Layer(material), delta, beta, 0.
    else:
        thickness = get_thickness(config, layer_id, mode)
        return ba.Layer(material, thickness), delta, beta, thickness
    
def get_roughness(config, layer_id, mode, thickness = None):
    name = config['sample_layers'][layer_id]
    assert name != 'Air', f'{name} layer does not have roughness'
    mode = mode_check(name, mode)
    if mode == 'default':
        roughness = config['roughness_default'][name]
        hurst = config['hurst_default'][name]
        corrlen = config['corrlen_default'][name]
    elif mode == 'range':
        if name in ['Air', 'SiO2', 'Substrate']:
            roughness = uniform(*config['roughness_range'][name])
        else:
            roughness = uniform(config['roughness_range'][name][0], min(config['roughness_range'][name][1], thickness))
        hurst = uniform(*config['hurst_range'][name])
        corrlen = uniform(*config['corrlen_range'][name])
    if name not in ['Air', 'SiO2', 'Substrate']:    
        assert thickness > roughness, f'thickness: {thickness} < roughness: {roughness}'
    return ba.LayerRoughness(roughness, hurst, corrlen * ba.nm), roughness, hurst, corrlen

def create_multilayer(config, mode, n_layers):
    params = []
    names = []
    multiLayer = ba.MultiLayer()
    multiLayer.setCrossCorrLength(2000)
    
    layer, _, _, _ = get_layer(config, 0, mode)
    multiLayer.addLayer(layer)
    
    for layer_id in range(1, 15):
        if layer_id <= n_layers or layer_id > 12:
            name = config['sample_layers'][layer_id]
            layer, delta, beta, thickness = get_layer(config, layer_id, mode)
            layer_roughness, roughness, hurst, corrlen = get_roughness(config, layer_id, mode, thickness)
            multiLayer.addLayerWithTopRoughness(layer, layer_roughness)
            if name not in ['Air', 'SiO2', 'Substrate']: 
                names.append(name)
                params.append([delta, beta, thickness, roughness, hurst, corrlen])
        
    return multiLayer, params, names

def mode_check(name, mode):
    if name in ['Air', 'SiO2', 'Substrate']:
        if mode != 'default':
            warn(f'no range is given for the material {name}, use default mode instead')
        mode = 'default'
    return mode

def normalize(x, config, layer_id, parameter):
    name = config['sample_layers'][layer_id]
    a, b = config[f'{parameter}_range'][name]
    if a != b:
        return (x-a)/(b-a)
    else:
        return 1.