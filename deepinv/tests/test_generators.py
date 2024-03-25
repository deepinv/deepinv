import pytest
import torch
import numpy as np
import deepinv as dinv
import itertools

# Generators to test (make sure they appear in find_generator as well)
GENERATORS = [
    "MotionBlurGenerator",
    "DiffractionBlurGenerator"
]

MIXTURES = list(itertools.combinations(GENERATORS, 2))

SIZES = [
    5,
    (5,),
    (5, 5),
    (3, 5, 5),
    (2, 3, 5, 5)
]


list_2d = [[(GENERATORS[i], SIZES[j]) for j in range(len(SIZES)) ] for i in range(len(GENERATORS))]
list_names_shape = [item for row in list_2d for item in row]

list_2d = [[(MIXTURES[i], SIZES[j]) for j in range(len(SIZES)) ] for i in range(len(MIXTURES))]
list_names_shape_mix = [item for row in list_2d for item in row]

## I want to test that the shapes work properly for all user input

def find_generator(name, filter_shape, device):
    r"""
    Chooses operator

    :param name: operator name
    :param device: (torch.device) cpu or cuda
    :return: (deepinv.physics.Physics) forward operator.
    """
    if name == "MotionBlurGenerator":
        g = dinv.physics.MotionBlurGenerator(shape=filter_shape, device=device)
    elif name == "DiffractionBlurGenerator":
        g = dinv.physics.DiffractionBlurGenerator(shape=filter_shape, device=device)
    else:
        raise Exception("The generator chosen doesn't exist")
    return g, filter_shape



@pytest.mark.parametrize("name, filter_shape", list_names_shape)
def test_shape(name, filter_shape, device):
    r"""
    Tests generators shape.
    """
    
    generator, filter_shape = find_generator(name, filter_shape, device)
    
    batch_size = 4
    
    w = generator.step(batch_size=batch_size)
    
    if type(filter_shape) == int:
        assert w.shape == (batch_size, ) + (1, filter_shape, filter_shape)
    elif type(filter_shape) == float:
        assert w.shape == (batch_size, ) + (1, int(filter_shape), int(filter_shape))
    elif type(filter_shape) == tuple:
        if len(filter_shape) == 1 :
            assert w.shape == (batch_size, ) + (1, filter_shape[-1], filter_shape[-1])
        elif len(filter_shape) == 2:
            assert w.shape == (batch_size, ) + (1, filter_shape[-2], filter_shape[-1])
        elif len(filter_shape) == 3:
            if name == 'MotionBlurGenerator':
                assert w.shape == (batch_size, ) + (1, filter_shape[-2], filter_shape[-1])
            elif name == 'DiffractionBlurGenerator':
                assert w.shape == (batch_size, ) + (filter_shape[-3], filter_shape[-2], filter_shape[-1])
        elif len(filter_shape) == 4:
            if name == 'MotionBlurGenerator':
                assert w.shape == (batch_size, ) + (1, filter_shape[-2], filter_shape[-1])
            elif name == 'DiffractionBlurGenerator':
                assert w.shape == (batch_size, ) + (filter_shape[-3], filter_shape[-2], filter_shape[-1])
                
              
@pytest.mark.parametrize("name, filter_shape", list_names_shape)
def test_generation(name, filter_shape, device):
    r"""
    Tests generators shape.
    """
    torch.manual_seed(0)
    generator, filter_shape = find_generator(name, filter_shape, device)
    batch_size = 1
    w = generator.step(batch_size=batch_size)

    if name == 'MotionBlurGenerator':
        wref = torch.Tensor([[[[0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000],
          [0.0000000000, 0.1509433985, 0.0000000000, 0.0000000000, 0.0000000000],
          [0.0000000000, 0.3081761003, 0.1572327018, 0.3836477995, 0.0000000000],
          [0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000],
          [0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000]]]]).to(device)
        assert torch.allclose(w, wref, atol=1e-10)
        
    elif name == 'DiffractionBlurGenerator':
        wref = torch.Tensor([[[[0.0204578601, 0.0265045520, 0.0261169113, 0.0289420150, 0.0199008565],
                  [0.0299979988, 0.0025556409, 0.0085813133, 0.0182024427, 0.0022124904],
                  [0.0489049256, 0.0196437053, 0.0345879272, 0.0402816199, 0.0308128875],
                  [0.0542067587, 0.0382712111, 0.0583972223, 0.0815427601, 0.1076817438],
                  [0.0292691737, 0.0180015489, 0.0343270153, 0.0808936730, 0.1397057325]]]]).to(device)
        assert torch.allclose(w, wref, atol=1e-10)           
                


@pytest.mark.parametrize("name_tuple, filter_shape", list_names_shape_mix)
def find_mixture(name_tuple, filter_shape, device):
    r"""
    Chooses operator

    :param name: operator name
    :param device: (torch.device) cpu or cuda
    :return: (deepinv.physics.Physics) forward operator.
    """
    gen_list = []
    for name in name_tuple:
    
        if name == "MotionBlurGenerator":
            gen_list.append( dinv.physics.MotionBlurGenerator(shape=filter_shape, device=device))
        elif name == "DiffractionBlurGenerator":
            gen_list.append(dinv.physics.DiffractionBlurGenerator(shape=filter_shape, device=device))
        else:
            raise Exception("The generator chosen doesn't exist")
        
    gm = dinv.physics.GeneratorMixture(gen_list, probs=[1. / len(gen_list) for _ in range(len(gen_list))])
        
    return gm, filter_shape



@pytest.mark.parametrize("name_tuple, filter_shape", list_names_shape_mix)
def test_mixture_shape(name_tuple, filter_shape, device):
    r"""
    Tests generators shape.
    """
    
    torch.manual_seed(0) ## for ('MotionBlurGenerator', 'DiffractionBlurGenerator') will select MotionBlurGenerator
    generator, filter_shape = find_mixture(name_tuple, filter_shape, device)
    batch_size = 4
    w = generator.step(batch_size=batch_size)
    
    if type(filter_shape) == int:
        assert w.shape == (batch_size, ) + (1, filter_shape, filter_shape)
    elif type(filter_shape) == float:
        assert w.shape == (batch_size, ) + (1, int(filter_shape), int(filter_shape))
    elif type(filter_shape) == tuple:
        if len(filter_shape) == 1 :
            assert w.shape == (batch_size, ) + (1, filter_shape[-1], filter_shape[-1])
        elif len(filter_shape) == 2:
            assert w.shape == (batch_size, ) + (1, filter_shape[-2], filter_shape[-1])
        elif len(filter_shape) == 3:
            if name_tuple == ('MotionBlurGenerator', 'DiffractionBlurGenerator'):
                assert w.shape == (batch_size, ) + (1, filter_shape[-2], filter_shape[-1])
        elif len(filter_shape) == 4:
            if name_tuple == ('MotionBlurGenerator', 'DiffractionBlurGenerator'):
                assert w.shape == (batch_size, ) + (1, filter_shape[-2], filter_shape[-1])
                
                
    torch.manual_seed(1) ## for ('MotionBlurGenerator', 'DiffractionBlurGenerator') will select MotionBlurGenerator
    generator, filter_shape = find_mixture(name_tuple, filter_shape, device)
    batch_size = 4
    w = generator.step(batch_size=batch_size)
    
    if type(filter_shape) == int:
        assert w.shape == (batch_size, ) + (1, filter_shape, filter_shape)
    elif type(filter_shape) == float:
        assert w.shape == (batch_size, ) + (1, int(filter_shape), int(filter_shape))
    elif type(filter_shape) == tuple:
        if len(filter_shape) == 1 :
            assert w.shape == (batch_size, ) + (1, filter_shape[-1], filter_shape[-1])
        elif len(filter_shape) == 2:
            assert w.shape == (batch_size, ) + (1, filter_shape[-2], filter_shape[-1])
        elif len(filter_shape) == 3:
            if name_tuple == ('MotionBlurGenerator', 'DiffractionBlurGenerator'):
                assert w.shape == (batch_size, ) + (filter_shape[-3], filter_shape[-2], filter_shape[-1])
        elif len(filter_shape) == 4:
            if name_tuple == ('MotionBlurGenerator', 'DiffractionBlurGenerator'):
                assert w.shape == (batch_size, ) + (filter_shape[-3], filter_shape[-2], filter_shape[-1])
    
    



@pytest.mark.parametrize("name_tuple, filter_shape", list_names_shape_mix)
def test_mixture_generation(name_tuple, filter_shape, device):
    r"""
    Tests generators shape.
    """
    torch.manual_seed(0) ## for ('MotionBlurGenerator', 'DiffractionBlurGenerator') will select MotionBlurGenerator
    generator, filter_shape = find_mixture(name_tuple, filter_shape, device)
    batch_size = 1
    w = generator.step(batch_size=batch_size)

    if name_tuple == ('MotionBlurGenerator', 'DiffractionBlurGenerator'):
        wref = torch.Tensor([[[[[0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000],
                  [0.0000000000, 0.1572327018, 0.0000000000, 0.0000000000, 0.0000000000],
                  [0.0000000000, 0.1132075489, 0.1572327018, 0.3522012532, 0.0000000000],
                  [0.0000000000, 0.0000000000, 0.2201257795, 0.0000000000, 0.0000000000],
                  [0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000]]]]]).to(device)
        assert torch.allclose(w, wref, atol=1e-10)
        

    torch.manual_seed(1) ## for ('MotionBlurGenerator', 'DiffractionBlurGenerator') will select DiffractionBlurGenerator
    generator, filter_shape = find_mixture(name_tuple, filter_shape, device)
    batch_size = 1
    w = generator.step(batch_size=batch_size)

    if name_tuple == ('MotionBlurGenerator', 'DiffractionBlurGenerator'):
        if type(filter_shape) == int:
            
            wref = torch.Tensor([[[[0.0274918228, 0.0143814832, 0.0140662920, 0.0074974545, 0.0513366051],
                  [0.0181005131, 0.0066175554, 0.0059073456, 0.0011248180, 0.0413780659],
                  [0.1133465692, 0.0765140280, 0.0271076560, 0.0018974071, 0.0162934363],
                  [0.1901196241, 0.1253128946, 0.0343741514, 0.0022714571, 0.0034683293],
                  [0.1266585141, 0.0746551305, 0.0100943763, 0.0025477419, 0.0074367356]]]]).to(device)
        
        elif type(filter_shape) == float:
              
            wref = torch.Tensor([[[[0.0274918228, 0.0143814832, 0.0140662920, 0.0074974545, 0.0513366051],
                  [0.0181005131, 0.0066175554, 0.0059073456, 0.0011248180, 0.0413780659],
                  [0.1133465692, 0.0765140280, 0.0271076560, 0.0018974071, 0.0162934363],
                  [0.1901196241, 0.1253128946, 0.0343741514, 0.0022714571, 0.0034683293],
                  [0.1266585141, 0.0746551305, 0.0100943763, 0.0025477419, 0.0074367356]]]]).to(device)     
        
        elif type(filter_shape) == tuple:
            if len(filter_shape) == 3:
                wref = torch.Tensor([[[[0.0274918228, 0.0143814832, 0.0140662920, 0.0074974545, 0.0513366051],
                  [0.0181005131, 0.0066175554, 0.0059073456, 0.0011248180, 0.0413780659],
                  [0.1133465692, 0.0765140280, 0.0271076560, 0.0018974071, 0.0162934363],
                  [0.1901196241, 0.1253128946, 0.0343741514, 0.0022714571, 0.0034683293],
                  [0.1266585141, 0.0746551305, 0.0100943763, 0.0025477419, 0.0074367356]],

                  [[0.0274918228, 0.0143814832, 0.0140662920, 0.0074974545, 0.0513366051],
                  [0.0181005131, 0.0066175554, 0.0059073456, 0.0011248180, 0.0413780659],
                  [0.1133465692, 0.0765140280, 0.0271076560, 0.0018974071, 0.0162934363],
                  [0.1901196241, 0.1253128946, 0.0343741514, 0.0022714571, 0.0034683293],
                  [0.1266585141, 0.0746551305, 0.0100943763, 0.0025477419, 0.0074367356]],

                  [[0.0274918228, 0.0143814832, 0.0140662920, 0.0074974545, 0.0513366051],
                  [0.0181005131, 0.0066175554, 0.0059073456, 0.0011248180, 0.0413780659],
                  [0.1133465692, 0.0765140280, 0.0271076560, 0.0018974071, 0.0162934363],
                  [0.1901196241, 0.1253128946, 0.0343741514, 0.0022714571, 0.0034683293],
                  [0.1266585141, 0.0746551305, 0.0100943763, 0.0025477419, 0.0074367356]]]]).to(device)
                 
            elif len(filter_shape) == 4:
                wref = torch.Tensor([[[[0.0274918228, 0.0143814832, 0.0140662920, 0.0074974545, 0.0513366051],
                  [0.0181005131, 0.0066175554, 0.0059073456, 0.0011248180, 0.0413780659],
                  [0.1133465692, 0.0765140280, 0.0271076560, 0.0018974071, 0.0162934363],
                  [0.1901196241, 0.1253128946, 0.0343741514, 0.0022714571, 0.0034683293],
                  [0.1266585141, 0.0746551305, 0.0100943763, 0.0025477419, 0.0074367356]],

                  [[0.0274918228, 0.0143814832, 0.0140662920, 0.0074974545, 0.0513366051],
                  [0.0181005131, 0.0066175554, 0.0059073456, 0.0011248180, 0.0413780659],
                  [0.1133465692, 0.0765140280, 0.0271076560, 0.0018974071, 0.0162934363],
                  [0.1901196241, 0.1253128946, 0.0343741514, 0.0022714571, 0.0034683293],
                  [0.1266585141, 0.0746551305, 0.0100943763, 0.0025477419, 0.0074367356]],

                  [[0.0274918228, 0.0143814832, 0.0140662920, 0.0074974545, 0.0513366051],
                  [0.0181005131, 0.0066175554, 0.0059073456, 0.0011248180, 0.0413780659],
                  [0.1133465692, 0.0765140280, 0.0271076560, 0.0018974071, 0.0162934363],
                  [0.1901196241, 0.1253128946, 0.0343741514, 0.0022714571, 0.0034683293],
                  [0.1266585141, 0.0746551305, 0.0100943763, 0.0025477419, 0.0074367356]]]]).to(device)
                
            else:
                wref = torch.Tensor([[[[0.0274918228, 0.0143814832, 0.0140662920, 0.0074974545, 0.0513366051],
                          [0.0181005131, 0.0066175554, 0.0059073456, 0.0011248180, 0.0413780659],
                          [0.1133465692, 0.0765140280, 0.0271076560, 0.0018974071, 0.0162934363],
                          [0.1901196241, 0.1253128946, 0.0343741514, 0.0022714571, 0.0034683293],
                          [0.1266585141, 0.0746551305, 0.0100943763, 0.0025477419, 0.0074367356]]]]).to(device)  
        

        assert torch.allclose(w, wref, atol=1e-10)  
    
    
                



