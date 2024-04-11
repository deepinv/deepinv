import pytest
import torch
import deepinv as dinv
import itertools

# Generators to test (make sure they appear in find_generator as well)
GENERATORS = [
    "MotionBlurGenerator",
    "DiffractionBlurGenerator",
    "AccelerationMaskGenerator",
    "SigmaGenerator",
]

MIXTURES = list(itertools.combinations(GENERATORS, 2))
SIZES = [(5, 5), (6, 6)]
NUM_CHANNELS = [1, 3]


def find_generator(name, size, num_channels, device):
    r"""
    Chooses operator

    :param name: operator name
    :param device: (torch.device) cpu or cuda:0
    :return: (deepinv.physics.Physics) forward operator.
    """
    if name == "MotionBlurGenerator":
        g = dinv.physics.generator.MotionBlurGenerator(
            psf_size=size, num_channels=num_channels, device=device
        )
        keys = ["filter"]
    elif name == "DiffractionBlurGenerator":
        g = dinv.physics.generator.DiffractionBlurGenerator(
            psf_size=size,
            device=device,
            num_channels=num_channels,
        )
        keys = ["filter", "coeff", "pupil"]
    elif name == "AccelerationMaskGenerator":
        g = dinv.physics.generator.AccelerationMaskGenerator(
            img_size=size, device=device
        )
        keys = ["mask"]
    elif name == "SigmaGenerator":
        g = dinv.physics.generator.SigmaGenerator(device=device)
        keys = ["sigma"]
    else:
        raise Exception("The generator chosen doesn't exist")
    return g, size, keys


@pytest.mark.parametrize("name", GENERATORS)
@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("num_channels", NUM_CHANNELS)
def test_shape(name, size, num_channels, device):
    r"""
    Tests generators shape.
    """

    generator, size, keys = find_generator(name, size, num_channels, device)
    batch_size = 4

    params = generator.step(batch_size=batch_size)

    assert list(params.keys()) == keys

    if "filter" in params.keys():
        assert params["filter"].shape == (batch_size, num_channels, size[0], size[1])

    if "mask" in params.keys():
        assert params["mask"].shape == (batch_size, 2, size[0], size[1])


@pytest.mark.parametrize("name", GENERATORS)
def test_generation(name, device):
    r"""
    Tests generators shape.
    """
    size = (5, 5)
    generator, size, _ = find_generator(name, size, 1, device)
    batch_size = 1
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    params = generator.step(batch_size=batch_size)

    if name == "MotionBlurGenerator":
        w = params["filter"]
        if device.type == "cpu":
            wref = torch.tensor(
                [
                    [
                        [
                            [
                                0.0000000000,
                                0.0000000000,
                                0.0000000000,
                                0.0000000000,
                                0.0000000000,
                            ],
                            [
                                0.0000000000,
                                0.1509433985,
                                0.0000000000,
                                0.0000000000,
                                0.0000000000,
                            ],
                            [
                                0.0000000000,
                                0.3081761003,
                                0.1572327018,
                                0.3836477995,
                                0.0000000000,
                            ],
                            [
                                0.0000000000,
                                0.0000000000,
                                0.0000000000,
                                0.0000000000,
                                0.0000000000,
                            ],
                            [
                                0.0000000000,
                                0.0000000000,
                                0.0000000000,
                                0.0000000000,
                                0.0000000000,
                            ],
                        ]
                    ]
                ]
            )
        elif device.type == "cuda":
            wref = torch.tensor(
                [
                    [
                        [
                            [
                                0.0000000000,
                                0.0000000000,
                                0.0000000000,
                                0.0000000000,
                                0.0000000000,
                            ],
                            [
                                0.0000000000,
                                0.0691823885,
                                0.0628930852,
                                0.0000000000,
                                0.0000000000,
                            ],
                            [
                                0.0000000000,
                                0.0503144637,
                                0.4842767417,
                                0.0943396240,
                                0.0000000000,
                            ],
                            [
                                0.0000000000,
                                0.0000000000,
                                0.1069182381,
                                0.1320754737,
                                0.0000000000,
                            ],
                            [
                                0.0000000000,
                                0.0000000000,
                                0.0000000000,
                                0.0000000000,
                                0.0000000000,
                            ],
                        ]
                    ]
                ],
            ).to(device)

    elif name == "DiffractionBlurGenerator":
        w = params["filter"]
        if device.type == "cpu":
            wref = torch.tensor(
                [
                    [
                        [
                            [
                                0.0113882571,
                                0.0531018935,
                                0.0675100237,
                                0.0303402841,
                                0.0033624785,
                            ],
                            [
                                0.0285054874,
                                0.1004439145,
                                0.1303785592,
                                0.0716396421,
                                0.0116784973,
                            ],
                            [
                                0.0275844987,
                                0.0919832960,
                                0.1246952936,
                                0.0736453235,
                                0.0134703806,
                            ],
                            [
                                0.0105234124,
                                0.0374408774,
                                0.0568509996,
                                0.0335799791,
                                0.0042723534,
                            ],
                            [
                                0.0024160261,
                                0.0023811366,
                                0.0076419995,
                                0.0046625556,
                                0.0005027915,
                            ],
                        ]
                    ]
                ]
            )
        elif device.type == "cuda":
            wref = torch.tensor(
                [
                    [
                        [
                            [
                                0.0095238974,
                                0.0175499711,
                                0.0286177993,
                                0.0064900601,
                                0.0026435892,
                            ],
                            [
                                0.0238581896,
                                0.0537733063,
                                0.0513569079,
                                0.0185344294,
                                0.0124229826,
                            ],
                            [
                                0.0368810110,
                                0.0751009807,
                                0.0805081055,
                                0.0695058778,
                                0.0502106547,
                            ],
                            [
                                0.0210823547,
                                0.0472785048,
                                0.0740763769,
                                0.0966628939,
                                0.0694876462,
                            ],
                            [
                                0.0038343454,
                                0.0082935337,
                                0.0336939581,
                                0.0635016710,
                                0.0451108776,
                            ],
                        ]
                    ]
                ]
            ).to(device)
    elif name == "AccelerationMaskGenerator":
        w = params["mask"]
        if device.type == "cpu":
            wref = torch.tensor(
                [
                    [
                        [
                            [0.0, 0.0, 1.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0, 1.0, 0.0],
                        ],
                        [
                            [0.0, 0.0, 1.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0, 1.0, 0.0],
                        ],
                    ]
                ]
            )
        elif device.type == "cuda":
            wref = torch.tensor(
                [
                    [
                        [
                            [0.0, 0.0, 1.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0, 1.0, 0.0],
                        ],
                        [
                            [0.0, 0.0, 1.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0, 1.0, 0.0],
                        ],
                    ]
                ]
            ).to(device)

    elif name == "SigmaGenerator":
        w = params["sigma"]
        if device.type == "cpu":
            wref = torch.tensor([0.2531657219])
        elif device.type == "cuda":
            wref = torch.tensor([0.2055327892]).to(device)

    assert torch.allclose(w, wref, atol=1e-6)


# @pytest.mark.parametrize("name_tuple, size", list_names_shape_mix)
# def find_mixture(name_tuple, size, device):
#     r"""
#     Chooses operator

#     :param name: operator name
#     :param device: (torch.device) cpu or cuda:0
#     :return: (deepinv.physics.Physics) forward operator.
#     """
#     gen_list = []
#     for name in name_tuple:
#         if name == "MotionBlurGenerator":
#             gen_list.append(dinv.physics.MotionBlurGenerator(shape=size, device=device))
#         elif name == "DiffractionBlurGenerator":
#             gen_list.append(
#                 dinv.physics.DiffractionBlurGenerator(shape=size, device=device)
#             )
#         else:
#             raise Exception("The generator chosen doesn't exist")

#     gm = dinv.physics.GeneratorMixture(
#         gen_list, probs=[1.0 / len(gen_list) for _ in range(len(gen_list))]
#     )

#     return gm, size


# @pytest.mark.parametrize("name_tuple, size", list_names_shape_mix)
# def test_mixture_shape(name_tuple, size, device):
#     r"""
#     Tests generators shape.
#     """

#     torch.manual_seed(
#         0
#     )  ## for ('MotionBlurGenerator', 'DiffractionBlurGenerator') will select MotionBlurGenerator
#     generator, size = find_mixture(name_tuple, size, device)
#     batch_size = 4
#     w = generator.step(batch_size=batch_size)

#     if type(size) == int:
#         assert w.shape == (batch_size,) + (1, size, size)
#     elif type(size) == float:
#         assert w.shape == (batch_size,) + (1, int(size), int(size))
#     elif type(size) == tuple:
#         if len(size) == 1:
#             assert w.shape == (batch_size,) + (1, size[-1], size[-1])
#         elif len(size) == 2:
#             assert w.shape == (batch_size,) + (1, size[-2], size[-1])
#         elif len(size) == 3:
#             if name_tuple == ("MotionBlurGenerator", "DiffractionBlurGenerator"):
#                 assert w.shape == (batch_size,) + (
#                     1,
#                     size[-2],
#                     size[-1],
#                 )
#         elif len(size) == 4:
#             if name_tuple == ("MotionBlurGenerator", "DiffractionBlurGenerator"):
#                 assert w.shape == (batch_size,) + (
#                     1,
#                     size[-2],
#                     size[-1],
#                 )

#     torch.manual_seed(
#         1
#     )  ## for ('MotionBlurGenerator', 'DiffractionBlurGenerator') will select MotionBlurGenerator
#     generator, size = find_mixture(name_tuple, size, device)
#     batch_size = 4
#     w = generator.step(batch_size=batch_size)

#     if type(size) == int:
#         assert w.shape == (batch_size,) + (1, size, size)
#     elif type(size) == float:
#         assert w.shape == (batch_size,) + (1, int(size), int(size))
#     elif type(size) == tuple:
#         if len(size) == 1:
#             assert w.shape == (batch_size,) + (1, size[-1], size[-1])
#         elif len(size) == 2:
#             assert w.shape == (batch_size,) + (1, size[-2], size[-1])
#         elif len(size) == 3:
#             if name_tuple == ("MotionBlurGenerator", "DiffractionBlurGenerator"):
#                 assert w.shape == (batch_size,) + (
#                     size[-3],
#                     size[-2],
#                     size[-1],
#                 )
#         elif len(size) == 4:
#             if name_tuple == ("MotionBlurGenerator", "DiffractionBlurGenerator"):
#                 assert w.shape == (batch_size,) + (
#                     size[-3],
#                     size[-2],
#                     size[-1],
#                 )


# @pytest.mark.parametrize("name_tuple, size", list_names_shape_mix)
# def test_mixture_generation(name_tuple, size, device):
#     r"""
#     Tests generators shape.
#     """
#     torch.manual_seed(
#         0
#     )  ## for ('MotionBlurGenerator', 'DiffractionBlurGenerator') will select MotionBlurGenerator
#     generator, size = find_mixture(name_tuple, size, device)
#     batch_size = 1
#     w = generator.step(batch_size=batch_size)

#     if name_tuple == ("MotionBlurGenerator", "DiffractionBlurGenerator"):
#         wref = torch.Tensor(
#             [
#                 [
#                     [
#                         [
#                             [
#                                 0.0000000000,
#                                 0.0000000000,
#                                 0.0000000000,
#                                 0.0000000000,
#                                 0.0000000000,
#                             ],
#                             [
#                                 0.0000000000,
#                                 0.1572327018,
#                                 0.0000000000,
#                                 0.0000000000,
#                                 0.0000000000,
#                             ],
#                             [
#                                 0.0000000000,
#                                 0.1132075489,
#                                 0.1572327018,
#                                 0.3522012532,
#                                 0.0000000000,
#                             ],
#                             [
#                                 0.0000000000,
#                                 0.0000000000,
#                                 0.2201257795,
#                                 0.0000000000,
#                                 0.0000000000,
#                             ],
#                             [
#                                 0.0000000000,
#                                 0.0000000000,
#                                 0.0000000000,
#                                 0.0000000000,
#                                 0.0000000000,
#                             ],
#                         ]
#                     ]
#                 ]
#             ]
#         ).to(device)
#         assert torch.allclose(w, wref, atol=1e-10)

#     torch.manual_seed(
#         1
#     )  ## for ('MotionBlurGenerator', 'DiffractionBlurGenerator') will select DiffractionBlurGenerator
#     generator, size = find_mixture(name_tuple, size, device)
#     batch_size = 1
#     w = generator.step(batch_size=batch_size)

#     if name_tuple == ("MotionBlurGenerator", "DiffractionBlurGenerator"):
#         if type(size) == int:
#             wref = torch.Tensor(
#                 [
#                     [
#                         [
#                             [
#                                 0.0274918228,
#                                 0.0143814832,
#                                 0.0140662920,
#                                 0.0074974545,
#                                 0.0513366051,
#                             ],
#                             [
#                                 0.0181005131,
#                                 0.0066175554,
#                                 0.0059073456,
#                                 0.0011248180,
#                                 0.0413780659,
#                             ],
#                             [
#                                 0.1133465692,
#                                 0.0765140280,
#                                 0.0271076560,
#                                 0.0018974071,
#                                 0.0162934363,
#                             ],
#                             [
#                                 0.1901196241,
#                                 0.1253128946,
#                                 0.0343741514,
#                                 0.0022714571,
#                                 0.0034683293,
#                             ],
#                             [
#                                 0.1266585141,
#                                 0.0746551305,
#                                 0.0100943763,
#                                 0.0025477419,
#                                 0.0074367356,
#                             ],
#                         ]
#                     ]
#                 ]
#             ).to(device)

#         elif type(size) == float:
#             wref = torch.Tensor(
#                 [
#                     [
#                         [
#                             [
#                                 0.0274918228,
#                                 0.0143814832,
#                                 0.0140662920,
#                                 0.0074974545,
#                                 0.0513366051,
#                             ],
#                             [
#                                 0.0181005131,
#                                 0.0066175554,
#                                 0.0059073456,
#                                 0.0011248180,
#                                 0.0413780659,
#                             ],
#                             [
#                                 0.1133465692,
#                                 0.0765140280,
#                                 0.0271076560,
#                                 0.0018974071,
#                                 0.0162934363,
#                             ],
#                             [
#                                 0.1901196241,
#                                 0.1253128946,
#                                 0.0343741514,
#                                 0.0022714571,
#                                 0.0034683293,
#                             ],
#                             [
#                                 0.1266585141,
#                                 0.0746551305,
#                                 0.0100943763,
#                                 0.0025477419,
#                                 0.0074367356,
#                             ],
#                         ]
#                     ]
#                 ]
#             ).to(device)

#         elif type(size) == tuple:
#             if len(size) == 3:
#                 wref = torch.Tensor(
#                     [
#                         [
#                             [
#                                 [
#                                     0.0274918228,
#                                     0.0143814832,
#                                     0.0140662920,
#                                     0.0074974545,
#                                     0.0513366051,
#                                 ],
#                                 [
#                                     0.0181005131,
#                                     0.0066175554,
#                                     0.0059073456,
#                                     0.0011248180,
#                                     0.0413780659,
#                                 ],
#                                 [
#                                     0.1133465692,
#                                     0.0765140280,
#                                     0.0271076560,
#                                     0.0018974071,
#                                     0.0162934363,
#                                 ],
#                                 [
#                                     0.1901196241,
#                                     0.1253128946,
#                                     0.0343741514,
#                                     0.0022714571,
#                                     0.0034683293,
#                                 ],
#                                 [
#                                     0.1266585141,
#                                     0.0746551305,
#                                     0.0100943763,
#                                     0.0025477419,
#                                     0.0074367356,
#                                 ],
#                             ],
#                             [
#                                 [
#                                     0.0274918228,
#                                     0.0143814832,
#                                     0.0140662920,
#                                     0.0074974545,
#                                     0.0513366051,
#                                 ],
#                                 [
#                                     0.0181005131,
#                                     0.0066175554,
#                                     0.0059073456,
#                                     0.0011248180,
#                                     0.0413780659,
#                                 ],
#                                 [
#                                     0.1133465692,
#                                     0.0765140280,
#                                     0.0271076560,
#                                     0.0018974071,
#                                     0.0162934363,
#                                 ],
#                                 [
#                                     0.1901196241,
#                                     0.1253128946,
#                                     0.0343741514,
#                                     0.0022714571,
#                                     0.0034683293,
#                                 ],
#                                 [
#                                     0.1266585141,
#                                     0.0746551305,
#                                     0.0100943763,
#                                     0.0025477419,
#                                     0.0074367356,
#                                 ],
#                             ],
#                             [
#                                 [
#                                     0.0274918228,
#                                     0.0143814832,
#                                     0.0140662920,
#                                     0.0074974545,
#                                     0.0513366051,
#                                 ],
#                                 [
#                                     0.0181005131,
#                                     0.0066175554,
#                                     0.0059073456,
#                                     0.0011248180,
#                                     0.0413780659,
#                                 ],
#                                 [
#                                     0.1133465692,
#                                     0.0765140280,
#                                     0.0271076560,
#                                     0.0018974071,
#                                     0.0162934363,
#                                 ],
#                                 [
#                                     0.1901196241,
#                                     0.1253128946,
#                                     0.0343741514,
#                                     0.0022714571,
#                                     0.0034683293,
#                                 ],
#                                 [
#                                     0.1266585141,
#                                     0.0746551305,
#                                     0.0100943763,
#                                     0.0025477419,
#                                     0.0074367356,
#                                 ],
#                             ],
#                         ]
#                     ]
#                 ).to(device)

#             elif len(size) == 4:
#                 wref = torch.Tensor(
#                     [
#                         [
#                             [
#                                 [
#                                     0.0274918228,
#                                     0.0143814832,
#                                     0.0140662920,
#                                     0.0074974545,
#                                     0.0513366051,
#                                 ],
#                                 [
#                                     0.0181005131,
#                                     0.0066175554,
#                                     0.0059073456,
#                                     0.0011248180,
#                                     0.0413780659,
#                                 ],
#                                 [
#                                     0.1133465692,
#                                     0.0765140280,
#                                     0.0271076560,
#                                     0.0018974071,
#                                     0.0162934363,
#                                 ],
#                                 [
#                                     0.1901196241,
#                                     0.1253128946,
#                                     0.0343741514,
#                                     0.0022714571,
#                                     0.0034683293,
#                                 ],
#                                 [
#                                     0.1266585141,
#                                     0.0746551305,
#                                     0.0100943763,
#                                     0.0025477419,
#                                     0.0074367356,
#                                 ],
#                             ],
#                             [
#                                 [
#                                     0.0274918228,
#                                     0.0143814832,
#                                     0.0140662920,
#                                     0.0074974545,
#                                     0.0513366051,
#                                 ],
#                                 [
#                                     0.0181005131,
#                                     0.0066175554,
#                                     0.0059073456,
#                                     0.0011248180,
#                                     0.0413780659,
#                                 ],
#                                 [
#                                     0.1133465692,
#                                     0.0765140280,
#                                     0.0271076560,
#                                     0.0018974071,
#                                     0.0162934363,
#                                 ],
#                                 [
#                                     0.1901196241,
#                                     0.1253128946,
#                                     0.0343741514,
#                                     0.0022714571,
#                                     0.0034683293,
#                                 ],
#                                 [
#                                     0.1266585141,
#                                     0.0746551305,
#                                     0.0100943763,
#                                     0.0025477419,
#                                     0.0074367356,
#                                 ],
#                             ],
#                             [
#                                 [
#                                     0.0274918228,
#                                     0.0143814832,
#                                     0.0140662920,
#                                     0.0074974545,
#                                     0.0513366051,
#                                 ],
#                                 [
#                                     0.0181005131,
#                                     0.0066175554,
#                                     0.0059073456,
#                                     0.0011248180,
#                                     0.0413780659,
#                                 ],
#                                 [
#                                     0.1133465692,
#                                     0.0765140280,
#                                     0.0271076560,
#                                     0.0018974071,
#                                     0.0162934363,
#                                 ],
#                                 [
#                                     0.1901196241,
#                                     0.1253128946,
#                                     0.0343741514,
#                                     0.0022714571,
#                                     0.0034683293,
#                                 ],
#                                 [
#                                     0.1266585141,
#                                     0.0746551305,
#                                     0.0100943763,
#                                     0.0025477419,
#                                     0.0074367356,
#                                 ],
#                             ],
#                         ]
#                     ]
#                 ).to(device)

#             else:
#                 wref = torch.Tensor(
#                     [
#                         [
#                             [
#                                 [
#                                     0.0274918228,
#                                     0.0143814832,
#                                     0.0140662920,
#                                     0.0074974545,
#                                     0.0513366051,
#                                 ],
#                                 [
#                                     0.0181005131,
#                                     0.0066175554,
#                                     0.0059073456,
#                                     0.0011248180,
#                                     0.0413780659,
#                                 ],
#                                 [
#                                     0.1133465692,
#                                     0.0765140280,
#                                     0.0271076560,
#                                     0.0018974071,
#                                     0.0162934363,
#                                 ],
#                                 [
#                                     0.1901196241,
#                                     0.1253128946,
#                                     0.0343741514,
#                                     0.0022714571,
#                                     0.0034683293,
#                                 ],
#                                 [
#                                     0.1266585141,
#                                     0.0746551305,
#                                     0.0100943763,
#                                     0.0025477419,
#                                     0.0074367356,
#                                 ],
#                             ]
#                         ]
#                     ]
#                 ).to(device)

#         assert torch.allclose(w, wref, atol=1e-10)
