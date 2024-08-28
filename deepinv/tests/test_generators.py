from deepinv.physics.generator import (
    GaussianMaskGenerator,
    EquispacedMaskGenerator,
    RandomMaskGenerator,
)
import pytest
import numpy as np
import torch
import deepinv as dinv
import itertools

# Avoiding nondeterministic algorithms
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True

# Generators to test (make sure they appear in find_generator as well)
GENERATORS = [
    "MotionBlurGenerator",
    "DiffractionBlurGenerator",
    "ProductConvolutionBlurGenerator",
    "SigmaGenerator",
]

MIXTURES = list(itertools.combinations(GENERATORS, 2))
SIZES = [(5, 5), (6, 6)]
NUM_CHANNELS = [1, 3]


# MRI Generators
C, T, H, W = 2, 12, 256, 512
MRI_GENERATORS = ["gaussian", "random", "uniform"]
MRI_IMG_SIZES = [(H, W), (C, H, W), (C, T, H, W), (64, 64)]
MRI_ACCELERATIONS = [4, 10, 12]
MRI_CENTER_FRACTIONS = [0, 0.04, 24 / 512]

# Inpainting/Splitting Generators
INPAINTING_IMG_SIZES = [
    (2, 64, 40),
    (2, 1000),
    (2, 3, 64, 40),
]  # (C,H,W), (C,M), (C,T,H,W)
INPAINTING_GENERATORS = ["bernoulli", "gaussian"]

# All devices to test
DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")


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
    elif name == "ProductConvolutionBlurGenerator":
        g = dinv.physics.generator.ProductConvolutionBlurGenerator(
            psf_generator=dinv.physics.generator.DiffractionBlurGenerator(
                psf_size=size,
                device=device,
                num_channels=num_channels,
            ),
            img_size=512,
            n_eigen_psf=10,
            device=device,
        )
        keys = ["filters", "multipliers", "padding"]
    elif name == "SigmaGenerator":
        g = dinv.physics.generator.SigmaGenerator(device=device)
        keys = ["sigma"]
    else:
        raise Exception("The generator chosen doesn't exist")
    return g, size, keys


@pytest.mark.parametrize("name", GENERATORS)
@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("num_channels", NUM_CHANNELS)
@pytest.mark.parametrize("device", DEVICES)
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


@pytest.mark.parametrize("name", GENERATORS)
@pytest.mark.parametrize("device", DEVICES)
def test_generation_newparams(name, device):
    r"""
    Tests generators shape.
    """
    size = (32, 32)
    generator, size, _ = find_generator(name, size, 1, device)
    batch_size = 1

    if name == "MotionBlurGenerator":
        param_key = ["filter"]
    elif name == "DiffractionBlurGenerator":
        param_key = ["filter"]
    elif name == "ProductConvolutionBlurGenerator":
        param_key = ["filters", "multipliers"]
    elif name == "SigmaGenerator":
        param_key = ["sigma"]

    params0 = generator.step(batch_size=batch_size, seed=0)
    params1 = generator.step(batch_size=batch_size, seed=1)

    for key in param_key:
        assert torch.any(params0[key] != params1[key])


@pytest.mark.parametrize("name", GENERATORS)
@pytest.mark.parametrize("device", DEVICES)
def test_generation_seed(name, device):
    r"""
    Tests generators shape.
    """
    size = (32, 32)
    generator, size, _ = find_generator(name, size, 1, device)
    batch_size = 1

    if name == "MotionBlurGenerator":
        param_key = ["filter"]
    elif name == "DiffractionBlurGenerator":
        param_key = ["filter"]
    elif name == "ProductConvolutionBlurGenerator":
        param_key = ["filters", "multipliers"]
    elif name == "SigmaGenerator":
        param_key = ["sigma"]

    params0 = generator.step(batch_size=batch_size, seed=42)
    params1 = generator.step(batch_size=batch_size, seed=42)

    for key in param_key:
        assert torch.allclose(params0[key], params1[key])


@pytest.mark.parametrize("name", GENERATORS)
@pytest.mark.parametrize("device", DEVICES)
def test_generation(name, device):
    r"""
    Tests generators shape.
    """
    size = (5, 5)
    generator, size, _ = find_generator(name, size, 1, device)
    batch_size = 1
    params = generator.step(batch_size=batch_size, seed=0)

    if name == "MotionBlurGenerator":
        atol = 1e-6
        w = params["filter"]
        if torch.device(device) == torch.device("cpu"):
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
        elif torch.device(device) == torch.device("cuda"):
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
        atol = 1e-6
        w = params["filter"]
        if torch.device(device) == torch.device("cpu"):
            wref = torch.tensor(
                [
                    [
                        [
                            [
                                0.0081667975,
                                0.0339039154,
                                0.0463643819,
                                0.0238370951,
                                0.0067134043,
                            ],
                            [
                                0.0235104840,
                                0.0769919083,
                                0.1068268567,
                                0.0638824701,
                                0.0154726375,
                            ],
                            [
                                0.0315882340,
                                0.0922789276,
                                0.1303794235,
                                0.0824520364,
                                0.0192645062,
                            ],
                            [
                                0.0210407600,
                                0.0526346825,
                                0.0764168128,
                                0.0469734631,
                                0.0075026853,
                            ],
                            [
                                0.0074746846,
                                0.0069856029,
                                0.0118351672,
                                0.0068998244,
                                0.0006031910,
                            ],
                        ]
                    ]
                ]
            )
        elif torch.device(device) == torch.device("cuda"):
            wref = torch.tensor(
                [
                    [
                        [
                            [
                                0.0032691115,
                                0.0060402630,
                                0.0175693501,
                                0.0059448336,
                                0.0007023035,
                            ],
                            [
                                0.0095862420,
                                0.0427204743,
                                0.0728377998,
                                0.0452912413,
                                0.0118838884,
                            ],
                            [
                                0.0251656137,
                                0.0764198229,
                                0.1213025227,
                                0.0953904763,
                                0.0436460413,
                            ],
                            [
                                0.0162273813,
                                0.0507672094,
                                0.0916804373,
                                0.0891042799,
                                0.0563216321,
                            ],
                            [
                                0.0018812985,
                                0.0099059129,
                                0.0294828881,
                                0.0403944701,
                                0.0364645906,
                            ],
                        ]
                    ]
                ]
            ).to(device)

    elif name == "ProductConvolutionBlurGenerator":
        w = params["filters"]
        # atol = 1e-5
        atol = 1e-2
        if torch.device(device) == torch.device("cpu"):
            wref = torch.tensor(
                [
                    [
                        [
                            [
                                [
                                    -0.0343423001,
                                    0.0499860235,
                                    0.0835180804,
                                    0.3166051805,
                                    -0.0027029207,
                                ],
                                [
                                    -0.0900977850,
                                    0.0646565780,
                                    0.3174273968,
                                    0.4801940620,
                                    -0.0544758141,
                                ],
                                [
                                    -0.0746181682,
                                    0.2372791916,
                                    0.0192378685,
                                    0.4329231083,
                                    -0.2117855251,
                                ],
                                [
                                    -0.2070546299,
                                    -0.1773867756,
                                    0.0963601321,
                                    0.0365741029,
                                    0.3449944556,
                                ],
                                [
                                    -0.1150196120,
                                    0.3202073872,
                                    -0.2164729238,
                                    0.1717140079,
                                    -0.4604763985,
                                ],
                            ],
                            [
                                [
                                    0.1073266491,
                                    -0.0756167993,
                                    -0.0670859516,
                                    -0.1510235071,
                                    0.0935383886,
                                ],
                                [
                                    -0.0794663876,
                                    0.2320474833,
                                    -0.3146029115,
                                    -0.2264315188,
                                    -0.2658869028,
                                ],
                                [
                                    0.1927164495,
                                    0.2216770053,
                                    0.1863117069,
                                    -0.1011029780,
                                    -0.2722999752,
                                ],
                                [
                                    -0.0329625830,
                                    0.0865557715,
                                    -0.1566917449,
                                    -0.2728624940,
                                    -0.0158728212,
                                ],
                                [
                                    0.0454258397,
                                    -0.0045348131,
                                    0.3586204350,
                                    0.3066178262,
                                    -0.0837633908,
                                ],
                            ],
                            [
                                [
                                    -0.0745730698,
                                    0.1263661087,
                                    0.2288170904,
                                    0.2371405661,
                                    0.2375129312,
                                ],
                                [
                                    -0.0134606939,
                                    0.1754960716,
                                    0.4330822527,
                                    -0.1783974171,
                                    -0.3347001076,
                                ],
                                [
                                    -0.2336662710,
                                    0.2627266049,
                                    0.1627702266,
                                    0.2913293540,
                                    0.0729261711,
                                ],
                                [
                                    -0.1588576287,
                                    0.0275929850,
                                    0.0494509228,
                                    -0.1789091080,
                                    -0.0063853385,
                                ],
                                [
                                    -0.3452173471,
                                    0.2812155783,
                                    -0.1393161267,
                                    0.0625517145,
                                    -0.1152934432,
                                ],
                            ],
                            [
                                [
                                    0.0023322322,
                                    -0.0210458767,
                                    -0.1803290993,
                                    -0.0074463435,
                                    -0.0798835233,
                                ],
                                [
                                    -0.2353824824,
                                    0.1489004046,
                                    -0.3527243137,
                                    -0.1963367462,
                                    0.0728322864,
                                ],
                                [
                                    -0.0199724324,
                                    0.0983824134,
                                    0.1473863870,
                                    -0.1338115036,
                                    0.0044367374,
                                ],
                                [
                                    -0.0696858615,
                                    0.0161693040,
                                    -0.2674804926,
                                    -0.1845093668,
                                    0.2691478431,
                                ],
                                [
                                    -0.1652642637,
                                    -0.0304192118,
                                    0.3568618298,
                                    -0.0089061316,
                                    0.4414129257,
                                ],
                            ],
                            [
                                [
                                    -0.1090380251,
                                    0.0938947499,
                                    0.2829086781,
                                    -0.0437412746,
                                    0.1607535630,
                                ],
                                [
                                    0.4368903637,
                                    0.1168525815,
                                    0.2489393800,
                                    -0.2477865666,
                                    0.0828180686,
                                ],
                                [
                                    -0.3369211555,
                                    0.1087786183,
                                    0.2858203351,
                                    -0.0709943026,
                                    0.1315235794,
                                ],
                                [
                                    0.1340721399,
                                    0.0719664022,
                                    -0.0564526021,
                                    0.0086258259,
                                    0.0673800260,
                                ],
                                [
                                    -0.4871386290,
                                    0.0206902549,
                                    0.0386303291,
                                    -0.0860586986,
                                    0.1346030235,
                                ],
                            ],
                            [
                                [
                                    -0.0470560044,
                                    0.0143988458,
                                    -0.2484803945,
                                    0.3393336833,
                                    -0.0804205388,
                                ],
                                [
                                    -0.3388066292,
                                    -0.1600835770,
                                    -0.2296808809,
                                    0.0356872790,
                                    0.2044885904,
                                ],
                                [
                                    -0.0451556370,
                                    -0.1514627337,
                                    -0.0332234502,
                                    0.0215671211,
                                    -0.0296727046,
                                ],
                                [
                                    -0.1049674079,
                                    -0.2294166535,
                                    -0.2630085945,
                                    0.1672945321,
                                    0.2410964817,
                                ],
                                [
                                    -0.0765956938,
                                    -0.3629496992,
                                    0.1739169061,
                                    -0.2168258280,
                                    0.1116329059,
                                ],
                            ],
                            [
                                [
                                    -0.0707437843,
                                    0.0054866900,
                                    0.2179695219,
                                    -0.1399696022,
                                    -0.1212633997,
                                ],
                                [
                                    0.4195635915,
                                    -0.3286511004,
                                    0.1124976799,
                                    -0.0185163617,
                                    0.3052802980,
                                ],
                                [
                                    -0.2229031622,
                                    -0.0797316730,
                                    0.2899165452,
                                    -0.1877875477,
                                    -0.1544894278,
                                ],
                                [
                                    0.0052086711,
                                    -0.2557645142,
                                    0.0382068567,
                                    -0.0669062659,
                                    0.0286107939,
                                ],
                                [
                                    -0.3308942616,
                                    -0.2201934606,
                                    0.1316154301,
                                    -0.1154549643,
                                    -0.0610672720,
                                ],
                            ],
                            [
                                [
                                    -0.1747713089,
                                    0.1397975683,
                                    -0.0757207051,
                                    0.0644572377,
                                    -0.0001406555,
                                ],
                                [
                                    -0.2315001786,
                                    -0.3676908612,
                                    -0.1078625545,
                                    0.1486423761,
                                    -0.0761282593,
                                ],
                                [
                                    0.1795608848,
                                    0.1372801661,
                                    -0.0871306956,
                                    -0.1101655513,
                                    0.0058511938,
                                ],
                                [
                                    -0.0762545988,
                                    -0.3396475315,
                                    -0.1857661605,
                                    0.3142660558,
                                    -0.0589785986,
                                ],
                                [
                                    0.2927993834,
                                    -0.1983406842,
                                    0.0674721599,
                                    -0.1320792437,
                                    -0.3321015537,
                                ],
                            ],
                            [
                                [
                                    -0.0286571849,
                                    -0.0194088258,
                                    0.0598221458,
                                    -0.1521518528,
                                    -0.1904205233,
                                ],
                                [
                                    0.0629114807,
                                    -0.3341853917,
                                    0.1756479144,
                                    0.2401934117,
                                    -0.0260598697,
                                ],
                                [
                                    -0.0628648028,
                                    -0.1272772551,
                                    0.1641642749,
                                    -0.1579861939,
                                    -0.3214652836,
                                ],
                                [
                                    -0.3186239600,
                                    -0.2821285427,
                                    0.2263276875,
                                    -0.1307759881,
                                    -0.2956997454,
                                ],
                                [
                                    -0.0975293368,
                                    -0.2381285727,
                                    0.1169830337,
                                    -0.1054816097,
                                    -0.2717215121,
                                ],
                            ],
                            [
                                [
                                    -0.3790825009,
                                    0.2477190048,
                                    0.1901373118,
                                    -0.2178372145,
                                    0.0565869622,
                                ],
                                [
                                    -0.0722707137,
                                    -0.2811512053,
                                    -0.0108981021,
                                    0.0714823976,
                                    -0.2634947896,
                                ],
                                [
                                    0.0946122035,
                                    0.4161617160,
                                    0.0925369486,
                                    -0.0336987041,
                                    0.3845770061,
                                ],
                                [
                                    -0.0375546627,
                                    -0.1836521327,
                                    -0.0594605431,
                                    0.2061686218,
                                    -0.1540993601,
                                ],
                                [
                                    0.2085018754,
                                    0.1271410882,
                                    0.1830290407,
                                    0.4160760641,
                                    -0.0043135798,
                                ],
                            ],
                        ]
                    ]
                ]
            )
        elif torch.device(device) == torch.device("cuda"):
            wref = torch.tensor(
                [
                    [
                        [
                            [
                                [
                                    0.0276096929,
                                    -0.0717773810,
                                    0.0917145610,
                                    -0.2993721962,
                                    -0.0363429748,
                                ],
                                [
                                    -0.0582000166,
                                    -0.2002883255,
                                    -0.0860655978,
                                    0.4101807177,
                                    0.0393529385,
                                ],
                                [
                                    0.0680099502,
                                    -0.2486239076,
                                    0.1535306275,
                                    -0.2970303297,
                                    0.1843222529,
                                ],
                                [
                                    0.1246490479,
                                    -0.3453255296,
                                    0.1883656979,
                                    -0.0012253744,
                                    0.1978117377,
                                ],
                                [
                                    0.1058738977,
                                    -0.3620114923,
                                    0.0326711386,
                                    -0.0333236940,
                                    0.3667499423,
                                ],
                            ],
                            [
                                [
                                    0.4036888480,
                                    -0.0239365287,
                                    0.0951733291,
                                    -0.0958235264,
                                    -0.1357195228,
                                ],
                                [
                                    0.0716638267,
                                    -0.2716240585,
                                    -0.0798844472,
                                    0.2285895050,
                                    0.3228994310,
                                ],
                                [
                                    0.1492844820,
                                    0.1070405617,
                                    -0.3178789914,
                                    0.0777360797,
                                    -0.4112769067,
                                ],
                                [
                                    0.0321757458,
                                    -0.0694509000,
                                    -0.0427257568,
                                    0.2569150925,
                                    0.0455271155,
                                ],
                                [
                                    -0.0208101068,
                                    -0.0873944834,
                                    -0.3024211824,
                                    0.4639047980,
                                    -0.0394729786,
                                ],
                            ],
                            [
                                [
                                    0.0634213239,
                                    -0.0810443237,
                                    0.2297264040,
                                    -0.3121828139,
                                    -0.1354659349,
                                ],
                                [
                                    -0.2458850592,
                                    -0.1681395322,
                                    -0.3385652304,
                                    0.0175531246,
                                    -0.3685525060,
                                ],
                                [
                                    0.2244260162,
                                    -0.2200513333,
                                    0.2305516750,
                                    -0.2805229425,
                                    -0.0141157778,
                                ],
                                [
                                    -0.1291317791,
                                    -0.1936571747,
                                    0.0267467871,
                                    -0.0891738385,
                                    0.0434010923,
                                ],
                                [
                                    0.3359231651,
                                    -0.3119476140,
                                    0.0155667150,
                                    0.0052201808,
                                    0.1054823026,
                                ],
                            ],
                            [
                                [
                                    0.0635403395,
                                    0.0736334473,
                                    0.1378314644,
                                    -0.0584154166,
                                    0.1859748214,
                                ],
                                [
                                    0.2284742594,
                                    -0.2468640804,
                                    -0.1737281382,
                                    0.2515604794,
                                    0.0855917186,
                                ],
                                [
                                    -0.1103037149,
                                    -0.0324281342,
                                    -0.2247159779,
                                    -0.1741518080,
                                    0.0830042735,
                                ],
                                [
                                    0.0712806731,
                                    -0.0841326043,
                                    -0.1842257679,
                                    0.2410607189,
                                    -0.0855439156,
                                ],
                                [
                                    -0.1781019121,
                                    -0.3655035496,
                                    -0.3385612369,
                                    -0.1143045723,
                                    0.2729386091,
                                ],
                            ],
                            [
                                [
                                    0.1002617255,
                                    0.0245944764,
                                    0.3150165081,
                                    -0.0557736903,
                                    -0.2202234864,
                                ],
                                [
                                    0.1062629968,
                                    0.1692755222,
                                    -0.4336427450,
                                    -0.2879211307,
                                    -0.0585210584,
                                ],
                                [
                                    0.3356589973,
                                    0.0064440765,
                                    0.2595517635,
                                    -0.0261770543,
                                    -0.1922556162,
                                ],
                                [
                                    -0.0366289876,
                                    0.1511871368,
                                    -0.0729549006,
                                    -0.0176989958,
                                    0.0154215805,
                                ],
                                [
                                    0.4886323214,
                                    -0.0236381292,
                                    -0.0284470059,
                                    0.0486573614,
                                    -0.1259857416,
                                ],
                            ],
                            [
                                [
                                    -0.1674654186,
                                    0.2127389312,
                                    0.1948453188,
                                    0.2447953075,
                                    0.0808801427,
                                ],
                                [
                                    0.3375539780,
                                    -0.0140990913,
                                    -0.2738401592,
                                    0.0761198997,
                                    -0.1719108522,
                                ],
                                [
                                    -0.0494174249,
                                    -0.0594578460,
                                    0.1058664247,
                                    0.0149588790,
                                    -0.0169615317,
                                ],
                                [
                                    0.1056448966,
                                    0.0254338272,
                                    -0.3097527027,
                                    0.0246985462,
                                    -0.2941854596,
                                ],
                                [
                                    0.1172447279,
                                    -0.4621493518,
                                    0.0111181503,
                                    -0.2456303686,
                                    -0.0191279687,
                                ],
                            ],
                            [
                                [
                                    0.0714409649,
                                    0.1349779069,
                                    0.3073345721,
                                    0.1968399882,
                                    -0.1515678018,
                                ],
                                [
                                    0.4657984376,
                                    0.0157919042,
                                    -0.1213344410,
                                    -0.1074782535,
                                    0.2748293579,
                                ],
                                [
                                    0.2323415726,
                                    0.2583765090,
                                    0.2751999795,
                                    0.1828601360,
                                    -0.0059689754,
                                ],
                                [
                                    0.1764015555,
                                    -0.0992728099,
                                    0.0940882191,
                                    -0.0687033385,
                                    -0.1596601456,
                                ],
                                [
                                    0.3434692323,
                                    0.2797037363,
                                    -0.0153920818,
                                    0.0051106866,
                                    0.1451397538,
                                ],
                            ],
                            [
                                [
                                    -0.1642148197,
                                    0.0707813501,
                                    0.0767122805,
                                    0.0585145801,
                                    -0.1652119756,
                                ],
                                [
                                    0.2368126065,
                                    0.2050102204,
                                    -0.2792106271,
                                    -0.1819262505,
                                    0.0390523896,
                                ],
                                [
                                    0.1736298501,
                                    0.1475563049,
                                    -0.0252305903,
                                    0.0201776866,
                                    -0.1126582772,
                                ],
                                [
                                    0.0764348432,
                                    0.1147030070,
                                    -0.3062642813,
                                    -0.2271936536,
                                    -0.2134804428,
                                ],
                                [
                                    0.4150142372,
                                    -0.1278586835,
                                    -0.0136230625,
                                    0.0041670389,
                                    -0.2803826928,
                                ],
                            ],
                            [
                                [
                                    0.0340809971,
                                    0.1057514399,
                                    0.1674485654,
                                    0.2122623473,
                                    -0.0073439116,
                                ],
                                [
                                    0.1983666122,
                                    -0.1943603158,
                                    -0.0254657026,
                                    0.4304075241,
                                    0.2351184636,
                                ],
                                [
                                    0.0743648186,
                                    0.2992247641,
                                    0.1822019815,
                                    0.1625616848,
                                    0.2512138784,
                                ],
                                [
                                    0.0196763147,
                                    -0.3972700238,
                                    0.1045062393,
                                    0.0379358158,
                                    -0.2274867147,
                                ],
                                [
                                    0.1123131216,
                                    0.3532764316,
                                    0.0026834551,
                                    -0.0721796826,
                                    0.4549141228,
                                ],
                            ],
                            [
                                [
                                    -0.2167591304,
                                    -0.1624610275,
                                    -0.1129787713,
                                    -0.2064430714,
                                    0.0629907176,
                                ],
                                [
                                    0.0801883489,
                                    0.2263888717,
                                    -0.1607946903,
                                    -0.2783388793,
                                    0.3173940480,
                                ],
                                [
                                    0.0869711936,
                                    0.1598061174,
                                    -0.3085705340,
                                    -0.0753590614,
                                    0.4057875872,
                                ],
                                [
                                    0.0376879275,
                                    0.0763962269,
                                    -0.1443094015,
                                    -0.2855783701,
                                    -0.0095453989,
                                ],
                                [
                                    0.2043217123,
                                    0.0217920225,
                                    -0.2521758378,
                                    0.2957173884,
                                    0.1018180326,
                                ],
                            ],
                        ]
                    ]
                ]
            ).to(device)

    elif name == "SigmaGenerator":
        w = params["sigma"]
        atol = 1e-6
        if torch.device(device) == torch.device("cpu"):
            wref = torch.tensor([0.2531657219])
        elif torch.device(device) == torch.device("cuda"):
            wref = torch.tensor([0.2055327892]).to(device)

    print(
        "----------------------------------------------------------------DEVICE: ",
        device,
    )
    print((w - wref).abs().max().item())
    print((w - wref).abs().mean().item())
    print((w - wref).abs().min().item())
    assert torch.allclose(w, wref, atol=atol)


######################
### MRI GENERATORS ###
######################


@pytest.fixture
def batch_size():
    return 2


def choose_mri_generator(generator_name, img_size, acc, center_fraction):
    if generator_name == "gaussian":
        g = GaussianMaskGenerator(
            img_size, acceleration=acc, center_fraction=center_fraction
        )
    elif generator_name == "random":
        g = RandomMaskGenerator(
            img_size, acceleration=acc, center_fraction=center_fraction
        )
    elif generator_name == "uniform":
        g = EquispacedMaskGenerator(
            img_size, acceleration=acc, center_fraction=center_fraction
        )
    return g


@pytest.mark.parametrize("generator_name", MRI_GENERATORS)
@pytest.mark.parametrize("img_size", MRI_IMG_SIZES)
@pytest.mark.parametrize("acc", MRI_ACCELERATIONS)
@pytest.mark.parametrize("center_fraction", MRI_CENTER_FRACTIONS)
def test_mri_generator(generator_name, img_size, batch_size, acc, center_fraction):
    generator = choose_mri_generator(generator_name, img_size, acc, center_fraction)
    # test across different accs and centre fracations
    H, W = img_size[-2:]
    assert W // generator.acc == (generator.n_lines + generator.n_center)

    mask = generator.step(batch_size=batch_size, seed=0)["mask"]

    if len(img_size) == 2:
        assert len(mask.shape) == 4
        C = 1
    elif len(img_size) == 3:
        assert len(mask.shape) == 4
        C = img_size[0]
    elif len(img_size) == 4:
        assert len(mask.shape) == 5
        C = img_size[0]
        assert mask.shape[2] == img_size[1]

    assert mask.shape[0] == batch_size
    assert mask.shape[1] == C
    assert mask.shape[-2:] == img_size[-2:]

    for b in range(batch_size):
        for c in range(C):
            if len(img_size) == 4:
                for t in range(img_size[1]):
                    mask[b, c, t, :, :].sum() * generator.acc == H * W
            else:
                mask[b, c, :, :].sum() * generator.acc == H * W

    mask2 = generator.step(batch_size=batch_size)["mask"]

    if generator.n_lines != 0 and generator_name != "uniform":
        assert not torch.allclose(mask, mask2)


#############################
### INPAINTING GENERATORS ###
#############################


def choose_inpainting_generator(name, img_size, split_ratio, pixelwise, device):
    if name == "bernoulli":
        return dinv.physics.generator.BernoulliSplittingMaskGenerator(
            tensor_size=img_size,
            split_ratio=split_ratio,
            device=device,
            pixelwise=pixelwise,
            rng=torch.Generator(device).manual_seed(0),
        )
    elif name == "gaussian":
        return dinv.physics.generator.GaussianSplittingMaskGenerator(
            tensor_size=img_size,
            split_ratio=split_ratio,
            device=device,
            pixelwise=pixelwise,
            rng=torch.Generator(device).manual_seed(0),
        )
    else:
        raise Exception("The generator chosen doesn't exist")


@pytest.mark.parametrize("generator_name", INPAINTING_GENERATORS)
@pytest.mark.parametrize("img_size", INPAINTING_IMG_SIZES)
@pytest.mark.parametrize("pixelwise", (False, True))
@pytest.mark.parametrize("split_ratio", (0.5,))
@pytest.mark.parametrize("device", DEVICES)
def test_inpainting_generators(
    generator_name, batch_size, img_size, pixelwise, split_ratio, device
):
    if generator_name == "gaussian" and len(img_size) < 3:
        pytest.skip(
            "Gaussian splitting mask not valid for images of shape smaller than (C, H, W)"
        )

    gen = choose_inpainting_generator(
        generator_name, img_size, split_ratio, pixelwise, device
    )  # Assume generator always receives "correct" img_size i.e. not one with dims missing

    def correct_ratio(ratio):
        assert torch.isclose(
            ratio,
            torch.tensor([split_ratio], device=device),
            rtol=1e-2,
            atol=1e-2,
        )

    def correct_pixelwise(mask):
        if pixelwise:
            assert torch.all(mask[:, 0, ...] == mask[:, 1, ...])
        else:
            assert not torch.all(mask[:, 0, ...] == mask[:, 1, ...])

    # Standard generate mask
    mask1 = gen.step(batch_size=batch_size, seed=0)["mask"]
    correct_ratio(mask1.sum() / np.prod((batch_size, *img_size)))
    correct_pixelwise(mask1)

    # Standard without batch dim
    mask1 = gen.step(batch_size=None, seed=0)["mask"]
    assert tuple(mask1.shape) == tuple(img_size)
    correct_ratio(mask1.sum() / np.prod(img_size))

    # Standard mask but by passing flat input_mask of ones
    input_mask = torch.ones(batch_size, *img_size)
    # should ignore batch_size
    mask2 = gen.step(batch_size=batch_size, input_mask=input_mask, seed=0)["mask"]
    correct_ratio(mask2.sum() / input_mask.sum())
    correct_pixelwise(mask2)

    # As above but with no batch dimension in input_mask
    input_mask = torch.ones(*img_size, device=device)
    mask2 = gen.step(batch_size=batch_size, input_mask=input_mask, seed=0)[
        "mask"
    ]  # should use batch_size
    correct_ratio(mask2.sum() / input_mask.sum() / batch_size)

    # As above but with img_size missing channel dimension (bad practice)
    input_mask = torch.ones(*img_size[1:], device=device)
    mask2 = gen.step(batch_size=batch_size, input_mask=input_mask, seed=0)["mask"]
    correct_ratio(mask2.sum() / input_mask.sum() / batch_size)

    # Generate splitting mask from already subsampled mask
    input_mask = torch.zeros(batch_size, *img_size, device=device)
    input_mask[..., 10:20] = 1
    mask3 = gen.step(batch_size=batch_size, input_mask=input_mask, seed=0)["mask"]
    correct_ratio(mask3.sum() / input_mask.sum())
    correct_pixelwise(mask3)
