from deepinv.models import NCSNpp


model = NCSNpp(
        img_resolution=64,
        in_channels=3,
        out_channels=3,
        pretrained='edm-ffhq64-uncond-ve'
    )