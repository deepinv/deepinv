from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PETExample:
    pet: object
    acquisition_model: object
    image_template: object
    acquisition_data: object
    data_path: str
    raw_data_file: str


def build_pet_ray_tracing_example() -> PETExample:
    import sirf.STIR as pet

    pet.AcquisitionData.set_storage_scheme("memory")
    data_path = pet.examples_data_path("PET")
    raw_data_file = pet.existing_filepath(data_path, "Utahscat600k_ca_seg4.hs")
    acquisition_data = pet.AcquisitionData(raw_data_file)

    image = pet.ImageData()
    image.initialise(dim=(31, 111, 111), vsize=(2.25, 2.25, 2.25))
    image.fill(1.0)

    acquisition_model = pet.AcquisitionModelUsingRayTracingMatrix()
    acquisition_model.set_up(acquisition_data, image)

    return PETExample(
        pet=pet,
        acquisition_model=acquisition_model,
        image_template=image,
        acquisition_data=acquisition_data,
        data_path=data_path,
        raw_data_file=raw_data_file,
    )


@dataclass
class MRExample:
    mr: object
    acquisition_model: object
    processed_data: object
    image_template: object
    data_path: str
    raw_data_file: str


def build_mr_cartesian_example() -> MRExample:
    import sirf.Gadgetron as mr

    mr.AcquisitionData.set_storage_scheme("memory")
    data_path = mr.examples_data_path("MR")
    raw_data_file = mr.existing_filepath(data_path, "simulated_MR_2D_cartesian.h5")
    acquisition_data = mr.AcquisitionData(raw_data_file)
    processed_data = mr.preprocess_acquisition_data(acquisition_data)

    if acquisition_data.is_undersampled():
        recon = mr.CartesianGRAPPAReconstructor()
        recon.compute_gfactors(False)
    else:
        recon = mr.FullySampledReconstructor()
    recon.set_input(processed_data)
    recon.process()
    image_template = recon.get_output()

    coil_maps = mr.CoilSensitivityData()
    coil_maps.calculate(processed_data)

    acquisition_model = mr.AcquisitionModel(processed_data, image_template)
    acquisition_model.set_coil_sensitivity_maps(coil_maps)

    return MRExample(
        mr=mr,
        acquisition_model=acquisition_model,
        processed_data=processed_data,
        image_template=image_template,
        data_path=data_path,
        raw_data_file=raw_data_file,
    )

