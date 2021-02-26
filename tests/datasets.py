import albumentations

from mvtec.datasets import MVTecDataset


data_dir = "~/github/mvtec-utils/data"
queries = ["category==bottle"]
transforms = albumentations.Compose(
    [
        albumentations.Resize(256, 256),
        albumentations.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        albumentations.pytorch.ToTensorV2(),
    ]
)
MVTecDataset(data_dir, queries, transforms, download=True)
