# Dataset Preparation
## Prerequisite
For dataset preparation and preprocessing, we assume you already finished intallation steps in `README.md`. ~~For data preprocessing, you also need to download [this .npy file](https://github.com/zju3dv/EasyMocap/blob/98a229f2ab7647f14ac9693eab00639337274b49/data/smplx/J_regressor_body25.npy) from the [EasyMocap Repository](https://github.com/zju3dv/EasyMocap), and put it under `body_models/misc/`~~ this is not needed for PeopleSnapshot and RANA.

## PeopleSnapshot Dataset
We use the preprocessing script from [InstantAvatar](https://github.com/tijiang13/InstantAvatar):
```
# Step 1: Download data from: https://graphics.tu-bs.de/people-snapshot
# Step 2: Preprocess using InstantAvatar's script
python scripts/preprocess_PeopleSnapshot.py --root <PATH_TO_PEOPLESNAPSHOT> --subject male-3-casual
```
You can use similar command to preprocess subjects other than `male-3-casual`.
After preprocessing the `load/peoplesnapshot` directory should look like the following:
```
load
 └-- peoplesnapshot
    ├-- male-3-casual
    |  ├-- cameras.npz
    |  ├-- images
    |  ├-- masks
    |  ├-- poses
    |  └-- poses.npz
    |
    ├-- male-4-casual
    |  |-- ...
    |   
    ├-- female-3-casual
    |  |-- ...
    |   
    |-- ...
```

## RANA Dataset
Use the following commands to download and preprocess the RANA dataset:
```
# Step 1: Download RANA data from: https://nvlabs.github.io/RANA/
# Step 2: Preprocess using our script
## Training set
python scripts/preprocess_RANA.py --data-dir <PATH_TO_RANA> --split train_p1 --out-dir load/rana --seqname subject_01
## Test set (will also download HDR images needed for relighting)
python scripts/preprocess_RANA.py --data-dir <PATH_TO_RANA> --split test --out-dir load/rana --seqname subject_01
```
You can use similar commands to preprocess subjects other than `subject_01`.
After preprocessing the `load/rana` directory should look like the following:
```
load
 └-- rana
    ├-- train_p1
    |  └-- subject_01
    |     ├-- albedos
    |     ├-- cameras.json
    |     ├-- images
    |     ├-- masks
    |     ├-- normals
    |     ├-- smpl_vis
    |     └-- poses.npz
    |
    ├-- hdri  # HDR images needed for relighting
    |
    └-- test
       └-- subject_01
          ├-- albedos
          ├-- cameras.json
          ├-- hdri_files.json  # list of HDR images for each test image
          ├-- images
          ├-- masks
          ├-- normals
          ├-- smpl_vis
          └-- poses.npz
```

## Synthetic-Human-relit and ZJU-MoCap
Coming soon!
