# ForgeData

[![Release](https://img.shields.io/github/v/release/OSU-SIMCenter/ForgeData)](https://img.shields.io/github/v/release/OSU-SIMCenter/ForgeData)
[![Build status](https://img.shields.io/github/actions/workflow/status/OSU-SIMCenter/ForgeData/main.yml?branch=main)](https://github.com/OSU-SIMCenter/ForgeData/actions/workflows/main.yml?query=branch%3Amain)
[![Commit activity](https://img.shields.io/github/commit-activity/m/OSU-SIMCenter/ForgeData)](https://img.shields.io/github/commit-activity/m/OSU-SIMCenter/ForgeData)

HAMMER-ERC incremental forming data management repo.

- **Github repository**: <https://github.com/OSU-SIMCenter/ForgeData/>

## Usage

Create a python virtual environment with `pip` or `uv`. Install [pytorch](https://pytorch.org/get-started/locally/) into the environment 

Install the package and dependencies in editable mode:

```
pip install -e .
```

Create a directory at `./data/raw` and place Agility Forge experimental data there.

An example tree print out on Windows11:

```
C:\USERS\USERNAME\GITHUB\FORGEDATA\DATA\RAW
├───2025_10_28 T16_47_30.E8 Tensile Bar 15-5PH
│   ├───TP0
│   │       5-8in round bar.obj
│   │       ASTM E8-E8M Subsize Tensile Bar Forge Size.obj
│   │       DestinationModelSaveData.txt
│   │       IntermediateModelSaveData.txt
│   │       LastModelSaveData.txt
│   │       ProjectedPointsSettings.txt
│   │       StockModelSaveData.txt
│   │       
│   ├───TP1
│   │   │   5-8in round bar.obj
│   │   │   ASTM E8-E8M Subsize Tensile Bar Forge Size.obj
│   │   │   DestinationModelSaveData.txt
│   │   │   IntermediateModelSaveData.txt
│   │   │   LastModelSaveData.txt
│   │   │   ProjectedPointsSettings.txt
│   │   │   StockModelSaveData.txt
│   │   │
│   │   ├───3D Scan Data
│   │   │       2025_10_28 T16_56_26.3D Scan at Hitpoint 31.csv
│   │   │
│   │   └───Load Stroke Data
│   │           2025_10_28 T16_49_06.LS data Hitpoint 0.csv
│   │           2025_10_28 T16_49_11.LS data Hitpoint 1.csv
│   │           2025_10_28 T16_49_16.LS data Hitpoint 2.csv
|   |           (continues)...
│   │
│   ├───TP2
│   │   │   5-8in round bar.obj
│   │   │   ASTM E8-E8M Subsize Tensile Bar Forge Size.obj
│   │   │   DestinationModelSaveData.txt
│   │   │   IntermediateModelSaveData.txt
│   │   │   LastModelSaveData.txt
│   │   │   ProjectedPointsSettings.txt
│   │   │   StockModelSaveData.txt
│   │   │
│   │   ├───3D Scan Data
│   │   │       2025_10_28 T17_10_04.3D Scan at Hitpoint 45.csv
│   │   │
│   │   └───Load Stroke Data
│   │           2025_10_28 T16_57_47.LS data Hitpoint 0.csv
│   │           2025_10_28 T16_57_52.LS data Hitpoint 1.csv
│   │           2025_10_28 T16_57_58.LS data Hitpoint 2.csv
|   |           (continues)...
│   │
│   ├─── (continues)...
│
└───2025_10_28 T18_01_24.E8 Tensile Bar 4140
    ├───TP0
    │       5-8in round bar.obj
    │       ASTM E8-E8M Subsize Tensile Bar Forge Size.obj
    │       DestinationModelSaveData.txt
    │       IntermediateModelSaveData.txt
    │       LastModelSaveData.txt
    │       ProjectedPointsSettings.txt
    │       StockModelSaveData.txt
    │
    ├───TP1
    │   │   5-8in round bar.obj
    │   │   ASTM E8-E8M Subsize Tensile Bar Forge Size.obj
    │   │   DestinationModelSaveData.txt
    │   │   IntermediateModelSaveData.txt
    │   │   LastModelSaveData.txt
    │   │   ProjectedPointsSettings.txt
    │   │   StockModelSaveData.txt
    │   │
    │   ├───3D Scan Data
    │   │       2025_10_28 T18_09_31.3D Scan at Hitpoint 31.csv
    │   │
    │   └───Load Stroke Data
    │           2025_10_28 T18_03_38.LS data Hitpoint 0.csv
    │           2025_10_28 T18_03_43.LS data Hitpoint 1.csv
    │           2025_10_28 T18_03_49.LS data Hitpoint 2.csv
    |           (continues)...
    |
    │
    ├───(continues)...
```

To parse raw data and generate HDF5/SQLite databases for use with Pytorch `dataset` classes, call:

```
python ./scripts/process_raw_data.py
```

NOTE: Irrelevant metadata, such as `ProjectedPointsSettings.txt`, are excluded from HDF5/SQLite databases. 

In the HDF5 file, mesh objects are saved as vertex and face matrices.
* vertices
    * Data Type: float64
    * Shape: (n, 3)

* faces
    * Data Type: int32
    * Shape: (m, 3)

To inspect the HDF5 file, use `hdf5view`:

```
pip install pyqt5 hdf5view
hdf5view -f ./data/processed/FILENAME.h5
```


## TODO

* ~~Fix raw data cosine error / ue multi-scan problem~~
* Pull action out of load-stroke
* ~~Have AgF dataset save zero to calibrate load-stroke into mesh reference frame~~
* Do we have ForgeDataset return action sequences if there isn't a geometry scan every hit?
* ~~SQLite is not done yet. H5 is much easier. rm sqlite?~~


---

Repository initiated with [fpgmaas/cookiecutter-uv](https://github.com/fpgmaas/cookiecutter-uv).
