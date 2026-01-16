## Environment Setup (Conda)

This project is tested with **Python 3.12** and a **Conda** environment.

### 1) Create the environment from `environment.yml`


```bash
# create the environment
conda env create -f environment.yml

# activate the environment
conda activate pyg_2
````

### 2) (Optional) Install pip packages

Depending on your platform, some packages (especially PyG extensions) may not be fully resolved by conda.
If you encounter missing-module errors, install the remaining dependencies via pip:

```bash
pip install -r requirements.txt
```

### 3) Verify installation

```bash
python -c "import torch; print(torch.__version__)"
python -c "import torch_geometric; print(torch_geometric.__version__)"
```

### 4) Run

```bash
python main.py
```
