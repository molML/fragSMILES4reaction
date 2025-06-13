# fragSMILES4Reactions

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3126/)
[![Jupyter Notebooks](https://img.shields.io/badge/Made%20with-Jupyter-orange)](https://jupyter.org/)

**fragSMILES4Reactions** is a scientific project focused on the analysis and modeling of chemical reactions using fragment-based SMILES representations and other notations like SMILES, SELFIES and SAFE. This repository contains all the necessary code, data, and scripts to reproduce the experiments and results described in the associated research work.

## 📁 Project Structure

- [`rawdata_reactions/`](./rawdata_reactions) – Raw reaction dataset already split into training, validation, and test sets.
- [`data_reactions/`](./data_reactions) – Processed reaction data used as input for experiments.
- [`experiments_reactions/`](./experiments_reactions) – Outputs from model training and prediction. Folder names follow the convention `{key}={value}-{key}={value}-...`.
- [`floats/`](./floats) – Figures (in PDF format) and tables (in LaTeX format) generated during analysis.
- [`notebooks/`](./notebooks) – Jupyter notebooks for data exploration and post-processing of prediction results.
- [`bestof_setup/`](./bestof_setup) – CSV files reporting the best configuration found for each model.
- [`scripts/`](./scripts) – Python scripts for preprocessing, training, prediction, and SMILES conversion tasks.
- [`src/`](./src) – Main source code of the project.
- [`extra/`](./extra) – Contains an example reaction used for creating the introductory figure/chart.
- [`shell/`](./shell) – Includes `run.sh`, a script to launch experiments using the best configurations for each model.
- [`requirements.txt`](./requirements.txt) – List of required Python dependencies for setting up the environment.
- [`chemicalgof/`](./chemicalgof) and [`safe`](./safe)/[`datamol/`](./datamol) are external static repositories adopted for this project. The `SAFE` package has been modified to detect and report reasons for invalid sampled sequences. [`chemicalgof/`](./chemicalgof) is the latest version to handle with fragSMILES notation.

## 🧪 Reproducibility

The output of the experiments is already included in [`experiments_reactions/`](./experiments_reactions), including model checkpoints (`.ckpt` files) adopted for analysis.
You can train the model yourself, making sure **not to resume from existing checkpoints**.  
The prediction phase (see the [scripts](#scripts) section) can be executed directly, if the trained models are stored in the appropriate `experiment` folder.
However, the results of such predictions have already been analyzed and are available in this repository.

To reproduce our experiments:

1. Clone the repository:

   ```bash
    git clone https://github.com/molML/fragSMILES4reaction.git
    cd fragSMILES4Reactions
    ```

2. Set up the Python environment:

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    ```

3. Experiments using the best configurations for each model can be run through a shell script.
   > **_NOTE:_** Set python environment path to be activated in file [`shell/run.sh line 3`](shell/run.sh#L3).

    ```bash
    bash shell/run.sh
    ```

    > :warning: These experiments were conducted using 4 GPUs in parallel.
        Running on fewer or lower-memory devices may result in out-of-memory errors.

4. Explore the Jupyter notebooks in [Notebooks](#-notebooks) section to analyze datasets and prediction results.

## Model parameters

| Parameter     | Description |
|---------------|-------------|
| `task`       | Task to perform: either `forward` (i.e., synthesis) or `backward` (i.e., retrosynthesis). |
| `notation`   | Molecular representation format used as input/output (i.e., smiles, selfies, safe or fragsmiles). |
| `model_dim` | Dimensionality of the model's hidden layers (e.g., transformer embedding size). |
| `num_heads` | Number of attention heads in multi-head attention mechanisms. |
| `num_layers` | Number of layers (e.g., encoder or decoder blocks) in the model architecture. |
| `batch_size` | Number of training samples processed simultaneously during one training step. |
| `lr`         | Learning rate used by the optimizer to update model weights. |
| `dropout`    | Dropout rate for regularization to prevent overfitting (only 0.3 value was adopted in this work) |

These parameters are used as arguments in the Python scripts (see relative [section](#scripts)) for [`training`](./scripts/training.py) and [`prediction`](./scripts/predict.py).

## Scripts

We recommend running scripts from the root directory.
Example:

```bash
python scripts/script_file.py --argument1 value1 --argument2 value2
```

- [`convert_dataset.py`](./scripts/convert_dataset.py)
  Prepares dataset adopted for the experiments starting from rawdata. Please, explore arguments (`python scripts/convert_dataset.py --help`) to be provided when command is called. Most important arguments are "notation", "split", "ncpus" (for multiprocessing computation). When a dataset notation-based is obtained, a csv file is written to track sequence lengths.

- [`train.py`](./scripts/train.py)
  Trains a model using the selected configuration (see dedicated [section](#model-parameters)).
  Model checkpoint (.ckpt file format) will be stored in the corresponding experiment folder.
  Vocabulary file (vocab.pt) will be stored in the respective notation folder.

- [`predict.py`](./scripts/predict.py)
  Predict the test set with trained model by using the selected configuration (see dedicated [section](#model-parameters)).
  The output includes encoded predicted sequences stored in the respective experiment folder, with filenames containing _tokens_ substring.

- [`convert_prediction_strict.py`](./scripts/convert_prediction_strict.py)
  Convert encoded predicted sequences obtained by model specifying its parameters. Invalid decoded sequences include the erroneus chirality label assigned to atoms.
- [`convert_prediction_strict_from_path.py`](./scripts/convert_prediction.py)
  Same as above, but only requires the path to the experiment folder.

- [`convert_prediction.py`](./scripts/convert_prediction.py) and [`convert_prediction_from_path.py`](./scripts/convert_prediction_from_path.py) : Similar to the strict versions, but invalid sequences do not include erroneus chirality label assigned to atoms.

- [`fragment_dataset.py`](./scripts/fragment_dataset.py)
  Fragment SMILES of data to obtain relative Scaffold, cycles, and acyclic chains of them.
  Used only on the test set, as demonstrated in [`05_struggle.ipynb`](notebooks/05_struggle.ipynb)

## 📓 Notebooks

The Jupyter notebooks in [`notebooks/`](./notebooks/) provide an interactive way to explore datasets and experiment outputs.
> **_NOTE:_** IPython package is required to handle with notebooks.

1. [`data_analysis.ipynb`](notebooks/01_data_analysis.ipynb)
   Can be explored before experiments running. It includes sequence length and dataset size per split.
2. [`bestof_selection.ipynb`](notebooks/02_bestof_selection.ipynb)
   Visualizes and compares loss curves for different hyperparameter settings to identify optimal configurations.
3. [`accuracy.ipynb`](notebooks/03_accuracy.ipynb)
   Computes performance metrics for the best models and outputs tables ready for publication.
4. [`similarity.ipynb`](notebooks/04_similarity.ipynb)
   Analyzes similarity distributions between incorrect but valid predictions and their target molecules (forward task only).
5. [`struggle.ipynb`](notebooks/05_struggle.ipynb)
   Investigates failure cases in prediction, including reasons for invalid samples and substructure matching in erroneous predictions.

## 💬 Citation

> **(No published paper)**

## 📜 License

This project is licensed under the MIT License.
See the [LICENSE](./LICENSE) file for details.
