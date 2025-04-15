# fragSMILES4Reactions

**fragSMILES4Reactions** is a scientific project focused on the analysis and modeling of chemical reactions using fragment-based SMILES representations and other notations like SMILES, SELFIES and SAFE. This repository contains all the necessary code, data, and scripts to reproduce the experiments and results described in the associated research work.

## 📁 Repository Structure

- `rawdata_reactions/` – Raw reaction dataset already splitted into training, validation and test sets.
- `data_reactions/` – Processed data related to reaction experiments.
- `experiments_reactions/` – Configuration files and outputs from reaction experiments.
- `floats/` – Figures and Tables (latex format) resulted from analysis.
- `notebooks/` – Jupyter notebooks for exploratory data analysis e processing results of predictions.
- `scripts/` – Scripts for preprocessing, training, prediction, conversion, etc.
- `src/` – Main source code for the project.
- `requirements.txt` – Dependencies required to run the project.

## 🧪 Reproducibility

To reproduce the experiments:

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

3. Run the desired scripts or notebooks located in notebooks/ or scripts/.

4. For SLURM-based computation, check the slurm/ folder for job templates.

You can then run the notebooks in `notebooks/` or execute the scripts in `scripts/`.  
For HPC execution using SLURM, refer to the job templates in the `slurm/` directory.

## 📓 Notebooks

The Jupyter notebooks in `notebooks/` provide an interactive way to explore datasets and experiment configurations.  
You may use `prova.ipynb` as a starting point.

## 💬 Citation

If you use this codebase or data in your research, please consider citing the corresponding publication:

> **(Add BibTeX entry or DOI link here once available)**

## 🤝 Contributing

This project is research-focused, but contributions are welcome.  
Feel free to open issues, submit pull requests, or discuss improvements.

## 📜 License

**(Specify the license here, e.g., MIT, GPL-3.0, etc.)**
