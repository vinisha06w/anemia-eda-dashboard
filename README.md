# Anemia EDA Dashboard — Streamlit

This project converts the analysis in `projectDS (2).ipynb` into an interactive Streamlit dashboard.

## 1. Prerequisites

Install:
- Python 3.10 or newer
- Visual Studio Code
- VS Code Python extension
- Internet connection for the initial package installation and dataset download

## 2. Download the dataset

Download `anemia.csv` from the Kaggle dataset:

https://www.kaggle.com/datasets/biswaranjanrao/anemia-dataset

Extract the downloaded archive if it is zipped. Copy `anemia.csv` into this project folder, beside `app.py`.

The app expects these columns:
`Gender`, `Hemoglobin`, `MCH`, `MCHC`, `MCV`, `Result`

In this project, `Result=0` means No Anemia and `Result=1` means Anemia. Gender values are intentionally displayed using the original dataset coding; do not assume a mapping unless it is confirmed from the dataset documentation.

## 3. Open the project in VS Code

1. Extract this project ZIP.
2. Open VS Code.
3. Select File > Open Folder.
4. Select the extracted `anemia_streamlit_project` folder.
5. Copy `anemia.csv` into this same folder.

## 4. Create a virtual environment

Open Terminal > New Terminal in VS Code.

Windows PowerShell:
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt:
```cmd
.venv\Scripts\activate.bat
```

## 5. Install libraries

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 6. Run the app

```bash
python -m streamlit run app.py
```

Streamlit should open a browser tab. If it does not, open:

http://localhost:8501

To stop the app, return to the VS Code terminal and press Ctrl+C.

## 7. Dataset / notebook clarification

- You do NOT need to download or run the Colab notebook to run this Streamlit app.
- You DO need the CSV dataset. The simplest method is to download `anemia.csv` from Kaggle and place it beside `app.py`.
- The notebook is useful as your original analysis/reference, but Streamlit runs `app.py`, not the notebook.
- Alternatively, start the app and upload `anemia.csv` using the sidebar uploader.

## 8. Dashboard sections

- Overview & Statistics: preview, structure, missing values, descriptive statistics, gender counts, exploratory Hemoglobin quantile groups.
- Distributions: histogram + KDE, KDE curve, IQR outlier summary, Shapiro-Wilk test, skewness.
- Anemia Comparisons: anemia status counts, feature boxplot, scatter relationships, gender vs anemia status.
- Correlation: numerical correlation heatmap.
- Data Explorer: filtered records and CSV download.

The sidebar allows gender/result filtering and optional exclusion of duplicate rows. Duplicate exclusion is off by default so the dashboard does not silently alter the original dataset.

## 9. Troubleshooting

**`streamlit` is not recognized**
Use:
```bash
python -m streamlit run app.py
```

**`ModuleNotFoundError`**
Activate `.venv` and run:
```bash
pip install -r requirements.txt
```

**`FileNotFoundError` / dataset not found**
Confirm that the filename is exactly `anemia.csv` and it is in the same folder as `app.py`. Alternatively, upload it in the sidebar.

**Port 8501 is already in use**
Run:
```bash
python -m streamlit run app.py --server.port 8502
```
Then open http://localhost:8502.
