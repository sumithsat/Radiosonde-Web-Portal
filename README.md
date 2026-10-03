# Intelligent Radiosonde Analytics System

A Streamlit-based web application for downloading and analysing University of Wyoming (UWyo) radiosonde soundings.

## Features

- Select common WMO radiosonde stations or enter a custom 5-digit WMO station ID.
- Select a start date, end date, and observation time (00 UTC, 12 UTC, or both).
- Download UWyo TEXT:LIST sounding data.
- Export combined sounding data as CSV.
- Export individual sounding files as a ZIP archive.
- Display temperature/dew-point, relative-humidity, and wind-speed profiles.
- Generate a Skew-T / Log-P diagram with parcel and wind-barb information when MetPy is available.
- Calculate atmospheric diagnostics including:
  - Surface pressure, temperature, dew point and relative humidity
  - Maximum sounding height
  - SBCAPE and SBCIN
  - Lifted Index
  - K-Index
  - Precipitable Water
  - Freezing level
  - Pressure-layer wind shear
- Download a processing/download log.

## Data source

The application retrieves upper-air sounding data from the University of Wyoming sounding service:

https://weather.uwyo.edu/wsgi/sounding

Third-party datasets and services remain subject to their respective terms, licenses, and attribution requirements.

## Local installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/UWyo-Radiosonde-Web-Portal.git
cd UWyo-Radiosonde-Web-Portal
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

## Streamlit Community Cloud deployment

1. Create a GitHub repository, for example:
   `UWyo-Radiosonde-Web-Portal`
2. Upload:
   - `app.py`
   - `requirements.txt`
   - `README.md`
3. Open Streamlit Community Cloud:
   https://share.streamlit.io/
4. Sign in with GitHub.
5. Select **Create app**.
6. Select the GitHub repository.
7. Select branch `main`.
8. Set the main file to:
   `app.py`
9. Use a supported Python version such as Python 3.12 if prompted.
10. Deploy the application.

After deployment, Streamlit will provide a public `streamlit.app` URL.

## Important deployment notes

The application retrieves sounding data over HTTPS using the `requests` package, so it does not require the local Windows `D:` drive or local data files for its core UWyo download workflow.

Do not place passwords, API keys, tokens, or other private credentials in `app.py` or commit them to GitHub.

## Author

**Developed by: Dr. Sumith Satheendran S. PhD**

Intelligent Radiosonde Analytics System

## Copyright / attribution

Original project implementation, analysis, coding and visualizations © Sumith Satheendran S.

Third-party datasets remain subject to their respective licenses and attribution requirements.
