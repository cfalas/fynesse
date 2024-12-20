import requests
from pathlib import Path
import zipfile
import io
from tqdm.auto import tqdm


def download_file(url: str, output_file: Path):
    """Download a file from the given URL
    :param url: URL of the file to download
    :param output_dir: directory to store the downloaded file
    """

    # Create the output directory if it doesn't exist
    output_path = output_file.parent
    output_path.mkdir(parents=True, exist_ok=True)

    # Check if the file is already downloaded
    if output_file.exists():
        return "skipped"

    # Download the file
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, stream=True)

    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024

    with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
        with open(output_file, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)

    if total_size != 0 and progress_bar.n != total_size:
        raise RuntimeError("Could not download file")
    else:
        return "success"


def download_zip(url: str, output_dir: Path):
    download_file(url, output_dir / "zipfile.zip")
    with zipfile.ZipFile(output_dir / "zipfile.zip", "r") as zip_ref:
        zip_ref.extractall(output_dir)


def price_paid(years: list[int], output_dir: str = "access_data"):
    """Download UK house price data for given years
    :param years: list of years to download data for
    :param output_dir: directory to store the downloaded data
    """

    # Base URL where the dataset is stored
    base_url = (
        "http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com"
    )

    # Create the output directory if it doesn't exist
    output_path = Path(output_dir) / "pp"
    output_path.mkdir(parents=True, exist_ok=True)

    # File name with placeholders
    stats = {"success": 0, "skipped": 0, "failed": 0}
    for year in years:
        for part in range(1, 3):
            file_name = f"pp-{year}-part{part}.csv"
            output_path = Path(output_dir) / "price_paid"
            res = download_file(f"{base_url}/{file_name}", output_path / file_name)
            stats[res] += 1
    print(
        f"Downloaded files for price-paid: {stats['success']} files downloaded, "
        f"{stats['skipped']} files already existed, {stats['failed']} files failed"
    )


def postcode_data(output_dir: str = "access_data"):
    """Download UK postcode data
    :param output_dir: directory to store the downloaded data
    """

    output_path = Path(output_dir) / "postcode_data"
    output_path.mkdir(parents=True, exist_ok=True)

    url = "https://www.getthedata.com/downloads/open_postcode_geo.csv.zip"

    download_file(url, output_path / "open_postcode_geo.csv.zip")

    if not (output_path / "open_postcode_geo.csv").exists():
        with zipfile.ZipFile(output_path / "open_postcode_geo.csv.zip") as zip_ref:
            zip_ref.extractall(output_path)


def census2021(code: str, output_dir: str = "access_data"):
    """Download UK Census data for given code
    :param code: census code to download (e.g. TS062 for NS-SEC, or TS003 for household composition)
    """

    output_path = Path(output_dir) / "census2021" / code.lower()
    output_path.mkdir(parents=True, exist_ok=True)

    url = f"https://www.nomisweb.co.uk/output/census/2021/census2021-{code.lower()}.zip"

    if list(output_path.iterdir()):
        print(f"Data for {code} already exists, skipping download")
        return

    response = requests.get(url)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
        zip_ref.extractall(output_path)

    print(f"Files extracted to: {output_path}")


def data_gov_uk(dataset_id: str, output_path: Path, format: str, unzip: bool = False):
    """Download data from data.gov.uk
    This uses the data.gov.uk API to search for a given dataset and downloads the resources in the given format.
    :param dataset_id: ID of the dataset to download
    :param output_path: file to store the downloaded data
    :param format: format of the file to download (e.g. CSV, ZIP). This should be as it appears on the data.gov.uk website
    """

    result = requests.get(
        f"https://data.gov.uk/api/action/package_show?id={dataset_id}"
    ).json()["result"]

    format_found = False

    for fmt in result["resources"]:
        if fmt["name"] == format:
            download_file(fmt["url"], output_path)
            format_found = True
            break

    if not format_found:
        raise ValueError(f"Format {format} not found in dataset")

    if unzip:
        folder = output_path.with_suffix("")
        folder.mkdir(exist_ok=True)
        with zipfile.ZipFile(output_path, "r") as zip_ref:
            zip_ref.extractall(folder)
