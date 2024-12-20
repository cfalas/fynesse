from sqlalchemy import func, select
from . import download
from . import db

import pandas as pd
import geopandas
import itertools
from pathlib import Path
from typing import cast, Iterable, Any
from geoalchemy2 import Geometry
import geoalchemy2.shape
from tqdm.auto import tqdm
import osmium
from shapely.geometry import Point
from itertools import chain
import csv

CHUNK_SIZE = 100000


class Dataset:
    def __init__(
        self,
        name: str,
        conn: db.Connection,
        table: db.Table,
    ):
        self.name = name
        self.table = table
        self.conn = conn

    @staticmethod
    def from_table(table_name: str, conn: db.Connection):
        # TODO: Make sure that the table exists
        table = conn.reflect_table(table_name)
        return Dataset(table_name, conn, table)

    @staticmethod
    def from_iterable(
        name: str,
        generator: Iterable[pd.DataFrame],
        conn: db.Connection,
        dtypes: dict[str, db.Datatype] = {},
    ):
        conn.upload_df(generator, name, dtypes)
        table = conn.reflect_table(name)
        dataset = Dataset(name, conn, table)
        return dataset

    def sample(self, p: float = 0.1):
        return self.conn.sample_table(self.name, p)

    @property
    def c(self):
        return self.table.c

    @property
    def geo_c(self):
        return [col for col in self.c if isinstance(col.type, db.Geometry)]

    def _process_query_df(self, df: pd.DataFrame, simplify: bool = True):
        for col in self.table.c:
            if isinstance(col.type, db.Geometry):
                df[col.name] = df[col.name].apply(geoalchemy2.shape.to_shape)
                if not isinstance(df, geopandas.GeoDataFrame):
                    df = geopandas.GeoDataFrame(df, geometry=col.name)
                    df.set_crs("EPSG:4326", inplace=True)
                    if simplify:
                        df.set_geometry(df.simplify(0.002), inplace=True)
        return df

    def query(self, query: str | db.Select[Any] | db.Query[Any], simplify: bool = True):
        df = pd.read_sql(query, self.conn.engine)
        return self._process_query_df(df, simplify)

    def query_batched(
        self,
        query: str | db.Select[Any] | db.Query[Any],
        chunksize: int,
        simplify: bool = True,
    ):
        dfs = pd.read_sql(query, self.conn.engine, chunksize=chunksize)
        for df in dfs:
            yield self._process_query_df(df)

    def count_by(self, col: db.Column[Any] | Iterable[db.Column[Any]]):
        if isinstance(col, db.Column):
            col = [col]
        df = self.query(select(*col, func.count("*")).group_by(*col))
        df = df.set_index([col.name for col in col])
        return df

    def filter_by(self, **kwargs: Any):
        return self.query(self.table.select().filter_by(**kwargs))

    def filter(self, filter: Any):
        return self.query(self.table.select().filter(filter))

    def batch(self, batch_size: int = 1000):
        num_rows = self.count()
        for i in tqdm(range(0, num_rows, batch_size)):
            yield self.query(self.table.select().offset(i).limit(batch_size))

    def all(self):
        return self.query(self.table.select())

    def count(self) -> int:
        return self.conn.num_rows(self.name)

    def join(
        self,
        other: "Dataset",
        on_left: str,
        on_right: str,
        dataset_name: str | None = None,
    ):
        if dataset_name and self.conn.num_rows(dataset_name) > 0:
            print(f"Table {dataset_name} already exists, skipping creation!")
            return Dataset.from_table(dataset_name, self.conn)
        query = self.conn.join_query([self.name, other.name], [on_left, on_right])
        table_name = dataset_name if dataset_name else f"{self.name}_{other.name}"

        geo_cols = [
            db.Column(f"{self.name}_{col.name}", col.type) for col in self.geo_c
        ] + [db.Column(f"{other.name}_{col.name}", col.type) for col in other.geo_c]
        self.conn.table_from_query(
            table_name,
            query,
            f"{self.name}_{on_left}",
            non_null=geo_cols,
        )

        dataset = Dataset.from_table(table_name, self.conn)

        for col in dataset.geo_c:
            print("Adding spatial index to", col.name)
            self.conn.add_index(table_name, col.name, kind="SPATIAL")

        return Dataset.from_table(table_name, self.conn)

    def drop(self):
        self.conn.drop_table(self.name)


def census2021(
    code: str,
    conn: db.Connection,
    level: str = "oa",
    columns: dict[str, str] | None = None,
    table_name: str = "census21",
) -> Dataset:
    """Download 2021 UK Census data for given code
    :param code: census code to download (e.g. TS062 for NS-SEC, or TS003 for household composition)
    :param level: level of geography to download data for
    :param columns: dictionary mapping for renaming and filtering columns, or `None`. If this is `None`, all columns are returned. Otherwise, only the columns which are keys in the dictionary are returned, and they are renamed to the corresponding values.
    """
    download.census2021(code)

    filename = (
        Path("access_data")
        / "census2021"
        / code.lower()
        / f"census2021-{code.lower()}-{level}.csv"
    )

    def get_data():
        dfs = pd.read_csv(filename, index_col="geography code", chunksize=CHUNK_SIZE)
        for df in dfs:
            df = cast(pd.DataFrame, df)
            df.index.name = "geo_code"
            df = df.drop(columns=["geography", "date"])
            if columns:
                df.rename(columns=columns, inplace=True)
                df = df[columns.values()]
            yield df

    return Dataset.from_iterable(table_name, get_data(), conn)


def census2011(
    code: str,
    conn: db.Connection,
    table_name: str = "census11",
):
    """Download 2011 UK Census data for given code"""

    download.download_zip(
        f"https://www.nomisweb.co.uk/output/census/2011/{code.lower()}_2011_ward.zip",
        Path("access_data") / "census2011" / code.lower(),
    )

    filename = Path("access_data") / "census2011" / code.lower()

    assert filename.is_dir()
    filename = [x for x in filename.iterdir() if x.is_dir()][
        0
    ] / f"{code.upper()}DATA.CSV"

    def get_data():
        dfs = pd.read_csv(
            filename, index_col="GeographyCode", chunksize=CHUNK_SIZE, na_values=".."
        )
        for df in dfs:
            df = cast(pd.DataFrame, df)
            yield df

    return Dataset.from_iterable(
        table_name, get_data(), conn, dtypes={"GeographyCode": db.VARCHAR(9)}
    )


def census21_oa_boundaries(conn: db.Connection) -> Dataset:
    return _boundaries(
        "4d4e021d-fe98-4a0e-88e2-3ead84538537",
        "census21_boundaries",
        Path("access_data") / "census21" / "oa_boundaries.zip",
        conn,
        index_col=("OA21CD", db.VARCHAR(9)),
    )


def census11_boundaries(conn: db.Connection) -> Dataset:
    return _boundaries(
        "716fe23a-275d-48c1-824a-f2ff8f1a67fe",
        "census11_boundaries",
        Path("access_data") / "census2011" / "oa_boundaries.zip",
        conn,
        index_col=("wd11cd", db.VARCHAR(9)),
    )


def elections15_boundaries(conn: db.Connection) -> Dataset:
    return _boundaries(
        # "7f994782-13dd-42a1-b0e1-991ad1e01a9b",
        "098584d8-3bd6-4710-9d9d-02d67d7b79eb",
        "elections2015_boundaries",
        Path("access_data") / "elections" / "2015" / "boundaries.zip",
        conn,
        index_col=("pcon16cd", db.VARCHAR(9)),
    )


def referendum_boundaries(conn: db.Connection) -> Dataset:
    return _boundaries(
        "f84341f3-ee53-4470-8fa7-e860595291e4",
        "referendum_boundaries",
        Path("access_data") / "referendum" / "boundaries.zip",
        conn,
        index_col=("lad16cd", db.VARCHAR(9)),
    )


def _boundaries(
    dataset_id: str,
    table_name: str,
    filename: Path,
    conn: db.Connection,
    index_col: tuple[str, db.Datatype],
) -> Dataset:
    """Download boundary data for a given dataset ID from data.gov.uk
    This can be reused for different geographical boundary datasets (electoral, census, etc.),
    as long as the given dataset_id corresponds to a dataset with a "Shapefile" format.
    """
    download.data_gov_uk(dataset_id, filename, format="Shapefile")

    def get_data():
        slice_start = 0
        while True:
            df = geopandas.GeoDataFrame.from_file(
                filename,
                rows=slice(slice_start, slice_start + CHUNK_SIZE),
                engine="fiona",
            )
            df = cast(geopandas.GeoDataFrame, df)
            if len(df) == 0:
                break
            slice_start += CHUNK_SIZE

            # Convert the coordinates from BNG to coords
            df.set_crs("EPSG:27700", inplace=True)
            df["geometry"] = df["geometry"].to_crs("EPSG:4326")

            # Set index
            df.index = df[index_col[0]]
            df.index.name = index_col[0]
            df.drop(columns=[index_col[0]], inplace=True)

            yield df

    return Dataset.from_iterable(
        table_name,
        get_data(),
        conn,
        dtypes={
            "geometry": db.Geometry("GEOMETRY", srid=4326),
            index_col[0]: index_col[1],
        },
    )


def price_paid(
    years: list[int], conn: db.Connection, data_dir: str = "access_data"
) -> Dataset:
    """Access UK Price Paid Data for a set of years
    Unlike other functions, this will return an iterable of DataFrames, one for each part of the data.
    This allows the user to process the data in chunks, rather than loading everything into memory at once.

    :param data_dir: directory to store the downloaded data
    :return: iterable of DataFrames - each DataFrame corresponds to one part of the data
    """
    download.price_paid(years, output_dir=data_dir)

    num_files = len(years) * 2
    files = itertools.product(years, range(1, 3))

    # Since this dataset is quite large, I added a progress bar so that the user
    # can see if there is any progress being made.
    def get_data():
        for year, part in tqdm(files, total=num_files, desc="Loading price paid data"):
            filename = Path(data_dir) / "price_paid" / f"pp-{year}-part{part}.csv"
            dfs = pd.read_csv(
                filename,
                chunksize=CHUNK_SIZE,
                names=[
                    "transaction_unique_identifier",
                    "price",
                    "date_of_transfer",
                    "postcode",
                    "property_type",
                    "new_build_flag",
                    "tenure_type",
                    "primary_addressable_object_name",
                    "secondary_addressable_object_name",
                    "street",
                    "locality",
                    "town_city",
                    "district",
                    "county",
                    "ppd_category_type",
                    "record_status",
                ],
            )
            for df in dfs:
                df = cast(pd.DataFrame, df)
                df.index.name = "db_id"
                df.index = (str(year) + str(part) + df.index.astype(str)).astype(int)
                df.drop(columns=["transaction_unique_identifier"], inplace=True)
                yield df

    return Dataset.from_iterable(
        "price_paid",
        get_data(),
        conn,
        dtypes={
            "price": db.Integer,
            "date_of_transfer": db.DATE,
            "postcode": db.VARCHAR(8),
            "property_type": db.VARCHAR(1),
            "new_build_flag": db.VARCHAR(1),
            "tenure_type": db.VARCHAR(1),
            "ppd_category_type": db.VARCHAR(2),
            "record_status": db.VARCHAR(2),
        },
    )


def england_osm(data_dir: str = "access_data") -> Dataset:
    """Download OpenStreetMap data for England
    The data is stored in a geodatabase file, where each row corresponds to a feature in the map. This is tabulated into a geodataframe, where each row corresponds to one feature, and the geometry represents the shape of the feature.
    :param data_dir: directory to store the downloaded data
    """
    filename = Path(data_dir) / "osm" / "england-latest.osm.pbf"

    url = "https://download.geofabrik.de/europe/united-kingdom/england-latest.osm.pbf"

    print(f"Downloading OpenStreetMap data...", end="\r")
    download.download_file(url, filename)
    print(f"Downloaded OpenStreetMap data to {filename}")

    current_nodes = {}
    for obj in (
        osmium.FileProcessor(filename)
        .with_filter(osmium.filter.EmptyTagFilter())
        .with_locations()
    ):
        if len(obj.tags) <= 1:
            continue
        if obj.is_way():
            try:
                lat = sum((node.location.lat for node in obj.nodes)) / len(obj.nodes)
                lon = sum((node.location.lon for node in obj.nodes)) / len(obj.nodes)
            except Exception as e:
                print("Failed to get location for way", obj.id, e)
                continue
        elif obj.is_node():
            lat = obj.location.lat
            lon = obj.location.lon
        else:
            continue
        for tag in obj.tags:
            obj_id = (obj.type_str(), obj.id, tag.k.lower())
            current_nodes[obj_id] = {}
            current_nodes[obj_id]["value"] = tag.v
            current_nodes[obj_id]["geometry"] = Point(lon, lat)
        if len(current_nodes) >= CHUNK_SIZE:
            df = geopandas.GeoDataFrame.from_dict(current_nodes, orient="index")
            df.index = df.index.set_names(["osm_type", "osm_id", "tag_key"])
            df = df.reindex(sorted(df.columns), axis=1)
            yield df
            current_nodes = {}


def election_results(
    year: int, conn: db.Connection, data_dir: str = "access_data"
) -> Dataset:
    filename = Path(data_dir) / "elections" / "election_results.csv"
    url = "https://researchbriefings.files.parliament.uk/documents/CBP-8647/1918-2019election_results.csv"
    download.download_file(url, filename)

    dfs = pd.read_csv(
        filename, chunksize=CHUNK_SIZE, encoding="ISO-8859-1", index_col=False
    )

    def get_data():
        for chunk in dfs:
            chunk = cast(pd.DataFrame, chunk)
            df = chunk[chunk.election == str(year)].copy()
            if len(df) == 0:
                continue
            df.rename(columns=lambda x: x.strip(), inplace=True)
            df.set_index("constituency_id", inplace=True)
            yield df

    return Dataset.from_iterable(
        f"elections{year}_results",
        get_data(),
        conn,
        dtypes={"constituency_id": db.VARCHAR(10), "electorate": db.Integer},
    )


def brexit_referendum(conn: db.Connection, data_dir: str = "access_data") -> Dataset:
    filename = Path(data_dir) / "brexit" / "results.csv"
    url = "https://www.electoralcommission.org.uk/sites/default/files/2019-07/EU-referendum-result-data.csv"
    download.download_file(url, filename)

    # For this particular dataset, there are only ~400 rows, so we can load the entire dataset at once
    # and avoid using chunks
    df = pd.read_csv(filename, encoding="ISO-8859-1", index_col="Area_Code")
    df.rename(columns=lambda x: x.strip(), inplace=True)

    return Dataset.from_iterable(
        "referendum_results", [df], conn, dtypes={"Area_Code": db.VARCHAR(10)}
    )
