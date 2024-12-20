from __future__ import annotations
from geopandas.geoseries import GeometryDtype
from shapely.geometry import Point
from .config import *

from . import access

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Callable, cast, Any
import tqdm.auto as tqdm
from sqlalchemy.sql import select, func
import seaborn as sns

# import osmnx as ox
import warnings

from sqlalchemy import text


def histogram_from_distribution(n_samples: int, bins: list[Any], p: list[float]):
    assert len(p) == len(bins), "Length of p must be equal to length of bins"
    total_p = sum(p)
    return {bins[i]: int(n_samples * p[i] / total_p) for i in range(len(bins))}


def random_points_in_geometry(
    num_points: int,
    geometry: GeometryDtype,
    conn: access.db.Connection,
    extra_columns: tuple[str, str, str] | None = None,
):
    """This function generates random points within the given geometries.

    Note: This should not be used for large numbers of points (e.g. more than 5000),
          as it is limited by the dummy table size
    """
    bbox = geometry.bounds
    bbox = cast(tuple[float, float, float, float], bbox)

    valid_points = []

    while len(valid_points) < 2 * num_points:
        lats = np.random.uniform(bbox[1], bbox[3], num_points * 4)
        lons = np.random.uniform(bbox[0], bbox[2], num_points * 4)
        candidate_points = np.column_stack((lons, lats))
        mask = np.array([geometry.contains(Point(x, y)) for x, y in candidate_points])
        valid_points.extend(candidate_points[mask])

    df = pd.DataFrame(valid_points, columns=["lon", "lat"])

    if extra_columns:
        table, geom_column, id_column = extra_columns
        df[id_column] = df.apply(
            lambda x: get_area_at_point(
                x.lat, x.lon, conn, [id_column], geom_column, table
            ),
            axis=1,
        )

    df = df.dropna()[:num_points]

    return df


def get_area_at_point(
    lat: float,
    lon: float,
    conn: access.db.Connection,
    columns: list[str],
    geom_column: str,
    table: str,
):
    assert conn.engine is not None, "No connection established!"

    with conn.engine.connect() as con:
        res = con.execute(
            text(
                f"SELECT {', '.join(columns)} FROM {table} WHERE ST_CONTAINS(`{table}`.`{geom_column}`, ST_GeomFromText('POINT({lon} {lat})'))"
            )
        ).fetchall()
    if len(res) == 0:
        return None
    return res[0][0]


def _random_points_in_geometry_sql(
    num_points: int,
    geometry: GeometryDtype,
    conn: access.db.Connection,
    extra_columns: tuple[str, str, str] | None = None,
):
    """This function generates random points within the given geometries.

    Note: This should not be used for large numbers of points (e.g. more than 5000),
          as it is limited by the dummy table size
    """
    bbox = geometry.bounds

    if extra_columns is None:
        extra_table = ""
        extra_clause = ""
        extra_select = ""

    else:
        table, geom_column, id_column = extra_columns

        extra_table = f", {table}"
        extra_clause = f"AND ST_CONTAINS({table}.`{geom_column}`, POINT(x,y)) "
        extra_select = f", {table}.`{geom_column}`, {table}.`{id_column}`"

    query = (
        f"SELECT RAND() * (({bbox[2]}) - ({bbox[0]})) + ({bbox[0]}) AS x, "
        f"RAND() * (({bbox[3]}) - ({bbox[1]})) + {bbox[1]} AS y"
        f"{extra_select} "
        f"FROM dummy{extra_table} "
        f" HAVING ST_CONTAINS(ST_GEOMFROMTEXT('{geometry}'), POINT(x, y)) "
        f"{extra_clause}"
        f"LIMIT {num_points}"
    )

    print(query)

    df = pd.read_sql(query, conn.engine)
    return df


def elections_referendum_per_party(random_voters: access.datasets.Dataset):
    parties = [
        "Conservative",
        "Liberal Democrats",
        "Labour",
        "SNP / Plaid Cymru",
        "Others",
    ]

    ref = ["Remain", "Leave"]

    party_counts = random_voters.count_by(
        [random_voters.c.elections_vote, random_voters.c.referendum_vote]
    )
    party_counts.rename(dict(enumerate(parties)), inplace=True, level=0)
    party_counts.rename(dict(enumerate(ref)), inplace=True, level=1)
    total_per_party = party_counts.groupby(level=0).sum()
    totals_ref = party_counts.groupby(level=1).sum()

    average_ratio = totals_ref.values[0] / totals_ref.sum().values[0]

    party_prop = party_counts / total_per_party
    ax = party_prop.unstack().plot.bar(stacked=True)
    ax.axhline(average_ratio, color="red", linestyle="--", label="Average")
    # ax.axhline(1 - average_ratio, color="green", linestyle="--", label="Average leave")
    ax.legend(loc="lower left")
    ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_ylim([0.35, 0.65])
    return party_counts


def referendum_plot_correlation(
    random_voters: access.datasets.Dataset,
    column: access.db.Column[Any],
    values: list[str],
):
    df = random_voters.query(
        select(
            *[func.avg(column == i) for i in range(len(values))],
            func.avg(random_voters.c.referendum_vote),
        ).group_by(random_voters.c.referendum_results_Area_Code)
    )

    columns = {"avg_" + str(i + 1): v for i, v in enumerate(values)}
    columns[f"avg_{len(values)+1}"] = "Leave"
    df = df.rename(columns=columns)

    fig, axs = plt.subplots(
        (len(values) + 3) // 4,
        min(4, len(values)),
        figsize=(20, 4 * (len(values) + 3) // 4),
    )
    axs = axs.flatten()
    for i, party in enumerate(columns.values()):
        if party == "Leave":
            continue
        # df_curr = df[(df[party] != 0) & (df[party] != 1)]
        df_curr = df
        corr = df_curr[party].corr(df_curr["Leave"])
        sns.regplot(
            x=party,
            y="Leave",
            data=df_curr,
            label="R: {:.2f}".format(corr),
            fit_reg=True,
            ax=axs[i],
            line_kws=dict(color="r"),
        )
        axs[i].set_xlabel(f"% {party}")
        axs[i].set_ylabel("% Leave")
        axs[i].legend()


def random_voters(
    elections: access.datasets.Dataset,
    referendum: access.datasets.Dataset,
    num_voters: int,
):
    elections_df = elections.filter(
        elections.c.elections2015_results_constituency_id.not_like("N%")
    )

    referendum_df = referendum.all()

    # Calculate how many points to place in each constituency
    samples_per_constituency = histogram_from_distribution(
        num_voters,
        elections_df.index.to_list(),
        elections_df["elections2015_results_electorate"].to_list(),
    )

    num_voters = sum(samples_per_constituency.values())

    num_existing_rows = elections.conn.num_rows("random_voters")
    if num_existing_rows >= num_voters:
        print("Skipping generation of random voters, table already exists")
        return access.datasets.Dataset.from_table("random_voters", elections.conn)
    elif num_existing_rows > 0:
        print("Deleting existing random voters table, as it is not complete")
        elections.conn.drop_table("random_voters")

    def get_data():
        # Place the points
        next_index = 0
        with tqdm.tqdm(total=num_voters) as pbar:
            for constituency, n_points in samples_per_constituency.items():
                if n_points == 0:
                    continue

                random_points = random_points_in_geometry(
                    n_points,
                    elections_df.loc[constituency].elections2015_boundaries_geometry,
                    elections.conn,
                    extra_columns=(
                        "referendum",
                        "referendum_boundaries_geometry",
                        "referendum_results_Area_Code",
                    ),
                )

                if len(random_points) != n_points:
                    print(
                        f"Could not place {n_points - len(random_points)} points in {constituency}"
                    )

                random_points.index += next_index

                # Add constituency information to the points, and cast votes
                random_points["elections_area"] = constituency

                current_constituency = elections_df.loc[constituency]

                # Fill NaNs with 0
                current_constituency = current_constituency.fillna(0)

                election_dist = np.array(
                    [
                        current_constituency.elections2015_results_con_share,
                        current_constituency.elections2015_results_lib_share,
                        current_constituency.elections2015_results_lab_share,
                        current_constituency.elections2015_results_natSW_share,
                        current_constituency.elections2015_results_oth_share,
                    ]
                )
                election_dist /= election_dist.sum()

                # Add vote information
                random_points["elections_vote"] = np.random.choice(
                    list(range(len(election_dist))),
                    n_points,
                    p=election_dist,
                )

                random_points["referendum_vote"] = random_points.apply(
                    lambda x: np.random.choice(
                        [0, 1],
                        p=[
                            referendum_df.loc[
                                x.referendum_results_Area_Code
                            ].referendum_results_Pct_Remain
                            / 100,
                            referendum_df.loc[
                                x.referendum_results_Area_Code
                            ].referendum_results_Pct_Leave
                            / 100,
                        ],
                    ),
                    axis=1,
                )
                yield random_points
                next_index = max(next_index, random_points.index.max() + 1)
                pbar.update(n_points)

    dataset = access.datasets.Dataset.from_iterable(
        "random_voters", get_data(), elections.conn
    )

    dataset.conn.add_index("random_voters", "elections_area")
    dataset.conn.add_index("random_voters", "referendum_results_Area_Code")
    dataset.conn.add_index("random_voters", "elections_vote")
    dataset.conn.add_index("random_voters", "referendum_vote")

    return dataset


def random_voters_census_ward(
    random_voters: access.datasets.Dataset, census_boundaries: access.datasets.Dataset
):
    random_voters.table = random_voters.conn.reflect_table(
        random_voters.name, clear=True
    )
    if "wd11cd" in random_voters.c:
        print("Skipping finding census ward for random voters, already exists")
        return

    # Create column
    with random_voters.conn.engine.connect() as con:
        con.execute(text("ALTER TABLE random_voters ADD COLUMN wd11cd VARCHAR(9)"))

    random_voters.table = random_voters.conn.reflect_table(
        random_voters.name, clear=True
    )

    random_voters.conn.add_index("random_voters", "wd11cd")
    # Read random_voters in batches
    for df in random_voters.batch():
        # Generate random census data
        # Find cenus ward
        df["wd11cd"] = df.apply(
            lambda x: get_area_at_point(
                x.lat,
                x.lon,
                random_voters.conn,
                ["wd11cd"],
                "geometry",
                census_boundaries.name,
            ),
            axis=1,
        )

        with random_voters.conn.engine.begin() as con:
            for index, row in df.iterrows():
                con.execute(
                    text(
                        "UPDATE random_voters SET wd11cd = :wd11cd WHERE `index` = :u_index"
                    ),
                    {"u_index": index, "wd11cd": row.wd11cd},
                )


def generate_random_voters_census_property(
    random_voters: access.datasets.Dataset,
    census11: access.datasets.Dataset,
    variables: dict[str, str],
    column_name: str,
    distribution_fn: Callable[[pd.Series], int],
):
    random_voters.table = random_voters.conn.reflect_table(
        random_voters.name, clear=True
    )
    if column_name in random_voters.c:
        print("Skipping finding census ward for random voters, already exists")
        return

    # Create column
    with random_voters.conn.engine.connect() as con:
        con.execute(
            text(f"ALTER TABLE random_voters ADD COLUMN `{column_name}` INTEGER")
        )

    random_voters.table = random_voters.conn.reflect_table(
        random_voters.name, clear=True
    )

    random_voters.conn.add_index("random_voters", column_name)

    joins = random_voters.query_batched(
        select("*")
        .select_from(random_voters.table)
        .join(
            census11.table,
            random_voters.c.wd11cd == census11.c.GeographyCode,
        ),
        chunksize=1000,
    )

    # Read random_voters in batches
    for df in joins:
        # Generate random census data
        df[column_name] = df.apply(
            distribution_fn,
            axis=1,
        )

        with random_voters.conn.engine.begin() as con:
            for _, row in df.iterrows():
                con.execute(
                    text(
                        f"UPDATE random_voters SET {column_name} = :value WHERE `index` = :u_index"
                    ),
                    {"u_index": row["index"], "value": row[column_name]},
                )


def random_voters_gen_origin_country(
    random_voters: access.datasets.Dataset, census11_country: access.datasets.Dataset
):
    variables = {"eu": "QS203EW0016", "uk": "QS203EW0003", "total": "QS203EW0001"}

    def choose(x: pd.Series[int]):
        return np.random.choice(
            range(3),
            p=[
                x[variables["uk"]] / x[variables["total"]],
                x[variables["eu"]] / x[variables["total"]],
                (x[variables["total"]] - x[variables["eu"]] - x[variables["uk"]])
                / x[variables["total"]],
            ]
            if not x[variables.values()].hasnans
            else [0.84, 0.06, 0.1],
        )

    generate_random_voters_census_property(
        random_voters, census11_country, variables, "country_of_birth", choose
    )


def random_voters_gen_nssec(
    random_voters: access.datasets.Dataset, census11_nssec: access.datasets.Dataset
):
    variables = {
        "Higher managerial, administrative and professional occupations": "QS607EW0002",
        "Lower managerial, administrative and professional occupations": "QS607EW0011",
        "Intermediate occupations": "QS607EW0019",
        "Small employers and own account workers": "QS607EW0024",
        "Lower supervisory and technical occupations": "QS607EW0031",
        "Semi-routine occupations": "QS607EW0036",
        "Routine occupations": "QS607EW0044",
        "Never worked and long-term unemployed": "QS607EW0050",
        "Full-time students": "QS607EW0054",
    }

    def choose(x: pd.Series[int]):
        if x[variables.values()].hasnans:
            p = np.array(
                [
                    24127673,
                    48767151,
                    29780050,
                    22012093,
                    16118172,
                    32693363,
                    25743418,
                    13105898,
                    21018724,
                ]
            )
            p = p / p.sum()
        else:
            total = x[variables.values()].sum()
            p = x[variables.values()] / total

        return np.random.choice(
            range(len(variables)),
            p=p,
        )

    generate_random_voters_census_property(
        random_voters, census11_nssec, variables, "nssec", choose
    )


def random_voters_gen_age(
    random_voters: access.datasets.Dataset, census11_age: access.datasets.Dataset
):
    variables = {
        "Age 18 to 19": "KS102EW0008",
        "Age 20 to 24": "KS102EW0009",
        "Age 25 to 29": "KS102EW0010",
        "Age 30 to 44": "KS102EW0011",
        "Age 45 to 59": "KS102EW0012",
        "Age 60 to 64": "KS102EW0013",
        "Age 65 to 74": "KS102EW0014",
        "Age 75 to 84": "KS102EW0015",
        "Age 85 to 89": "KS102EW0016",
        "Age 90 and over": "KS102EW0017",
    }

    def choose(x: pd.Series[int]):
        if x[variables.values()].hasnans:
            p = np.array(
                [
                    1_375_315 + 3_595_321,
                    3_650_881 + 10_944_271,
                    10_276_902 + 3_172_277,
                    4_552_283 + 2_928_118 + 776_311 + 403_817,
                ]
            )
            p = p / p.sum()
        else:
            total = x[variables.values()].sum()
            p = (x[variables.values()] / total).to_numpy()
            p = [p[0] + p[1], p[2] + p[3], p[4] + p[5], p[6] + p[7] + p[8] + p[9]]

        return np.random.choice(
            range(len(p)),
            p=p,
        )

    generate_random_voters_census_property(
        random_voters, census11_age, variables, "age", choose
    )


def random_voters_median_age(
    random_voters: access.datasets.Dataset, census11_age: access.datasets.Dataset
):
    df = random_voters.query(
        select(func.avg(random_voters.c.referendum_vote), census11_age.c.KS102EW0019)
        .select_from(random_voters.table)
        .join(
            census11_age.table,
            random_voters.c.wd11cd == census11_age.c.GeographyCode,
        )
        .group_by(census11_age.c.GeographyCode, census11_age.c.KS102EW0019),
    )
    ax = sns.regplot(
        x="KS102EW0019", y="avg_1", data=df, fit_reg=True, line_kws=dict(color="r")
    )
    ax.set_xlabel("Median age")
    ax.set_ylabel("% Leave")


def random_in_england(conn: access.db.Connection, columns: list[str]):
    england = (49.895, -6.237, 58.635, 1.768)
    point = (
        np.random.uniform(england[0], england[2]),
        np.random.uniform(england[1], england[3]),
    )
    res = get_oa_at_point(point[0], point[1], conn, columns)
    while res is None:
        point = (
            np.random.uniform(england[0], england[2]),
            np.random.uniform(england[1], england[3]),
        )
        res = get_oa_at_point(point[0], point[1], conn, columns)
    return *point, *res


def get_oa_at_point(
    lat: float, lon: float, conn: access.db.Connection, columns: list[str]
):
    assert conn.engine is not None, "No connection established!"

    with conn.engine.connect() as con:
        res = con.execute(
            text(
                f"SELECT {', '.join(columns)} FROM census21 WHERE ST_CONTAINS(`census21`.`geometry`, ST_GeomFromText('POINT({lon} {lat})'))"
            )
        ).fetchall()
    if len(res) == 0:
        return None
    return res[0]


def count_osm_in_oa(
    oa_table: str,
    conn: access.db.Connection,
    osm_filter: str,
    boundary_col: str = "boundary",
):
    with conn.engine.connect() as con:
        query = f"""SELECT `{oa_table}`.oa, COUNT(DISTINCT osm.osm_id) AS osm_count
            FROM osm, `{oa_table}` 
            WHERE ST_CONTAINS(`{oa_table}`.`{boundary_col}`, osm.`geometry`) AND ({osm_filter})
            GROUP BY `{oa_table}`.L15, `{oa_table}`.oa"""
        df = pd.read_sql(
            query,
            con,
            index_col="oa",
        )
    return df.osm_count


def plot_buildings(buildings: pd.DataFrame):
    has_address = ~(buildings["addr:street"].isna()) & (
        ~(buildings["addr:housenumber"].isna()) | ~(buildings["addr:housename"].isna())
    )

    _, ax = plt.subplots()
    buildings[has_address].plot(ax=ax, color="blue", alpha=0.7, markersize=10)
    buildings[~has_address].plot(ax=ax, color="red", alpha=0.7, markersize=10)

    ax.legend(["With address", "Without address"])

    print(
        f"Found {has_address.sum()} buildings with an address out of {len(buildings)}"
    )

    plt.show()


def match_building_to_transactions(transactions, buildings):
    transactions["osmid"] = np.nan
    transactions["osm_element_type"] = ""

    for index, transaction in transactions.iterrows():
        matching_postcode_exact = (
            buildings["addr:postcode"] == transaction["postcode"].iloc[0]
        )
        matching_postcode_seven = (
            buildings["addr:postcode"] == transaction["postcode_fixed_width_seven"]
        )
        matching_postcode = matching_postcode_exact | matching_postcode_seven

        primary_object_name = transaction["primary_addressable_object_name"].lower()

        # In some cases, the primary addressable object name has the format house name, house number
        primary_object_name_parts = primary_object_name.split(", ")
        primary_object_name = primary_object_name_parts[0]
        house_number = primary_object_name_parts[-1]

        if transaction.secondary_addressable_object_name:
            continue

        matching_streetno = (buildings["addr:housenumber"] == primary_object_name) | (
            buildings["addr:housenumber"] == house_number
        )
        matching_house = (buildings["addr:housename"] == primary_object_name) | (
            buildings["name"] == primary_object_name
        )

        matching_buildings = buildings[
            matching_postcode & (matching_streetno | matching_house)
        ]
        if len(matching_buildings) == 1:
            transactions.loc[index, "osm_element_type"] = matching_buildings.index[0][0]
            transactions.loc[index, "osmid"] = matching_buildings.index[0][1]
            continue

        # Sometimes, street numbers have a letter suffix to indicate the flat etc, which is at a granularity lower than a building, so we can ignore it
        stripped_house_number = primary_object_name.rstrip("abcdef")
        matching_streetno_stripped = (
            buildings["addr:housenumber"] == stripped_house_number
        )
        matching_buildings = buildings[matching_postcode & (matching_streetno_stripped)]
        if len(matching_buildings) == 1:
            transactions.loc[index, "osm_element_type"] = matching_buildings.index[0][0]
            transactions.loc[index, "osmid"] = matching_buildings.index[0][1]
            continue

        # Try to match buildings which don't have a postcode
        matching_street = buildings["addr:street"] == transaction["street"].lower()
        matching_buildings = buildings[matching_street & (matching_streetno)]
        if len(matching_buildings) == 1:
            transactions.loc[index, "osm_element_type"] = matching_buildings.index[0][0]
            transactions.loc[index, "osmid"] = matching_buildings.index[0][1]
            continue

    print(
        f"Found {(~transactions.osmid.isna()).sum()} matches out "
        f"of {len(transactions)} transactions"
    )
    return transactions


def get_building_areas(buildings):
    building_projections = ox.projection.project_gdf(buildings)
    return building_projections.area


def adjust_for_inflation(price, date):
    # This is a very crude approximation
    pass


def plot_correlation(dataframe: pd.DataFrame, x: str, y: str, **kwargs: Any):
    dataframe.plot.scatter(x=x, y=y, **kwargs)
    print(
        f"There is a {dataframe[x].corr(dataframe[y])} correlation "
        f"between {x} and {y}"
    )


def plot_pois_near_coordinates(
    latitude: float,
    longitude: float,
    tags: dict[str, str | list[str]],
    distance_km: float = 1.0,
):
    position = (latitude, longitude)
    pois = ox.features.features_from_point(position, tags, dist=distance_km * 1000)

    # Get place boundary related to the place name as a geodataframe

    warnings.filterwarnings("ignore", category=FutureWarning)
    graph = ox.graph_from_point(position, dist=distance_km * 1000)

    # Retrieve nodes and edges
    nodes, edges = ox.graph_to_gdfs(graph)

    fig, ax = plt.subplots()

    edges.plot(ax=ax, linewidth=1, edgecolor="dimgray")

    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")

    # Plot all POIs
    pois.plot(ax=ax, color="blue", alpha=0.7, markersize=10)
    plt.tight_layout()
