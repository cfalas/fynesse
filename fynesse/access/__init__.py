from . import db
from . import datasets

import pymysql
import pymysql.cursors
import csv
import pandas as pd
from typing import Generator

# import osmnx as ox
import warnings

__all__ = ["datasets", "db"]
# This file accesses the data

"""Place commands in this file to access the data electronically. Don't remove any missing values, or deal with outliers. Make sure you have legalities correct, both intellectual property and personal data privacy rights. Beyond the legal side also think about the ethical issues around this data. """


def housing_upload_join_data(conn: pymysql.Connection, year: int):
    start_date = str(year) + "-01-01"
    end_date = str(year) + "-12-31"

    cur = conn.cursor()
    print("Selecting data for year: " + str(year))
    cur.execute(
        "SELECT pp.price, pp.date_of_transfer, po.postcode, pp.property_type, "
        "pp.new_build_flag, pp.tenure_type, pp.locality, pp.town_city, "
        "pp.district, pp.county, po.country, po.latitude, po.longitude FROM "
        "(SELECT price, date_of_transfer, postcode, property_type, "
        "new_build_flag, tenure_type, locality, town_city, district, county "
        f'FROM pp_data WHERE date_of_transfer BETWEEN "{start_date}" AND '
        f'"{end_date}") AS pp '
        "INNER JOIN postcode_data AS po ON pp.postcode = po.postcode"
    )
    rows = cur.fetchall()

    csv_file_path = "output_file.csv"

    # Write the rows to the CSV file
    with open(csv_file_path, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write the data rows
        csv_writer.writerows(rows)
    print("Storing data for year: " + str(year))
    cur.execute(
        f"LOAD DATA LOCAL INFILE '{csv_file_path}' INTO TABLE "
        "`prices_coordinates_data` FIELDS TERMINATED BY ',' OPTIONALLY "
        "ENCLOSED by '\"' LINES STARTING BY '' TERMINATED BY '\n';"
    )
    print("Data stored for year: " + str(year))
    conn.commit()


def get_square_around_coordinate(latitude: float, longitude: float, width: float = 1):
    import math

    latitude_min = latitude - width / (4 * 110.574)
    latitude_max = latitude + width / (4 * 110.574)
    longitude_min = longitude - width / (
        2 * 111.320 * math.cos(latitude / 180 * math.pi)
    )
    longitude_max = longitude + width / (
        2 * 111.320 * math.cos(latitude / 180 * math.pi)
    )

    return (latitude_min, latitude_max, longitude_min, longitude_max)


def housing_data_around_coordinate(conn, latitude, longitude):
    warnings.filterwarnings("ignore", category=UserWarning)
    transactions = pd.read_sql(
        "select * from pp_data as pp inner join postcode_data as po on pp.postcode = po.postcode "
        "WHERE latitude > %s AND latitude < %s AND longitude > %s and longitude < %s",
        conn,
        params=get_square_around_coordinate(latitude, longitude, width=2),
    )
    transactions["date_of_transfer"] = pd.to_datetime(transactions["date_of_transfer"])

    return transactions


def get_buildings_around_coordinate(latitude, longitude):
    tags = {"building": True}

    buildings = ox.features.features_from_point((latitude, longitude), tags, dist=1000)
    buildings["addr:housename"] = buildings["addr:housename"].str.lower()
    buildings["name"] = buildings["name"].str.lower()
    return buildings


def count_pois_near_coordinates(
    latitude: float, longitude: float, tags: dict, distance_km: float = 1.0
) -> dict:
    """
    Count Points of Interest (POIs) near a given pair of coordinates within a specified distance.
    Args:
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.
        tags (dict): A dictionary of OSM tags to filter the POIs (e.g., {'amenity': True, 'tourism': True}).
        distance_km (float): The distance around the location in kilometers. Default is 1 km. Note: this does not form a circle, but rather a square with sides 2*distance_km
    Returns:
        dict: A dictionary where keys are the OSM tags and values are the counts of POIs for each tag.
    """
    pois = ox.features.features_from_point(
        (latitude, longitude), tags, dist=distance_km * 1000
    )
    pois_df = pd.DataFrame(pois)
    poi_counts = {}

    # I believe the original code is slightly incorrect when the tags values are not just True
    # For example, when querying for amenity=library, we might include some non-libraries in the count,
    # since they still have the amenity tag (albeit with a different value), since they were selected
    # due to some other filter
    extended_tags = []
    for tag in tags:
        if isinstance(tags[tag], list):
            for value in tags[tag]:
                extended_tags.append((tag, value))
        else:
            extended_tags.append((tag, tags[tag]))

    for tag, value in extended_tags:
        tag_name = tag if isinstance(value, bool) else f"{tag}_{value}"
        if tag not in pois_df.columns:
            poi_counts[tag_name] = 0
        elif isinstance(value, bool):
            # Select only those with the exact value
            poi_counts[tag_name] = pois_df[tag].notnull().sum()
        else:
            poi_counts[tag_name] = pois_df[pois_df[tag] == value][tag].notnull().sum()
    return poi_counts
