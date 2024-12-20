from geoalchemy2 import Geometry as Geometry
from typing import Iterable, Any, cast
from sqlalchemy.schema import SchemaItem
from sqlalchemy import Column, create_engine, text, VARCHAR, Integer, DATE
from sqlalchemy import MetaData, Table, Select
from sqlalchemy.orm import Query, Session
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.exc import SAWarning
import yaml

from sqlalchemy.exc import SQLAlchemyError
import pandas as pd


class RawGeometry(Geometry):
    as_binary = "geometry_id"


type Datatype = VARCHAR | RawGeometry | Integer | DATE


class Connection:
    def __init__(
        self, user: str, password: str, host: str, database: str, port: int = 3306
    ):
        """Create a database connection to the MariaDB database
            specified by the host URL and database name.
        :param user: username
        :param password: password
        :param host: host URL
        :param database: database name
        :param port: port number
        """
        self.engine = create_engine(
            f"mariadb+pymysql://{user}:{password}@{host}:{port}",
            echo=False,
        )

        # Create the database if it doesn't exist
        with self.engine.connect() as conn:
            conn.execute(text(f"CREATE DATABASE IF NOT EXISTS {database}"))
            conn.execute(text(f"USE {database}"))
        print("Connection established!")

        self.engine = create_engine(
            f"mariadb+pymysql://{user}:{password}@{host}:{port}/{database}",
            echo=False,
        )

        with self.engine.connect() as conn:
            # Create dummy functions, dummy table
            conn.execute(
                text(
                    "CREATE FUNCTION IF NOT EXISTS geometry_id(arg1 GEOMETRY) "
                    "RETURNS GEOMETRY DETERMINISTIC BEGIN "
                    "RETURN arg1; END"
                )
            )

            if not self.check_table_exists("dummy"):
                df = pd.DataFrame([0] * 10000, columns=["zero"])
                df.to_sql("dummy", self.engine, if_exists="replace")

        self.schema = database
        self.metadata = MetaData()

    @staticmethod
    def from_file(file_path: str):
        """Create a database connection to the MariaDB database
            specified by the host URL and database name.
        :param file_path: path to the yaml file
        :return: Connection object
        """
        with open(file_path, "r") as file:
            config = yaml.safe_load(file)
        return Connection(**config)

    def check_table_exists(self, table_name: str):
        assert self.engine is not None, "No connection established!"
        with self.engine.connect() as conn:
            try:
                conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                return True
            except Exception as _:
                return False

    def upload_df(
        self,
        dfs: Iterable[pd.DataFrame],
        table_name: str,
        dtype: dict[str, Any],
    ):
        """Upload a pandas DataFrame to the database
        :param df: pandas DataFrame
        :param table_name: name of the table where the data will be stored
        """
        # Check if table exists
        assert self.engine is not None, "No connection established!"

        if isinstance(dfs, pd.DataFrame):
            # This is a single DataFrame
            dfs = [dfs]

        table_exists = self.check_table_exists(table_name)
        with self.engine.connect() as conn:
            total_chunks = 0
            uploaded_chunks = 0
            id_column = None
            for df in dfs:
                id_column = cast(str, df.index.name) or "index"
                if isinstance(df.index, pd.MultiIndex):
                    id_columns = list(df.index.names)
                else:
                    id_columns = [id_column]

                # Check if this chunk of data already exists in the table
                try:
                    query = text(
                        f"SELECT COUNT(*) FROM `{table_name}` WHERE {' AND '.join([f'`{id_column}` = :{id_column}' for id_column in id_columns])}",
                    )
                    result = conn.execute(
                        query,
                        {id_column: df.index[0]}
                        if len(id_columns) == 1
                        else dict(zip(id_columns, df.index[0])),
                    ).fetchone()
                except Exception as e:
                    # Table does not exist or data not present
                    result = None

                total_chunks += 1
                if not result or result[0] == 0:
                    uploaded_chunks += 1
                    df.to_sql(
                        table_name,
                        self.engine,
                        if_exists="append",
                        schema=self.schema,
                        chunksize=10000,
                        dtype=dtype,
                    )
            print(
                f"Uploaded {uploaded_chunks} new out of {total_chunks} total chunks to table {table_name}"
            )

            # Add a PK to the table - without this, the tables are not detected
            # later on during reflection when joining tables/adding indices
            if not table_exists and id_column:
                conn.execute(
                    text(f"ALTER TABLE `{table_name}` ADD PRIMARY KEY(`{id_column}`)")
                )
                print(f"Added primary key to table {table_name}")

    def reflect_tables(self):
        import warnings

        warnings.filterwarnings("ignore", category=SAWarning)
        Base = automap_base()
        Base.prepare(autoload_with=self.engine, reflect=True)

        self.tables = Base.classes

    def reflect_table(self, table_name: str, clear: bool = False):
        assert self.engine is not None, "No connection established!"
        import warnings

        # This is needed because it complains about not knowing about Geometry columns,
        # but it works because of geoalchemy2
        warnings.filterwarnings("ignore", category=SAWarning)

        if clear:
            self.metadata.clear()

        table = Table(table_name, self.metadata, autoload_with=self.engine)
        return table

    def add_index(
        self, table_name: str, column: str | Iterable[str], kind: str = "BTREE"
    ):
        assert self.engine is not None, "No connection established!"

        if kind == "SPATIAL":
            if not isinstance(column, str):
                raise ValueError(
                    "Only single-column indexes are supported for SPATIAL index"
                )
            index = f"CREATE SPATIAL INDEX IF NOT EXISTS idx_{table_name}_{column} ON {table_name}({column})"
        else:
            columns = (
                ", ".join([f"`{col}`" for col in column])
                if isinstance(column, list)
                else f"`{column}`"
            )
            index = f"CREATE INDEX IF NOT EXISTS idx_{table_name}_{column} ON {table_name}({columns}) USING {kind}"
        with self.engine.connect() as conn:
            conn.execute(text(index))
            print(f"Added index to table {table_name} for column {column}")

    def join_query(
        self, tables: list[str], on: list[str], columns: list[str] | None = None
    ):
        if not hasattr(self, "tables"):
            self.reflect_tables()

        reflected_tables_dict = {
            tbl: Table(tbl, self.metadata, autoload_with=self.engine) for tbl in tables
        }

        for table_name, table in reflected_tables_dict.items():
            additional_reflect_cols: list[SchemaItem] = []
            for col in table.columns:
                if isinstance(col.type, Geometry):
                    additional_reflect_cols.append(
                        Column(col.name, RawGeometry(col.name), nullable=False)
                    )

            if additional_reflect_cols:
                reflected_tables_dict[table_name] = Table(
                    table_name,
                    self.metadata,
                    *additional_reflect_cols,
                    autoload_with=self.engine,
                    extend_existing=True,
                )

        reflected_tables = list(reflected_tables_dict.values())

        session = Session(self.engine)
        query = session.query(*reflected_tables)
        for tbl, col in zip(reflected_tables[1:], on[1:]):
            query = query.join(
                tbl, reflected_tables[0].columns[on[0]] == tbl.columns[col]
            )

        return query

    def table_from_query(
        self,
        table_name: str,
        query: Query[Any],
        pk: str,
        non_null: list[Column[Any]] | None = None,
    ):
        assert self.engine is not None, "No connection established!"

        if self.check_table_exists(table_name):
            print(f"Table {table_name} already exists, skipping creation!")
            return

        with self.engine.connect() as conn:
            conn.execute(text(f"CREATE TABLE `{table_name}` AS ({query})"))

            # Add a PK to the table
            conn.execute(text(f"ALTER TABLE `{table_name}` ADD PRIMARY KEY(`{pk}`)"))
            print(f"Created table {table_name} from query")

            if non_null:
                for col in non_null:
                    if isinstance(col.type, Geometry):
                        dtype = "GEOMETRY"
                    else:
                        dtype = col.type.compile()
                    conn.execute(
                        text(
                            f"ALTER TABLE `{table_name}` MODIFY `{col.name}` "
                            f"{dtype} NOT NULL"
                        )
                    )

    def num_rows(self, table_name: str) -> int:
        assert self.engine is not None, "No connection established!"
        with self.engine.connect() as conn:
            try:
                result = conn.execute(
                    text(f"SELECT COUNT(*) FROM `{table_name}`")
                ).fetchone()
                if result is None:
                    return 0
                rows = cast(int, result[0])
                return rows
            except Exception as e:
                return 0

    def sample_table(
        self, table_name: str, prob: float = 0.1, columns: list[str] | None = None
    ) -> pd.DataFrame:
        assert self.engine is not None, "No connection established!"
        with self.engine.connect() as conn:
            try:
                columns_sql = "*" if columns is None else ", ".join(f"`{columns}`")
                sql = f"SELECT {columns_sql} FROM `{table_name}` WHERE RAND() < {prob}"
                return pd.read_sql(sql, conn)
            except Exception as _:
                return pd.DataFrame()

    def drop_table(self, table_name: str):
        assert self.engine is not None, "No connection established!"
        with self.engine.connect() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS `{table_name}`"))
            print(f"Dropped table {table_name}")


class MockConnection(Connection):
    """Mock connection class for testing purposes"""

    def __init__(self):
        pass

    def upload_df(
        self,
        dfs: Iterable[pd.DataFrame],
        table_name: str,
        dtype: dict[str, Any],
    ):
        for df in dfs:
            print(f"Got chunk with {len(df)} rows to table {table_name}")
        pass
