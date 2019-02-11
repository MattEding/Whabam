import numpy as np
import psycopg2 as pg


TABLES = ['rmse', 'spectral_bandwidth', 'spectral_centroid',
          'spectral_flatness', 'spectral_rolloff', 'zero_crossing_rate',
          'x_perf5', 'y_perf5', 'x_min3', 'y_min3', 'x_maj3', 'y_maj3',
          'bound_width', 'pitch_0', 'pitch_1', 'pitch_2', 'pitch_3',
          'pitch_4', 'pitch_5', 'pitch_6', 'pitch_7', 'pitch_8', 'pitch_9',
          'pitch_10', 'pitch_11', 'tempogram', 'dynamic_tempo']


def map_float_sql(flt):
    """Coerce python NaN and +/-Inf values to SQL values and return"""

    if np.isnan(flt):
        return "'NaN'"
    if np.isposinf(flt):
        return "'+infinity'"
    if np.isneginf(flt):
        return "'-infinity'"
    return flt


def connect_db(*, dbname='audio', host='localhost', port=5432):
    """Return connection to database"""

    connection_args = {'host': host, 'dbname': dbname, 'port': port}
    connection = pg.connect(**connection_args)
    return connection


def reset_tables(*, dbname='audio', host='localhost', port=5432):
    """Reset all tables in database"""

    drop_tables(dbname=dbname, host=host, port=port)
    create_tables(dbname=dbname, host=host, port=port)


def drop_tables(*, dbname='audio', host='localhost', port=5432):
    """Drop all tables in database"""

    inpt = ''
    while inpt not in 'ny':
        inpt = input('drop tables? y/n')

    if inpt != 'y':
        raise KeyboardInterrupt
        return

    connection = connect_db(dbname=dbname, host=host, port=port)
    cursor = connection.cursor()

    for table in TABLES + ['tempo']:
        query = f"DROP TABLE {table}"
        cursor.execute(query)

    cursor.execute('COMMIT;')
    cursor.close()
    connection.close()


def create_tables(*, dbname='audio', host='localhost', port=5432):
    """Create all tables in database"""

    connection = connect_db(dbname=dbname, host=host, port=port)
    cursor = connection.cursor()

    for table in TABLES:
        query = f"""
        CREATE TABLE {table}(
            song VARCHAR (64),
            segment INTEGER,
            mean FLOAT,
            median FLOAT,
            mode FLOAT,
            amax FLOAT,
            amin FLOAT,
            std FLOAT,
            "sum" FLOAT,
            prod FLOAT
        );
        """
        cursor.execute(query)

    query = """
    CREATE TABLE tempo(
        song VARCHAR (64),
        segment INTEGER,
        val FLOAT
    );
    """
    cursor.execute(query)

    cursor.execute('COMMIT;')
    cursor.close()
    connection.close()


def insert_features(name, seg_id, feature_dct, * ,dbname='audio', host='localhost', port=5432):
    """Insert audio features into database"""

    connection = connect_db(dbname=dbname, host=host, port=port)
    cursor = connection.cursor()

    tempo = np.asscalar(feature_dct.pop('tempo'))
    query = f"INSERT INTO tempo (song, segment, val) VALUES ('{name}', {seg_id}, {tempo});"
    cursor.execute(query)

    for feature, aggs_nm_tpl in list(feature_dct.items()):
        aggs_dct = aggs_nm_tpl._asdict()
        agg_keys = ', '.join(aggs_dct.keys())
        agg_vals = ', '.join(f'{map_float_sql(val)}' for val in aggs_dct.values())
        query = f"INSERT INTO {feature} (song, segment, {agg_keys}) VALUES ('{name}', {seg_id}, {agg_vals});"
        cursor.execute(query)

    cursor.execute('COMMIT;')
    cursor.close()
    connection.close()
