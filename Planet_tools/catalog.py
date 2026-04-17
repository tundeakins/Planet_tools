import numpy as np
import matplotlib.pyplot as plt
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
from astropy import units as u


def get_catalog(catalog="NASAExoplanetArchive", table="pscomppars", select="*", transiting=False, return_type="table"):
    """ Get exoplanet data from a specified catalog and table.
    
    Parameters
    ----------
    catalog : str
        The catalog to query. Must be either "NASAExoplanetArchive" or "Exoplanet.eu".
    table : str
        The table to query within the catalog. For "NASAExoplanetArchive", must be either "pscomppars" or "exoplanets". For "Exoplanet.eu", not yet implemented.
    select : str
        The columns to select from the table. Default is "*", which selects all columns. e.g. "pl_name, pl_orbper, pl_rade" to select only the planet name, orbital period, and radius columns.
        see https://exoplanetarchive.ipac.caltech.edu/docs/API_PS_columns.html for available columns in the "pscomppars" table.
    transiting : bool
        If True, only return transiting planets. If False, return all planets. Default is False.
    return_type : str
        The format of the returned data. Must be either "table" (default) or "pandas". If "table", returns an Astropy Table. If "pandas", returns a Pandas DataFrame.

    Returns
    -------
    data : astropy.table.Table or pandas.DataFrame
        The queried data from the specified catalog and table.

    Examples
    --------
    >>> data = get_catalog(catalog="NASAExoplanetArchive", table="pscomppars", select="pl_name, pl_orbper, pl_rade", 
    ...                     transiting=True, return_type="pandas")
    >>> print(data.head())

    """
    assert catalog in ["NASAExoplanetArchive","Exoplanet.eu"], "Catalog must be either 'NASAExoplanetArchive' or 'Exoplanet.eu'."
    if catalog == "Exoplanet.eu":
        raise NotImplementedError("Exoplanet.eu catalog is not yet implemented.")
    else:
        assert table in ["pscomppars", "exoplanets"], "Table must be either 'pscomppars' or 'exoplanets'."
    
    if catalog == "NASAExoplanetArchive":
        if transiting: 
            data = NasaExoplanetArchive.query_criteria(table=table, select=select, where="tran_flag=1")
        else:
            data = NasaExoplanetArchive.query_criteria(table=table, select=select)
    
    if return_type == "pandas":
        data = data.to_pandas()
    return data