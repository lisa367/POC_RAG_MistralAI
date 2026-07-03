import pytest 
import pandas as pd
from datetime import datetime as dt
from vectorization import clean_events_data


run_date = dt.now().strftime("%Y-%m-%d")
data_path = "openagenda/evenements-publics-openagenda.json"
df_events = clean_events_data(data_path)

def test_data_region():
    """Test pour vérifier que tous les évènements importés se trouvent bien en région Ile-de-France."""
    df_events = clean_events_data(data_path)
    events_regions = df_events.location_region.unique().tolist()

    assert (len(events_regions) == 1 and events_regions[0] == 'Ile-de-France')

def test_data_less_than_a_year():
    """Test pour vérifier que tous les évènements importés datent au plus d'un an."""
    df_events["firstdate_begin"] = pd.to_datetime(df_events["firstdate_begin"], errors="coerce", utc=True)
    df_events["less_than_a_year_old"] = df_events.apply(
        lambda row: row.firstdate_begin >= pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=365),
        axis=1
    )
    events_age = df_events.query("less_than_a_year_old == False")
    assert events_age.empty
    
