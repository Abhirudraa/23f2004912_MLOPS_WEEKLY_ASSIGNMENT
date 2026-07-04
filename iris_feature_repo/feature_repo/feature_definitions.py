from datetime import timedelta

from feast import Entity, FeatureView, FileSource, Field
from feast.types import Float32

# -------------------------------------------------------
# Entity
# -------------------------------------------------------
iris = Entity(
    name="iris",
    join_keys=["iris_id"],
)

# -------------------------------------------------------
# Data Source
# -------------------------------------------------------
iris_source = FileSource(
    name="iris_source",
    path="data/iris_features.parquet",
    timestamp_field="event_timestamp",
)

# -------------------------------------------------------
# Feature View
# -------------------------------------------------------
iris_feature_view = FeatureView(
    name="iris_features",
    entities=[iris],
    ttl=timedelta(days=365),
    schema=[
        Field(name="sepal_length", dtype=Float32),
        Field(name="sepal_width", dtype=Float32),
        Field(name="petal_length", dtype=Float32),
        Field(name="petal_width", dtype=Float32),
    ],
    source=iris_source,
    online=True,
)
