from feast import FeatureStore

store = FeatureStore(
    repo_path="iris_feature_repo/feature_repo"
)

print(store.list_entities())
print(store.list_feature_views())
