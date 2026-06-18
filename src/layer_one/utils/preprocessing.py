from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import TargetEncoder, StandardScaler, RobustScaler


column_transformer = ColumnTransformer(
    [
        (
            "target_encoder",
            TargetEncoder(),
            ['spectral_type', 'galaxy_population']
        ),
        (
            "standard_scaler",
            StandardScaler(),
            ['alpha', 'delta']
        ),
    ],
    remainder=RobustScaler()
)
