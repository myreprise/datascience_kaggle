from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer

# Custom transformer for cleaning and preprocessing the dataset
class CustomPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        # Impute missing values
        # For 'fuel_type', impute with mode
        fuel_type_mode = df['fuel_type'].mode()[0]
        df['fuel_type'].fillna(fuel_type_mode, inplace=True)

        # For 'accident', impute with 'None reported'
        df['accident'].fillna('None reported', inplace=True)

        # For 'clean_title', impute with mode
        clean_title_mode = df['clean_title'].mode()[0]
        df['clean_title'].fillna(clean_title_mode, inplace=True)

        # Clean categorical features
        categorical_features = df.select_dtypes(include=['object']).columns
        for col in categorical_features:
            # Convert to lowercase and strip leading/trailing whitespace
            df[col] = df[col].str.lower().str.strip()

        # Group rare models
        model_counts = df['model'].value_counts()
        rare_models = model_counts[model_counts < 100].index
        df['model'] = df['model'].apply(lambda x: 'other' if x in rare_models else x)

        # Clean 'fuel_type' feature
        df['fuel_type'] = df['fuel_type'].replace(['–', 'not supported'], 'unknown')

        # Group rare fuel types
        df['fuel_type'] = df['fuel_type'].replace(['plug-in hybrid', 'unknown'], 'other')

        # Extract features from 'engine'
        # Extract horsepower
        df['horsepower'] = df['engine'].str.extract(r'(\d+\.\d+)hp').astype(float)

        # Extract engine size in liters
        df['engine_size'] = df['engine'].str.extract(r'(\d+\.\d+)l').astype(float)

        # Extract number of cylinders
        df['num_cylinders'] = df['engine'].str.extract(r'(\d+) cylinder').astype(float)

        # Fill any missing values in the extracted features with median
        df['horsepower'].fillna(df['horsepower'].median(), inplace=True)
        df['engine_size'].fillna(df['engine_size'].median(), inplace=True)
        df['num_cylinders'].fillna(df['num_cylinders'].median(), inplace=True)

        # Drop the original 'engine' column as it is now represented by extracted features
        df.drop(columns=['engine'], inplace=True)

        # Extract features from 'transmission'
        # Extract transmission type (automatic or manual)
        df['transmission_type'] = df['transmission'].apply(
            lambda x: 'automatic' if re.search(r'a/t|automatic|cvt', x) else 'manual'
        )

        # Extract number of speeds
        df['num_speeds'] = df['transmission'].str.extract(r'(\d+)-speed').astype(float)

        # Fill any missing values in the extracted features with median
        df['num_speeds'].fillna(df['num_speeds'].median(), inplace=True)

        # Drop the original 'transmission' column as it is now represented by extracted features
        df.drop(columns=['transmission'], inplace=True)

        # Clean 'ext_col' feature
        # Reduce categories in 'ext_col' based on a threshold count of 100
        ext_col_counts = df['ext_col'].value_counts()
        rare_ext_cols = ext_col_counts[ext_col_counts < 100].index
        df['ext_col'] = df['ext_col'].apply(lambda x: 'other' if x in rare_ext_cols else x)

        # Clean 'int_col' feature
        # Reduce categories in 'int_col' based on a threshold count of 100
        int_col_counts = df['int_col'].value_counts()
        rare_int_cols = int_col_counts[int_col_counts < 100].index
        df['int_col'] = df['int_col'].apply(lambda x: 'other' if x in rare_int_cols else x)

        # Encode 'accident' feature as binary
        df['accident'] = df['accident'].apply(lambda x: 0 if x == 'none reported' else 1)

        return df


# Load the dataset
train_data = pd.read_csv('train_dataset.csv')
test_data = pd.read_csv('test_dataset.csv')

# Create the sklearn pipeline
pipeline = Pipeline([
    ('preprocessor', CustomPreprocessor()),
])

# Preprocess training and test datasets using the pipeline
train_data = pipeline.fit_transform(train_data)
test_data = pipeline.transform(test_data)