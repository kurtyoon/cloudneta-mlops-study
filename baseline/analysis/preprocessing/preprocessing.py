import pandas as pd
import warnings

from transformers import BertTokenizer
from util import malicious_translation
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE


warnings.filterwarnings("ignore", category=FutureWarning)


class Preprocessing:
    def __init__(self, url_path, feature_path):
        self.url = pd.read_csv(url_path)
        self.feature = pd.read_csv(feature_path)
        self.feature.dropna(inplace=True)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    def tokenization(self, string):
        tokens_ids_list = []
        tokens = self.tokenizer.tokenize(string)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        tokens_ids_list.append(token_ids)
        return tokens_ids_list

    def url_preprocessing(self):
        self.url['label'] = self.url['type'].map(malicious_translation)
        self.url['token_ids'] = self.url['url'].apply(self.tokenization)
        self.url.drop(columns=['url', 'type'], inplace=True)
        self.url.to_csv('data/url_ids.csv', index=False)

    def feature_selection(self):
        X = self.feature.drop(columns=['phishing'], axis=1)
        y = self.feature['phishing']

        sm = SMOTE(random_state=42)
        X_smoted, y_smoted = sm.fit_resample(X, y)

        forest = RandomForestClassifier(n_estimators=100, random_state=42)
        forest.fit(X_smoted, y_smoted)

        feature_importances = pd.Series(forest.feature_importances_, index=X_smoted.columns)
        important_features = feature_importances[feature_importances > 0.009]

        X_train_important = X_smoted[important_features.index]

        def feature_normalization(value):
            if isinstance(value, float) and value.is_integer():
                return str(int(value))
            return str(value)

        feature_strings = ["/".join(feature_normalization(value) for value in row)
                           for index, row in X_train_important.iterrows()]

        self.feature = pd.DataFrame({
            'label': y_smoted,
            'feature_string': feature_strings
        })

    def feature_preprocessing(self):
        self.feature_selection()
        self.feature['token_idx'] = self.feature['feature_string'].apply(self.tokenization)
        self.feature.drop(columns=['feature_string'], inplace=True)
        self.feature.to_csv('data/feature_ids.csv', index=False)

    def preprocessing(self):
        self.url_preprocessing()
        self.feature_preprocessing()


if __name__ == '__main__':
    url_path = 'data/malicious_phish.csv'
    feature_path = 'data/dataset_full.csv'
    preprocessing = Preprocessing(url_path, feature_path)
    preprocessing.preprocessing()
