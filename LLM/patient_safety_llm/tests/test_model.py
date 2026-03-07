import os
import joblib
from src.train import train
from src.data_ingest import load_csv


def test_train_creates_model(tmp_path, monkeypatch):
    # create a tiny csv
    p = tmp_path / "tiny.csv"
    p.write_text('id,text\n1,Patient had an error with medication dosing\n2,No adverse events reported\n')

    # run training
    args = type('A', (), {'input': str(p), 'text_col': 'text', 'label_col': None, 'extra_names_col': None})
    train(args)

    # expect model files
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'src', '..', 'models')
    models_dir = os.path.abspath(models_dir)
    model_path = os.path.join(models_dir, 'risk_model.pkl')
    vect_path = os.path.join(models_dir, 'vectorizer.pkl')

    assert os.path.exists(model_path)
    assert os.path.exists(vect_path)

    clf = joblib.load(model_path)
    vect = joblib.load(vect_path)
    # basic smoke check
    X = vect.transform(["Medication error occurred"]) 
    if hasattr(clf, 'predict'):
        pred = clf.predict(X)
        assert len(pred) == 1
