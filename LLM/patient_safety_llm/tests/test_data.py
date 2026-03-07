from src.data_ingest import load_csv, preprocess_dataframe, split_dataset


def test_load_and_preprocess(tmp_path):
    p = tmp_path / "sample.csv"
    p.write_text("id,text,extra_names\n1,Patient email is test.user@example.com,Test User\n")

    df = load_csv(str(p))
    assert "text" in df.columns

    out = preprocess_dataframe(df, text_column="text", extra_names_column="extra_names")
    assert "processed_text" in out.columns
    # de-identification should redact email
    assert "[EMAIL]" in out.at[0, "processed_text"]


def test_split_dataset():
    import pandas as pd
    df = pd.DataFrame({"processed_text": ["a", "b", "c", "d", "e"]})
    tr, te = split_dataset(df, test_size=0.4, random_state=1)
    assert len(tr) + len(te) == len(df)
    assert 0 < len(te) < len(df)
