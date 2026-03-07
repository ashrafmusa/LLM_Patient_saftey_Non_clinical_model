"""
Open-source database integration for training data collection.

Integrates with:
- MIMIC-III (medical records) - requires access request
- PhysioNet (clinical datasets)
- PubMed (medical literature)
- NIH NLM (medical NLP datasets)
- HuggingFace Datasets (curated medical datasets)
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
import csv
from datetime import datetime

import pandas as pd
import requests
from datasets import load_dataset

logger = logging.getLogger(__name__)


@dataclass
class DataSource:
    """Configuration for a data source."""

    name: str
    url: str
    description: str
    access_type: str  # "public", "authentication_required", "request_only"
    data_format: str  # "csv", "json", "parquet", "xml"


class DatabaseIntegrator:
    """Integrate with open-source databases for training data."""

    # Publicly available sources
    PUBLIC_SOURCES = {
        "bioasq": DataSource(
            name="BioASQ",
            url="https://huggingface.co/datasets/bioasq",
            description="Biomedical semantic QA dataset",
            access_type="public",
            data_format="json",
        ),
        "medical_qa": DataSource(
            name="Medical QA",
            url="https://huggingface.co/datasets/keivalya/medical-qa",
            description="Medical question-answering dataset",
            access_type="public",
            data_format="json",
        ),
        "pubmed_central": DataSource(
            name="PubMed Central",
            url="https://www.ncbi.nlm.nih.gov/pmc/",
            description="Free full-text biomedical literature",
            access_type="public",
            data_format="xml",
        ),
        "mednli": DataSource(
            name="MedNLI",
            url="https://huggingface.co/datasets/lfcc/mednli",
            description="Medical NLI dataset",
            access_type="public",
            data_format="json",
        ),
        "radiology_notes": DataSource(
            name="Radiology Notes",
            url="https://huggingface.co/datasets/allenai/med-qa",
            description="Radiology and medical notes dataset",
            access_type="public",
            data_format="json",
        ),
        "clinical_notes": DataSource(
            name="Clinical Notes Dataset",
            url="https://huggingface.co/datasets/physionet/mimic-cxr",
            description="Clinical notes and imaging",
            access_type="authentication_required",
            data_format="json",
        ),
    }

    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize database integrator."""
        self.cache_dir = Path(cache_dir or "./data/downloads")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata = {}
        logger.info(f"Database integrator initialized with cache: {self.cache_dir}")

    def list_available_sources(self) -> Dict[str, Dict]:
        """List available data sources."""
        sources = {}
        for key, source in self.PUBLIC_SOURCES.items():
            sources[key] = {
                "name": source.name,
                "description": source.description,
                "access_type": source.access_type,
                "url": source.url,
            }
        return sources

    def fetch_bioasq(self, task: str = "A", split: str = "train") -> pd.DataFrame:
        """
        Fetch BioASQ dataset.

        Args:
            task: BioASQ task (A, B, C)
            split: train or test

        Returns:
            DataFrame with questions and answers
        """
        logger.info(f"Fetching BioASQ task {task}...")
        try:
            dataset = load_dataset("bioasq", f"task_{task}", split=split)
            df = pd.DataFrame(dataset)
            logger.info(f"Fetched {len(df)} records from BioASQ")
            return df
        except Exception as e:
            logger.error(f"Error fetching BioASQ: {e}")
            return pd.DataFrame()

    def fetch_medical_qa(self) -> pd.DataFrame:
        """Fetch Medical QA dataset."""
        logger.info("Fetching Medical QA dataset...")
        try:
            dataset = load_dataset("keivalya/medical-qa")
            df = pd.concat(
                [pd.DataFrame(dataset[split]) for split in dataset.keys()],
                ignore_index=True,
            )
            logger.info(f"Fetched {len(df)} records from Medical QA")
            return df
        except Exception as e:
            logger.error(f"Error fetching Medical QA: {e}")
            return pd.DataFrame()

    def fetch_mednli(self) -> pd.DataFrame:
        """Fetch MedNLI dataset."""
        logger.info("Fetching MedNLI dataset...")
        try:
            dataset = load_dataset("lfcc/mednli")
            df = pd.concat(
                [pd.DataFrame(dataset[split]) for split in dataset.keys()],
                ignore_index=True,
            )
            logger.info(f"Fetched {len(df)} records from MedNLI")
            return df
        except Exception as e:
            logger.error(f"Error fetching MedNLI: {e}")
            return pd.DataFrame()

    def fetch_pubmed_abstracts(self, query: str, max_records: int = 1000) -> pd.DataFrame:
        """
        Fetch PubMed abstracts using NCBI E-utilities API.

        Args:
            query: PubMed search query
            max_records: Maximum records to fetch

        Returns:
            DataFrame with PubMed articles
        """
        logger.info(f"Fetching PubMed abstracts for query: {query}")

        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        search_url = f"{base_url}/esearch.fcgi"
        fetch_url = f"{base_url}/efetch.fcgi"

        try:
            # Search
            search_params = {
                "db": "pubmed",
                "term": query,
                "retmax": min(max_records, 100000),
                "rettype": "json",
            }
            search_resp = requests.get(search_url, params=search_params, timeout=30)
            search_data = search_resp.json()

            uids = search_data.get("esearchresult", {}).get("idlist", [])[:max_records]
            logger.info(f"Found {len(uids)} PubMed articles")

            if not uids:
                return pd.DataFrame()

            # Fetch abstracts
            fetch_params = {
                "db": "pubmed",
                "id": ",".join(uids),
                "rettype": "json",
            }
            fetch_resp = requests.get(fetch_url, params=fetch_params, timeout=60)
            articles = fetch_resp.json().get("result", {})

            # Parse articles
            records = []
            for uid in uids:
                if uid in articles and uid != "uids":
                    article = articles[uid]
                    records.append({
                        "pmid": uid,
                        "title": article.get("title", ""),
                        "abstract": article.get("abstract", ""),
                        "authors": article.get("authors", []),
                    })

            df = pd.DataFrame(records)
            logger.info(f"Fetched {len(df)} PubMed articles with abstracts")
            return df

        except Exception as e:
            logger.error(f"Error fetching PubMed: {e}")
            return pd.DataFrame()

    def fetch_medical_datasets_huggingface(self, dataset_name: str) -> pd.DataFrame:
        """
        Fetch medical datasets from HuggingFace.

        Args:
            dataset_name: HuggingFace dataset identifier

        Returns:
            DataFrame with dataset
        """
        logger.info(f"Fetching HuggingFace dataset: {dataset_name}")
        try:
            dataset = load_dataset(dataset_name)
            df = pd.concat(
                [pd.DataFrame(dataset[split]) for split in dataset.keys()],
                ignore_index=True,
            )
            logger.info(f"Fetched {len(df)} records from {dataset_name}")
            return df
        except Exception as e:
            logger.error(f"Error fetching {dataset_name}: {e}")
            return pd.DataFrame()

    def save_training_dataset(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        output_format: str = "csv",
    ) -> str:
        """
        Save dataset locally for training.

        Args:
            df: DataFrame to save
            dataset_name: Name for the dataset
            output_format: csv, json, or parquet

        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{dataset_name}_{timestamp}.{output_format}"
        filepath = self.cache_dir / filename

        try:
            if output_format == "csv":
                df.to_csv(filepath, index=False)
            elif output_format == "json":
                df.to_json(filepath, orient="records")
            elif output_format == "parquet":
                df.to_parquet(filepath, index=False)

            logger.info(f"Saved dataset to {filepath} ({len(df)} records)")
            return str(filepath)
        except Exception as e:
            logger.error(f"Error saving dataset: {e}")
            return ""

    def combine_datasets(self, dataframes: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Combine multiple datasets.

        Args:
            dataframes: List of DataFrames to combine

        Returns:
            Combined DataFrame
        """
        combined = pd.concat(dataframes, ignore_index=True)
        logger.info(f"Combined {len(dataframes)} datasets ({len(combined)} total records)")
        return combined

    def get_dataset_statistics(self, df: pd.DataFrame) -> Dict:
        """Get statistics about a dataset."""
        return {
            "total_records": len(df),
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.to_dict(),
            "null_counts": df.isnull().sum().to_dict(),
            "sample_size": min(5, len(df)),
            "sample_data": df.head().to_dict("records"),
        }

    def prepare_for_training(
        self,
        df: pd.DataFrame,
        text_column: str,
        label_column: Optional[str] = None,
        test_size: float = 0.2,
    ) -> Tuple[List[str], List[int], List[str], List[int]]:
        """
        Prepare dataset for fine-tuning.

        Args:
            df: Input DataFrame
            text_column: Column name for text
            label_column: Column name for labels
            test_size: Test split fraction

        Returns:
            Tuple of (train_texts, train_labels, test_texts, test_labels)
        """
        if text_column not in df.columns:
            raise ValueError(f"Column {text_column} not found in dataset")

        texts = df[text_column].astype(str).tolist()

        if label_column and label_column in df.columns:
            labels = df[label_column].astype(int).tolist()
        else:
            labels = [0] * len(texts)  # Default labels

        # Split
        split_idx = int(len(texts) * (1 - test_size))
        train_texts = texts[:split_idx]
        train_labels = labels[:split_idx]
        test_texts = texts[split_idx:]
        test_labels = labels[split_idx:]

        logger.info(
            f"Dataset prepared: {len(train_texts)} train, {len(test_texts)} test"
        )

        return train_texts, train_labels, test_texts, test_labels
