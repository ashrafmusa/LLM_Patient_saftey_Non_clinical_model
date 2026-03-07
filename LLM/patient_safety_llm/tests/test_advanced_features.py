"""Tests for advanced models and fine-tuning."""

import pytest
import torch
from unittest.mock import Mock, patch

from src.advanced_models import AdvancedLLMModel
from src.database_integrator import DatabaseIntegrator
from src.model_management import ModelRegistry, DeploymentManager
from src.transfer_learning import TransferLearningTrainer


class TestAdvancedLLMModel:
    """Test advanced LLM model."""

    @pytest.fixture
    def model(self):
        """Create test model."""
        return AdvancedLLMModel(
            model_name="distilbert",
            num_labels=2,
            device="cpu",
        )

    def test_model_initialization(self, model):
        """Test model initialization."""
        assert model.model_name == "distilbert"
        assert model.num_labels == 2
        assert model.device == "cpu"
        assert model.model is not None
        assert model.tokenizer is not None

    def test_available_models(self):
        """Test available models."""
        models = AdvancedLLMModel.AVAILABLE_MODELS
        assert "clinical_bert" in models
        assert "biobert" in models
        assert "distilbert" in models

    def test_predict(self, model):
        """Test prediction."""
        texts = ["Patient has fever", "Normal vitals"]
        predictions = model.predict(texts, batch_size=2)

        assert len(predictions) == 2
        assert all("label" in p for p in predictions)
        assert all("score" in p for p in predictions)

    def test_get_model_info(self, model):
        """Test getting model info."""
        info = model.get_model_info()

        assert info["model_name"] == "distilbert"
        assert info["num_labels"] == 2
        assert "total_parameters" in info
        assert "trainable_parameters" in info


class TestDatabaseIntegrator:
    """Test database integration."""

    @pytest.fixture
    def integrator(self):
        """Create test integrator."""
        return DatabaseIntegrator(cache_dir="/tmp/test_cache")

    def test_initialization(self, integrator):
        """Test integrator initialization."""
        assert integrator.cache_dir is not None
        assert integrator.metadata == {}

    def test_list_sources(self, integrator):
        """Test listing available sources."""
        sources = integrator.list_available_sources()

        assert "bioasq" in sources
        assert "medical_qa" in sources
        assert "mednli" in sources
        assert "pubmed_central" in sources

    def test_source_structure(self, integrator):
        """Test data source structure."""
        sources = integrator.list_available_sources()
        bioasq = sources["bioasq"]

        assert bioasq["name"] == "BioASQ"
        assert bioasq["access_type"] in ["public", "authentication_required"]
        assert "description" in bioasq

    @patch("src.database_integrator.load_dataset")
    def test_fetch_medical_qa(self, mock_load_dataset, integrator):
        """Test fetching Medical QA."""
        import pandas as pd

        # Mock dataset
        mock_data = {
            "train": [{"question": "What is sepsis?", "answer": "..."}],
            "test": [{"question": "What is ARDS?", "answer": "..."}],
        }
        mock_load_dataset.return_value = mock_data

        df = integrator.fetch_medical_qa()

        assert len(df) > 0
        assert "question" in df.columns

    def test_dataset_statistics(self, integrator):
        """Test dataset statistics."""
        import pandas as pd

        df = pd.DataFrame({
            "text": ["Sample 1", "Sample 2", "Sample 3"],
            "label": [0, 1, 1],
        })

        stats = integrator.get_dataset_statistics(df)

        assert stats["total_records"] == 3
        assert stats["columns"] == ["text", "label"]
        assert "null_counts" in stats

    def test_prepare_for_training(self, integrator):
        """Test training data preparation."""
        import pandas as pd

        df = pd.DataFrame({
            "text": ["Text 1"] * 10 + ["Text 2"] * 10,
            "label": [0] * 10 + [1] * 10,
        })

        train_texts, train_labels, test_texts, test_labels = (
            integrator.prepare_for_training(
                df,
                text_column="text",
                label_column="label",
                test_size=0.2,
            )
        )

        assert len(train_texts) == 16
        assert len(test_texts) == 4
        assert len(train_labels) == len(train_texts)
        assert len(test_labels) == len(test_texts)


class TestModelRegistry:
    """Test model registry."""

    @pytest.fixture
    def registry(self, tmp_path):
        """Create test registry."""
        return ModelRegistry(registry_dir=str(tmp_path))

    def test_register_model(self, registry):
        """Test model registration."""
        model_id = registry.register_model(
            model_name="test_model",
            model_path="./models/test",
            model_type="fine_tuned",
            version="1.0.0",
            metadata={"accuracy": 0.95},
        )

        assert model_id == "test_model:1.0.0"
        assert registry.get_model(model_id) is not None

    def test_get_model(self, registry):
        """Test getting model."""
        registry.register_model(
            model_name="model1",
            model_path="./path",
            model_type="pretrained",
            version="1.0.0",
        )

        model = registry.get_model("model1:1.0.0")
        assert model["name"] == "model1"
        assert model["version"] == "1.0.0"

    def test_list_models(self, registry):
        """Test listing models."""
        registry.register_model("m1", "./p1", "type1", "1.0.0")
        registry.register_model("m1", "./p2", "type1", "2.0.0")
        registry.register_model("m2", "./p3", "type2", "1.0.0")

        # List all
        all_models = registry.list_models()
        assert len(all_models) == 3

        # List by name
        m1_models = registry.list_models(model_name="m1")
        assert len(m1_models) == 2

    def test_get_statistics(self, registry):
        """Test registry statistics."""
        registry.register_model("m1", "./p", "pretrained", "1.0.0")
        registry.register_model("m2", "./p", "fine_tuned", "1.0.0")

        stats = registry.get_statistics()

        assert stats["total_models"] == 2
        assert "model_types" in stats


class TestDeploymentManager:
    """Test deployment manager."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create test manager."""
        return DeploymentManager(deployments_dir=str(tmp_path))

    def test_create_deployment(self, manager):
        """Test creating deployment."""
        dep_id = manager.create_deployment(
            deployment_name="prod_model",
            model_id="model1:1.0.0",
            environment="production",
            config={"batch_size": 32},
        )

        assert dep_id == "prod_model:production"
        assert manager.get_deployment(dep_id) is not None

    def test_update_status(self, manager):
        """Test updating deployment status."""
        dep_id = manager.create_deployment(
            "model",
            "m1:1.0",
            "prod",
            {},
        )

        manager.update_deployment_status(
            dep_id,
            status="running",
            endpoint="http://localhost:8000",
        )

        dep = manager.get_deployment(dep_id)
        assert dep["status"] == "running"
        assert dep["endpoint"] == "http://localhost:8000"


class TestTransferLearning:
    """Test transfer learning."""

    def test_layer_extraction(self):
        """Test layer number extraction."""
        from src.transfer_learning import TransferLearningTrainer

        trainer = TransferLearningTrainer(None, None)

        assert trainer._extract_layer_num("encoder.layer.0.weight") == 0
        assert trainer._extract_layer_num("encoder.layer.12.weight") == 12
        assert trainer._extract_layer_num("classifier.weight") == 0

    def test_curriculum_learning(self):
        """Test curriculum learning."""
        trainer = TransferLearningTrainer(None, None)

        texts = [
            "Short text",
            "This is a longer text with more information",
            "Medium length text here",
        ]
        labels = [0, 1, 1]

        sorted_texts, sorted_labels = trainer.curriculum_learning(texts, labels)

        assert len(sorted_texts) == 3
        assert len(sorted_labels) == 3
