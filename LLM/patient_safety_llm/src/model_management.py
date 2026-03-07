"""
Model management system for versioning and lifecycle management.

Features:
- Model registry and versioning
- Training metadata tracking
- Model comparison and evaluation
- Deployment management
- Fine-tuning history
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import hashlib
import shutil

import pandas as pd

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Manage model versions, training, and metadata."""

    def __init__(self, registry_dir: str = "./models"):
        """
        Initialize model registry.

        Args:
            registry_dir: Directory for storing model metadata and versions
        """
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)

        self.metadata_file = self.registry_dir / "registry.json"
        self.models = self._load_registry()

        logger.info(f"Model registry initialized at {self.registry_dir}")

    def _load_registry(self) -> Dict:
        """Load registry from disk."""
        if self.metadata_file.exists():
            with open(self.metadata_file) as f:
                return json.load(f)
        return {}

    def _save_registry(self) -> None:
        """Save registry to disk."""
        with open(self.metadata_file, "w") as f:
            json.dump(self.models, f, indent=2, default=str)

    def register_model(
        self,
        model_name: str,
        model_path: str,
        model_type: str,
        version: str,
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        Register a new model version.

        Args:
            model_name: Name of the model
            model_path: Path to model directory
            model_type: Type (pretrained, fine_tuned, custom)
            version: Version identifier
            metadata: Additional metadata

        Returns:
            Model ID
        """
        model_id = f"{model_name}:{version}"

        model_info = {
            "id": model_id,
            "name": model_name,
            "version": version,
            "type": model_type,
            "path": model_path,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "metadata": metadata or {},
            "status": "active",
        }

        self.models[model_id] = model_info
        self._save_registry()

        logger.info(f"Registered model: {model_id}")
        return model_id

    def create_version(
        self,
        model_name: str,
        base_version: str,
        new_version: str,
        changes: Dict,
    ) -> str:
        """
        Create a new version from existing model.

        Args:
            model_name: Model name
            base_version: Base version to fork from
            new_version: New version identifier
            changes: What changed (training results, etc.)

        Returns:
            New model ID
        """
        base_id = f"{model_name}:{base_version}"

        if base_id not in self.models:
            raise ValueError(f"Base model {base_id} not found")

        base_model = self.models[base_id]
        new_id = f"{model_name}:{new_version}"

        new_model_info = {
            **base_model,
            "id": new_id,
            "version": new_version,
            "created_at": datetime.now().isoformat(),
            "parent_version": base_version,
            "changes": changes,
        }

        self.models[new_id] = new_model_info
        self._save_registry()

        logger.info(f"Created new version: {new_id} (parent: {base_id})")
        return new_id

    def get_model(self, model_id: str) -> Optional[Dict]:
        """Get model info by ID."""
        return self.models.get(model_id)

    def list_models(self, model_name: Optional[str] = None) -> List[Dict]:
        """
        List all models or versions of a specific model.

        Args:
            model_name: Filter by model name (optional)

        Returns:
            List of model info dicts
        """
        if model_name:
            return [
                m for m in self.models.values()
                if m["name"] == model_name
            ]
        return list(self.models.values())

    def set_status(self, model_id: str, status: str) -> None:
        """
        Set model status (active, archived, deprecated).

        Args:
            model_id: Model ID
            status: New status
        """
        if model_id in self.models:
            self.models[model_id]["status"] = status
            self.models[model_id]["updated_at"] = datetime.now().isoformat()
            self._save_registry()
            logger.info(f"Model {model_id} status set to {status}")

    def add_training_run(
        self,
        model_id: str,
        training_config: Dict,
        results: Dict,
    ) -> None:
        """
        Log a training run for a model.

        Args:
            model_id: Model ID
            training_config: Training configuration
            results: Training results
        """
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")

        model = self.models[model_id]

        if "training_history" not in model:
            model["training_history"] = []

        training_run = {
            "timestamp": datetime.now().isoformat(),
            "config": training_config,
            "results": results,
        }

        model["training_history"].append(training_run)
        model["updated_at"] = datetime.now().isoformat()
        self._save_registry()

        logger.info(f"Added training run to {model_id}")

    def compare_models(self, model_ids: List[str]) -> pd.DataFrame:
        """
        Compare multiple models.

        Args:
            model_ids: List of model IDs

        Returns:
            DataFrame with comparison metrics
        """
        comparison_data = []

        for model_id in model_ids:
            if model_id in self.models:
                model = self.models[model_id]
                comparison_data.append({
                    "Model": model_id,
                    "Type": model.get("type"),
                    "Created": model.get("created_at"),
                    "Status": model.get("status"),
                    "Training Runs": len(model.get("training_history", [])),
                })

        return pd.DataFrame(comparison_data)

    def export_model_metadata(self, model_id: str, output_path: str) -> None:
        """
        Export model metadata to JSON.

        Args:
            model_id: Model ID
            output_path: Path to save JSON
        """
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")

        with open(output_path, "w") as f:
            json.dump(self.models[model_id], f, indent=2, default=str)

        logger.info(f"Exported model metadata to {output_path}")

    def delete_model(self, model_id: str, delete_files: bool = False) -> None:
        """
        Delete a model from registry.

        Args:
            model_id: Model ID
            delete_files: Also delete model files
        """
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")

        if delete_files:
            model_path = self.models[model_id].get("path")
            if model_path and Path(model_path).exists():
                shutil.rmtree(model_path)
                logger.info(f"Deleted model files: {model_path}")

        del self.models[model_id]
        self._save_registry()
        logger.info(f"Deleted model {model_id}")

    def get_statistics(self) -> Dict:
        """Get registry statistics."""
        return {
            "total_models": len(self.models),
            "active_models": len([m for m in self.models.values() if m["status"] == "active"]),
            "archived_models": len([m for m in self.models.values() if m["status"] == "archived"]),
            "model_types": list(set(m["type"] for m in self.models.values())),
            "total_training_runs": sum(
                len(m.get("training_history", [])) for m in self.models.values()
            ),
        }


class DeploymentManager:
    """Manage model deployments."""

    def __init__(self, deployments_dir: str = "./deployments"):
        """Initialize deployment manager."""
        self.deployments_dir = Path(deployments_dir)
        self.deployments_dir.mkdir(parents=True, exist_ok=True)
        self.deployments = {}
        logger.info(f"Deployment manager initialized at {self.deployments_dir}")

    def create_deployment(
        self,
        deployment_name: str,
        model_id: str,
        environment: str,
        config: Dict,
    ) -> str:
        """
        Create a model deployment.

        Args:
            deployment_name: Name of deployment
            model_id: Model to deploy
            environment: Environment (dev, staging, prod)
            config: Deployment configuration

        Returns:
            Deployment ID
        """
        deployment_id = f"{deployment_name}:{environment}"

        deployment_info = {
            "id": deployment_id,
            "name": deployment_name,
            "model_id": model_id,
            "environment": environment,
            "config": config,
            "created_at": datetime.now().isoformat(),
            "status": "pending",
            "endpoint": None,
        }

        self.deployments[deployment_id] = deployment_info
        logger.info(f"Created deployment: {deployment_id}")

        return deployment_id

    def update_deployment_status(
        self,
        deployment_id: str,
        status: str,
        endpoint: Optional[str] = None,
    ) -> None:
        """Update deployment status and endpoint."""
        if deployment_id in self.deployments:
            self.deployments[deployment_id]["status"] = status
            if endpoint:
                self.deployments[deployment_id]["endpoint"] = endpoint
            logger.info(f"Deployment {deployment_id} status: {status}")

    def list_deployments(self, environment: Optional[str] = None) -> List[Dict]:
        """List deployments, optionally filtered by environment."""
        if environment:
            return [
                d for d in self.deployments.values()
                if d["environment"] == environment
            ]
        return list(self.deployments.values())

    def get_deployment(self, deployment_id: str) -> Optional[Dict]:
        """Get deployment info."""
        return self.deployments.get(deployment_id)
