import sys
from pathlib import Path

from src.components.Retriver import Retriever
from src.exception import MyException
from src.logger import logger


class MultimodalSearch:
    def __init__(self, artifact_root: str = "artifact") -> None:
        try:
            self.artifact_root = Path(artifact_root)
        except Exception as e:
            raise MyException(e, sys)

    def _latest_artifact_dir(self) -> Path:
        if not self.artifact_root.exists():
            raise FileNotFoundError("The artifact directory does not exist yet. Run training first.")

        artifact_dirs = sorted(
            [path for path in self.artifact_root.iterdir() if path.is_dir()],
            reverse=True,
        )
        if not artifact_dirs:
            raise FileNotFoundError("No training artifacts were found. Run the training pipeline first.")

        return artifact_dirs[0]

    @staticmethod
    def _first_existing_path(candidates: list[Path], description: str) -> Path:
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"Missing {description}. Expected one of: {', '.join(str(path) for path in candidates)}"
        )

    def _resolve_assets(self) -> tuple[Path, Path, Path]:
        artifact_dir = self._latest_artifact_dir()

        model_path = self._first_existing_path(
            [
                artifact_dir / "Fine_Tuned_Model" / "Fine_Tuned_Model",
                artifact_dir / "Fine_Tuned_Model" / "Trained_Model",
                artifact_dir / "Loaded_Model" / "CLIP_MODEL_UNTRAINED",
            ],
            "model directory",
        )
        faiss_path = self._first_existing_path(
            [
                artifact_dir / "Embeddings" / "index",
                artifact_dir / "index",
            ],
            "FAISS index",
        )
        mapping_path = self._first_existing_path(
            [
                artifact_dir / "Embeddings" / "Mappnig",
                artifact_dir / "Embeddings" / "mapping.json",
            ],
            "mapping file",
        )

        return model_path, faiss_path, mapping_path

    def predict(self, query, top_k: int = 5):
        try:
            logger.info("Entered predict method of MultimodalSearch class")
            model_path, faiss_path, mapping_path = self._resolve_assets()
            retriever = Retriever(
                Faiss_path=faiss_path,
                mapping_path=mapping_path,
                Model_Path=model_path,
            )
            return retriever.predict(query, top_k=top_k)
        except Exception as e:
            raise MyException(e, sys)
