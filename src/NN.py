from typing_extensions import List

import chromadb
from chromadb.config import Settings, DEFAULT_DATABASE, DEFAULT_TENANT

""" Example fingerprint
{
    "id": "ExpertModel1:27B"
    "local": True
    "model_name": "BioExpert:27B"
    "ip": "http://localhost:11434/"
    "description": ["biology", "micro-organisms", "medicine", "natural science"] # Embedded keywords
    "model_size_B": [27]
}
"""

class NearestNeighbour():
    def __init__(self):
        self.__client  = chromadb.EphemeralClient(
            Settings(),
            tenant=DEFAULT_TENANT,
            database=DEFAULT_DATABASE
        )

        self.__collection = self.__client.get_or_create_collection(name="fingerprint_collection")
        self.fingerprints: dict = {}

    def add_or_update(self, fingerprint: dict) -> None:
        _id: int | str | None = fingerprint.get("id", None)
        if _id is None:
            raise KeyError("Fingerprint must contain 'id' key")

        self.fingerprints[_id] = fingerprint

        REQUIRED_KEYS = ["description", "local", "model_name", "ip", "model_size_B"]

        missing_keys: list = [key for key in REQUIRED_KEYS if key not in fingerprint]
        if missing_keys:
            raise KeyError(f"Fingerprint missing required keys: {missing_keys}")

        try:
            self.__collection.upsert(
                ids=[_id],
                documents=fingerprint["description"],
                metadatas=[{
                    "local": fingerprint["local"],
                    "model_name": fingerprint["model_name"],
                    "ip": fingerprint["ip"],
                    "model_size_B": fingerprint["model_size_B"],
                }]
            )
        except KeyError as e:
            raise KeyError(f"Missing key during collection add: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error while adding fingerprint (id={_id}): {e}") from e
        
    def delete(self, id: int | str) -> None:
        self.__collection.delete(ids=[id]) # type: ignore

    def task_agent(self, task) -> dict:
        fingerprints = self.__collection.query(
            query_texts=[task],
            include=["documents", "metadatas"]
        )

        return {
            _id: {"document": doc, "metadata": meta}
            for _id, doc, meta in zip(
                fingerprints["ids"][0],
                fingerprints["documents"][0], # type: ignore
                fingerprints["metadatas"][0] # type: ignore
            )
        }