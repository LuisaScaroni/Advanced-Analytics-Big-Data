from __future__ import annotations
import chromadb
import ollama
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from steam_sqlite import load_games_from_sqlite


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = Path(os.environ.get("RAGLOOKER_DB_PATH", BASE_DIR / "steam_games_reviews_25.sqlite"))
MAX_GAMES = 5000


chroma_client = chromadb.PersistentClient(path="./chroma_data")
collection = chroma_client.get_or_create_collection(name='steam_games')


def create_search_engine() -> "GameSearchEngine":
    return GameSearchEngine(DB_PATH)

@dataclass
class GameRecord:
    app_id: str
    raw: dict[str, Any]

    @property
    def name(self) -> str:
        return self.raw.get("name", "Unknown title")

    @property
    def short_description(self) -> str:
        return self.raw.get("short_description", "")

    def to_result(self, score: float) -> dict[str, Any]:
        return {
            "app_id": self.app_id,
            "name": self.name,
            "score": round(score, 4),
            "short_description": self.short_description,
            "genres": self.raw.get("genres", []),
            "tags": self._normalize_tags(self.raw.get("tags")),
            "price": self.raw.get("price"),
            "release_date": self.raw.get("release_date"),
            "header_image": self.raw.get("header_image"),
            "store_page": f"https://store.steampowered.com/app/{self.app_id}",
            "platforms": {
                "windows": bool(self.raw.get("windows")),
                "mac": bool(self.raw.get("mac")),
                "linux": bool(self.raw.get("linux")),
            },
        }

    @staticmethod
    def _normalize_tags(tags: Any) -> list[str]:
        if isinstance(tags, dict):
            return list(tags.keys())[:8]
        if isinstance(tags, list):
            return tags[:8]
        return []

class GameSearchEngine:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.records = self.load_records()

    def load_records(self) -> list[GameRecord]:
        records: list[GameRecord] = []
        for app_id, raw in load_games_from_sqlite(self.db_path, MAX_GAMES):
            records.append(GameRecord(app_id=app_id, raw=raw))
        return records

    def search(self, query: str):
        
        candidates = self.retrieve_candidates(query)
        ranked_candidates = self.rank_candidates(candidates)
        
        
        answer = self.generate_answer(query, ranked_candidates)
        
        
        frontend_candidates = []
        for cand in ranked_candidates:
            for record in self.records:
                if str(record.app_id) == str(cand['id']):
                    frontend_candidates.append(record.to_result(score=1.0))
                    break
                    
    
        return {
            "matches": frontend_candidates,
            "answer": answer,
            "meta": {
                "indexed_games": len(self.records),
                "retrieval_mode": "RAG con ChromaDB e Ollama",
                "note": "Raccomandazioni generate dall'LLM locale"
            }
        }
    def retrieve_candidates(self, query: str, k: int = 10):
        results = collection.query(
            query_texts=[query],
            n_results=k
        )
        candidates = []
        if results['metadatas'] and results['documents']:
            for i in range(len(results['metadatas'][0])):
                candidates.append({
                    'id': results['ids'][0][i],
                    'title': results['metadatas'][0][i]['title'],
                    'context': results['documents'][0][i]
                })
        return candidates

    def rank_candidates(self, candidates: list):
        return candidates[:5]

    def generate_answer(self, query: str, ranked_candidates: list) -> str:
        context_strings = []
        for game in ranked_candidates:
            context_strings.append(f"- {game['title']}: {game['context']}")
        
        context_text = "\n".join(context_strings)
        prompt = f"""The user is looking for recommendations based on this query: "{query}"
Here are the top matching games from our database:
{context_text}
Based ONLY on the context provided above, recommend 2-3 games that best fit the user's query. 
Explain why you are recommending them in a friendly way.
"""
        try:
            response = ollama.chat(model='phi3.5', messages=[
                {'role': 'user', 'content': prompt}
            ])
            return response['message']['content']
        except Exception as e:
            print(f"Errore Ollama: {e}")
            return "Sorry, I couldn't generate a recommendation."