"""
Diagnostic server for Qazcode challenge.

Usage:
    uv run uvicorn src.mock_server:app --host 127.0.0.1 --port 8080
"""

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from src.llm_client import LLMClient
from src.pipeline import RetrievalPipeline


class DiagnoseRequest(BaseModel):
    symptoms: Optional[str] = ""
    top_k: int = 3


class Diagnosis(BaseModel):
    rank: int
    diagnosis: str
    icd10_code: str
    explanation: str


class DiagnoseResponse(BaseModel):
    diagnoses: list[Diagnosis]


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading retrieval pipeline from data/model.pkl ...")
    app.state.retriever = RetrievalPipeline.from_artifact("data/model.pkl")
    print("Initializing LLM client ...")
    app.state.llm = LLMClient()
    print("Server ready: POST /diagnose")
    yield


app = FastAPI(title="Medical Diagnosis Assistant", lifespan=lifespan)


@app.post("/diagnose", response_model=DiagnoseResponse)
async def handle_diagnose(request: DiagnoseRequest) -> DiagnoseResponse:
    symptoms = (request.symptoms or "").strip()
    top_k = max(1, min(int(request.top_k or 3), 10))

    if not symptoms:
        return DiagnoseResponse(diagnoses=[])

    retrieved = app.state.retriever.retrieve(symptoms, top_k=8)
    diagnoses = app.state.llm.rank_diagnoses(symptoms, retrieved_context=retrieved, top_k=top_k)
    return DiagnoseResponse(diagnoses=[Diagnosis(**d) for d in diagnoses])
