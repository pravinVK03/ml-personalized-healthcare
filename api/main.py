"""Small API serving the React research dashboard."""

from __future__ import annotations

from pathlib import Path

from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Route
from starlette.staticfiles import StaticFiles

from .catalog import DATASETS, ROOT, all_datasets, dataset_summary
from . import shanghai


def response(data: object) -> JSONResponse:
    return JSONResponse(data)


async def datasets(_request):
    return response({"datasets": all_datasets()})


async def dataset_overview(request):
    dataset_id = request.path_params["dataset_id"]
    try:
        data = dataset_summary(dataset_id)
    except KeyError:
        return response({"error": "Unknown dataset"}, status_code=404)
    if dataset_id == "shanghai_t2d" and data["status"] == "ready":
        data["evaluation"] = shanghai.overview()
    return response(data)


def require_shanghai(dataset_id: str):
    if dataset_id != "shanghai_t2d":
        return JSONResponse(
            {"error": "This dataset is not configured locally. Add its documented source files before requesting participant-level results."},
            status_code=409,
        )
    if dataset_summary(dataset_id)["status"] != "ready":
        return JSONResponse({"error": "Run the Shanghai pipeline before opening results."}, status_code=409)
    return None


async def participants(request):
    blocked = require_shanghai(request.path_params["dataset_id"])
    return blocked or response({"participants": shanghai.participant_list()})


async def participant_episodes(request):
    blocked = require_shanghai(request.path_params["dataset_id"])
    if blocked:
        return blocked
    subject_id = request.path_params["subject_id"]
    return response({"episodes": shanghai.episodes(subject_id)})


async def participant_state(request):
    blocked = require_shanghai(request.path_params["dataset_id"])
    if blocked:
        return blocked
    query = request.query_params
    try:
        data = shanghai.state_and_forecast(
            request.path_params["subject_id"], query["episode"], query.get("model", "gradient_boosting"), query.get("origin"),
        )
    except (KeyError, ValueError) as error:
        return response({"error": str(error)}, status_code=400)
    return response(data)


async def participant_error(request):
    blocked = require_shanghai(request.path_params["dataset_id"])
    if blocked:
        return blocked
    try:
        return response({"rows": shanghai.participant_error(request.query_params.get("model", "gradient_boosting"))})
    except ValueError as error:
        return response({"error": str(error)}, status_code=400)


async def range_agreement(request):
    blocked = require_shanghai(request.path_params["dataset_id"])
    if blocked:
        return blocked
    try:
        horizon = int(request.query_params.get("horizon", "60"))
        return response({"rows": shanghai.range_agreement(request.query_params.get("model", "gradient_boosting"), horizon)})
    except ValueError as error:
        return response({"error": str(error)}, status_code=400)


async def index(_request):
    index_path = ROOT / "frontend" / "dist" / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return JSONResponse({"message": "React frontend is not built. Run npm --prefix frontend install and npm --prefix frontend run build."})


routes = [
    Route("/api/datasets", datasets),
    Route("/api/datasets/{dataset_id}", dataset_overview),
    Route("/api/datasets/{dataset_id}/participants", participants),
    Route("/api/datasets/{dataset_id}/participants/{subject_id}/episodes", participant_episodes),
    Route("/api/datasets/{dataset_id}/participants/{subject_id}/state", participant_state),
    Route("/api/datasets/{dataset_id}/participant-error", participant_error),
    Route("/api/datasets/{dataset_id}/range-agreement", range_agreement),
    Route("/", index),
]

app = Starlette(debug=True, routes=routes)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"], allow_methods=["*"], allow_headers=["*"])

dist = ROOT / "frontend" / "dist" / "assets"
if dist.exists():
    app.mount("/assets", StaticFiles(directory=dist), name="assets")
