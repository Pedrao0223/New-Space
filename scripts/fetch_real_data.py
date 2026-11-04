"""Download official open scientific datasets into data/raw/ with SHA-256 provenance manifest.

Sources:
1. NASA/Caltech JPL NHATS API (Accessible Near-Earth Asteroids & Trajectory Delta-V)
2. NASA/Caltech JPL Small-Body Database Query API (All cataloged Near-Earth Objects)
3. CelesTrak / USSPACECOM Official NORAD SATCAT Catalog (All tracked orbital objects since 1957)
4. Harvard-Smithsonian CfA GCAT Launch Log (Historical orbital launch attempts since 1957)
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from newspace.config import DATA_RAW_DIR, ENDPOINTS


SBDB_FIELDS = (
    "spkid,full_name,pdes,name,neo,pha,H,diameter,albedo,spec_B,spec_T,"
    "e,a,q,i,moid,per_y,data_arc,n_obs_used,condition_code,first_obs,last_obs"
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def fetch_with_retry(
    client: httpx.Client,
    url: str,
    params: dict[str, str] | None = None,
    retries: int = 3,
) -> httpx.Response:
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = client.get(url, params=params)
            resp.raise_for_status()
            return resp
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if attempt < retries:
                time.sleep(2.0 * attempt)
    raise RuntimeError(f"Failed to download {url} after {retries} attempts: {last_err}") from last_err


def download_all_sources(force: bool = False) -> Path:
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = DATA_RAW_DIR / "manifest_provenance.json"

    targets = [
        {
            "id": "nasa_jpl_nhats",
            "description": "NASA/Caltech JPL NHATS Accessible Near-Earth Asteroids",
            "url": ENDPOINTS.nasa_nhats_api,
            "params": None,
            "filename": "nasa_jpl_nhats.json",
        },
        {
            "id": "nasa_jpl_sbdb_neo",
            "description": "NASA/Caltech JPL Small-Body Database (All Near-Earth Objects)",
            "url": ENDPOINTS.nasa_sbdb_query_api,
            "params": {"sb-group": "neo", "fields": SBDB_FIELDS},
            "filename": "nasa_jpl_sbdb_neo.json",
        },
        {
            "id": "celestrak_norad_satcat",
            "description": "CelesTrak / USSPACECOM Official NORAD Satellite Catalog (SATCAT)",
            "url": ENDPOINTS.celestrak_satcat_csv,
            "params": None,
            "filename": "celestrak_satcat.csv",
        },
        {
            "id": "gcat_orbital_launchlog",
            "description": "Harvard-Smithsonian CfA GCAT Historical Orbital Launch Log (J. McDowell)",
            "url": ENDPOINTS.gcat_launchlog_tsv,
            "params": None,
            "filename": "gcat_launchlog.tsv",
        },
    ]

    headers = {
        "User-Agent": "NewSpace-Academic-Benchmark/1.0 (Scientific Data Mining Research)"
    }

    provenance_records: list[dict[str, object]] = []
    with httpx.Client(timeout=120.0, follow_redirects=True, headers=headers) as client:
        for item in targets:
            out_file = DATA_RAW_DIR / str(item["filename"])
            if out_file.exists() and out_file.stat().st_size > 1024 and not force:
                raw_bytes = out_file.read_bytes()
                status_code = 200
                elapsed_ms = 0.0
                cached = True
                resolved_url = str(item["url"])
            else:
                t0 = time.perf_counter()
                resp = fetch_with_retry(
                    client,
                    str(item["url"]),
                    params=item["params"],  # type: ignore[arg-type]
                )
                elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                raw_bytes = resp.content
                out_file.write_bytes(raw_bytes)
                status_code = resp.status_code
                resolved_url = str(resp.url)
                cached = False

            record = {
                "id": item["id"],
                "description": item["description"],
                "source_url": resolved_url,
                "local_file": out_file.name,
                "size_bytes": len(raw_bytes),
                "size_mb": round(len(raw_bytes) / (1024 * 1024), 3),
                "sha256": sha256_bytes(raw_bytes),
                "http_status": status_code,
                "download_latency_ms": elapsed_ms,
                "from_local_cache": cached,
                "verified_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            provenance_records.append(record)
            print(
                f"[OK] {item['id']}: {record['size_mb']} MB | "
                f"SHA256={str(record['sha256'])[:12]}... | cached={cached}"
            )

    manifest = {
        "project": "NewSpace Analytics & Data Mining Benchmark",
        "data_integrity_policy": "100% Real Empirical Public Datasets - Zero Synthetic/Fictional Records",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "datasets": provenance_records,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nManifest saved to: {manifest_path}")
    return manifest_path


if __name__ == "__main__":
    download_all_sources()
