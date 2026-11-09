"""Central configuration, official brand palette, paths, and data source definitions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote


def resolve_project_root() -> Path:
    """Resolve repository root reliably regardless of working directory (root or notebooks/)."""
    here = Path(__file__).resolve()
    return here.parents[2]


PROJECT_ROOT: Path = resolve_project_root()
DATA_DIR: Path = PROJECT_ROOT / "data"
DATA_RAW_DIR: Path = DATA_DIR / "raw"
DATA_PROCESSED_DIR: Path = DATA_DIR / "processed"
DB_PATH: Path = DATA_PROCESSED_DIR / "newspace_analytics.db"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
DOCS_DIR: Path = PROJECT_ROOT / "docs"
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"


FAVICON_SVG: str = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
  <defs>
    <radialGradient id="bg" cx="30%" cy="25%" r="85%">
      <stop offset="0%" stop-color="#1E1B4B"/>
      <stop offset="100%" stop-color="#05070D"/>
    </radialGradient>
    <linearGradient id="orb" x1="0%" y1="100%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#6366F1"/>
      <stop offset="50%" stop-color="#8B5CF6"/>
      <stop offset="100%" stop-color="#38BDF8"/>
    </linearGradient>
  </defs>
  <rect width="64" height="64" rx="16" fill="url(#bg)"/>
  <ellipse cx="32" cy="32" rx="23" ry="9" transform="rotate(-28 32 32)" fill="none" stroke="url(#orb)" stroke-width="2.6"/>
  <circle cx="32" cy="32" r="8.5" fill="url(#orb)"/>
  <circle cx="50" cy="20" r="2.8" fill="#E2E8F0"/>
  <circle cx="14" cy="46" r="1.2" fill="#E2E8F0" opacity="0.6"/>
</svg>"""

FAVICON_DATA_URI: str = "data:image/svg+xml," + quote(FAVICON_SVG)


@dataclass(frozen=True)
class BrandPalette:
    """Modern "Deep Space" palette: void background, indigo/violet nebula tones and star-blue accent.

    Field names are kept stable for backward compatibility; each field documents its semantic role.
    """

    cosmic_bg: str = "#05070D"       # Void black-blue        - Page / figure background
    space_violet: str = "#1E1B4B"    # Indigo nebula (950)     - Deep surface / colorscale low end
    orbital_blue: str = "#6366F1"    # Indigo (500)            - Secondary analytical series
    deep_teal: str = "#8B5CF6"       # Violet (500)            - Tertiary analytical series
    vibrant_cyan: str = "#38BDF8"    # Star blue (sky 400)     - Primary accent / KPI / trend line

    # Neutral UI tones & semantic status colors for the dark space theme
    text_primary: str = "#E2E8F0"
    text_muted: str = "#7C8599"
    grid_line: str = "#161B2B"
    accent_gold: str = "#FBBF24"     # Inflection points / warnings
    accent_coral: str = "#FB7185"    # High collision / actuarial risk alert
    accent_mint: str = "#34D399"     # Optimal viability confirmation

    @property
    def sequential_scale(self) -> list[str]:
        return [
            self.cosmic_bg,
            self.space_violet,
            self.orbital_blue,
            self.deep_teal,
            self.vibrant_cyan,
        ]

    @property
    def categorical_series(self) -> list[str]:
        return [
            self.vibrant_cyan,
            "#A78BFA",
            self.orbital_blue,
            "#22D3EE",
            self.accent_gold,
            "#F472B6",
            self.deep_teal,
        ]


PALETTE = BrandPalette()


@dataclass(frozen=True)
class DataSourceEndpoints:
    """Authoritative, open, peer-reviewed public aerospace and astronomical endpoints."""

    # 1. NASA/Caltech JPL NHATS (Human/Robotic Accessible Near-Earth Asteroids & Delta-V)
    nasa_nhats_api: str = "https://ssd-api.jpl.nasa.gov/nhats.api"

    # 2. NASA/Caltech JPL Small-Body Database (SBDB) Query API (Near-Earth Asteroids)
    nasa_sbdb_query_api: str = "https://ssd-api.jpl.nasa.gov/sbdb_query.api"

    # 3. CelesTrak / USSPACECOM Official NORAD Satellite Catalog (SATCAT CSV)
    celestrak_satcat_csv: str = "https://celestrak.org/pub/satcat.csv"

    # 4. Jonathan McDowell's GCAT (Harvard-Smithsonian CfA) Orbital Launch Log (1957-Present)
    gcat_launchlog_tsv: str = "https://planet4589.org/space/gcat/tsv/derived/launchlog.tsv"


ENDPOINTS = DataSourceEndpoints()


@dataclass(frozen=True)
class DefaultSelectionCriteria:
    """Default domain-grounded selection parameters for NewSpace decision support."""

    max_delta_v_kms: float = 6.5          # Comparable to Lunar surface/Cislunar transfer budget
    max_duration_days: int = 450          # Operational round-trip window limit
    max_orbit_uncertainty: int = 3        # NASA Orbit Condition Code (0 = best, 9 = highest uncertainty)
    min_observations: int = 50            # Minimum astrometric observations for reliable targeting
    min_launcher_flights: int = 10        # Minimum historical orbital attempts for actuarial credibility
    min_wilson_reliability: float = 0.85  # 95% Wilson lower bound reliability threshold
    analysis_start_year: int = 1957
    newspace_inflection_year: int = 2015  # Commercial reusability & megaconstellation inflection


DEFAULT_CRITERIA = DefaultSelectionCriteria()
