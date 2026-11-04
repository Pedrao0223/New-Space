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
    <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#160D18"/>
      <stop offset="55%" stop-color="#23145B"/>
      <stop offset="100%" stop-color="#09456C"/>
    </linearGradient>
    <linearGradient id="orb" x1="0%" y1="100%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#026F6E"/>
      <stop offset="100%" stop-color="#1CA39E"/>
    </linearGradient>
  </defs>
  <rect width="64" height="64" rx="14" fill="url(#bg)" stroke="#1CA39E" stroke-width="2"/>
  <ellipse cx="32" cy="32" rx="24" ry="10" transform="rotate(-32 32 32)" fill="none" stroke="url(#orb)" stroke-width="3.2"/>
  <circle cx="32" cy="32" r="9.5" fill="#1CA39E"/>
  <circle cx="29" cy="29.5" r="2.2" fill="#09456C" opacity="0.65"/>
  <circle cx="36" cy="35" r="1.6" fill="#09456C" opacity="0.65"/>
  <circle cx="49" cy="21" r="3.6" fill="#F1F5F9"/>
</svg>"""

FAVICON_DATA_URI: str = "data:image/svg+xml," + quote(FAVICON_SVG)


@dataclass(frozen=True)
class BrandPalette:
    """Official color palette extracted from team design specification (download.pdf)."""

    cosmic_bg: str = "#160D18"       # RGB(22, 13, 24)  - Deep Cosmic Background
    space_violet: str = "#23145B"    # RGB(35, 20, 91)  - Surface / Secondary Dark
    orbital_blue: str = "#09456C"    # RGB(9, 69, 108)  - Primary Container / Structural Accent
    deep_teal: str = "#026F6E"       # RGB(2, 111, 110) - Secondary Analytical Series
    vibrant_cyan: str = "#1CA39E"    # RGB(28, 163, 158)- Primary Highlight / KPI / Trend Line

    # Complementary high-contrast UI & semantic status colors for dark cosmic theme
    text_primary: str = "#F1F5F9"
    text_muted: str = "#94A3B8"
    grid_line: str = "#2A2438"
    accent_gold: str = "#F59E0B"     # Highlighting inflection points / warnings
    accent_coral: str = "#EF4444"    # High collision / actuarial risk alert
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
            "#38BDF8",
            self.deep_teal,
            "#818CF8",
            self.accent_gold,
            "#F472B6",
            self.orbital_blue,
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
