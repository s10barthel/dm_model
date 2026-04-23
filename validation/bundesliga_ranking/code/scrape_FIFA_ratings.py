from pathlib import Path
import os
import re
import sys
import tempfile

RANKING_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_ROOT = Path(__file__).resolve().parent
CACHE_ROOT = RANKING_ROOT / "soccerdata_cache"
TEMP_ROOT = RANKING_ROOT / "tmp"
BROWSER_PROFILE_ROOT = WORKSPACE_ROOT / "data" / "chrome_profiles" / "sofifa"
OUTPUT_CSV = RANKING_ROOT / "output" / "fc25_ratings.csv"
LEAGUE = "GER-Bundesliga"
VERSION = 250044
METADATA_COLUMNS = [
    "player_id",
    "player",
    "team",
    "league",
    "version_id",
    "fifa_edition",
    "update",
]
EXPECTED_RATING_COLUMNS = [
    "overallrating",
    "potential",
    "crossing",
    "finishing",
    "headingaccuracy",
    "shortpassing",
    "volleys",
    "dribbling",
    "curve",
    "fk_accuracy",
    "longpassing",
    "ballcontrol",
    "acceleration",
    "sprintspeed",
    "agility",
    "reactions",
    "balance",
    "shotpower",
    "jumping",
    "stamina",
    "strength",
    "longshots",
    "aggression",
    "interceptions",
    "positioning",
    "vision",
    "penalties",
    "composure",
    "defensiveawareness",
    "standingtackle",
    "slidingtackle",
    "gk_diving",
    "gk_handling",
    "gk_kicking",
    "gk_positioning",
    "gk_reflexes",
]
RATING_LABEL_TO_COLUMN = {
    "Overall rating": "overallrating",
    "Gesamtbewertung": "overallrating",
    "Potential": "potential",
    "Potenzial": "potential",
    "Crossing": "crossing",
    "Flanken": "crossing",
    "Finishing": "finishing",
    "Abschluss": "finishing",
    "Heading accuracy": "headingaccuracy",
    "Kopfball-Präzision": "headingaccuracy",
    "Kopfballpräzision": "headingaccuracy",
    "Short passing": "shortpassing",
    "Kurzpässe": "shortpassing",
    "Volleys": "volleys",
    "Dribbling": "dribbling",
    "Curve": "curve",
    "Effet": "curve",
    "FK Accuracy": "fk_accuracy",
    "Freistoß-Präzision": "fk_accuracy",
    "Long passing": "longpassing",
    "Lange Pässe": "longpassing",
    "Ball control": "ballcontrol",
    "Ballkontrolle": "ballcontrol",
    "Acceleration": "acceleration",
    "Beschleunigung": "acceleration",
    "Sprint speed": "sprintspeed",
    "Sprintgeschwindigkeit": "sprintspeed",
    "Agility": "agility",
    "Beweglichkeit": "agility",
    "Reactions": "reactions",
    "Reaktionen": "reactions",
    "Balance": "balance",
    "Shot power": "shotpower",
    "Schusskraft": "shotpower",
    "Jumping": "jumping",
    "Sprungkraft": "jumping",
    "Stamina": "stamina",
    "Ausdauer": "stamina",
    "Strength": "strength",
    "Stärke": "strength",
    "Long shots": "longshots",
    "Fernschüsse": "longshots",
    "Aggression": "aggression",
    "Aggressivität": "aggression",
    "Interceptions": "interceptions",
    "Abfangen": "interceptions",
    "Positioning": "positioning",
    "Stellungssp.": "positioning",
    "Vision": "vision",
    "Übersicht": "vision",
    "Penalties": "penalties",
    "Elfmeter": "penalties",
    "Composure": "composure",
    "Ruhe": "composure",
    "Defensive awareness": "defensiveawareness",
    "Defensives Bewusstsein": "defensiveawareness",
    "Standing tackle": "standingtackle",
    "Faire Zweikämpfe": "standingtackle",
    "Sliding tackle": "slidingtackle",
    "Grätsche": "slidingtackle",
    "GK Diving": "gk_diving",
    "TH Flugparaden": "gk_diving",
    "GK Handling": "gk_handling",
    "TH Fangsicherheit": "gk_handling",
    "GK Kicking": "gk_kicking",
    "TH Abschlag": "gk_kicking",
    "GK Positioning": "gk_positioning",
    "TH Stellungsspiel": "gk_positioning",
    "GK Reflexes": "gk_reflexes",
    "TH Reflexe": "gk_reflexes",
}
BLOCK_PAGE_MARKERS = [
    "cf-error-details",
    "sorry, you have been blocked",
    "you are unable to access",
    "cloudflare ray id",
]

# soccerdata creates its cache/log directories at import time, so the workspace
# cache root must be configured before importing the package.
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
TEMP_ROOT.mkdir(parents=True, exist_ok=True)
BROWSER_PROFILE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ["SOCCERDATA_DIR"] = str(CACHE_ROOT)
os.environ["TEMP"] = str(TEMP_ROOT)
os.environ["TMP"] = str(TEMP_ROOT)
os.environ["TMPDIR"] = str(TEMP_ROOT)
tempfile.tempdir = str(TEMP_ROOT)

from soccerdata import SoFIFA
import pandas as pd
import seleniumbase as sb
from lxml import html


class WorkspaceSoFIFA(SoFIFA):
    @classmethod
    def _all_leagues(cls):
        return SoFIFA._all_leagues()

    def _init_webdriver(self):
        if hasattr(self, "_driver"):
            self._driver.quit()

        proxy_str = self.proxy()
        resolver_rules = None
        if proxy_str is not None:
            resolver_rules = "MAP * ~NOTFOUND , EXCLUDE 127.0.0.1"

        return sb.Driver(
            uc=True,
            headless=self.headless,
            binary_location=self.path_to_browser,
            host_resolver_rules=resolver_rules,
            proxy=proxy_str,
            user_data_dir=str(BROWSER_PROFILE_ROOT),
        )

    def _validate_page(self, url: str) -> str:
        page_text = super()._validate_page(url)
        validate_not_block_page(page_text, url)
        return page_text


def build_reader() -> WorkspaceSoFIFA:
    return WorkspaceSoFIFA(leagues=LEAGUE, versions=VERSION, headless=True)


def fetch_roster(reader: SoFIFA) -> pd.DataFrame:
    roster = reader.read_players().reset_index()
    roster["version_id"] = VERSION
    roster = roster.drop_duplicates(subset=["player_id"]).reset_index(drop=True)
    return roster


def safe_console_text(value: object) -> str:
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    return str(value).encode(encoding, errors="replace").decode(encoding)


def normalize_label(value: object) -> str:
    return " ".join(str(value).split())


def validate_not_block_page(page_text: str | bytes, source: object) -> None:
    if isinstance(page_text, bytes):
        searchable_text = page_text.decode("utf-8", errors="replace")
    else:
        searchable_text = str(page_text)

    lowered_text = searchable_text.lower()
    if any(marker in lowered_text for marker in BLOCK_PAGE_MARKERS):
        raise RuntimeError(
            f"SoFIFA returned a Cloudflare/block page for {source}. "
            "The page was not parsed because it is not valid SoFIFA data."
        )


def cached_player_page_path(reader: SoFIFA, player_id: int, version_id: int = VERSION) -> Path:
    return reader.data_dir / f"player_{player_id}_{version_id}.html"


def read_page_bytes(reader) -> bytes:
    try:
        return reader.read()
    finally:
        reader.close()


def parse_rating_value(node) -> int | None:
    raw_value = normalize_label(node.xpath("string(.//em[1]/@title)"))
    if not raw_value:
        raw_value = normalize_label(node.xpath("string(.//em[1])"))
    if not re.fullmatch(r"\d+", raw_value):
        return None
    return int(raw_value)


def add_rating_if_known(ratings: dict[str, int], label: str, value: int | None) -> None:
    column = RATING_LABEL_TO_COLUMN.get(normalize_label(label))
    if column is not None and value is not None:
        ratings[column] = value


def extract_player_name(tree) -> str:
    profile_names = tree.xpath("//div[contains(@class, 'profile')]/h1")
    if not profile_names:
        raise RuntimeError("Could not find player profile name in SoFIFA page.")

    name_parts = [
        normalize_label(text)
        for text in profile_names[0].itertext()
        if normalize_label(text) and normalize_label(text) != "\xa0"
    ]
    if not name_parts:
        raise RuntimeError("Could not parse player profile name in SoFIFA page.")
    return name_parts[-1]


def parse_player_rating_page(page_bytes: bytes, player_id: int, source: object) -> dict[str, object]:
    validate_not_block_page(page_bytes, source)
    tree = html.fromstring(page_bytes, parser=html.HTMLParser(encoding="utf8"))
    ratings: dict[str, int] = {}

    profile_rating_nodes = tree.xpath(
        "//div[contains(concat(' ', normalize-space(@class), ' '), ' grid ')]"
        "/div[contains(concat(' ', normalize-space(@class), ' '), ' col ')][.//div[contains(@class, 'sub')]]"
    )
    for node in profile_rating_nodes:
        label = normalize_label(node.xpath("string(.//div[contains(@class, 'sub')])"))
        add_rating_if_known(ratings, label, parse_rating_value(node))

    attribute_nodes = tree.xpath(
        "//div[contains(concat(' ', normalize-space(@class), ' '), ' attribute ')]"
        "//p[.//em]"
    )
    for node in attribute_nodes:
        label_nodes = node.xpath("./span")
        if not label_nodes:
            continue
        label = normalize_label(label_nodes[-1].xpath("string(.)"))
        add_rating_if_known(ratings, label, parse_rating_value(node))

    missing_columns = [col for col in EXPECTED_RATING_COLUMNS if col not in ratings]
    if missing_columns:
        page_language = normalize_label(tree.xpath("string(//body/@lang)")) or "unknown"
        raise RuntimeError(
            f"SoFIFA rating extraction failed for player_id={player_id} "
            f"from {source}; body lang={page_language!r}; missing columns: "
            + ", ".join(missing_columns)
        )

    return {
        "player": extract_player_name(tree),
        **{column: ratings[column] for column in EXPECTED_RATING_COLUMNS},
    }


def fetch_player_rating_row(reader: SoFIFA, player_row: dict[str, object]) -> pd.DataFrame:
    player_id = int(player_row["player_id"])
    if len(reader.versions) != 1:
        raise RuntimeError("This scraper expects exactly one SoFIFA version.")

    version_id = int(reader.versions.index[0])
    version = reader.versions.iloc[0].to_dict()
    page_path = cached_player_page_path(reader, player_id, version_id)
    page_url = f"https://sofifa.com/player/{player_id}/?r={version_id}&set=true"
    page_bytes = read_page_bytes(reader.get(page_url, page_path))
    parsed_ratings = parse_player_rating_page(page_bytes, player_id, page_path)

    row = {
        "player_id": player_id,
        **parsed_ratings,
        "team": player_row["team"],
        "league": player_row["league"],
        "version_id": version_id,
        **version,
    }
    return pd.DataFrame([row])


def order_columns(df: pd.DataFrame) -> pd.DataFrame:
    remaining_columns = [
        col
        for col in df.columns
        if col not in METADATA_COLUMNS and col not in EXPECTED_RATING_COLUMNS
    ]
    return df.loc[:, [*METADATA_COLUMNS, *EXPECTED_RATING_COLUMNS, *remaining_columns]]


def validate_rating_columns(df: pd.DataFrame) -> None:
    missing_columns = [col for col in EXPECTED_RATING_COLUMNS if col not in df.columns]
    if missing_columns:
        raise RuntimeError(
            "SoFIFA output is missing expected rating columns: "
            + ", ".join(missing_columns)
        )

    missing_values_by_column = {
        col: int(df[col].isna().sum())
        for col in EXPECTED_RATING_COLUMNS
        if df[col].isna().any()
    }
    if missing_values_by_column:
        missing_summary = ", ".join(
            f"{col}={missing_count}"
            for col, missing_count in missing_values_by_column.items()
        )
        raise RuntimeError(
            "SoFIFA rating extraction produced missing values. "
            f"Missing counts by column: {missing_summary}"
        )


def fetch_fc25_ratings() -> pd.DataFrame:
    reader = build_reader()
    roster = fetch_roster(reader)
    player_rows = roster.to_dict(orient="records")

    rating_frames: list[pd.DataFrame] = []
    total_players = len(player_rows)
    for idx, player_row in enumerate(player_rows, start=1):
        print(
            f"[{idx}/{total_players}] Scraping ratings for "
            f"{safe_console_text(player_row['player'])} "
            f"({safe_console_text(player_row['team'])})",
            flush=True,
        )
        rating_frames.append(fetch_player_rating_row(reader, player_row))

    if not rating_frames:
        raise RuntimeError("No Bundesliga players were returned by SoFIFA.")

    ratings = pd.concat(rating_frames, ignore_index=True)
    ratings = order_columns(ratings)
    validate_rating_columns(ratings)
    ratings = ratings.sort_values(["team", "player", "player_id"]).reset_index(drop=True)
    return ratings


def export_ratings(df: pd.DataFrame, output_path: Path = OUTPUT_CSV) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    return output_path


def main() -> int:
    ratings = fetch_fc25_ratings()
    output_path = export_ratings(ratings)
    print(f"Saved {len(ratings)} player rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
