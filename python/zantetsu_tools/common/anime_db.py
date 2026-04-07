from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from .paths import DEFAULT_ANIMEDB_BASE


@dataclass
class AnimeDbClient:
    base_url: str = DEFAULT_ANIMEDB_BASE
    user_agent: str = "Zantetsu/1.0"

    def _request_json(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        query = urllib.parse.urlencode(params)
        url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}?{query}"
        request = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        with urllib.request.urlopen(request, timeout=15) as response:
            return json.loads(response.read())

    def fetch_anilist_media_page(
        self,
        page: int,
        *,
        page_size: int = 100,
        media_type: str = "ANIME",
    ) -> list[dict[str, Any]]:
        data = self._request_json(
            "/anilist/media",
            {"page": page, "page_size": page_size, "type": media_type},
        )
        return data.get("data", [])

    def fetch_all_anime_titles(
        self,
        *,
        max_pages: int = 50,
        page_size: int = 500,
        delay: float = 0.5,
        progress: bool = True,
    ) -> list[dict[str, Any]]:
        all_titles: list[dict[str, Any]] = []
        seen_ids: set[int] = set()

        if progress:
            print("Fetching anime titles from AnimeDB API...")

        for page in range(1, max_pages + 1):
            try:
                titles = self.fetch_anilist_media_page(page, page_size=page_size)
            except Exception as exc:
                print(f"  Error fetching page {page}: {exc}", file=sys.stderr)
                break

            if not titles:
                break

            for title in titles:
                anime_id = title.get("id")
                if anime_id in seen_ids:
                    continue
                seen_ids.add(anime_id)
                all_titles.append(title)

            if progress:
                print(
                    f"  Fetched page {page}: {len(titles)} titles (total: {len(all_titles)})"
                )

            if delay > 0:
                time.sleep(delay)

        return all_titles

    def search_realtime(
        self,
        title: str,
        *,
        source: str = "both",
        limit: int = 5,
    ) -> dict[str, Any] | None:
        if not title.strip():
            return None
        try:
            data = self._request_json(
                "/search/realtime",
                {"q": title[:200], "source": source, "limit": limit},
            )
        except Exception as exc:
            print(f"  API error: {exc}", file=sys.stderr)
            return None

        results = data.get("data") or []
        return results[0] if results else None
