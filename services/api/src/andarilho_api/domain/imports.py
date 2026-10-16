"""Existing-tour sources: resolve a Matterport link or a HomeQ listing (README section 15).

Only two inputs are understood: a Matterport Showcase link
(``https://my.matterport.com/show/?m=<model>``), which is parsed without any
network access, and a HomeQ listing page, which is fetched once to find the
Matterport tours it embeds. The listing fetch is the deliberate exception to
"no arbitrary fetch URLs" and is guarded against SSRF: HTTPS only, exact
allowlisted hosts on the default port, no credentials in the URL, every
resolved address public, redirects re-checked hop by hop, no cookies or
application credentials sent, and a bounded response size and time.

The result is only ever a provider *embed*: Matterport keeps presenting its own
tour. A native Andarilho scene needs an owner-authorized export (MatterPak or
model access), which this module never tries to obtain from the provider's CDN.
"""

from __future__ import annotations

import html
import ipaddress
import re
import socket
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal
from urllib.parse import parse_qs, urljoin, urlsplit

import httpx

MATTERPORT_HOST = "my.matterport.com"
LISTING_HOSTS = frozenset({"www.homeq.se", "homeq.se"})
MODEL_ID = re.compile(r"[A-Za-z0-9]{11}")
_MODEL_LINK = re.compile(
    r"https://my\.matterport\.com/show/?\?(?:[^\"'\s<>]*?&)?m=([A-Za-z0-9]{11})(?![A-Za-z0-9])"
)
MAX_PAGE_BYTES = 3 * 1024 * 1024
MAX_REDIRECTS = 3
MAX_TOURS = 10
FETCH_TIMEOUT_SECONDS = 10.0
NATIVE_NOTE = (
    "This shows the provider's own tour inside Andarilho. Walking it natively needs "
    "the owner's MatterPak or model export."
)

Resolver = Callable[[str], list[str]]  # host -> IP addresses


class UnsupportedSource(Exception):
    """The URL is not a supported provider link or listing page."""


class SourceUnavailable(Exception):
    """The listing could not be fetched safely (network, status, size, redirect)."""


@dataclass(frozen=True)
class Tour:
    provider: Literal["matterport"]
    model_id: str
    canonical_url: str
    modes: tuple[str, ...] = ("embed",)

    def as_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "modelId": self.model_id,
            "canonicalUrl": self.canonical_url,
            "modes": list(self.modes),
        }


@dataclass(frozen=True)
class Resolution:
    source: Literal["matterport_url", "homeq_listing"]
    listing_url: str | None
    tours: list[Tour]
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "listingUrl": self.listing_url,
            "tours": [t.as_dict() for t in self.tours],
            "notes": self.notes,
        }


def canonical_matterport_url(model_id: str) -> str:
    if not MODEL_ID.fullmatch(model_id):
        raise UnsupportedSource("invalid Matterport model id")
    return f"https://{MATTERPORT_HOST}/show/?m={model_id}"


def _https_host(url: str) -> tuple[str, str]:
    """``(host, normalized url)`` for a plain HTTPS URL on the default port."""
    if len(url) > 2048 or any(ord(c) < 33 for c in url):
        raise UnsupportedSource("not a plain URL")
    parts = urlsplit(url)
    try:
        port = parts.port
    except ValueError:
        raise UnsupportedSource("bad port") from None
    if parts.scheme != "https" or parts.username or parts.password or port not in (None, 443):
        raise UnsupportedSource("only plain https URLs on the default port")
    host = (parts.hostname or "").lower().rstrip(".")
    return host, parts.geturl()


def parse_matterport_url(url: str) -> str | None:
    """The model id of a Matterport Showcase link, or None for any other URL."""
    host, _ = _https_host(url)
    if host != MATTERPORT_HOST:
        return None
    parts = urlsplit(url)
    if parts.path not in ("/show", "/show/"):
        raise UnsupportedSource("not a Matterport Showcase link")
    ids = parse_qs(parts.query).get("m", [])
    if len(ids) != 1 or not MODEL_ID.fullmatch(ids[0]):
        raise UnsupportedSource("the link has no valid model id")
    return ids[0]


def check_listing_url(url: str) -> str:
    host, normalized = _https_host(url)
    if host not in LISTING_HOSTS:
        raise UnsupportedSource("not an allowlisted listing host")
    return normalized


def models_in_page(page: str) -> list[str]:
    """Matterport model ids linked from a page, in order, without duplicates.

    Listing pages embed the tour in JSON page data, so JSON-escaped slashes and
    HTML-escaped ampersands are undone before matching.
    """
    text = html.unescape(page.replace("\\/", "/").replace("\\u0026", "&"))
    found: list[str] = []
    for match in _MODEL_LINK.finditer(text):
        model = match.group(1)
        if model not in found:
            found.append(model)
        if len(found) >= MAX_TOURS:
            break
    return found


def system_resolver(host: str) -> list[str]:
    try:
        infos = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    except OSError:
        raise SourceUnavailable("name resolution failed") from None
    return [str(info[4][0]) for info in infos]


def _require_public(host: str, resolve: Resolver) -> None:
    addresses = resolve(host)
    if not addresses:
        raise SourceUnavailable("no address")
    for address in addresses:
        ip = ipaddress.ip_address(address.split("%", 1)[0])
        if not ip.is_global or ip.is_multicast:
            raise SourceUnavailable("the host resolves to a non-public address")


def fetch_listing(
    url: str,
    *,
    resolve: Resolver = system_resolver,
    transport: httpx.BaseTransport | None = None,
) -> str:
    """Fetch an allowlisted listing page under the SSRF rules in the module docstring."""
    current = check_listing_url(url)
    for _ in range(MAX_REDIRECTS + 1):
        host, current = _https_host(current)
        if host not in LISTING_HOSTS:
            raise SourceUnavailable("redirected away from the listing host")
        _require_public(host, resolve)
        try:
            with (
                httpx.Client(
                    transport=transport,
                    timeout=httpx.Timeout(FETCH_TIMEOUT_SECONDS),
                    follow_redirects=False,
                    headers={"User-Agent": "Andarilho-import/0.1", "Accept": "text/html"},
                ) as client,
                client.stream("GET", current) as response,
            ):
                if response.status_code in (301, 302, 303, 307, 308):
                    location = response.headers.get("location")
                    if not location:
                        raise SourceUnavailable("redirect without a location")
                    current = urljoin(current, location)
                    continue
                if response.status_code != 200:
                    raise SourceUnavailable(f"status {response.status_code}")
                body = bytearray()
                for chunk in response.iter_bytes():
                    body += chunk
                    if len(body) > MAX_PAGE_BYTES:
                        raise SourceUnavailable("the page is too large")
                return body.decode(response.encoding or "utf-8", errors="replace")
        except httpx.HTTPError:
            raise SourceUnavailable("the page could not be fetched") from None
    raise SourceUnavailable("too many redirects")


def resolve_source(url: str, *, fetch: Callable[[str], str] = fetch_listing) -> Resolution:
    """What ``url`` offers: the tours it points to and how each can be shown."""
    url = url.strip()
    model = parse_matterport_url(url)
    if model is not None:
        tour = Tour("matterport", model, canonical_matterport_url(model))
        return Resolution("matterport_url", None, [tour], [NATIVE_NOTE])
    listing = check_listing_url(url)
    models = models_in_page(fetch(listing))
    tours = [Tour("matterport", m, canonical_matterport_url(m)) for m in models]
    notes = [NATIVE_NOTE] if tours else ["No Matterport tour was found on this listing page."]
    return Resolution("homeq_listing", listing, tours, notes)
