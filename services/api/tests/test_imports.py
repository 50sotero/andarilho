"""Existing tours: link/listing resolution under SSRF rules, embed records, isolation."""

from __future__ import annotations

import httpx
import pytest
from factories import create_project
from fastapi.testclient import TestClient
from sqlalchemy import select

from andarilho_api.domain.imports import (
    MAX_PAGE_BYTES,
    SourceUnavailable,
    UnsupportedSource,
    fetch_listing,
    models_in_page,
    parse_matterport_url,
    resolve_source,
)
from andarilho_api.main import create_app
from andarilho_api.models import AuditEvent

MODEL = "GzAxTe2zAQs"
LISTING = (
    "https://www.homeq.se/lagenhet/286693-2rum-goteborg-vastra-gotalands-lan-nedre-furuskogsvagen-4"
)
# Listing pages carry the tour in JSON page data, with escaped slashes.
PAGE = (
    '<html><script id="__NEXT_DATA__">{"three_d_views":[{"url":'
    f'"https:\\/\\/my.matterport.com\\/show\\/?m={MODEL}&play=1"}}],'
    '"images":["https:\\/\\/cdn.homeq.se\\/a.jpg"]}</script>'
    f'<a href="https://my.matterport.com/show/?m={MODEL}">tour</a></html>'
)


def public(host: str) -> list[str]:
    return ["93.184.216.34"]


def page_transport(*responses: httpx.Response, seen: list[str] | None = None):
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(str(request.url))
        assert "cookie" not in request.headers and "authorization" not in request.headers
        return queue.pop(0)

    return httpx.MockTransport(handler)


# -- resolver -----------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        f"https://my.matterport.com/show/?m={MODEL}",
        f"https://my.matterport.com/show?m={MODEL}&play=1",
        f"https://MY.matterport.com./show/?play=1&m={MODEL}",
    ],
)
def test_matterport_links_resolve_without_network(url):
    resolution = resolve_source(url, fetch=lambda u: pytest.fail("no fetch for a tour link"))
    assert resolution.source == "matterport_url"
    assert [t.as_dict() for t in resolution.tours] == [{
        "provider": "matterport", "modelId": MODEL, "modes": ["embed"],
        "canonicalUrl": f"https://my.matterport.com/show/?m={MODEL}",
    }]  # fmt: skip


@pytest.mark.parametrize(
    "url",
    [
        f"http://my.matterport.com/show/?m={MODEL}",  # not https
        f"https://user@my.matterport.com/show/?m={MODEL}",  # credentials
        f"https://my.matterport.com:8443/show/?m={MODEL}",  # port
        f"https://my.matterport.com/models/?m={MODEL}",  # not a Showcase link
        "https://my.matterport.com/show/?m=short",  # bad id
        f"https://my.matterport.com/show/?m={MODEL}&m=AAAAAAAAAAA",  # ambiguous
        f"https://evil.example/show/?m={MODEL}",  # other host
        "https://homeq.se.evil.example/lagenhet/1",  # lookalike listing host
        "file:///etc/passwd",
        f"https://my.matterport.com/show/?m={MODEL}%0A",  # encoded newline after the id
    ],
)
def test_other_urls_are_unsupported(url):
    with pytest.raises(UnsupportedSource):
        resolve_source(url, fetch=lambda u: pytest.fail("never fetched"))


def test_listing_tours_are_found_in_page_data():
    assert models_in_page(PAGE) == [MODEL]  # escaped and plain links, deduplicated
    assert parse_matterport_url(LISTING) is None
    fetched: list[str] = []
    resolution = resolve_source(LISTING, fetch=lambda u: fetched.append(u) or PAGE)
    assert fetched == [LISTING]
    assert resolution.source == "homeq_listing" and resolution.listing_url == LISTING
    assert [t.model_id for t in resolution.tours] == [MODEL]
    empty = resolve_source(LISTING, fetch=lambda u: "<html>no tour</html>")
    assert empty.tours == [] and "No Matterport tour" in empty.notes[0]


def test_listing_fetch_follows_only_allowlisted_redirects():
    seen: list[str] = []
    moved = httpx.Response(301, headers={"location": "/lagenhet/286693"})
    page = fetch_listing(
        "https://homeq.se/lagenhet/1",
        resolve=public,
        transport=page_transport(moved, httpx.Response(200, text=PAGE), seen=seen),
    )
    assert models_in_page(page) == [MODEL]
    assert seen == ["https://homeq.se/lagenhet/1", "https://homeq.se/lagenhet/286693"]
    away = httpx.Response(302, headers={"location": "https://169.254.169.254/latest/meta-data"})
    with pytest.raises(SourceUnavailable):
        fetch_listing(LISTING, resolve=public, transport=page_transport(away))
    insecure = httpx.Response(302, headers={"location": "http://www.homeq.se/lagenhet/1"})
    with pytest.raises((SourceUnavailable, UnsupportedSource)):
        fetch_listing(LISTING, resolve=public, transport=page_transport(insecure))


@pytest.mark.parametrize("address", ["127.0.0.1", "10.0.0.5", "169.254.169.254", "::1", "fd00::1"])
def test_listing_fetch_refuses_non_public_addresses(address):
    def handler(request):
        pytest.fail("no request to a private address")

    with pytest.raises(SourceUnavailable):
        fetch_listing(LISTING, resolve=lambda host: [address],
                      transport=httpx.MockTransport(handler))  # fmt: skip


def test_listing_fetch_bounds_size_and_status():
    huge = httpx.Response(200, content=b"x" * (MAX_PAGE_BYTES + 1))
    with pytest.raises(SourceUnavailable):
        fetch_listing(LISTING, resolve=public, transport=page_transport(huge))
    with pytest.raises(SourceUnavailable):
        fetch_listing(LISTING, resolve=public, transport=page_transport(httpx.Response(404)))


# -- API ----------------------------------------------------------------------


@pytest.fixture
def import_client(aws, settings, keys):
    pages = {LISTING: PAGE}

    def fetch(url: str) -> str:
        if url not in pages:
            raise SourceUnavailable("unknown page")
        return pages[url]

    app = create_app(settings, jwks_fetcher=keys.fetch, listing_fetcher=fetch)
    with TestClient(app) as client:
        yield client
    app.state.engine.dispose()


def test_resolve_then_attach_a_listing_tour(import_client, auth, db):
    pid = create_project(import_client, auth)["id"]
    resolved = import_client.post(
        f"/v1/projects/{pid}/imports/resolve", json={"url": LISTING}, headers=auth()
    )
    assert resolved.status_code == 200, resolved.text
    tour = resolved.json()["tours"][0]
    assert tour["modelId"] == MODEL and tour["modes"] == ["embed"]
    body = {"kind": "matterport_embed", "modelId": MODEL, "listingUrl": LISTING,
            "permissionConfirmed": True}  # fmt: skip
    created = import_client.post(f"/v1/projects/{pid}/imports", json=body, headers=auth())
    assert created.status_code == 201, created.text
    assert created.json()["canonicalUrl"] == f"https://my.matterport.com/show/?m={MODEL}"
    again = import_client.post(f"/v1/projects/{pid}/imports", json=body, headers=auth())
    assert again.status_code == 409
    listed = import_client.get(f"/v1/projects/{pid}/imports", headers=auth()).json()["items"]
    assert [i["modelId"] for i in listed] == [MODEL]
    actions = db.execute(select(AuditEvent.action)).scalars().all()
    assert "import.created" in actions

    one = f"/v1/imports/{listed[0]['id']}"
    assert import_client.delete(one, headers=auth("user-b")).status_code == 404
    assert import_client.delete(one, headers=auth()).status_code == 204
    assert import_client.get(f"/v1/projects/{pid}/imports", headers=auth()).json()["items"] == []


def test_import_requests_are_validated(import_client, auth):
    pid = create_project(import_client, auth)["id"]
    url = f"/v1/projects/{pid}/imports"
    base = {"kind": "matterport_embed", "modelId": MODEL, "permissionConfirmed": True}
    # The permission attestation is required and must be affirmed, whatever the client.
    assert import_client.post(url, json={**base, "permissionConfirmed": False},
                              headers=auth()).status_code == 422  # fmt: skip
    unattested = {k: v for k, v in base.items() if k != "permissionConfirmed"}
    assert import_client.post(url, json=unattested, headers=auth()).status_code == 422
    assert import_client.post(url, json={**base, "modelId": "../../x"},
                              headers=auth()).status_code == 422  # fmt: skip
    bad_listing = import_client.post(
        url, json={**base, "listingUrl": "https://evil.example/x"}, headers=auth()
    )
    assert bad_listing.status_code == 422
    assert bad_listing.json()["error"]["code"] == "IMPORT_UNSUPPORTED"
    resolve = f"/v1/projects/{pid}/imports/resolve"
    unsupported = import_client.post(resolve, json={"url": "https://example.com/tour"},
                                     headers=auth())  # fmt: skip
    assert unsupported.status_code == 422
    gone = import_client.post(resolve, json={"url": "https://www.homeq.se/lagenhet/404"},
                              headers=auth())  # fmt: skip
    assert gone.status_code == 502 and gone.json()["error"]["code"] == "IMPORT_SOURCE_UNAVAILABLE"
    other = import_client.post(resolve, json={"url": LISTING}, headers=auth("user-b"))
    assert other.status_code == 404
