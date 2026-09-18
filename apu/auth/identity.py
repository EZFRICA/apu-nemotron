"""AUTHENTICATION STUB. NOT SECURE. DO NOT DEPLOY AS IS.

Who the requester is gets read from a plain `X-Requester-Id` header, with no password, token
or signature. Anyone who can reach the API can claim to be any teacher or admin by setting
that header, and so read or resolve that person's escalations.

What IS enforced: once a requester id is taken as given, its role and scope come only from
the assignment registry (apu.auth.assignments) via authorize_view, never from anything the
caller declares. Replacing this module with real authentication (e.g. verifying a signed
token issued by the school's identity provider and taking requester_id from its subject)
is required before any real deployment; the rest of the authorization path does not change.
"""

from fastapi import Header, HTTPException, status

REQUESTER_HEADER = "X-Requester-Id"


async def requester_id_from_header(
    x_requester_id: str | None = Header(default=None, alias=REQUESTER_HEADER),
) -> str:
    if not x_requester_id or not x_requester_id.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Missing {REQUESTER_HEADER} header (authentication stub, not secure).",
        )
    return x_requester_id.strip()
