"""Shared FastAPI dependencies.

`Annotated` aliases rather than `Depends(...)` defaults -- the current FastAPI
idiom, and it keeps ruff's B008 quiet.
"""

from typing import Annotated

from fastapi import Depends
from sqlalchemy.orm import Session

from argus.core.config import Settings, get_settings
from argus.db.session import get_session

SessionDep = Annotated[Session, Depends(get_session)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
