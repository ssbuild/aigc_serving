# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/8/11 14:29
import typing
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseSettings


class AppSettings(BaseSettings):
    # The address of the model controller.
    api_keys: typing.List[str] = None

app_settings = AppSettings()
headers = {"User-Agent": "localserving"}
get_bearer_token = HTTPBearer(auto_error=False)

def auth_api_key(
    auth: typing.Optional[HTTPAuthorizationCredentials] = Depends(get_bearer_token),
) -> typing.Optional[str]:
    if app_settings.api_keys:
        if auth is None or (token := auth.credentials) not in app_settings.api_keys:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": {
                        "message": "",
                        "type": "invalid_request_error",
                        "param": None,
                        "code": "invalid_api_key",
                    }
                },
            )
        return token
    else:
        # api_keys not set; allow all
        return None