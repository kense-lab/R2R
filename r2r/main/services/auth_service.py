import uuid
from datetime import datetime
from typing import Any, Optional

from r2r.base import R2RException, RunLoggingSingleton, RunManager, Token
from r2r.base.api.models.auth.requests import CreateUserRequest
from r2r.base.api.models.auth.responses import UserResponse
from r2r.telemetry.telemetry_decorator import telemetry_event

from ..abstractions import R2RAgents, R2RPipelines, R2RProviders
from ..assembly.config import R2RConfig
from .base import Service


class AuthService(Service):
    def __init__(
        self,
        config: R2RConfig,
        providers: R2RProviders,
        pipelines: R2RPipelines,
        agents: R2RAgents,
        run_manager: RunManager,
        logging_connection: RunLoggingSingleton,
    ):
        super().__init__(
            config,
            providers,
            pipelines,
            agents,
            run_manager,
            logging_connection,
        )

    @telemetry_event("RegisterUser")
    async def register(self, user: CreateUserRequest) -> UserResponse:
        return self.providers.auth.register(user)

    @telemetry_event("VerifyEmail")
    async def verify_email(self, verification_code: str) -> bool:

        if not self.config.auth.require_email_verification:
            raise R2RException(
                status_code=400, message="Email verification is not required"
            )

        user_id = self.providers.database.relational.get_user_id_by_verification_code(
            verification_code
        )
        if not user_id:
            raise R2RException(
                status_code=400, message="Invalid or expired verification code"
            )

        self.providers.database.relational.mark_user_as_verified(user_id)
        self.providers.database.relational.remove_verification_code(
            verification_code
        )
        return {"message": f"User account {user_id} verified successfully."}

    @telemetry_event("Login")
    async def login(self, email: str, password: str) -> dict[str, Token]:
        return self.providers.auth.login(email, password)

    @telemetry_event("GetCurrentUser")
    async def user(self, token: str) -> UserResponse:
        token_data = self.providers.auth.decode_token(token)
        user = self.providers.database.relational.get_user_by_email(
            token_data.email
        )
        if user is None:
            raise R2RException(
                status_code=401, message="Invalid authentication credentials"
            )
        return user

    @telemetry_event("RefreshToken")
    async def refresh_access_token(
        self, user_email: str, refresh_token: str
    ) -> dict[str, Token]:
        return self.providers.auth.refresh_access_token(
            user_email, refresh_token
        )

    @telemetry_event("ChangePassword")
    async def change_password(
        self, user: UserResponse, current_password: str, new_password: str
    ) -> dict[str, str]:
        if not user:
            raise R2RException(status_code=404, message="User not found")
        return self.providers.auth.change_password(
            user, current_password, new_password
        )

    @telemetry_event("RequestPasswordReset")
    async def request_password_reset(self, email: str) -> dict[str, str]:
        return self.providers.auth.request_password_reset(email)

    @telemetry_event("ConfirmPasswordReset")
    async def confirm_password_reset(
        self, reset_token: str, new_password: str
    ) -> dict[str, str]:
        return self.providers.auth.confirm_password_reset(
            reset_token, new_password
        )

    @telemetry_event("Logout")
    async def logout(self, token: str) -> dict[str, str]:
        return self.providers.auth.logout(token)

    @telemetry_event("GetUserProfile")
    async def get_user_profile(self, user_id: uuid.UUID) -> UserResponse:
        user = self.providers.database.relational.get_user_by_id(user_id)
        if not user:
            raise R2RException(status_code=404, message="User not found")
        return user

    @telemetry_event("UpdateUserProfile")
    async def update_user(
        self, user_id: uuid.UUID, user_data: dict[str, Any]
    ) -> UserResponse:
        user = self.providers.database.relational.get_user_by_id(user_id)
        if not user:
            raise R2RException(status_code=404, message="User not found")
        for key, value in user_data.items():
            setattr(user, key, value)
        return self.providers.database.relational.update_user(user)

    @telemetry_event("DeleteUserAccount")
    async def delete_user(
        self,
        user_id: uuid.UUID,
        password: Optional[str] = None,
        is_superuser: bool = False,
    ) -> dict[str, str]:
        user = self.providers.database.relational.get_user_by_id(user_id)
        if not user:
            raise R2RException(status_code=404, message="User not found")
        if not (
            is_superuser
            or self.providers.auth.crypto_provider.verify_password(
                password, user.hashed_password
            )
        ):
            raise R2RException(status_code=400, message="Incorrect password")
        self.providers.database.relational.delete_user(user_id)
        return {"message": f"User account {user_id} deleted successfully."}

    @telemetry_event("CleanExpiredBlacklistedTokens")
    async def clean_expired_blacklisted_tokens(
        self, max_age_hours: int = 7 * 24, current_time: datetime = None
    ):
        self.providers.database.relational.clean_expired_blacklisted_tokens(
            max_age_hours, current_time
        )
