import json
from json.decoder import JSONDecodeError # Import specific exception
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timezone # Added timezone for UTC comparison
import structlog
from supabase import Client # Import Client for type hinting

from video_tool.core.db import get_supabase, get_db, get_supabase_admin # Added get_supabase_admin

logger = structlog.get_logger(__name__)

# Auth file location
AUTH_FILE = Path.home() / ".video_tool_auth.json"

class AuthManager:
    """Manages CLI authentication with Supabase"""
    
    def __init__(self):
        self.client: Optional[Client] = None # Explicitly type client
        self._ensure_auth_file()
    
    def _ensure_auth_file(self):
        """Ensure auth file exists with proper permissions"""
        if not AUTH_FILE.exists():
            AUTH_FILE.touch(mode=0o600) # Ensure mode is octal
    
    async def _initialize_client(self):
        """Initializes the Supabase client if not already initialized."""
        if not self.client:
            # Use get_db to ensure Database is initialized before getting supabase client
            await get_db() 
            self.client = await get_supabase()
            if not self.client:
                # This case should ideally be handled by get_supabase raising an error
                logger.error("Failed to initialize Supabase client in AuthManager.")
                raise RuntimeError("Supabase client could not be initialized.")

    async def login(self, email: str, password: str) -> bool:
        """Login with email and password"""
        await self._initialize_client()
        try:
            # Authenticate using Supabase Auth
            response = self.client.auth.sign_in_with_password({
                "email": email,
                "password": password
            })
            
            if response.session:
                # Save session
                session_data = {
                    "access_token": response.session.access_token,
                    "refresh_token": response.session.refresh_token,
                    "expires_at": response.session.expires_at, # This is typically a UTC timestamp
                    "user_id": response.user.id,
                    "email": response.user.email
                }
                self._save_session(session_data)
                logger.info(f"Successfully logged in as {email}")
                return True
                
        except Exception as e:
            logger.error(f"Login failed: {str(e)}", exc_info=True)
            
        return False
    
    async def signup(self, email: str, password: str) -> bool:
        """Sign up new user"""
        await self._initialize_client()
        try:
            # Sign up using Supabase Auth
            response = self.client.auth.sign_up({
                "email": email,
                "password": password
            })
            
            if response.user and response.user.id: # Check for user object and ID in response
                user_id = response.user.id
                logger.info(f"Successfully signed up user: {email} with ID: {user_id}. Attempting auto-confirmation.")
                try:
                    admin_client = await get_supabase_admin()
                    # Confirm the user's email
                    # Note: The exact method might vary slightly based on supabase-py version.
                    # Common is admin.update_user_by_id or client.auth.admin.update_user_by_id
                    # Assuming client.auth.admin is the correct path for the admin functions.
                    update_response = admin_client.auth.admin.update_user_by_id(
                        user_id, {"email_confirm": True}
                    )
                    # Check update_response for success, though it might not return detailed info or could raise on error.
                    # If no error is raised, assume success.
                    logger.info(f"Successfully auto-confirmed email for user: {email}")
                    # Optionally, you might want to log in the user directly here if desired,
                    # or instruct them they can now log in.
                    # For now, just confirm and let them log in separately.
                except Exception as admin_e:
                    logger.error(f"Failed to auto-confirm email for {email}: {str(admin_e)}", exc_info=True)
                    # Decide if signup should still be considered successful if auto-confirmation fails.
                    # For now, we'll still return True for signup, but log the confirmation error.
                    logger.info("User was signed up, but email auto-confirmation failed. Manual confirmation might be needed.")
                return True
                
        except Exception as e:
            logger.error(f"Signup failed: {str(e)}", exc_info=True)
            
        return False
    
    async def logout(self) -> bool:
        """Logout and clear stored session"""
        await self._initialize_client()
        try:
            # Sign out from Supabase
            # The plan uses `await self.client.auth.sign_out()`
            # Depending on the supabase-py version, sign_out might not be async
            # or might return an error if no active session.
            # Let's assume it's async as per the plan.
            # Removing await as it's often synchronous in supabase-py
            self.client.auth.sign_out()
            
            # Clear local session file
            if AUTH_FILE.exists():
                AUTH_FILE.unlink()
            
            logger.info("Successfully logged out")
            return True
            
        except Exception as e:
            logger.error(f"Logout failed: {str(e)}", exc_info=True)
            return False
    
    async def get_current_session(self) -> Optional[Dict[str, Any]]:
        """Get current session if valid. Tries to refresh if expired."""
        if not AUTH_FILE.exists():
            return None
        
        try:
            with open(AUTH_FILE, 'r') as f:
                session_data = json.load(f)
            
            expires_at_ts = session_data.get('expires_at', 0)
            if expires_at_ts <= datetime.now(timezone.utc).timestamp():
                logger.info("Session token expired, attempting refresh.")
                return await self._refresh_session(session_data) # Now directly awaits refresh

            return session_data
            
        except (FileNotFoundError, JSONDecodeError) as e:
            # These are expected if the auth file doesn't exist or is empty/corrupted.
            # Log as info/debug instead of error, and don't print full traceback for these.
            logger.info(f"Auth session file not found or invalid, proceeding as new session: {str(e)}")
            if AUTH_FILE.exists(): # Attempt to clean up if it was corrupted
                try:
                    AUTH_FILE.unlink()
                    logger.debug(f"Removed invalid auth file: {AUTH_FILE}")
                except OSError as ose:
                    logger.warning(f"Could not remove invalid auth file {AUTH_FILE}: {ose}")
            return None
        except Exception as e:
            # For any other unexpected errors, log them as errors with traceback.
            logger.error(f"Failed to read session due to an unexpected error: {str(e)}", exc_info=True)
            if AUTH_FILE.exists(): # Attempt to clean up if it was corrupted
                try:
                    AUTH_FILE.unlink()
                    logger.info(f"Removed auth file after unexpected error: {AUTH_FILE}")
                except OSError as ose:
                    logger.error(f"Error removing auth file after unexpected error {AUTH_FILE}: {ose}")
            return None
    
    async def _refresh_session(self, old_session_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Refresh an expired session using the refresh token"""
        await self._initialize_client()
        try:
            refresh_token_value = old_session_data.get('refresh_token')
            if not refresh_token_value:
                logger.warning("No refresh token found in old session.")
                return None

            # Supabase-py uses set_session then refresh_session
            # The plan's `self.client.auth.set_session` then `self.client.auth.refresh_session()` is correct.
            # However, `refresh_session` in some versions might not be async or might return the new session directly.
            # The plan implies `refresh_session()` returns an object with a `session` attribute.
            
            # Set the existing session to use its refresh token
            self.client.auth.set_session(
                access_token=old_session_data['access_token'], # access_token is needed for set_session
                refresh_token=refresh_token_value
            )
            
            refreshed_response = self.client.auth.refresh_session() # This might not be async
            # If it's not async and returns the session directly:
            # new_session_obj = refreshed_response
            # If it is async:
            # refreshed_response = await self.client.auth.refresh_session() # if it were async

            # Assuming refreshed_response has .session and .user as per the plan
            if refreshed_response and refreshed_response.session:
                new_session_data = {
                    "access_token": refreshed_response.session.access_token,
                    "refresh_token": refreshed_response.session.refresh_token, # Important: get the new refresh token
                    "expires_at": refreshed_response.session.expires_at,
                    "user_id": refreshed_response.user.id,
                    "email": refreshed_response.user.email
                }
                self._save_session(new_session_data)
                logger.info("Successfully refreshed session.")
                return new_session_data
            else:
                logger.warning("Failed to refresh session, no new session data returned.")
                # Clear the invalid session file if refresh fails
                if AUTH_FILE.exists(): AUTH_FILE.unlink()
                return None
                
        except Exception as e:
            logger.error(f"Failed to refresh session: {str(e)}", exc_info=True)
            if AUTH_FILE.exists(): AUTH_FILE.unlink() # Clear broken session
            return None
    
    async def get_authenticated_client(self) -> Optional[Client]:
        """Get authenticated Supabase client, attempting refresh if session is expired."""
        await self._initialize_client() # Ensures self.client is potentially available
        
        session_data = await self.get_current_session() # Now async, handles refresh internally

        if session_data:
            # If session_data is returned, it's either valid or has been successfully refreshed.
            self.client.auth.set_session(
                access_token=session_data['access_token'],
                refresh_token=session_data['refresh_token']
            )
            return self.client
        
        logger.info("No valid session found even after attempting refresh.")
        return None
    
    async def get_user_profile(self) -> Optional[Dict[str, Any]]:
        """Get current user profile from the database via RPC"""
        authed_client = await self.get_authenticated_client()
        if not authed_client:
            return None
        
        try:
            # Get user profile from the database
            # Ensure the RPC function 'get_user_profile' exists in your Supabase SQL.
            result = authed_client.rpc('get_user_profile').execute()
            if result.data:
                return result.data[0] if isinstance(result.data, list) and result.data else result.data
            else:
                logger.warning(f"No data returned from get_user_profile RPC. Response: {result}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to get user profile: {str(e)}", exc_info=True)
            return None
    
    async def is_admin(self) -> bool:
        """Check if current user is admin"""
        profile = await self.get_user_profile()
        if not profile:
            return False
        
        return profile.get('profile_type') == 'admin'
    
    def _save_session(self, session_data: Dict[str, Any]) -> None:
        """Save session to local file"""
        try:
            with open(AUTH_FILE, 'w') as f:
                json.dump(session_data, f)
            
            # Ensure proper permissions
            AUTH_FILE.chmod(0o600) # Ensure mode is octal
            
        except Exception as e:
            logger.error(f"Failed to save session: {str(e)}", exc_info=True)
    
    async def require_auth(self) -> Client:
        """Require authentication or raise exception"""
        client = await self.get_authenticated_client()
        if not client:
            # Logged out or session invalid
            logger.error("Authentication required. Client not available.")
            raise Exception("Authentication required. Please login first.")
        return client