import os
from typing import Optional
from procrastinate import App, PsycopgConnector
from supabase import create_client, Client
from supabase.lib.client_options import ClientOptions
from postgrest import AsyncPostgrestClient # Import for async client
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import structlog
from urllib.parse import urlparse

load_dotenv()
logger = structlog.get_logger(__name__)

class Database:
    """Manages both Supabase client and Procrastinate app using same database"""
    
    def __init__(self):
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_ANON_KEY")
        self.supabase_db_url = os.getenv("SUPABASE_DB_URL")  # PostgreSQL connection string
        self.supabase_service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") # Added for admin client
        
        # Supabase client for API operations
        self.supabase: Optional[Client] = None
        self._supabase_initialized: bool = False

        # Supabase admin client
        self.supabase_admin: Optional[Client] = None
        self._supabase_admin_initialized: bool = False

        # Procrastinate app using same database
        self.procrastinate_app: Optional[App] = None
        self._procrastinate_initialized: bool = False
        
        # SQLAlchemy for direct database access
        self.engine = None
        self.SessionLocal = None
        self._sqlalchemy_initialized: bool = False

    async def initialize_supabase_client(self):
        """Initializes only the Supabase client."""
        if self._supabase_initialized and self.supabase:
            return
        try:
            if not self.supabase_url or not self.supabase_key:
                logger.error("Supabase URL or Anon Key not set for Supabase client initialization.")
                raise ValueError("SUPABASE_URL and SUPABASE_ANON_KEY must be set.")
            # Reverted to standard client initialization; await behavior will be tested
            self.supabase = create_client(self.supabase_url, self.supabase_key)
            self._supabase_initialized = True
            logger.info("Supabase client initialized (testing async behavior).")
        except Exception as e:
            logger.error(f"Supabase client initialization failed: {str(e)}", exc_info=True)
            self._supabase_initialized = False
            self.supabase = None # Ensure client is None on failure
            raise

    async def initialize_procrastinate(self):
        """Initializes only the Procrastinate app and ensures its pool is open."""
        # This method will now ensure the app from procrastinate_app.py is opened.
        # Assuming app.open_async() is idempotent or handles already-open state gracefully.
        try:
            # Import the app from the central definition
            from video_tool.procrastinate_app import app as central_procrastinate_app
            self.procrastinate_app = central_procrastinate_app
            
            logger.debug(f"Attempting to open Procrastinate app '{self.procrastinate_app.name if hasattr(self.procrastinate_app, 'name') else 'default_app'}'.")
            await self.procrastinate_app.open_async()
            # Check if the pool was actually created on the connector
            if hasattr(self.procrastinate_app.connector, '_async_pool') and \
               self.procrastinate_app.connector._async_pool is not None:
                logger.info("Procrastinate app (from procrastinate_app.py) open_async() called, connector pool should be active.")
                self._procrastinate_initialized = True # Mark as successfully initialized
            else:
                # This case means open_async might not have set up the pool as expected,
                # or our check for _async_pool is insufficient/wrong for the connector type.
                logger.error("CRITICAL: Procrastinate app connector's async pool is None or not found after open_async().")
                self._procrastinate_initialized = False # Mark as failed initialization
                # Optionally raise an error if this state is definitely problematic
                # raise RuntimeError("Procrastinate app connector pool failed to initialize.")

        except Exception as e:
            logger.error(f"Procrastinate app (from procrastinate_app.py) ensuring open failed: {str(e)}", exc_info=True)
            self._procrastinate_initialized = False
            # Do not nullify self.procrastinate_app if it's the shared instance
            # if self.procrastinate_app and self.procrastinate_app.open:
            #     try:
            #         await self.procrastinate_app.close_async()
            #     except Exception as close_e:
            #         logger.error(f"Error closing Procrastinate app: {close_e}", exc_info=True)
            raise

    async def initialize_sqlalchemy(self):
        """Initializes only the SQLAlchemy engine."""
        if self._sqlalchemy_initialized and self.engine:
            return
        try:
            if not self.supabase_db_url:
                logger.error("SUPABASE_DB_URL not set for SQLAlchemy initialization.")
                raise ValueError("SUPABASE_DB_URL must be set for SQLAlchemy.")

            async_sqlalchemy_url = self.supabase_db_url
            if async_sqlalchemy_url.startswith('postgresql://'):
                async_sqlalchemy_url = async_sqlalchemy_url.replace('postgresql://', 'postgresql+asyncpg://', 1)
            
            self.engine = create_async_engine(async_sqlalchemy_url, echo=False)
            self.SessionLocal = sessionmaker(
                self.engine, class_=AsyncSession, expire_on_commit=False
            )
            self._sqlalchemy_initialized = True
            logger.info("SQLAlchemy engine initialized successfully.")
        except Exception as e:
            logger.error(f"SQLAlchemy initialization failed: {str(e)}", exc_info=True)
            self._sqlalchemy_initialized = False
            self.engine = None
            self.SessionLocal = None
            raise

    async def initialize(self):
        """Initializes the core Supabase client. Other components are lazy-loaded via their getters."""
        if not self._supabase_initialized: # Only init if not already done
            await self.initialize_supabase_client()
        logger.debug("Core Database object ensured (Supabase client ready or attempted).")


    async def close(self):
        """Close all potentially open connections."""
        try:
            if self.procrastinate_app: # Removed is_open check
                await self.procrastinate_app.close_async()
                logger.info("Procrastinate app connection closed.")
            if self.engine: # SQLAlchemy engine
                await self.engine.dispose()
                logger.info("SQLAlchemy engine disposed.")
            # Supabase client (self.supabase) does not have an explicit close method for HTTP.
        except Exception as e:
            logger.error(f"Error closing database connections: {str(e)}", exc_info=True)
        finally: # Reset flags to allow re-initialization if needed
            self._procrastinate_initialized = False
            self.procrastinate_app = None
            self._sqlalchemy_initialized = False
            self.engine = None
            self.SessionLocal = None
            # self._supabase_initialized = False # Keep supabase client as it's lightweight
            # self.supabase = None
            self._supabase_admin_initialized = False # Reset admin client flag
            self.supabase_admin = None # Reset admin client

_db_instance: Optional[Database] = None

async def get_db() -> Database:
    """Get initialized database instance using singleton pattern. Ensures core Supabase client is ready."""
    global _db_instance
    if _db_instance is None:
        _db_instance = Database()
        await _db_instance.initialize() # Initializes Supabase client part
    elif not _db_instance._supabase_initialized or _db_instance.supabase is None:
        # This case handles if the instance exists but supabase client part failed/wasn't set
        logger.warning("Supabase client in DB instance was not initialized. Attempting to initialize.")
        await _db_instance.initialize_supabase_client()
    return _db_instance

async def get_supabase() -> Client:
    """Get Supabase client, ensuring it's initialized."""
    db = await get_db() # Ensures Database object exists
    if not db._supabase_initialized or db.supabase is None:
        # This call ensures the Supabase client part of the Database object is initialized
        await db.initialize_supabase_client()
    if db.supabase is None: # Should not happen if initialize_supabase_client is successful
         logger.critical("Supabase client is still None after explicit initialization attempt in get_supabase.")
         raise RuntimeError("Supabase client could not be initialized.")
    return db.supabase

async def get_procrastinate_app() -> App:
    """Get Procrastinate app, ensuring it's initialized and comes from procrastinate_app.py."""
    db = await get_db() # Ensures Database object and its _db_instance exists
    
    # Import the app from the central definition
    try:
        from video_tool.procrastinate_app import app as central_procrastinate_app
    except ImportError as e:
        logger.error("Failed to import central Procrastinate app from video_tool.procrastinate_app", exc_info=True)
        raise RuntimeError("Central Procrastinate app could not be imported.") from e

    db.procrastinate_app = central_procrastinate_app # Assign to the db instance's attribute for consistency if needed elsewhere

    if not db.procrastinate_app: # Should not happen if import is successful
        logger.critical("Central Procrastinate app is None after import.")
        raise RuntimeError("Central Procrastinate app is None.")

    # Ensure the app's connection pool is open.
    # The initialize_procrastinate method now handles this by calling open_async.
    # We rely on initialize_procrastinate to set _procrastinate_initialized correctly.
    
    # Check if the app object exists and if it was marked as initialized.
    # The actual check for the pool's readiness is now within initialize_procrastinate.
    if not db.procrastinate_app or not db._procrastinate_initialized:
        logger.info("Procrastinate app not available or not initialized. Attempting initialization.")
        await db.initialize_procrastinate() # This will use the central app and call open_async.

    # After attempting initialization, re-check if it's now considered initialized.
    # The critical error if the pool is still not active will be logged by initialize_procrastinate.
    if not (db.procrastinate_app and db._procrastinate_initialized and \
            hasattr(db.procrastinate_app.connector, '_async_pool') and \
            db.procrastinate_app.connector._async_pool is not None):
        logger.warning("Procrastinate app or its connector pool might not be fully initialized after get_procrastinate_app.")
        # Depending on strictness, could raise an error here.
        # For now, we'll return the app object, and operations will fail if the pool isn't truly ready.
        
    if not db.procrastinate_app: # Should not happen if import from procrastinate_app.py is okay
        raise RuntimeError("Failed to obtain Procrastinate app instance.")

    return db.procrastinate_app

async def get_sqlalchemy_session() -> AsyncSession:
    """Get an SQLAlchemy session, ensuring the engine is initialized."""
    db = await get_db()
    if not db._sqlalchemy_initialized or db.engine is None or db.SessionLocal is None:
        logger.info("SQLAlchemy engine not yet initialized. Initializing now.")
        await db.initialize_sqlalchemy()
    if db.SessionLocal is None:
        logger.critical("SQLAlchemy SessionLocal is still None after explicit initialization attempt.")
        raise RuntimeError("SQLAlchemy could not be initialized.")
    return db.SessionLocal()

    async def initialize_supabase_admin_client(self):
        """Initializes only the Supabase admin client."""
        if self._supabase_admin_initialized and self.supabase_admin:
            return
        try:
            if not self.supabase_url or not self.supabase_service_key:
                logger.error("Supabase URL or Service Key not set for Supabase admin client initialization.")
                raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set for admin client.")
            # Reverted to standard client initialization; await behavior will be tested
            self.supabase_admin = create_client(self.supabase_url, self.supabase_service_key)
            self._supabase_admin_initialized = True
            logger.info("Supabase admin client initialized (testing async behavior).")
        except Exception as e:
            logger.error(f"Supabase admin client initialization failed: {str(e)}", exc_info=True)
            self._supabase_admin_initialized = False
            self.supabase_admin = None # Ensure client is None on failure
            raise

async def get_supabase_admin() -> Client:
    """Get Supabase admin client, ensuring it's initialized."""
    db = await get_db() # Ensures Database object exists
    if not db._supabase_admin_initialized or db.supabase_admin is None:
        await db.initialize_supabase_admin_client()
    if db.supabase_admin is None:
         logger.critical("Supabase admin client is still None after explicit initialization attempt.")
         raise RuntimeError("Supabase admin client could not be initialized.")
    return db.supabase_admin

async def cleanup_db():
    """Cleanup database connections"""
    global _db_instance
    if _db_instance:
        await _db_instance.close()
        _db_instance = None
        logger.info("Database instance cleaned up.")