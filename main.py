from typing import List, Dict, Tuple, Optional, Any, Set
from pydantic import BaseModel, Field, SecretStr
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
import chromadb
from chromadb.config import Settings
import PyPDF2
import os
from loguru import logger
import uvicorn
import secrets
import json
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager
import threading
import signal
import sys

# Enhanced logging configuration
LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {process}:{thread} | {name}:{function}:{line} | {message}"

logger.remove()  # Remove default handler
logger.add(
    "logs/app_{time:YYYY-MM-DD}.log",
    rotation="00:00",  # Rotate daily
    retention="30 days",
    level="INFO",
    format=LOG_FORMAT,
    backtrace=True,
    diagnose=True
)
logger.add(lambda msg: print(msg), level="INFO", format=LOG_FORMAT)

# Models
class APIKey(BaseModel):
    """API key model."""
    key: str
    created_at: datetime
    name: str
    is_active: bool = True

class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    query: str = Field(..., description="The search query text")
    n_results: int = Field(default=5, ge=1, le=20, description="Number of results to return")
    doc_limit: int = Field(default=4, ge=1, le=10, description="Maximum number of sentences to return per document")

class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    matches: List[str] = Field(..., description="Matching text segments")
    scores: List[float] = Field(..., description="Similarity scores")
    documents: List[str] = Field(..., description="Source document names")

class APIKeyManager:
    """Manages API key creation, validation, and storage."""
    
    def __init__(self, keys_file: str = "config/api_keys.json"):
        self.keys_file = Path(keys_file)
        self.keys: Dict[str, APIKey] = {}
        self.lock = threading.Lock()
        self._load_keys()

    def _load_keys(self) -> None:
        """Load API keys from file."""
        if self.keys_file.exists():
            try:
                with open(self.keys_file, 'r') as f:
                    keys_data = json.load(f)
                    self.keys = {
                        k: APIKey(
                            key=k,
                            created_at=datetime.fromisoformat(v['created_at']),
                            name=v['name'],
                            is_active=v['is_active']
                        )
                        for k, v in keys_data.items()
                    }
                logger.info(f"Loaded {len(self.keys)} API keys")
            except Exception as e:
                logger.error(f"Error loading API keys: {e}")
                raise

    def _save_keys(self) -> None:
        """Save API keys to file."""
        self.keys_file.parent.mkdir(exist_ok=True)
        with open(self.keys_file, 'w') as f:
            json.dump({
                k: {
                    'created_at': v.created_at.isoformat(),
                    'name': v.name,
                    'is_active': v.is_active
                }
                for k, v in self.keys.items()
            }, f, indent=2)

    def create_key(self, name: str) -> str:
        """Create a new API key."""
        with self.lock:
            if len([k for k in self.keys.values() if k.is_active]) >= 2:
                raise ValueError("Maximum number of active API keys reached")
            
            key = secrets.token_urlsafe(32)
            self.keys[key] = APIKey(
                key=key,
                created_at=datetime.utcnow(),
                name=name
            )
            self._save_keys()
            logger.info(f"Created new API key: {name}")
            return key

    def validate_key(self, key: str) -> bool:
        """Validate an API key."""
        return key in self.keys and self.keys[key].is_active

    def revoke_key(self, key: str) -> None:
        """Revoke an API key."""
        with self.lock:
            if key in self.keys:
                self.keys[key].is_active = False
                self._save_keys()
                logger.info(f"Revoked API key: {self.keys[key].name}")

class VectorDatabase:
    """Vector database management class using ChromaDB."""
    
    def __init__(self, persist_directory: str = "chroma_db") -> None:
        """Initialize the vector database."""
        logger.info(f"Initializing vector database at {persist_directory}")
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        self.client: Any = chromadb.Client(Settings(
            persist_directory=str(self.persist_directory),
            anonymized_telemetry=False
        ))
        self.collection = self.client.get_or_create_collection("pdf_collection")
        logger.info("Vector database initialized successfully")

    @contextmanager
    def error_handling(self, operation: str):
        """Context manager for error handling."""
        try:
            yield
        except Exception as e:
            logger.error(f"Error during {operation}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Database error during {operation}")

    def add_documents(self, texts: List[str], metadata: List[Dict[str, str]]) -> None:
        """Add documents to the vector database."""
        with self.error_handling("document addition"):
            ids = [str(i) for i in range(len(texts))]
            self.collection.add(
                documents=texts,
                metadatas=metadata,
                ids=ids
            )
            logger.info(f"Added {len(texts)} documents to the database")

    def query(self, query_text: str, n_results: int = 5, doc_limit: int = 4) -> Tuple[List[str], List[float], List[str]]:
        """Query the vector database."""
        with self.error_handling("query"):
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results
            )
            
            limited_docs = []
            for doc in results['documents'][0]:
                sentences = doc.split('. ')
                limited_doc = '. '.join(sentences[:doc_limit])
                if limited_doc and not limited_doc.endswith('.'):
                    limited_doc += '.'
                limited_docs.append(limited_doc)
            
            return (
                limited_docs,
                results['distances'][0] if 'distances' in results else [],
                [meta['source'] for meta in results['metadatas'][0]]
            )

class PDFProcessor:
    """PDF processing utility class."""
    
    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> str:
        """Extract text content from a PDF file."""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = '\n'.join(
                    page.extract_text()
                    for page in pdf_reader.pages
                )
            return text
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
            raise

    @staticmethod
    def process_directory(directory: str) -> Tuple[List[str], List[Dict[str, str]]]:
        """Process all PDFs in a directory."""
        texts: List[str] = []
        metadata: List[Dict[str, str]] = []
        
        logger.info(f"Processing PDF directory: {directory}")
        
        directory_path = Path(directory)
        for file_path in directory_path.glob("*.pdf"):
            try:
                text = PDFProcessor.extract_text_from_pdf(str(file_path))
                texts.append(text)
                metadata.append({"source": file_path.name})
                logger.info(f"Processed {file_path.name} successfully")
            except Exception as e:
                logger.error(f"Failed to process {file_path.name}: {str(e)}")
        
        return texts, metadata

# Initialize FastAPI app
app = FastAPI(
    title="PDF Vector Search API",
    description="Search through PDF documents using vector similarity",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
api_key_header = APIKeyHeader(name="X-API-Key")

# Initialize global variables
db: Optional[VectorDatabase] = None
pdf_processor: Optional[PDFProcessor] = None
api_key_manager: Optional[APIKeyManager] = None

def get_api_key(api_key: str = Security(api_key_header)) -> str:
    """Validate API key."""
    if not api_key_manager or not api_key_manager.validate_key(api_key):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return api_key

@app.on_event("startup")
async def startup_event() -> None:
    """Initialize the application on startup."""
    global db, pdf_processor, api_key_manager
    
    logger.info("Starting application...")
    
    # Initialize components
    db = VectorDatabase()
    pdf_processor = PDFProcessor()
    api_key_manager = APIKeyManager()
    
    # Create initial API keys if none exist
    if not api_key_manager.keys:
        try:
            key1 = api_key_manager.create_key("primary_key")
            key2 = api_key_manager.create_key("secondary_key")
            logger.info("Created initial API keys:")
            logger.info(f"Primary Key: {key1}")
            logger.info(f"Secondary Key: {key2}")
        except Exception as e:
            logger.error(f"Error creating initial API keys: {e}")
    
    # Setup PDF directory
    pdf_directory = Path(os.getenv("PDF_DIRECTORY", "pdf_data"))
    pdf_directory.mkdir(exist_ok=True)
    logger.info(f"Using PDF directory: {pdf_directory}")
    
    # Process and index PDFs
    try:
        texts, metadata = pdf_processor.process_directory(str(pdf_directory))
        if texts:
            db.add_documents(texts, metadata)
            logger.info(f"Successfully indexed {len(texts)} PDF documents")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise

@app.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    api_key: str = Depends(get_api_key)
) -> QueryResponse:
    """Query the vector database for similar documents."""
    try:
        logger.info(f"Processing query request: {request.query}")
        
        if not db:
            raise HTTPException(status_code=500, detail="Database not initialized")
        
        matches, scores, documents = db.query(
            request.query,
            n_results=request.n_results,
            doc_limit=request.doc_limit
        )
        
        return QueryResponse(
            matches=matches,
            scores=scores,
            documents=documents
        )
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    logger.info(f"Received signal {signum}")
    sys.exit(0)

def main() -> None:
    """Run the application."""
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run the application
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=False,  # Disable reload in production
        workers=int(os.getenv("WORKERS", "1")),
        log_level="info"
    )

if __name__ == "__main__":
    main()
