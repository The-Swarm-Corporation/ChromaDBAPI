from typing import List, Dict, Tuple, Optional, Any
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import chromadb
from chromadb.config import Settings
import PyPDF2
import os
from loguru import logger
import uvicorn

# Setup loguru logger
logger.remove()  # Remove default handler
logger.add(
    "app.log",
    rotation="500 MB",
    retention="10 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
)
logger.add(lambda msg: print(msg), level="INFO")  # Console output

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

class VectorDatabase:
    """Vector database management class using ChromaDB."""
    
    def __init__(self, persist_directory: str = "chroma_db") -> None:
        """Initialize the vector database."""
        logger.info(f"Initializing vector database at {persist_directory}")
        self.client: Any = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))
        self.collection = self.client.get_or_create_collection("pdf_collection")
        logger.info("Vector database initialized successfully")

    def add_documents(self, texts: List[str], metadata: List[Dict[str, str]]) -> None:
        """
        Add documents to the vector database.
        
        Args:
            texts: List of document texts
            metadata: List of metadata dictionaries
        """
        try:
            ids = [str(i) for i in range(len(texts))]
            self.collection.add(
                documents=texts,
                metadatas=metadata,
                ids=ids
            )
            logger.info(f"Added {len(texts)} documents to the database")
        except Exception as e:
            logger.error(f"Error adding documents to database: {str(e)}")
            raise

    def query(self, query_text: str, n_results: int = 5, doc_limit: int = 4) -> Tuple[List[str], List[float], List[str]]:
        """
        Query the vector database.
        
        Args:
            query_text: Search query
            n_results: Number of results to return
            doc_limit: Maximum number of sentences to return per document
            
        Returns:
            Tuple of (matches, scores, source_documents)
        """
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results
            )
            
            # Limit the length of returned documents
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
        except Exception as e:
            logger.error(f"Error querying database: {str(e)}")
            raise

class PDFProcessor:
    """PDF processing utility class."""
    
    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> str:
        """
        Extract text content from a PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text content
        """
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
        """
        Process all PDFs in a directory.
        
        Args:
            directory: Directory containing PDF files
            
        Returns:
            Tuple of (texts, metadata)
        """
        texts: List[str] = []
        metadata: List[Dict[str, str]] = []
        
        logger.info(f"Processing PDF directory: {directory}")
        
        for filename in os.listdir(directory):
            if filename.lower().endswith('.pdf'):
                file_path = os.path.join(directory, filename)
                try:
                    text = PDFProcessor.extract_text_from_pdf(file_path)
                    texts.append(text)
                    metadata.append({"source": filename})
                    logger.info(f"Processed {filename} successfully")
                except Exception as e:
                    logger.error(f"Failed to process {filename}: {str(e)}")
        
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

# Initialize global variables
db: Optional[VectorDatabase] = None
pdf_processor: Optional[PDFProcessor] = None

@app.on_event("startup")
async def startup_event() -> None:
    """Initialize the database and processor on startup."""
    global db, pdf_processor
    
    logger.info("Starting application...")
    
    # Initialize components
    db = VectorDatabase()
    pdf_processor = PDFProcessor()
    
    # Setup PDF directory
    pdf_directory = os.getenv("PDF_DIRECTORY", "pdf_data")
    if not os.path.exists(pdf_directory):
        os.makedirs(pdf_directory)
        logger.info(f"Created PDF directory: {pdf_directory}")
        return
    
    # Process and index PDFs
    try:
        texts, metadata = pdf_processor.process_directory(pdf_directory)
        if texts:
            db.add_documents(texts, metadata)
            logger.info(f"Successfully indexed {len(texts)} PDF documents")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest) -> QueryResponse:
    """
    Query the vector database for similar documents.
    
    Args:
        request: Query request containing search text, number of results, and document limit
        
    Returns:
        QueryResponse with matches, scores, and source documents
    """
    try:
        logger.info(f"Processing query request: {request.query} (n_results={request.n_results}, doc_limit={request.doc_limit})")
        
        if not db:
            raise HTTPException(status_code=500, detail="Database not initialized")
        
        matches, scores, documents = db.query(
            request.query,
            n_results=request.n_results,
            doc_limit=request.doc_limit
        )
        
        logger.info(f"Query successful, found {len(matches)} matches")
        
        return QueryResponse(
            matches=matches,
            scores=scores,
            documents=documents
        )
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def main() -> None:
    """Run the application."""
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()
