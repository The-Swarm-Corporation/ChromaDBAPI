
# PDF Vector Search API


[![Join our Discord](https://img.shields.io/badge/Discord-Join%20our%20server-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/agora-999382051935506503) [![Subscribe on YouTube](https://img.shields.io/badge/YouTube-Subscribe-red?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@kyegomez3242) [![Connect on LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kye-g-38759a207/) [![Follow on X.com](https://img.shields.io/badge/X.com-Follow-1DA1F2?style=for-the-badge&logo=x&logoColor=white)](https://x.com/kyegomezb)


[![GitHub stars](https://img.shields.io/github/stars/The-Swarm-Corporation/Legal-Swarm-Template?style=social)](https://github.com/The-Swarm-Corporation/Legal-Swarm-Template)
[![Swarms Framework](https://img.shields.io/badge/Built%20with-Swarms-blue)](https://github.com/kyegomez/swarms)

A production-ready FastAPI application that creates a searchable vector database from PDF documents using ChromaDB.

## Features

- PDF text extraction and vectorization
- Semantic search capabilities
- Automatic document indexing
- Configurable search results
- Comprehensive logging
- CORS support
- Type-safe implementation

## Prerequisites

- Python 3.8+
- Adequate storage space for PDF documents and vector database
- Access permissions to create directories and files

## Installation

1. Clone the repository:
```bash
git clone (https://github.com/The-Swarm-Corporation/ChromaDBAPI/)
cd pdf-vector-search-api
```

2. Install required dependencies:
```bash
pip install fastapi uvicorn chromadb PyPDF2 python-multipart pydantic loguru
```

3. Create a directory for your PDF documents:
```bash
mkdir pdf_data
```

## Configuration

The application uses several environment variables that can be configured:

- `PDF_DIRECTORY`: Directory containing PDF files (default: "pdf_data")
- `HOST`: Server host (default: "0.0.0.0")
- `PORT`: Server port (default: 8000)

## Usage

1. Place your PDF documents in the `pdf_data` directory.

2. Start the server:
```bash
python main.py
```

The API will automatically:
- Create necessary directories if they don't exist
- Process and index all PDFs in the specified directory
- Start the FastAPI server
- Initialize logging

## API Endpoints

### POST /query

Search through indexed PDF documents using semantic similarity.

#### Request Body

```json
{
    "query": "your search query",
    "n_results": 5,
    "doc_limit": 4
}
```

Parameters:
- `query` (string, required): The search query text
- `n_results` (integer, optional): Number of results to return (default: 5, max: 20)
- `doc_limit` (integer, optional): Maximum number of sentences per result (default: 4, max: 10)

#### Response

```json
{
    "matches": ["matching text segments..."],
    "scores": [0.8, 0.7, ...],
    "documents": ["doc1.pdf", "doc2.pdf", ...]
}
```

Fields:
- `matches`: List of text segments matching the query
- `scores`: Similarity scores for each match (0-1)
- `documents`: Source PDF filenames

## Logging

The application uses loguru for comprehensive logging:

- Logs are written to `app.log`
- Log files rotate at 500MB
- 10-day retention period
- Console output for development

Example log entry:
```
2024-12-17 14:30:00 | INFO | Processing query request: market analysis (n_results=5, doc_limit=4)
```

## Error Handling

The API includes extensive error handling:

- Detailed error messages in responses
- Error logging with stack traces
- Graceful handling of missing files/directories
- Input validation using Pydantic

## Performance Considerations

- PDF processing occurs at startup
- Vector database is persisted to disk
- Queries are processed in-memory for speed
- Document content is limited to prevent oversized responses

## Development

For development mode with auto-reload:
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

## API Documentation

Once running, access the auto-generated API documentation:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Submit a pull request

## Limitations

- Maximum file size depends on available memory
- Processing time increases with document size
- Query response time depends on database size
- Only text content is extracted from PDFs

## License

[Your chosen license]

## Support

For issues and feature requests, please [create an issue](your-issue-tracker-url).
