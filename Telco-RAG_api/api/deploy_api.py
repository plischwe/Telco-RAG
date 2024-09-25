from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from api.pipeline import TelcoRAG 

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

@app.get('/')
def hello_world():
    return{'hello':'world'}

@app.get('/favicon.ico')
async def favicon():
    file_name = "favicon.ico"
    file_path = os.path.join(app.root_path, "static", file_name)
    return FileResponse(path=file_path, headers={"Content-Disposition": "attachment; filename=" + file_name})

# Setup CORS policy for the application
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryData(BaseModel):
    query: str
    model_name: str

@app.post("/process_query/")
async def process_query(data: QueryData):
    """Processes incoming queries using the TelcoRAG."""
    try:
        os.environ["KMP_DUPLICATE_LIB_OK"] = 'TRUE'
        
        print("Here is data server recieved: ", data)
        # Generate response using the TelcoRAG model
        response, retrieval, query = await TelcoRAG(query= data.query, model_name= data.model_name)

        return json.dumps({"result": response, "retrieval": retrieval, "query": query})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
