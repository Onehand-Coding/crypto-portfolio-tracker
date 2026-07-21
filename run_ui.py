"""Start the React UI. Build the frontend first with: npm --prefix frontend run build"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=False)
