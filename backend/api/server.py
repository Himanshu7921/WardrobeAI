from fastapi import FastAPI
app = FastAPI(title='WardrobeAI')

@app.get('/health')
async def health():
    return {'status':'ok'}
