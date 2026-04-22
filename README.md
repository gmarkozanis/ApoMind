# ApoMind (Vercel-ready)

ApoMind is a knowledge-grounded chatbot that answers **only** from `knowledge.docx` using the OpenAI Responses API and File Search.

## What changed for Vercel

This version is ready for deployment on Vercel with:
- static UI served from `public/`
- API served by `server.mjs`
- no dependency on local writable storage
- optional one-time bootstrap endpoint to create the vector store

## Local run

```bash
npm install
cp .env.example .env
npm start
```

Open:

```bash
http://localhost:3000
```

## Required environment variables

- `OPENAI_API_KEY` - your OpenAI API key
- `MODEL` - optional, defaults to `gpt-5.4-mini`
- `VECTOR_STORE_NAME` - optional
- `KNOWLEDGE_FILE` - optional, defaults to `./knowledge.docx`
- `OPENAI_VECTOR_STORE_ID` - strongly recommended on Vercel after first bootstrap

## First bootstrap

If `OPENAI_VECTOR_STORE_ID` is empty, call:

```bash
curl -X POST http://localhost:3000/api/bootstrap-knowledge
```

or after deployment:

```bash
curl -X POST https://YOUR-VERCEL-URL/api/bootstrap-knowledge
```

Response example:

```json
{
  "ok": true,
  "message": "Το knowledge.docx ανέβηκε και έγινε index επιτυχώς. Αποθήκευσε το vector_store_id στο OPENAI_VECTOR_STORE_ID στο Vercel.",
  "vector_store_id": "vs_..."
}
```

Then save that `vector_store_id` in Vercel as `OPENAI_VECTOR_STORE_ID` and redeploy.

## Endpoints

### Health

```bash
GET /api/health
```

### Knowledge status

```bash
GET /api/knowledge/status
```

### Bootstrap knowledge

```bash
POST /api/bootstrap-knowledge
```

### Chat

```bash
POST /api/chat
Content-Type: application/json

{
  "message": "Τι λέει το έγγραφο για ...;"
}
```

## Deploy to Vercel

1. Push the project to GitHub
2. Import the repo into Vercel
3. Add `OPENAI_API_KEY` in Project Settings -> Environment Variables
4. Deploy
5. Call `POST /api/bootstrap-knowledge` once
6. Copy the returned `vector_store_id`
7. Add it as `OPENAI_VECTOR_STORE_ID` in Vercel
8. Redeploy

## Important note

Vercel does not guarantee persistent local file writes across function invocations, so this project does **not** rely on `.knowledge-store.json` anymore. Instead, it uses `OPENAI_VECTOR_STORE_ID` from environment variables, with an in-memory bootstrap fallback for first setup.
