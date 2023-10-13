# FastAPI Playground

Various tests with Fast API.

To deploy `main:app` in the `app` route using `uvicorn` on port 8192 (as an example), run the following.

```
uvicorn --host 0.0.0.0 --port 8192 app.main:app --reload
```

The Python package `TenSEAL` may need to be compiled from source if the correct package for the target processor architecture (e.g., arm64) is not available. See: https://github.com/OpenMined/TenSEAL/issues/234#issuecomment-1761707506.
