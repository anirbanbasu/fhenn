# Vectors and matrices in CKKS

Various tests on vectors and matrices with CKKS using Fast API.

To deploy `main:app` in the `app` route using `uvicorn` on port 8192 (as an example), run the following.

```
uvicorn --host 0.0.0.0 --port 8192 app.main:app --reload
```

The Python package `TenSEAL` may need to be compiled from source if the correct package for the target processor architecture (e.g., arm64) is not available by running the following. See: https://github.com/OpenMined/TenSEAL/issues/234#issuecomment-1761707506.

```
git clone --recursive https://github.com/OpenMined/TenSEAL.git
cd TenSEAL/
git submodule init
git submodule update
pip install .
```

Alternatively, executing `pip install git+https://github.com/OpenMined/TenSEAL.git#egg=tenseal` may also work.
