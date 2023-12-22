from typing import Annotated
from fastapi.responses import RedirectResponse
import tenseal as ts
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

description = """
Exciting stuff.

## Operations

You will be able to:

* Get a wise welcome message.
* Try basic computations over random arrays and a matrix using the CKKS fully homomorphic cryptosystem.

[CKKS](https://eprint.iacr.org/2016/421.pdf) is a RingLWE-based cryptosystem that allows for approximate computations over encrypted data. It is a great tool for privacy-preserving machine learning. This API uses the [TenSEAL](https://github.com/OpenMined/TenSEAL) library implementation of the CKKS cryptosystem. The TenSEAL context settings for CKKS in this demo are as follows.
```
{
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[60, 40, 40, 60]
}
```
"""

app = FastAPI(
    title="CKKS vector and matrix operations",
    description=description,
    summary="Test stuff with CKKS vectors and matrices.",
    version="0.0.1",
    contact={
        "name": "Anirban Basu",
    },
)


class TryCKKSResult(BaseModel):
    v1: Annotated[str, Field(description="First random Numpy vector.")] = None
    v2: Annotated[str, Field(description="Second random Numpy vector.")] = None
    matrix: Annotated[str, Field(description="Random Numpy matrix.")] = None
    v1_sum_v2: Annotated[str, Field(
        description="Sum of v1 and v2, computed over the CKKS encrypted domain and then decrypted.")] = None
    v1_dot_v2: Annotated[str, Field(
        description="Dot product of v1 and v2, computed over the CKKS encrypted domain and then decrypted.")] = None
    v1_mult_matrix: Annotated[str, Field(
        description="Product of v1 and the matrix, computed over the CKKS encrypted domain and then decrypted.")] = None


@app.get("/", summary="Redirects to the docs.")
async def root():
    return RedirectResponse(url="/docs")


@app.get("/hello", summary="Get a hello world message!")
async def hello():
    return {"message": "There is no rain that won't stop!"}


@app.get("/try_ckks",
         summary="Performs some basic operations on two random vectors and one random matrix over the CKKS fully homomorphic cryptosystem.",
         description="The operations are: [a] homomorphic addition of two vectors (v1 and v2), [b] homomorphic dot product of two vectors (v1 and v2), and [c] homomorphic multiplication of an encrypted vector (v1) with a plaintext matrix.",
         response_description="The vectors, the matrix, and the results of the operations on them. Notice the approximate results due to errors arising from the RingLWE construction of the CKKS.")
async def try_ckks() -> TryCKKSResult:
    # Setup TenSEAL context
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    context.generate_galois_keys()
    context.global_scale = 2**40

    retval = TryCKKSResult()

    v1 = np.random.random(5)
    v2 = np.random.random(5)
    retval.v1 = map(str, v1)  # np.array2string(v1)
    retval.v2 = map(str, v2)  # np.array2string(v2)

    # encrypted vectors
    enc_v1 = ts.ckks_vector(context, v1)
    enc_v2 = ts.ckks_vector(context, v2)

    result = enc_v1 + enc_v2
    retval.v1_sum_v2 = map(str, result.decrypt())

    result = enc_v1.dot(enc_v2)
    retval.v1_dot_v2 = map(str, result.decrypt())

    matrix = np.random.random((5, 3))
    retval.matrix = map(str, matrix)  # np.array2string(matrix)
    result = enc_v1.matmul(matrix)
    retval.v1_mult_matrix = map(str, result.decrypt())

    return retval
