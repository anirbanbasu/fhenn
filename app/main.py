import tenseal as ts
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

description = """
Fast API Playground helps you play with exciting stuff. 🚀

## Operations

You will be able to:

* Get a welcome message.
* Try basic computations over random arrays and a matrix using the CKKS fully homomorphic cryptosystem.
"""

app = FastAPI(
    title="FastAPI Playground",
    description=description,
    summary="Freedom to play.",
    version="0.0.1",
    contact={
        "name": "Anirban Basu",
        "url": "https://www.anirbanbasu.com",
        "email": "0x0@anirbanbasu.com",
    },
)


class TryCKKSResult(BaseModel):
    v1: str = None
    v2: str = None
    matrix: str = None
    v1_sum_v2: str = None
    v1_dot_v2: str = None
    v1_mult_matrix: str = None


@app.get("/", summary="Get a hello world message!")
async def root():
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

    v1 = np.random.randint(0, 10, 5)
    v2 = np.random.randint(0, 10, 5)
    retval.v1 = np.array2string(v1)
    retval.v2 = np.array2string(v2)

    # encrypted vectors
    enc_v1 = ts.ckks_vector(context, v1)
    enc_v2 = ts.ckks_vector(context, v2)

    result = enc_v1 + enc_v2
    retval.v1_sum_v2 = result.decrypt()

    result = enc_v1.dot(enc_v2)
    retval.v1_dot_v2 = result.decrypt()

    matrix = np.random.randint(0, 10, (5, 3))
    retval.matrix = np.array2string(matrix)
    result = enc_v1.matmul(matrix)
    retval.v1_mult_matrix = result.decrypt()

    return retval
