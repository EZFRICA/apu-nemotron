"""Cloud registry settings (publishing side).

The embedding model is deliberately NOT declared here. The pipeline reads
apu.config (LOCAL_EMBEDDING_MODEL / LOCAL_EMBEDDING_DIM), the same module the student
device reads. In Akili this file once held its own remote-embedder literal, and the
registry was published at 3072 dimensions while devices queried at 384, with nothing
detecting it. Never redeclare the model here.
"""

import os

from dotenv import load_dotenv

load_dotenv()

# Bucket the registry is published to by `batch_pipeline.py --upload`. Devices find it
# through apu.config.REGISTRY_MANIFEST_URL instead.
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")

# Service-account JSON. Unset: Application Default Credentials.
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
