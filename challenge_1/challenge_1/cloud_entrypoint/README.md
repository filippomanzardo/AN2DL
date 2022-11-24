# Cloud Training Entrypoint
This folder contains the code to run the training job on Cloud ML Engine.
It starts from templates:
- template.py
- hypertune.py

And formats them in the desired way. Then, it prepares the `entrypoint.py` that will be attached to the remote build and executed as a job in Google AI Platform. 
