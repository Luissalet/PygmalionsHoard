"""Scripts run by the trainer environment's Python (``python workers/<name>.py --args <job_dir>/args.json``).

They never import the app package. Heavy libraries (torch, transformers, peft) are imported inside functions, so every module here
can be imported by the tests, which exercise their pure helpers without a GPU.
"""
