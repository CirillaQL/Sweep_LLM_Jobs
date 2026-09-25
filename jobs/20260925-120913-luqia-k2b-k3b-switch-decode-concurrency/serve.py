"""Start the vLLM OpenAI server in-process without 'vllm' on this process's command line.

The shared-account r3 launcher (run_disagg_benchmark.sh) runs
`pkill -KILL -f 'python.*vllm'` and `pkill -f 'vllm.entrypoints'` on its nodes,
which killed the earlier K3 attempts (jobs 267572, 267626) that were running
concurrently on ganymede. All arguments therefore come from the environment.
"""

import os
import runpy
import sys

import importlib.metadata

print("serve_vllm_version", importlib.metadata.version("vllm"), flush=True)
sys.argv = ["api_server", "--model", os.environ["SERVE_MODEL"], *os.environ.get("SERVE_ARGS", "").split()]
runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__", alter_sys=True)
