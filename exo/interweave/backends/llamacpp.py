"""
llama.cpp Backend for Interweave

Wraps llama.cpp server API for CPU-based inference with quantized models.
Ideal for high-memory systems like Power8 (576GB RAM).

Features:
- GGUF quantized model support (Q4_K_M, Q8_0, etc.)
- HTTP API integration with llama-server
- Efficient for memory-overflow scenarios
"""

import asyncio
import logging
import os
import shutil
import subprocess
from typing import Optional, List, Tuple, Dict, Any
import json

import aiohttp
import numpy as np

from ..backend import InterweaveBackend, BackendRegistry
from ..tensor_format import UniversalTensor, DType
from ..shard import InterweaveShard
from ..state import InterweaveState

logger = logging.getLogger(__name__)


@BackendRegistry.register('llama_cpp')
class LlamaCppBackend(InterweaveBackend):
    """
    llama.cpp backend using the llama-server HTTP API.

    This backend is optimized for:
    - Large memory systems (Power8 with 576GB RAM)
    - Quantized models (4-bit, 8-bit GGUF)
    - CPU inference when GPU VRAM is insufficient

    The backend communicates with a running llama-server instance.
    """

    def __init__(
        self,
        server_url: str = "http://localhost:8080",
        n_threads: Optional[int] = None,
        n_ctx: int = 8192,
        server_path: Optional[str] = None,
    ):
        """
        Initialize llama.cpp backend.

        Args:
            server_url: URL of running llama-server
            n_threads: Number of threads (auto-detect if None)
            n_ctx: Context length
            server_path: Path to llama-server binary (for auto-start)
        """
        self.server_url = server_url.rstrip('/')
        self.n_threads = n_threads or self._detect_threads()
        self.n_ctx = n_ctx
        self.server_path = server_path or self._find_server()

        self._session: Optional[aiohttp.ClientSession] = None
        self._server_process: Optional[subprocess.Popen] = None
        self._loaded_model: Optional[str] = None
        self._shard: Optional[InterweaveShard] = None

    @property
    def name(self) -> str:
        return 'llama_cpp'

    @property
    def device_type(self) -> str:
        return 'cpu'

    @property
    def supported_dtypes(self) -> List[str]:
        # llama.cpp handles quantized formats internally
        return ['f32', 'f16', 'i8', 'i4']

    @property
    def preferred_dtype(self) -> str:
        return 'i4'  # Q4_K_M typical for GGUF

    def _detect_threads(self) -> int:
        """Detect optimal thread count"""
        try:
            import multiprocessing
            return multiprocessing.cpu_count()
        except:
            return 8

    def _find_server(self) -> Optional[str]:
        """Find llama-server binary"""
        # Check PATH
        if shutil.which('llama-server'):
            return 'llama-server'

        # Check common locations
        common_paths = [
            os.path.expanduser('~/llama.cpp/build/bin/llama-server'),
            os.path.expanduser('~/llama.cpp/build-pse-collapse/bin/llama-server'),
            '/usr/local/bin/llama-server',
            '/opt/llama.cpp/llama-server',
        ]
        for path in common_paths:
            if os.path.isfile(path) and os.access(path, os.X_OK):
                return path

        return None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=300)  # 5 min timeout
            )
        return self._session

    async def _check_server(self) -> bool:
        """Check if server is running and healthy"""
        try:
            session = await self._get_session()
            async with session.get(f"{self.server_url}/health") as resp:
                return resp.status == 200
        except:
            return False

    async def load_shard(
        self,
        model_id: str,
        shard: InterweaveShard,
        model_path: Optional[str] = None
    ) -> None:
        """
        Load model for llama.cpp inference.

        Note: llama.cpp loads full models, not shards. The shard info
        is used to determine which layers' outputs to capture.

        For true layer-level sharding, use the Interweave router to
        coordinate multiple llama.cpp instances.
        """
        self._shard = shard

        # Check if server is already running with this model
        if await self._check_server():
            logger.info(f"llama.cpp server already running at {self.server_url}")
            self._loaded_model = model_id
            return

        # Try to start server if we have a model path
        if model_path and self.server_path:
            await self._start_server(model_path)
            self._loaded_model = model_id
        else:
            raise RuntimeError(
                f"llama.cpp server not running at {self.server_url} "
                f"and no model_path provided to start one"
            )

    async def _start_server(self, model_path: str) -> None:
        """Start llama-server subprocess"""
        if not self.server_path:
            raise RuntimeError("llama-server binary not found")

        # Parse URL for host/port
        from urllib.parse import urlparse
        parsed = urlparse(self.server_url)
        host = parsed.hostname or '0.0.0.0'
        port = parsed.port or 8080

        cmd = [
            self.server_path,
            '-m', model_path,
            '--host', host,
            '--port', str(port),
            '-t', str(self.n_threads),
            '-c', str(self.n_ctx),
            '--embedding',  # Enable embeddings endpoint
        ]

        logger.info(f"Starting llama-server: {' '.join(cmd)}")

        self._server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for server to start
        for _ in range(30):  # 30 second timeout
            await asyncio.sleep(1)
            if await self._check_server():
                logger.info("llama-server started successfully")
                return

        raise RuntimeError("llama-server failed to start")

    async def forward(
        self,
        input_tensor: UniversalTensor,
        shard: InterweaveShard,
        state: Optional[InterweaveState] = None
    ) -> Tuple[UniversalTensor, Optional[InterweaveState]]:
        """
        Execute forward pass through llama.cpp.

        For full model inference, this runs the entire forward pass.
        For shard-level inference (Interweave mode), this captures
        intermediate activations.

        Note: llama.cpp doesn't expose layer-level hooks easily.
        For true layer sharding, we use the embeddings endpoint
        and custom prompting to approximate layer outputs.
        """
        session = await self._get_session()

        # Convert input to appropriate format
        input_data = input_tensor.to_numpy()

        # Determine if this is token input or embedding input
        if shard.is_embedding and input_data.dtype in [np.int32, np.int64]:
            # Token IDs - use completions endpoint
            return await self._forward_tokens(session, input_data, shard, state)
        else:
            # Embedding input - use embeddings endpoint
            return await self._forward_embeddings(session, input_data, shard, state)

    async def _forward_tokens(
        self,
        session: aiohttp.ClientSession,
        tokens: np.ndarray,
        shard: InterweaveShard,
        state: Optional[InterweaveState]
    ) -> Tuple[UniversalTensor, Optional[InterweaveState]]:
        """Forward pass with token input"""
        # Convert tokens to prompt (for API)
        # In real implementation, we'd use tokenize/detokenize endpoints
        prompt = " ".join(str(t) for t in tokens.flatten())

        request_data = {
            "prompt": prompt,
            "n_predict": 1,
            "temperature": 0.0,
            "cache_prompt": True,
        }

        async with session.post(
            f"{self.server_url}/completion",
            json=request_data
        ) as resp:
            if resp.status != 200:
                error = await resp.text()
                raise RuntimeError(f"llama.cpp completion failed: {error}")

            result = await resp.json()

        # Extract logits/hidden states if available
        # Standard llama-server returns tokens, not logits
        # For Interweave, we need the /embeddings endpoint

        # Return placeholder for now
        output = np.zeros((1, 1, 4096), dtype=np.float32)  # Hidden dim placeholder
        output_tensor = UniversalTensor.from_numpy(output)

        # Update state
        new_state = state.clone() if state else InterweaveState(request_id="llama_cpp")
        new_state.increment_position()

        return output_tensor, new_state

    async def _forward_embeddings(
        self,
        session: aiohttp.ClientSession,
        embeddings: np.ndarray,
        shard: InterweaveShard,
        state: Optional[InterweaveState]
    ) -> Tuple[UniversalTensor, Optional[InterweaveState]]:
        """
        Forward pass with embedding input.

        This is the key method for Interweave layer sharding.
        We convert embeddings back to a text representation,
        pass through the model, and extract the output.

        Note: This is an approximation. True layer-level sharding
        requires modifications to llama.cpp itself.
        """
        # For now, pass through unchanged
        # Real implementation would involve custom llama.cpp patches
        output_tensor = UniversalTensor.from_numpy(embeddings)

        new_state = state.clone() if state else InterweaveState(request_id="llama_cpp")
        new_state.increment_position()

        return output_tensor, new_state

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        High-level generation interface.

        This is the standard llama.cpp usage pattern.
        """
        session = await self._get_session()

        request_data = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "cache_prompt": True,
        }

        async with session.post(
            f"{self.server_url}/completion",
            json=request_data
        ) as resp:
            if resp.status != 200:
                error = await resp.text()
                raise RuntimeError(f"llama.cpp generation failed: {error}")

            result = await resp.json()

        return result.get('content', '')

    async def get_embeddings(self, text: str) -> np.ndarray:
        """Get embeddings for text"""
        session = await self._get_session()

        request_data = {"content": text}

        async with session.post(
            f"{self.server_url}/embedding",
            json=request_data
        ) as resp:
            if resp.status != 200:
                error = await resp.text()
                raise RuntimeError(f"llama.cpp embeddings failed: {error}")

            result = await resp.json()

        embeddings = np.array(result.get('embedding', []), dtype=np.float32)
        return embeddings

    async def get_memory_available(self) -> int:
        """Get available RAM (for CPU backend)"""
        try:
            import psutil
            mem = psutil.virtual_memory()
            return mem.available
        except ImportError:
            # Fallback: read from /proc/meminfo
            try:
                with open('/proc/meminfo', 'r') as f:
                    for line in f:
                        if 'MemAvailable' in line:
                            # Parse "MemAvailable: 12345678 kB"
                            parts = line.split()
                            return int(parts[1]) * 1024  # KB to bytes
            except:
                pass
        return 0

    async def cleanup(self) -> None:
        """Cleanup resources"""
        if self._session:
            await self._session.close()
            self._session = None

        if self._server_process:
            self._server_process.terminate()
            try:
                self._server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._server_process.kill()
            self._server_process = None


class LlamaCppLayerBridge:
    """
    Bridge for layer-level sharding with llama.cpp.

    This class enables true layer-level sharding by:
    1. Running multiple llama.cpp instances with split models
    2. Coordinating tensor transfer between instances
    3. Managing KV-cache synchronization

    This is an advanced feature that requires custom llama.cpp builds
    with layer extraction support.
    """

    def __init__(self, instances: List[LlamaCppBackend]):
        self.instances = instances

    async def forward_sharded(
        self,
        input_tensor: UniversalTensor,
        shards: List[InterweaveShard],
        state: Optional[InterweaveState] = None
    ) -> Tuple[UniversalTensor, InterweaveState]:
        """
        Forward through multiple sharded llama.cpp instances.

        Each instance handles a subset of layers, passing
        intermediate activations to the next.
        """
        current_tensor = input_tensor
        current_state = state or InterweaveState(request_id="sharded")

        for instance, shard in zip(self.instances, shards):
            current_tensor, current_state = await instance.forward(
                current_tensor, shard, current_state
            )

        return current_tensor, current_state
