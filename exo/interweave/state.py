"""
Interweave State Management

Handles KV-cache synchronization across heterogeneous backends during
distributed inference.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, Any, TYPE_CHECKING
import struct
import json
import zlib

from .tensor_format import UniversalTensor, DType

if TYPE_CHECKING:
    from .backend import InterweaveBackend


@dataclass
class InterweaveState:
    """
    Synchronized inference state for cross-backend KV-cache management.

    This state travels with tensors as they move between nodes/backends,
    carrying the KV-cache and position information needed for autoregressive
    generation.

    Attributes:
        request_id: Unique identifier for this inference request
        sequence_position: Current position in the generated sequence
        kv_cache: Dict mapping layer index to (K, V) UniversalTensor pairs
        attention_mask: Optional attention mask tensor
        position_ids: Optional position IDs tensor
        metadata: Additional backend-specific metadata (JSON-serializable)
    """

    request_id: str
    sequence_position: int = 0

    # KV-cache: layer_idx -> (key_cache, value_cache)
    kv_cache: Dict[int, Tuple[UniversalTensor, UniversalTensor]] = field(default_factory=dict)

    # Optional tensors
    attention_mask: Optional[UniversalTensor] = None
    position_ids: Optional[UniversalTensor] = None

    # Extensible metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_cache_for_layer(self, layer_idx: int) -> Optional[Tuple[UniversalTensor, UniversalTensor]]:
        """Get KV-cache for a specific layer"""
        return self.kv_cache.get(layer_idx)

    def set_cache_for_layer(
        self,
        layer_idx: int,
        key_cache: UniversalTensor,
        value_cache: UniversalTensor
    ) -> None:
        """Set KV-cache for a specific layer"""
        self.kv_cache[layer_idx] = (key_cache, value_cache)

    def get_cache_for_range(
        self,
        start_layer: int,
        end_layer: int
    ) -> Dict[int, Tuple[UniversalTensor, UniversalTensor]]:
        """Get KV-cache for a range of layers"""
        return {
            layer: cache
            for layer, cache in self.kv_cache.items()
            if start_layer <= layer <= end_layer
        }

    def increment_position(self, count: int = 1) -> None:
        """Advance sequence position"""
        self.sequence_position += count

    @property
    def cache_size_bytes(self) -> int:
        """Total size of KV-cache in bytes"""
        total = 0
        for k, v in self.kv_cache.values():
            total += len(k.data) + len(v.data)
        return total

    @property
    def num_cached_layers(self) -> int:
        """Number of layers with cached KV"""
        return len(self.kv_cache)

    def convert_for_backend(self, backend: 'InterweaveBackend') -> dict:
        """
        Convert state to backend-native format.

        Returns a dict that can be passed directly to the backend's inference
        function as **kwargs or explicit state parameter.
        """
        native_cache = {}

        for layer_idx, (k, v) in self.kv_cache.items():
            # Convert based on backend type
            if backend.name == 'mlx':
                native_cache[layer_idx] = (k.to_mlx(), v.to_mlx())
            elif backend.name.startswith('tinygrad'):
                native_cache[layer_idx] = (k.to_tinygrad(), v.to_tinygrad())
            elif backend.name == 'llama_cpp':
                # llama.cpp handles KV-cache internally
                # We pass position info instead
                pass
            else:
                # Default to numpy
                native_cache[layer_idx] = (k.to_numpy(), v.to_numpy())

        result = {
            'cache': native_cache,
            'position': self.sequence_position,
            'request_id': self.request_id,
        }

        # Add attention mask if present
        if self.attention_mask is not None:
            if backend.name == 'mlx':
                result['attention_mask'] = self.attention_mask.to_mlx()
            elif backend.name.startswith('tinygrad'):
                result['attention_mask'] = self.attention_mask.to_tinygrad()
            else:
                result['attention_mask'] = self.attention_mask.to_numpy()

        # Add position IDs if present
        if self.position_ids is not None:
            if backend.name == 'mlx':
                result['position_ids'] = self.position_ids.to_mlx()
            elif backend.name.startswith('tinygrad'):
                result['position_ids'] = self.position_ids.to_tinygrad()
            else:
                result['position_ids'] = self.position_ids.to_numpy()

        # Add any extra metadata
        result.update(self.metadata)

        return result

    @classmethod
    def from_native(
        cls,
        native_state: dict,
        backend: 'InterweaveBackend',
        request_id: str
    ) -> 'InterweaveState':
        """
        Create InterweaveState from backend-native state dict.

        Used after a backend executes forward() to capture updated state.
        """
        state = cls(request_id=request_id)

        # Extract position
        state.sequence_position = native_state.get('position', 0)

        # Convert cache back to universal format
        native_cache = native_state.get('cache', {})
        for layer_idx, (k, v) in native_cache.items():
            if backend.name == 'mlx':
                k_ut = UniversalTensor.from_mlx(k)
                v_ut = UniversalTensor.from_mlx(v)
            elif backend.name.startswith('tinygrad'):
                k_ut = UniversalTensor.from_tinygrad(k)
                v_ut = UniversalTensor.from_tinygrad(v)
            else:
                import numpy as np
                k_ut = UniversalTensor.from_numpy(np.asarray(k))
                v_ut = UniversalTensor.from_numpy(np.asarray(v))

            state.kv_cache[int(layer_idx)] = (k_ut, v_ut)

        # Handle attention mask
        if 'attention_mask' in native_state:
            mask = native_state['attention_mask']
            if backend.name == 'mlx':
                state.attention_mask = UniversalTensor.from_mlx(mask)
            elif backend.name.startswith('tinygrad'):
                state.attention_mask = UniversalTensor.from_tinygrad(mask)
            else:
                import numpy as np
                state.attention_mask = UniversalTensor.from_numpy(np.asarray(mask))

        return state

    # ==================== SERIALIZATION ====================

    def serialize(self, compress: bool = True) -> bytes:
        """
        Serialize state to bytes for wire transfer.

        Format:
        - 4 bytes: magic (0x49575354 = "IWST")
        - 1 byte: version
        - 1 byte: flags (compressed, has_mask, has_pos_ids)
        - 4 bytes: request_id length
        - N bytes: request_id (utf-8)
        - 4 bytes: sequence_position
        - 4 bytes: num_cache_entries
        - For each cache entry:
            - 4 bytes: layer_idx
            - serialized K tensor
            - serialized V tensor
        - [optional] attention_mask tensor
        - [optional] position_ids tensor
        - 4 bytes: metadata JSON length
        - N bytes: metadata JSON (utf-8)
        """
        parts = []

        # Header
        parts.append(struct.pack('<I', 0x49575354))  # Magic "IWST"
        parts.append(struct.pack('<B', 1))  # Version

        # Flags
        flags = 0
        if compress:
            flags |= 1
        if self.attention_mask is not None:
            flags |= 2
        if self.position_ids is not None:
            flags |= 4
        parts.append(struct.pack('<B', flags))

        # Request ID
        req_id_bytes = self.request_id.encode('utf-8')
        parts.append(struct.pack('<I', len(req_id_bytes)))
        parts.append(req_id_bytes)

        # Sequence position
        parts.append(struct.pack('<i', self.sequence_position))

        # KV-cache
        parts.append(struct.pack('<I', len(self.kv_cache)))
        for layer_idx, (k, v) in sorted(self.kv_cache.items()):
            parts.append(struct.pack('<i', layer_idx))
            k_bytes = k.serialize()
            parts.append(struct.pack('<I', len(k_bytes)))
            parts.append(k_bytes)
            v_bytes = v.serialize()
            parts.append(struct.pack('<I', len(v_bytes)))
            parts.append(v_bytes)

        # Optional tensors
        if self.attention_mask is not None:
            mask_bytes = self.attention_mask.serialize()
            parts.append(struct.pack('<I', len(mask_bytes)))
            parts.append(mask_bytes)

        if self.position_ids is not None:
            pos_bytes = self.position_ids.serialize()
            parts.append(struct.pack('<I', len(pos_bytes)))
            parts.append(pos_bytes)

        # Metadata
        meta_json = json.dumps(self.metadata).encode('utf-8')
        parts.append(struct.pack('<I', len(meta_json)))
        parts.append(meta_json)

        data = b''.join(parts)

        if compress:
            data = zlib.compress(data, level=1)  # Fast compression

        return data

    @classmethod
    def deserialize(cls, data: bytes) -> 'InterweaveState':
        """Deserialize from bytes"""
        # Check if compressed (try decompressing)
        try:
            decompressed = zlib.decompress(data)
            data = decompressed
        except zlib.error:
            pass  # Not compressed

        offset = 0

        # Magic
        magic, = struct.unpack_from('<I', data, offset)
        offset += 4
        if magic != 0x49575354:
            raise ValueError(f"Invalid state magic: {hex(magic)}")

        # Version
        version, = struct.unpack_from('<B', data, offset)
        offset += 1

        # Flags
        flags, = struct.unpack_from('<B', data, offset)
        offset += 1
        has_mask = bool(flags & 2)
        has_pos_ids = bool(flags & 4)

        # Request ID
        req_id_len, = struct.unpack_from('<I', data, offset)
        offset += 4
        request_id = data[offset:offset + req_id_len].decode('utf-8')
        offset += req_id_len

        # Sequence position
        sequence_position, = struct.unpack_from('<i', data, offset)
        offset += 4

        # KV-cache
        num_cache, = struct.unpack_from('<I', data, offset)
        offset += 4

        kv_cache = {}
        for _ in range(num_cache):
            layer_idx, = struct.unpack_from('<i', data, offset)
            offset += 4

            k_len, = struct.unpack_from('<I', data, offset)
            offset += 4
            k = UniversalTensor.deserialize(data[offset:offset + k_len])
            offset += k_len

            v_len, = struct.unpack_from('<I', data, offset)
            offset += 4
            v = UniversalTensor.deserialize(data[offset:offset + v_len])
            offset += v_len

            kv_cache[layer_idx] = (k, v)

        # Optional tensors
        attention_mask = None
        if has_mask:
            mask_len, = struct.unpack_from('<I', data, offset)
            offset += 4
            attention_mask = UniversalTensor.deserialize(data[offset:offset + mask_len])
            offset += mask_len

        position_ids = None
        if has_pos_ids:
            pos_len, = struct.unpack_from('<I', data, offset)
            offset += 4
            position_ids = UniversalTensor.deserialize(data[offset:offset + pos_len])
            offset += pos_len

        # Metadata
        meta_len, = struct.unpack_from('<I', data, offset)
        offset += 4
        metadata = json.loads(data[offset:offset + meta_len].decode('utf-8'))

        return cls(
            request_id=request_id,
            sequence_position=sequence_position,
            kv_cache=kv_cache,
            attention_mask=attention_mask,
            position_ids=position_ids,
            metadata=metadata,
        )

    def clone(self) -> 'InterweaveState':
        """Create a deep copy of this state"""
        return InterweaveState.deserialize(self.serialize(compress=False))

    def __repr__(self) -> str:
        cache_layers = list(self.kv_cache.keys())
        cache_summary = f"layers={cache_layers}" if cache_layers else "empty"
        return (
            f"InterweaveState(request={self.request_id[:8]}..., "
            f"pos={self.sequence_position}, cache={cache_summary}, "
            f"size={self.cache_size_bytes // 1024}KB)"
        )


class StateManager:
    """
    Manages InterweaveState instances across multiple concurrent requests.

    Provides:
    - LRU eviction for memory management
    - Request-scoped state isolation
    - Automatic cleanup of old states
    """

    def __init__(self, max_states: int = 16, max_memory_bytes: int = 8 * 1024**3):
        """
        Args:
            max_states: Maximum number of concurrent states to keep
            max_memory_bytes: Maximum total memory for all states (default 8GB)
        """
        self.max_states = max_states
        self.max_memory_bytes = max_memory_bytes
        self._states: Dict[str, InterweaveState] = {}
        self._access_order: list[str] = []  # Most recent at end

    def get(self, request_id: str) -> Optional[InterweaveState]:
        """Get state for a request, updating access order"""
        state = self._states.get(request_id)
        if state:
            # Move to end (most recent)
            if request_id in self._access_order:
                self._access_order.remove(request_id)
            self._access_order.append(request_id)
        return state

    def set(self, state: InterweaveState) -> None:
        """Store or update state for a request"""
        request_id = state.request_id

        # Add/update state
        self._states[request_id] = state

        # Update access order
        if request_id in self._access_order:
            self._access_order.remove(request_id)
        self._access_order.append(request_id)

        # Evict if needed
        self._evict_if_needed()

    def remove(self, request_id: str) -> Optional[InterweaveState]:
        """Remove and return state for a request"""
        state = self._states.pop(request_id, None)
        if request_id in self._access_order:
            self._access_order.remove(request_id)
        return state

    def _evict_if_needed(self) -> None:
        """Evict oldest states if limits exceeded"""
        # Check count limit
        while len(self._states) > self.max_states:
            oldest = self._access_order.pop(0)
            del self._states[oldest]

        # Check memory limit
        while self._total_memory() > self.max_memory_bytes and self._access_order:
            oldest = self._access_order.pop(0)
            del self._states[oldest]

    def _total_memory(self) -> int:
        """Calculate total memory used by all states"""
        return sum(s.cache_size_bytes for s in self._states.values())

    def clear(self) -> None:
        """Clear all states"""
        self._states.clear()
        self._access_order.clear()

    def __len__(self) -> int:
        return len(self._states)

    def __contains__(self, request_id: str) -> bool:
        return request_id in self._states
