"""
Interweave Protocol Buffers

Auto-generated from interweave.proto
"""

from .interweave_pb2 import (
    UniversalTensorProto,
    InterweaveShardProto,
    KVCacheEntry,
    InterweaveStateProto,
    BackendCapabilities,
    NodeCapabilities,
    ForwardRequest,
    ForwardResponse,
    StreamForwardResponse,
    RouteQueryRequest,
    RouteQueryResponse,
    RouteCandidate,
    RegisterNodeRequest,
    RegisterNodeResponse,
    InterweaveHealthRequest,
    InterweaveHealthResponse,
    GetTopologyRequest,
    GetTopologyResponse,
    NodePeers,
    NetworkMetric,
    GenerateRequest,
    GenerateResponse,
)

from .interweave_pb2_grpc import (
    InterweaveServiceStub,
    InterweaveServiceServicer,
    add_InterweaveServiceServicer_to_server,
)

__all__ = [
    # Proto messages
    'UniversalTensorProto',
    'InterweaveShardProto',
    'KVCacheEntry',
    'InterweaveStateProto',
    'BackendCapabilities',
    'NodeCapabilities',
    'ForwardRequest',
    'ForwardResponse',
    'StreamForwardResponse',
    'RouteQueryRequest',
    'RouteQueryResponse',
    'RouteCandidate',
    'RegisterNodeRequest',
    'RegisterNodeResponse',
    'InterweaveHealthRequest',
    'InterweaveHealthResponse',
    'GetTopologyRequest',
    'GetTopologyResponse',
    'NodePeers',
    'NetworkMetric',
    'GenerateRequest',
    'GenerateResponse',
    # gRPC
    'InterweaveServiceStub',
    'InterweaveServiceServicer',
    'add_InterweaveServiceServicer_to_server',
]
