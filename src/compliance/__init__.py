"""
TRIALPULSE NEXUS 10X - Compliance Module Initialization
"""

from .electronic_signature import (
    ElectronicSignatureService,
    SignatureRecord,
    VerificationResult,
    SignatureMeaning,
    get_signature_service
)

__all__ = [
    'ElectronicSignatureService',
    'SignatureRecord',
    'VerificationResult',
    'SignatureMeaning',
    'get_signature_service'
]
