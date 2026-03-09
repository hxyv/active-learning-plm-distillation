"""ESM3 teacher wrapper.

Supports:
- local HuggingFace-backed weights via `ESM3.from_pretrained(...)`
- EvolutionaryScale Forge API via `ESM3ForgeInferenceClient`
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import torch

from teacher.base import TeacherBase

logger = logging.getLogger(__name__)


class ESM3Teacher(TeacherBase):
    def __init__(
        self,
        model_name: str = "esm3_sm_open_v1",
        device: str = "cuda",
        temperature: float = 1.0,
        backend: str = "auto",
        forge_url: str = "https://forge.evolutionaryscale.ai",
        forge_token_env: str = "ESM_API_TOKEN",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.temperature = float(temperature)
        self.backend = backend
        self.forge_url = forge_url
        self.forge_token_env = forge_token_env

        self._model = None
        self._api_kind = None
        self._init_model()

    def _init_model(self) -> None:
        errors = []

        if self.backend not in {"auto", "local", "forge"}:
            raise ValueError(f"Invalid ESM3 backend: {self.backend}")

        if self.backend in {"auto", "local"}:
            try:
                from esm.models.esm3 import ESM3  # type: ignore

                model = ESM3.from_pretrained(self.model_name)
                model = model.to(self.device)
                model.eval()
                self._model = model
                self._api_kind = "esm3_local"
                logger.info("Initialized local ESM3 model: %s", self.model_name)
                return
            except Exception as exc:  # pragma: no cover - environment dependent
                msg = str(exc)
                if "401" in msg or "gated" in msg.lower() or "restricted" in msg.lower():
                    msg += (
                        " | HF auth/access issue. "
                        "Run `huggingface-cli login` and ensure access to the model repo, "
                        "or use backend=forge with ESM_API_TOKEN."
                    )
                errors.append(f"local backend init failed: {msg}")

        if self.backend in {"auto", "forge"}:
            try:
                from esm.sdk import ESM3ForgeInferenceClient  # type: ignore

                token = os.environ.get(self.forge_token_env, "")
                if self.backend == "forge" and not token:
                    raise RuntimeError(
                        f"Forge backend requested but env var {self.forge_token_env} is not set"
                    )
                if not token and self.backend == "auto":
                    raise RuntimeError(
                        f"No {self.forge_token_env} found for forge fallback"
                    )

                forge_model_name = self.model_name.replace("_", "-")
                client = ESM3ForgeInferenceClient(
                    model=forge_model_name,
                    url=self.forge_url,
                    token=token,
                )
                self._model = client
                self._api_kind = "esm3_forge"
                logger.info(
                    "Initialized ESM3 Forge client: model=%s url=%s token_env=%s",
                    forge_model_name,
                    self.forge_url,
                    self.forge_token_env,
                )
                return
            except Exception as exc:  # pragma: no cover - environment dependent
                errors.append(f"forge backend init failed: {exc}")

        raise RuntimeError(
            "Unable to initialize ESM3 teacher. "
            "Use backend=local with HF-authenticated access OR backend=forge with ESM_API_TOKEN. "
            + " | ".join(errors)
        )

    def _extract_secondary_structure_logits(self, output) -> torch.Tensor:
        """Best-effort extraction of [L, 8] logits from SDK output object."""
        candidate_paths = [
            ("secondary_structure_logits",),
            ("ss8_logits",),
            ("logits", "secondary_structure"),
            ("logits", "ss8"),
            ("outputs", "secondary_structure"),
            ("secondary_structure",),
        ]

        for path in candidate_paths:
            obj = output
            ok = True
            for key in path:
                if isinstance(obj, dict):
                    if key not in obj:
                        ok = False
                        break
                    obj = obj[key]
                else:
                    if not hasattr(obj, key):
                        ok = False
                        break
                    obj = getattr(obj, key)
            if ok:
                tensor = obj
                if isinstance(tensor, np.ndarray):
                    tensor = torch.from_numpy(tensor)
                if isinstance(tensor, list):
                    tensor = torch.tensor(tensor)
                if torch.is_tensor(tensor):
                    return tensor

        raise RuntimeError(
            "Could not find secondary-structure logits in ESM3 output. "
            "Update teacher/esm3_teacher.py for your SDK response shape."
        )

    def _call_logits(self, sequence: str, backbone_coords_ang: Optional[np.ndarray] = None):
        """Call ESM3 to get secondary-structure logits.

        Args:
            sequence: amino acid sequence string.
            backbone_coords_ang: optional (L, 3, 3) float32 array of N/CA/C coordinates
                in Angstroms. When provided, ESM3 conditions its SS8 predictions on the
                structure, making teacher labels consistent with mdtraj DSSP from the
                same coordinates (paper-faithful setup).
        """
        from esm.sdk.api import ESMProtein, ESMProteinError, LogitsConfig  # type: ignore

        if backbone_coords_ang is not None:
            try:
                from esm.utils.structure.protein_chain import ProteinChain  # type: ignore
                chain = ProteinChain.from_backbone_atom_coordinates(
                    backbone_coords_ang, sequence=sequence
                )
                protein = ESMProtein.from_protein_chain(chain)
            except Exception as exc:
                logger.warning(
                    "Could not build structure-conditioned ESMProtein (%s); "
                    "falling back to sequence-only.", exc
                )
                protein = ESMProtein(sequence=sequence)
        else:
            protein = ESMProtein(sequence=sequence)

        encoded = self._model.encode(protein)
        if isinstance(encoded, ESMProteinError):
            raise RuntimeError(f"ESM3 encode failed: {encoded.error_code} {encoded.error_msg}")

        if self._api_kind == "esm3_local" and hasattr(encoded, "to"):
            encoded = encoded.to(self.device)

        logits_cfg = LogitsConfig(sequence=False, structure=False, secondary_structure=True)
        if self._api_kind == "esm3_forge":
            try:
                output = self._model.logits(encoded, logits_cfg, return_bytes=False)
            except TypeError:
                output = self._model.logits(encoded, logits_cfg)
        else:
            output = self._model.logits(encoded, logits_cfg)
        if isinstance(output, ESMProteinError):
            raise RuntimeError(f"ESM3 logits failed: {output.error_code} {output.error_msg}")
        return output

    def _align_logits_length(self, logits: torch.Tensor, sequence_length: int) -> torch.Tensor:
        n = int(logits.shape[0])
        L = int(sequence_length)
        if n == L:
            return logits
        if n == L + 2:
            return logits[1:-1]
        if n == L + 1:
            return logits[1:]
        if n > L:
            return logits[:L]
        raise RuntimeError(f"Logit length {n} is shorter than sequence length {L}")

    def _project_to_ss8_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Project model secondary-structure logits to pure SS8 class logits.

        In esm==3.2.x, SS8 track vocab is 11 tokens:
        [<pad>, <motif>, <unk>, G, H, I, T, E, B, S, C]
        We drop special tokens and keep [G,H,I,T,E,B,S,C] in that order.
        """
        if logits.shape[-1] == 8:
            return logits

        if logits.shape[-1] == 11:
            try:
                from esm.tokenization.ss_tokenizer import SecondaryStructureTokenizer  # type: ignore

                tok = SecondaryStructureTokenizer(kind="ss8")
                keep_tokens = list("GHITEBSC")
                keep_idx = [tok.vocab_to_index[t] for t in keep_tokens]
                return logits[..., keep_idx]
            except Exception:
                # Safe fallback for known esm3 tokenizer layout.
                return logits[..., 3:11]

        raise RuntimeError(
            f"Unsupported secondary-structure class dimension {logits.shape[-1]} "
            "(expected 8 or 11)."
        )

    def predict_ss8_probs(
        self,
        sequence: str,
        sample_id: Optional[str] = None,
        backbone_coords_ang: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if self._api_kind is None:
            raise RuntimeError("No active ESM3 backend")

        with torch.no_grad():
            output = self._call_logits(sequence, backbone_coords_ang=backbone_coords_ang)
            logits = self._extract_secondary_structure_logits(output)

            if logits.ndim == 3:
                logits = logits[0]
            logits = self._align_logits_length(logits, len(sequence))
            logits = self._project_to_ss8_logits(logits)

            logits = logits.float() / max(self.temperature, 1e-6)
            probs = torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float32)

        if probs.shape[0] != len(sequence):
            raise RuntimeError(
                f"ESM3 output length mismatch for {sample_id}: sequence={len(sequence)}, probs={probs.shape}"
            )

        return probs
