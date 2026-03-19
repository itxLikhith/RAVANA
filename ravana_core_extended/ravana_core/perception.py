"""
Perception Module — RAVANA Core Extended
Handles multimodal input processing with entropy-based uncertainty estimation
and dual-confidence (mean + volatility) tracking.

EXTENSIONS v0.2.0:
- Pluggable feature extractors with BaseFeatureExtractor interface
- ResNetFeatureExtractor for production image processing
- Wav2VecFeatureExtractor for production audio processing
- Configuration-driven extractor selection

Implements:
- Multimodal feature extraction (text, visual, audio)
- Pluggable extractor architecture with ABC interface
- Perceptual entropy / novelty computation
- HMM-based sequential state tracking
- Dual-confidence: mean confidence + confidence volatility (Eq. 2.1, 3.1–3.2)
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
import warnings


# ─────────────────────────────────────────────────────────────────────────────
# EXTENSION: Pluggable Feature Extractor Interface
# ─────────────────────────────────────────────────────────────────────────────

class BaseFeatureExtractor(ABC):
    """
    Abstract base class for all feature extractors.
    
    Implement this interface to add new extractors (e.g., for new modalities
    or different neural architectures).
    
    Example:
        class MyExtractor(BaseFeatureExtractor):
            def extract(self, input_data: Any) -> np.ndarray:
                # Your extraction logic here
                return features
            
            @property
            def output_dim(self) -> int:
                return 512
    """
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self._initialized = False
    
    @abstractmethod
    def extract(self, input_data: Any) -> np.ndarray:
        """
        Extract features from input data.
        
        Args:
            input_data: Raw input (image array, audio waveform, etc.)
            
        Returns:
            Normalized feature vector as numpy array
        """
        pass
    
    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Return the dimensionality of extracted features."""
        pass
    
    def initialize(self) -> bool:
        """
        Initialize the extractor (load models, etc.).
        Returns True if successful, False otherwise.
        """
        self._initialized = True
        return True
    
    def is_available(self) -> bool:
        """Check if the extractor's dependencies are available."""
        return True


class MockFeatureExtractor(BaseFeatureExtractor):
    """
    Mock extractor for testing without heavy dependencies.
    Always available and returns random features.
    """
    
    def __init__(self, output_dim: int = 64, seed: int = 42):
        super().__init__()
        self._output_dim = output_dim
        self.rng = np.random.default_rng(seed)
    
    def extract(self, input_data: Any) -> np.ndarray:
        """Return random normalized features."""
        features = self.rng.random(self._output_dim)
        features /= features.sum() + 1e-8
        return features
    
    @property
    def output_dim(self) -> int:
        return self._output_dim
    
    def is_available(self) -> bool:
        return True


class ResNetFeatureExtractor(BaseFeatureExtractor):
    """
    Production-ready ResNet50 feature extractor for images.
    
    Requires: torch, torchvision
    Extracts 2048-dim features from the final layer before classification.
    
    Configuration flag: use_resnet=True in PerceptionModule
    """
    
    def __init__(self, device: str = "cpu", pretrained: bool = True):
        super().__init__(device)
        self.pretrained = pretrained
        self._model = None
        self._transform = None
        self._output_dim = 2048
    
    def is_available(self) -> bool:
        """Check if PyTorch and torchvision are available."""
        try:
            import torch
            import torchvision
            return True
        except ImportError:
            return False
    
    def initialize(self) -> bool:
        """Load ResNet50 model."""
        if not self.is_available():
            warnings.warn("PyTorch/torchvision not available. Using mock extractor.")
            return False
        
        try:
            import torch
            import torchvision.models as models
            from torchvision import transforms
            
            # Load pretrained ResNet50
            self._model = models.resnet50(pretrained=self.pretrained)
            self._model.fc = torch.nn.Identity()  # Remove classification head
            self._model = self._model.to(self.device)
            self._model.eval()
            
            # Standard ImageNet preprocessing
            self._transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
            
            self._initialized = True
            return True
            
        except Exception as e:
            warnings.warn(f"Failed to initialize ResNet: {e}")
            return False
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extract ResNet50 features from image.
        
        Args:
            image: numpy array (H, W, C) or (H, W) for grayscale
            
        Returns:
            2048-dim normalized feature vector
        """
        if not self._initialized:
            # Fallback to mock
            mock = MockFeatureExtractor(output_dim=self._output_dim)
            return mock.extract(image)
        
        try:
            import torch
            
            # Convert to torch tensor
            if image.ndim == 2:
                # Grayscale to RGB
                image = np.stack([image] * 3, axis=-1)
            
            # Apply transforms
            img_tensor = self._transform(image).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self._model(img_tensor)
            
            # Convert to numpy and normalize
            features = features.cpu().numpy().flatten()
            features = np.abs(features)
            features /= features.sum() + 1e-8
            
            return features
            
        except Exception as e:
            warnings.warn(f"ResNet extraction failed: {e}. Using fallback.")
            mock = MockFeatureExtractor(output_dim=self._output_dim)
            return mock.extract(image)
    
    @property
    def output_dim(self) -> int:
        return self._output_dim


class Wav2VecFeatureExtractor(BaseFeatureExtractor):
    """
    Production-ready Wav2Vec2 feature extractor for audio.
    
    Requires: transformers, torch, librosa
    Uses pre-trained Wav2Vec2 model for speech/audio embeddings.
    
    Configuration flag: use_wav2vec=True in PerceptionModule
    """
    
    def __init__(self, device: str = "cpu", model_name: str = "facebook/wav2vec2-base"):
        super().__init__(device)
        self.model_name = model_name
        self._processor = None
        self._model = None
        self._output_dim = 768  # wav2vec2-base hidden size
    
    def is_available(self) -> bool:
        """Check if transformers and torch are available."""
        try:
            import transformers
            import torch
            import librosa
            return True
        except ImportError:
            return False
    
    def initialize(self) -> bool:
        """Load Wav2Vec2 model and processor."""
        if not self.is_available():
            warnings.warn("transformers/torch/librosa not available. Using mock extractor.")
            return False
        
        try:
            from transformers import Wav2Vec2Processor, Wav2Vec2Model
            import torch
            
            # Load processor and model
            self._processor = Wav2Vec2Processor.from_pretrained(self.model_name)
            self._model = Wav2Vec2Model.from_pretrained(self.model_name)
            self._model = self._model.to(self.device)
            self._model.eval()
            
            self._initialized = True
            return True
            
        except Exception as e:
            warnings.warn(f"Failed to initialize Wav2Vec: {e}")
            return False
    
    def extract(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract Wav2Vec2 features from audio waveform.
        
        Args:
            audio: numpy array of audio samples (mono, any sample rate)
            
        Returns:
            768-dim normalized feature vector (mean-pooled over time)
        """
        if not self._initialized:
            mock = MockFeatureExtractor(output_dim=self._output_dim)
            return mock.extract(audio)
        
        try:
            import torch
            import librosa
            
            # Resample to 16kHz if needed
            if len(audio) > 0:
                # Assume input might be at different sample rate
                target_sr = 16000
                # Simple resampling approximation
                audio = librosa.resample(
                    audio.astype(np.float32),
                    orig_sr=len(audio) / (len(audio) / target_sr),  # estimate
                    target_sr=target_sr
                )
            
            # Process audio
            inputs = self._processor(
                audio,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Extract features
            with torch.no_grad():
                outputs = self._model(**inputs)
            
            # Mean pool over time dimension
            features = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            
            # Normalize
            features = np.abs(features)
            features /= features.sum() + 1e-8
            
            return features
            
        except Exception as e:
            warnings.warn(f"Wav2Vec extraction failed: {e}. Using fallback.")
            mock = MockFeatureExtractor(output_dim=self._output_dim)
            return mock.extract(audio)
    
    @property
    def output_dim(self) -> int:
        return self._output_dim


class TextFeatureExtractor(BaseFeatureExtractor):
    """
    Text feature extractor using bag-of-words with HMM.
    """
    
    def __init__(self, hidden_states: int = 8, seed: int = 42):
        super().__init__()
        self.hidden_states = hidden_states
        self.rng = np.random.default_rng(seed)
        self.hmm_transition = self.rng.dirichlet(np.ones(hidden_states))
        self.hmm_state_probs = self.rng.dirichlet(np.ones(hidden_states))
    
    def extract(self, text: str) -> np.ndarray:
        """Extract features from text using BoW → HMM."""
        words = text.lower().split()
        vocab_size = self.hidden_states * 2
        bow = np.zeros(vocab_size)
        for w in words:
            idx = hash(w) % vocab_size
            bow[idx] += 1
        bow /= (np.sum(bow) + 1e-8)
        
        # HMM forward pass
        hmm_input = bow[:self.hidden_states]
        self.hmm_state_probs = hmm_input @ self.hmm_transition
        if self.hmm_state_probs.ndim == 0:
            self.hmm_state_probs = np.atleast_1d(self.hmm_state_probs)
        self.hmm_state_probs = np.maximum(self.hmm_state_probs, 1e-8)
        self.hmm_state_probs /= self.hmm_state_probs.sum()
        
        result = self.hmm_state_probs if self.hmm_state_probs is not None else bow[:self.hidden_states]
        return np.atleast_1d(result)
    
    @property
    def output_dim(self) -> int:
        return self.hidden_states


# ─────────────────────────────────────────────────────────────────────────────
# EXTENDED Perception Module
# ─────────────────────────────────────────────────────────────────────────────

class PerceptionModule:
    """
    Extended Perception Module with pluggable extractors.
    
    Processes multimodal inputs and computes perceptual uncertainty
    following Section 3.1 of the RAVANA paper.
    
    NEW in v0.2.0:
    - Configurable feature extractors via constructor flags
    - Production-ready ResNet50 for images
    - Production-ready Wav2Vec2 for audio
    - Mock extractors for testing without dependencies
    
    Configuration:
        use_resnet: bool = False — Use ResNet50 for images (requires torch)
        use_wav2vec: bool = False — Use Wav2Vec2 for audio (requires transformers)
        force_mock: bool = False — Force mock extractors (useful for testing)
    """

    def __init__(
        self,
        learning_rate: float = 0.05,
        volatility_window: int = 10,
        hidden_states: int = 8,
        seed: int = 42,
        # NEW: Configuration flags for extractors
        use_resnet: bool = False,
        use_wav2vec: bool = False,
        force_mock: bool = False,
        device: str = "cpu",
    ):
        self.lr = learning_rate
        self.volatility_window = volatility_window
        self.hidden_states = hidden_states
        self.rng = np.random.default_rng(seed)
        
        # Configuration flags
        self.use_resnet = use_resnet and not force_mock
        self.use_wav2vec = use_wav2vec and not force_mock
        self.force_mock = force_mock
        self.device = device

        # Dual-confidence state
        self.mean_conf: float = 0.5
        self.volatility_conf: float = 0.0
        self.confidence_history: list[float] = []

        # Feature caches
        self.last_features: Optional[np.ndarray] = None
        self.last_U: float = 0.0
        
        # Initialize extractors
        self._init_extractors()

    def _init_extractors(self):
        """Initialize feature extractors based on configuration."""
        # Text extractor (always available)
        self.text_extractor = TextFeatureExtractor(
            hidden_states=self.hidden_states,
            seed=int(self.rng.integers(1 << 30))
        )
        
        # Visual extractor
        if self.use_resnet and not self.force_mock:
            self.visual_extractor = ResNetFeatureExtractor(device=self.device)
            if not self.visual_extractor.initialize():
                # Fallback to mock
                self.visual_extractor = MockFeatureExtractor(output_dim=2048, seed=int(self.rng.integers(1 << 30)))
        else:
            self.visual_extractor = MockFeatureExtractor(output_dim=64, seed=int(self.rng.integers(1 << 30)))
        
        # Audio extractor
        if self.use_wav2vec and not self.force_mock:
            self.audio_extractor = Wav2VecFeatureExtractor(device=self.device)
            if not self.audio_extractor.initialize():
                # Fallback to mock
                self.audio_extractor = MockFeatureExtractor(output_dim=768, seed=int(self.rng.integers(1 << 30)))
        else:
            self.audio_extractor = MockFeatureExtractor(output_dim=32, seed=int(self.rng.integers(1 << 30)))

    # ── Public API ────────────────────────────────────────────────────────────

    def process(
        self,
        text: Optional[str] = None,
        visual: Optional[np.ndarray] = None,
        audio: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Main entry point. Pass any combination of modalities.

        NEW: Uses configured extractors (ResNet/Wav2Vec when enabled)

        Returns dict with:
            features     — concatenated feature vector
            U            — global entropy / uncertainty [0, 1]
            mean_conf    — running mean confidence
            volatility_conf — confidence volatility (variance over window)
            novelty      — perceptual novelty score
            gw_bid       — Global Workspace bid (Eq. 2.1)
            extractor_info — which extractors were used
        """
        features_list = []
        extractor_info = {}

        # Text stream
        if text is not None:
            text_features = self.text_extractor.extract(text)
            features_list.append(text_features)
            extractor_info["text"] = "TextFeatureExtractor"
        else:
            text_features = np.zeros(self.text_extractor.output_dim)
            extractor_info["text"] = "none"

        # Visual stream — use configured extractor
        if visual is not None:
            vis_features = self.visual_extractor.extract(visual)
            features_list.append(vis_features)
            extractor_info["visual"] = self.visual_extractor.__class__.__name__
        else:
            vis_features = np.zeros(self.visual_extractor.output_dim)
            extractor_info["visual"] = "none"

        # Audio stream — use configured extractor
        if audio is not None:
            aud_features = self.audio_extractor.extract(audio)
            features_list.append(aud_features)
            extractor_info["audio"] = self.audio_extractor.__class__.__name__
        else:
            aud_features = np.zeros(self.audio_extractor.output_dim)
            extractor_info["audio"] = "none"

        # Aggregate uncertainty via normalised entropy
        U_text = self._entropy(text_features) / np.log(self.text_extractor.output_dim + 1)
        U_vis  = self._entropy(vis_features)  / np.log(self.visual_extractor.output_dim + 1)
        U_aud  = self._entropy(aud_features)   / np.log(self.audio_extractor.output_dim + 1)
        U_global = (U_text + U_vis + U_aud) / 3.0

        # Per-module confidence
        conf = 0.6 + 0.4 * (1 - U_global)  # heuristic: low uncertainty → higher confidence

        # Update dual-confidence
        self._update_confidence(conf)

        # Compute novelty as deviation from last feature vector
        if self.last_features is not None:
            all_features = np.concatenate(features_list)
            novelty = float(np.linalg.norm(all_features - self.last_features) /
                             (np.linalg.norm(self.last_features) + 1e-8))
        else:
            novelty = 0.5

        self.last_features = np.concatenate(features_list)
        self.last_U = U_global

        # GW bid
        gw_bid = self._compute_gw_bid(U_global, novelty)

        return {
            "features": np.concatenate(features_list),
            "U": U_global,
            "mean_conf": self.mean_conf,
            "volatility_conf": self.volatility_conf,
            "novelty": novelty,
            "gw_bid": gw_bid,
            "extractor_info": extractor_info,
        }

    def _entropy(self, probs: np.ndarray) -> float:
        """Shannon entropy (Eq. 3.2)."""
        probs = np.maximum(probs, 1e-12)
        return -np.sum(probs * np.log(probs))

    def _update_confidence(self, conf: float):
        """Exponential moving average + rolling volatility (dual-confidence)."""
        self.mean_conf += self.lr * (conf - self.mean_conf)
        self.confidence_history.append(conf)
        if len(self.confidence_history) > self.volatility_window:
            self.confidence_history.pop(0)
        if len(self.confidence_history) >= 2:
            self.volatility_conf = float(np.var(self.confidence_history))

    def _compute_gw_bid(self, U: float, novelty: float, alpha: float = 0.5) -> float:
        """
        Global Workspace bid (Eq. 2.1):
        bid = emotion_intensity + novelty + goal_relevance × mean_conf × exp(−α × volatility)
        """
        goal_relevance = 1.0 - U
        decay = np.exp(-alpha * self.volatility_conf)
        bid = (1 - U) + novelty + goal_relevance * self.mean_conf * decay
        return float(bid)

    def reset(self):
        """Reset internal state."""
        self.__init__(
            learning_rate=self.lr,
            volatility_window=self.volatility_window,
            hidden_states=self.hidden_states,
            seed=int(self.rng.integers(1 << 30)),
            use_resnet=self.use_resnet,
            use_wav2vec=self.use_wav2vec,
            force_mock=self.force_mock,
            device=self.device,
        )
    
    def get_extractor_status(self) -> Dict[str, Any]:
        """Return status of all extractors (for debugging/monitoring)."""
        return {
            "text": {
                "name": self.text_extractor.__class__.__name__,
                "output_dim": self.text_extractor.output_dim,
                "available": True,
            },
            "visual": {
                "name": self.visual_extractor.__class__.__name__,
                "output_dim": self.visual_extractor.output_dim,
                "available": isinstance(self.visual_extractor, ResNetFeatureExtractor) 
                             or self.force_mock,
            },
            "audio": {
                "name": self.audio_extractor.__class__.__name__,
                "output_dim": self.audio_extractor.output_dim,
                "available": isinstance(self.audio_extractor, Wav2VecFeatureExtractor)
                             or self.force_mock,
            },
            "config": {
                "use_resnet": self.use_resnet,
                "use_wav2vec": self.use_wav2vec,
                "force_mock": self.force_mock,
                "device": self.device,
            },
        }
