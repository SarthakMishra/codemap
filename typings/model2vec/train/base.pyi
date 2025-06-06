"""
This type stub file was generated by pyright.
"""

import torch
from typing import Any, TypeVar
from tokenizers import Tokenizer
from torch import nn
from torch.utils.data import DataLoader, Dataset
from model2vec import StaticModel

class FinetunableStaticModel(nn.Module):
    def __init__(self, *, vectors: torch.Tensor, tokenizer: Tokenizer, out_dim: int = ..., pad_id: int = ...) -> None:
        """
        Initialize a trainable StaticModel from a StaticModel.

        :param vectors: The embeddings of the staticmodel.
        :param tokenizer: The tokenizer.
        :param out_dim: The output dimension of the head.
        :param pad_id: The padding id. This is set to 0 in almost all model2vec models
        """
        ...
    
    def construct_weights(self) -> nn.Parameter:
        """Construct the weights for the model."""
        ...
    
    def construct_head(self) -> nn.Sequential:
        """Method should be overridden for various other classes."""
        ...
    
    @classmethod
    def from_pretrained(cls: type[ModelType], *, out_dim: int = ..., model_name: str = ..., **kwargs: Any) -> ModelType:
        """Load the model from a pretrained model2vec model."""
        ...
    
    @classmethod
    def from_static_model(cls: type[ModelType], *, model: StaticModel, out_dim: int = ..., **kwargs: Any) -> ModelType:
        """Load the model from a static model."""
        ...
    
    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the mean, and a classifier layer after."""
        ...
    
    def tokenize(self, texts: list[str], max_length: int | None = ...) -> torch.Tensor:
        """
        Tokenize a bunch of strings into a single padded 2D tensor.

        Note that this is not used during training.

        :param texts: The texts to tokenize.
        :param max_length: If this is None, the sequence lengths are truncated to 512.
        :return: A 2D padded tensor
        """
        ...
    
    @property
    def device(self) -> str:
        """Get the device of the model."""
        ...
    
    def to_static_model(self) -> StaticModel:
        """Convert the model to a static model."""
        ...
    


class TextDataset(Dataset):
    def __init__(self, tokenized_texts: list[list[int]], targets: torch.Tensor) -> None:
        """
        A dataset of texts.

        :param tokenized_texts: The tokenized texts. Each text is a list of token ids.
        :param targets: The targets.
        :raises ValueError: If the number of labels does not match the number of texts.
        """
        ...
    
    def __len__(self) -> int:
        """Return the length of the dataset."""
        ...
    
    def __getitem__(self, index: int) -> tuple[list[int], torch.Tensor]:
        """Gets an item."""
        ...
    
    @staticmethod
    def collate_fn(batch: list[tuple[list[list[int]], int]]) -> tuple[torch.Tensor, torch.Tensor]:
        """Collate function."""
        ...
    
    def to_dataloader(self, shuffle: bool, batch_size: int = ...) -> DataLoader:
        """Convert the dataset to a DataLoader."""
        ...
    


ModelType = TypeVar("ModelType", bound=FinetunableStaticModel)
