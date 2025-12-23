import torch
import torch.nn as nn
import logging

from tevatron.retriever.arguments import ModelArguments
from transformers import AutoModel
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

from .dense import DenseModel
from .encoder import EncoderOutput


logger = logging.getLogger(__name__)


class PriorMLP(nn.Module):
    """
    MLP that maps document embeddings to prior values.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256, n_layers: int = 2, 
                 use_tanh: bool = False,):
        super().__init__()
        layers = []
        for i in range(n_layers - 1):
            layers.append(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim if n_layers > 1 else input_dim, 1))
        self.network = nn.Sequential(*layers)
        self.use_tanh = use_tanh
    
    def forward(self, embeddings):
        """
        Args:
            embeddings: Document embeddings of shape (batch_size, embedding_dim)
        
        Returns:
            Prior values of shape (batch_size,)
        """
        priors = self.network(embeddings).squeeze(-1)
        if self.use_tanh:
            priors = torch.tanh(priors)
            # changed_indices = (priors_ != priors).nonzero(as_tuple=True)[0]
        return priors


class DenseModelWithPriors(DenseModel):
    """
    Dense biencoder model with learned document-level priors.
    
    The model learns to predict document priors using an MLP that maps
    document embeddings to scalar values. The final score for a (query, document)
    pair is computed as:
        score(q, d) = similarity(q, d) + prior(d)
    
    where similarity is the standard dot product and prior(d) is the learned
    document prior.
    """
    
    def __init__(
        self,
        encoder,
        pooling: str = 'cls',
        normalize: bool = False,
        temperature: float = 1.0,
        prior_hidden_dim: int = 256,
        prior_n_layers: int = 2,
        prior_use_tanh: bool = False,
        prior_reg_weight: float = 0.01,
    ):
        super().__init__(encoder, pooling, normalize, temperature)
        
        # Initialize the prior MLP
        embedding_dim = self.config.hidden_size
        self.prior_head = PriorMLP(embedding_dim, prior_hidden_dim, prior_n_layers, prior_use_tanh)
        
        # Regularization weight for prior values
        self.prior_reg_weight = prior_reg_weight
    
    def compute_similarity(self, q_reps, p_reps):
        return torch.matmul(q_reps, p_reps.transpose(0, 1))
    
    def compute_loss(self, scores, target, doc_priors):
        """
        Compute loss with regularization on prior values.
        """
        ce_loss = self.cross_entropy(scores, target)
        if self.prior_reg_weight == 0.0:
            return ce_loss
        prior_reg_loss = torch.mean(doc_priors ** 2)
        total_loss = ce_loss + self.prior_reg_weight * prior_reg_loss
        
        # # L2 Regularization on priors to prevent them from getting too large
        # # NOTE In the forward pass, we compute priors on p_reps. Here we need to access the priors that were just computed
        # if hasattr(self, '_last_doc_priors') and self._last_doc_priors is not None:
        #     prior_reg_loss = torch.mean(self._last_doc_priors ** 2)
        #     total_loss = ce_loss + self.prior_reg_weight * prior_reg_loss
        #     # print(f"[DEBUG] CE Loss: {ce_loss.item():.4f}, Total Loss: {total_loss.item():.4f}")
        #     # print(f"[DEBUG] CE Loss: {ce_loss.item():.4f}, Prior Reg: {prior_reg_loss.item():.4f}, "
        #         #   f"Reg Weight: {self.prior_reg_weight}, Total Loss: {total_loss.item():.4f}")
        #     # # Log the components (only on process 0 if DDP)
        #     # if not self.is_ddp or self.process_rank == 0:
        #     #     logger.debug(f"CE Loss: {ce_loss.item():.4f}, Prior Reg: {prior_reg_loss.item():.4f}, "
        #     #                f"Reg Weight: {self.prior_reg_weight}, Total Loss: {total_loss.item():.4f}")
        # else:
        #     total_loss = ce_loss
        
        return total_loss
    
    def forward(self, query=None, passage=None):
        """
        Forward pass with prior computation and caching.
        """
        q_reps = self.encode_query(query) if query else None
        p_reps = self.encode_passage(passage) if passage else None

        # for inference
        if q_reps is None or p_reps is None:
            return EncoderOutput(q_reps=q_reps, p_reps=p_reps)

        # for training
        if self.training:
            if self.is_ddp:
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)

            semantic_scores = self.compute_similarity(q_reps, p_reps)
            doc_priors = self.prior_head(p_reps)
            scores = semantic_scores + doc_priors.unsqueeze(0)
            
            # Compute any tracking metrics before loss
            with torch.no_grad():
                # self._last_doc_priors = doc_priors.detach()
                
                batch_size = q_reps.size(0)
                train_group_size = p_reps.size(0) // batch_size
                self._tracking_metrics = self._compute_tracking_metrics(
                    semantic_scores, doc_priors, batch_size, train_group_size
                )
            
            # Compute loss
            scores = scores.view(q_reps.size(0), -1)
            target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
            target = target * (p_reps.size(0) // q_reps.size(0))
            loss = self.compute_loss(scores / self.temperature, target, doc_priors)
            if self.is_ddp:
                loss = loss * self.world_size  # counter average weight reduction
        
        # for eval TODO not used yet
        else:
            scores = self.compute_similarity(q_reps, p_reps)
            loss = None
            
        return EncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
        )
    
    def get_document_priors(self, p_reps):
        """
        Get document priors for given passage representations.
        Useful for analysis and debugging.
        
        Args:
            p_reps: Passage representations of shape (num_passages, embedding_dim)
        
        Returns:
            Prior values of shape (num_passages,)
        """
        return self.prior_head(p_reps)
    
    def save(self, output_dir: str):
        """
        Save the model to a directory.
        Saves encoder via save_pretrained and prior_head separately.
        """
        import os
        # Save encoder using standard HuggingFace method
        self.encoder.save_pretrained(output_dir)
        # Save prior_head weights
        prior_head_path = os.path.join(output_dir, 'prior_head.pt')
        torch.save(self.prior_head.state_dict(), prior_head_path)
        logger.info(f"Saved prior head to {prior_head_path}")

    def _compute_tracking_metrics(self, semantic_scores, doc_priors, batch_size, train_group_size):
        """
        Compute tracking metrics for monitoring during training.
        Extends base class method to include prior-specific metrics.
        
        Args:
            semantic_scores: Scores without priors, shape (batch_size, num_passages)
            doc_priors: Document prior values, shape (num_passages,)
            batch_size: Number of queries
            train_group_size: Number of passages per query (1 pos + n negs)
        
        Returns:
            Dictionary with tracking metrics (semantic scores + priors)
        """
        # Get indices: [0, train_group_size, 2*train_group_size, ...]
        positive_indices = torch.arange(batch_size, device=semantic_scores.device) * train_group_size
        
        # Extract positive priors
        positive_priors = doc_priors[positive_indices]
        sum_pos_priors = positive_priors.sum()
        avg_prior_pos = sum_pos_priors / batch_size
        
        # Calculate Negative Priors
        sum_priors = doc_priors.sum()
        sum_neg_priors = sum_priors - sum_pos_priors
        numel_priors = doc_priors.numel()
        avg_prior_neg = sum_neg_priors / (numel_priors - batch_size)

        # Compute base metrics (semantic scores) and add prior metrics
        prior_metrics = {
            'avg_prior_pos': avg_prior_pos.item(),
            'avg_prior_neg': avg_prior_neg.item(),
        }
        
        return super()._compute_tracking_metrics(
            semantic_scores, batch_size, train_group_size, additional_metrics=prior_metrics
        )
    
    @classmethod
    def build(
            cls,
            model_args,
            train_args,
            **hf_kwargs,
    ):
        """
        Build method that passes prior-specific arguments to the constructor.
        """
        base_model = AutoModel.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = 0
        
        if model_args.lora or model_args.lora_name_or_path:
            if train_args.gradient_checkpointing:
                base_model.enable_input_require_grads()
            if model_args.lora_name_or_path:
                lora_config = LoraConfig.from_pretrained(model_args.lora_name_or_path, **hf_kwargs)
                lora_model = PeftModel.from_pretrained(base_model, model_args.lora_name_or_path, is_trainable=True)
            else:
                lora_config = LoraConfig(
                    base_model_name_or_path=model_args.model_name_or_path,
                    task_type=TaskType.FEATURE_EXTRACTION,
                    r=model_args.lora_r,
                    lora_alpha=model_args.lora_alpha,
                    lora_dropout=model_args.lora_dropout,
                    target_modules=model_args.lora_target_modules.split(','),
                    inference_mode=False
                )
                lora_model = get_peft_model(base_model, lora_config)
            model = cls(
                encoder=lora_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature,
                prior_hidden_dim=model_args.prior_hidden_dim,
                prior_n_layers=model_args.prior_n_layers,
                prior_use_tanh=model_args.prior_use_tanh,
                prior_reg_weight=model_args.prior_reg_weight,
            )
        else:
            model = cls(
                encoder=base_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature,
                prior_hidden_dim=model_args.prior_hidden_dim,
                prior_n_layers=model_args.prior_n_layers,
                prior_use_tanh=model_args.prior_use_tanh,
                prior_reg_weight=model_args.prior_reg_weight,
            )
        return model
    
    @classmethod
    def load(
            cls,
            model_name_or_path: str,
            pooling: str = 'cls',
            normalize: bool = False,
            lora_name_or_path: str = None,  # type: ignore
            prior_hidden_dim: int = 256,
            prior_n_layers: int = 2,
            prior_use_tanh: bool = False,
            prior_reg_weight: float = 0.01,
            **hf_kwargs
    ):
        """
        Load a trained DenseModelWithPriors from a checkpoint.
        """
        import os
        from transformers import AutoModel
        from peft import LoraConfig, PeftModel
        
        base_model = AutoModel.from_pretrained(model_name_or_path, **hf_kwargs)
        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = 0
        
        if lora_name_or_path:
            lora_config = LoraConfig.from_pretrained(lora_name_or_path, **hf_kwargs)
            lora_model = PeftModel.from_pretrained(base_model, lora_name_or_path, config=lora_config)
            lora_model = lora_model.merge_and_unload()
            model = cls(
                encoder=lora_model,
                pooling=pooling,
                normalize=normalize,
                prior_hidden_dim=prior_hidden_dim,
                prior_n_layers=prior_n_layers,
                prior_use_tanh=prior_use_tanh,
                prior_reg_weight=prior_reg_weight,
            )
        else:
            model = cls(
                encoder=base_model,
                pooling=pooling,
                normalize=normalize,
                prior_hidden_dim=prior_hidden_dim,
                prior_n_layers=prior_n_layers,
                prior_use_tanh=prior_use_tanh,
                prior_reg_weight=prior_reg_weight,
            )
        
        # Load prior_head weights if they exist
        prior_head_path = os.path.join(model_name_or_path, 'prior_head.pt')
        if os.path.exists(prior_head_path):
            prior_head_state_dict = torch.load(prior_head_path, map_location='cpu')
            model.prior_head.load_state_dict(prior_head_state_dict)
            logger.info(f"Loaded prior head weights from {prior_head_path}")
        else:
            logger.warning(f"Prior head weights not found at {prior_head_path}. Using randomly initialized weights.")
        
        return model


