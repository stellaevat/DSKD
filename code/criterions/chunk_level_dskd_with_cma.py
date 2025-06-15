import math
import torch
from .dual_space_kd_with_cross_model_attention import DualSpaceKDWithCMA


class ChunkLevelDSKDWithCMA(DualSpaceKDWithCMA):
    def __init__(self, args, padding_id=-100) -> None:
        super().__init__(args, padding_id=padding_id)

    def get_chunk_alignment_mask(self, s_offsets, t_offsets, chunk_mask):
        # For each sample
        for i in range(len(s_offsets)):
            s_chunk_end, t_chunk_end = 0, 0
            while s_chunk_end < len(s_offsets[i]) and t_chunk_end < len(t_offsets[i]):
                # Start new chunk
                s_chunk_start, t_chunk_start = s_chunk_end, t_chunk_end
                s_token_end, t_token_end = s_offsets[i, s_chunk_end], t_offsets[i, t_chunk_end]

                # Find end token of current chunk
                while s_token_end != t_token_end:
                    if s_token_end < t_token_end:
                        # Moving from prompt to response
                        # # TODO: consider changing to token starts rather than ends, to be able to check for 0
                        if (s_chunk_end == len(s_offsets[i]) - 1) or (s_offsets[i, s_chunk_end] < s_token_end):
                            break
                        s_chunk_end += 1 
                        s_token_end = s_offsets[i, s_chunk_end]
                    else:
                        # Moving from prompt to response
                        if (t_chunk_end == len(t_offsets[i]) - 1) or (t_offsets[i, t_chunk_end] < t_token_end):
                            break
                        t_chunk_end += 1
                        t_token_end = t_offsets[i, t_chunk_end]

                # Update mask
                chunk_mask[s_chunk_start : s_chunk_end + 1, t_chunk_start : t_chunk_end + 1] = 1.
                s_chunk_end, t_chunk_end = s_chunk_end + 1, t_chunk_end + 1

        return chunk_mask


    def compute_dual_space_kd_loss_with_cma(
        self, outputs, teacher_outputs, input_data, output_data, distiller, log
    ):
        target = output_data["label"]
        teacher_target = output_data[f"teacher_{distiller.teacher_model_type}_label"]
        
        pad_mask = target.ne(self.padding_id)
        teacher_pad_mask = teacher_target.ne(self.padding_id)

        hiddens = outputs.hidden_states[-1]
        teacher_hiddens = teacher_outputs.hidden_states[-1]

        if hasattr(distiller.student_model, "model") \
            and hasattr(distiller.student_model.model, "embed_tokens"):
            stu_embed_tokens = distiller.student_model.model.embed_tokens
        elif hasattr(distiller.student_model, "model") \
            and hasattr(distiller.student_model.model, "model") \
            and hasattr(distiller.student_model.model.model, "embed_tokens"):
            stu_embed_tokens = distiller.student_model.model.model.embed_tokens
        elif hasattr(distiller.student_model, "transformer") \
            and hasattr(distiller.student_model.transformer, "wte"):
            stu_embed_tokens = distiller.student_model.transformer.wte
        else:
            raise NotImplementedError

        if hasattr(distiller.teacher_model, "model") \
            and hasattr(distiller.teacher_model.model, "embed_tokens"):
            tea_embed_tokens = distiller.teacher_model.model.embed_tokens
        elif hasattr(distiller.teacher_model, "model") \
            and hasattr(distiller.teacher_model.model, "model") \
            and hasattr(distiller.teacher_model.model.model, "embed_tokens"):
            tea_embed_tokens = distiller.teacher_model.model.model.embed_tokens
        elif hasattr(distiller.teacher_model, "transformer") \
            and hasattr(distiller.teacher_model.model, "wte"):
            tea_embed_tokens = distiller.teacher_model.transformer.wte
        else:
            raise NotImplementedError 

        formal_target = torch.where(pad_mask, target, torch.zeros_like(target))
        formal_input = torch.where(pad_mask, input_data["input_ids"], torch.zeros_like(target)) # TODO: Why same mask as target??
        stu_input_embeds = stu_embed_tokens(formal_input).detach()
        stu_target_embeds = stu_embed_tokens(formal_target).detach()

        formal_teacher_target = torch.where(teacher_pad_mask, teacher_target, torch.zeros_like(teacher_target))
        formal_teacher_input = torch.where(teacher_pad_mask, input_data[f"teacher_{distiller.teacher_model_type}_input_ids"], torch.zeros_like(teacher_target))
        tea_input_embeds = tea_embed_tokens(formal_teacher_input).detach()
        tea_target_embeds = tea_embed_tokens(formal_teacher_target).detach()

        stu_index_embeds = torch.cat([stu_input_embeds, stu_target_embeds], -1)
        tea_index_embeds = torch.cat([tea_input_embeds, tea_target_embeds], -1)

        norm_tea_index_embeds = tea_index_embeds / tea_index_embeds.std() # Normalise "for faster convergence"
        norm_tea_target_embeds = tea_target_embeds / tea_target_embeds.std()
        norm_teacher_hiddens = teacher_hiddens / teacher_hiddens.std()

        stu_q_hiddens = distiller.projectors["query"](stu_index_embeds).float()
        tea_k_hiddens = norm_tea_index_embeds.float()

        stu_v_hiddens = distiller.projectors["s2t"](hiddens).float()
        tea_v_hiddens = distiller.projectors["t2s"](
            norm_teacher_hiddens + norm_tea_target_embeds
        ).float()
        
        align = stu_q_hiddens.matmul(tea_k_hiddens.transpose(-1, -2))
        align = align / math.sqrt(2 * teacher_hiddens.shape[-1])
        align_mask = pad_mask.float().unsqueeze(-1) * teacher_pad_mask.float().unsqueeze(1)
        align = align + (1.0 - align_mask) * (-100000.) # Make masked values very small since logits

        stu_offsets = input_data["offsets"]
        tea_offsets = input_data[f"teacher_{distiller.teacher_model_type}_offsets"]
        chunk_mask = self.get_chunk_alignment_mask(stu_offsets, tea_offsets, torch.zeros_like(align))
        align = torch.where(chunk_mask.bool(), align, torch.full_like(align, -100000.))

        t2s_weight = torch.softmax(align, -1)        
        t2s_hiddens = t2s_weight.matmul(tea_v_hiddens).to(hiddens)
        t2s_logits = t2s_hiddens.matmul(
            distiller.student_model.lm_head.weight.detach().transpose(-1, -2)
        ) # Equation 4 (except where is the softmax?)
        t2s_ce_loss = self.compute_cross_entropy_loss(t2s_logits, target)[0] # Equation 5
        t2s_acc_mask = t2s_logits.argmax(-1).eq(target)
        t2s_acc = (t2s_acc_mask * pad_mask).sum() # TODO: Absolute value rather than ratio (see line 154)
        max_probs = (t2s_logits.softmax(-1).max(-1)[0] * pad_mask).sum()
        log["t2s_ce_loss"] = t2s_ce_loss
        log["t2s_acc"] = t2s_acc
        log["max_t2s_prob"] = max_probs
        
        if not self.args.only_save_projector:  # skip if only train projectors (pre-train projectors)
            t2s_kd_loss = self.dist_func(
                outputs.logits, t2s_logits.detach(), target, reduction="none", use_tea_temp=True
            ) # Equation 6
            t2s_kd_loss = (t2s_kd_loss * pad_mask * t2s_acc_mask).sum() 

            s2t_weight = torch.softmax(align.transpose(-1, -2), -1)
            s2t_hiddens = s2t_weight.matmul(stu_v_hiddens).to(hiddens)
            s2t_logits = s2t_hiddens.matmul(
            distiller.teacher_model.lm_head.weight.detach().transpose(-1, -2)
            ) # Equation 8 (except where is the softmax?)

            s2t_kd_loss = self.compute_forward_kl_divergence(
                s2t_logits, teacher_outputs.logits, teacher_target, reduction="none"
            ) # Equation 9
            s2t_kd_loss = (s2t_kd_loss * teacher_pad_mask).sum()
            s2t_acc = (s2t_logits.argmax(-1).eq(teacher_target) * teacher_pad_mask).sum() * pad_mask.sum() / teacher_pad_mask.sum() # TODO: Why mul/div by these?

            kd_loss = t2s_ce_loss + t2s_kd_loss + s2t_kd_loss
            log["t2s_kd_loss"] = t2s_kd_loss
            log["s2t_kd_loss"] = s2t_kd_loss
            log["s2t_acc"] = s2t_acc
        else:
            kd_loss = t2s_ce_loss

        log["kd_loss"] = kd_loss
        return kd_loss, log
    