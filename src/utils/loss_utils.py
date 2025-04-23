import torch

def contrastive_loss(query, relevant_doc, negative_doc, margin):
    """
    Contrastive loss function.
    Args:
        pos: Positive pair distance.
        neg: Negative pair distance.
        margin: Margin for the loss.
    Returns:
        Loss value.
    """

    # compute pos and neg with cosine similarity
    cos_pos = torch.nn.functional.cosine_similarity(query, relevant_doc, dim=1)
    cos_neg = torch.nn.functional.cosine_similarity(query, negative_doc, dim=1)

    return torch.mean(torch.clamp(margin + cos_pos - cos_neg, min=0.0))