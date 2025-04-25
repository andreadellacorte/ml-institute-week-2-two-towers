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
    sim_pos = torch.nn.functional.cosine_similarity(query, relevant_doc, dim=1)
    sim_neg = torch.nn.functional.cosine_similarity(query, negative_doc, dim=1)
    loss = torch.nn.functional.relu(margin - sim_pos + sim_neg)
    return loss.mean()