def reciprocal_rank_fusion(ranked_lists, k=60):
    """
    Implements Reciprocal Rank Fusion (RRF).
    
    Args:
        ranked_lists (list of lists): A list of ranked lists, where each sublist contains document IDs in rank order.
        k (int): The constant used in the RRF score calculation. Default is 60.
    
    Returns:
        dict: A dictionary where keys are document IDs and values are their RRF scores.
    """
    scores = {}

    # Iterate through each ranked list
    for ranked_list in ranked_lists:
        for rank, doc_id in enumerate(ranked_list):
            rrf_score = 1 / (k + rank + 1)
            if doc_id in scores:
                scores[doc_id] += rrf_score
            else:
                scores[doc_id] = rrf_score
    
    # Sort documents by their accumulated RRF score in descending order
    sorted_scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
    
    return sorted_scores