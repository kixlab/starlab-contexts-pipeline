def iterative_task_schema_discovery(video_corpus, unit_type):
    """
    Discovers the optimal task schema frontier through iterative refinement.
    """
    
    # --- Initialization ---
    # Knowledge items (K), Facets/Dimensions (C), Annotations (A)
    knowledge_base = construct_knowledge_base(video_corpus, unit_type)
    facets = set()
    annotations = {}
    
    # Compute the initial Pareto front of valid schemas
    frontier = compute_frontier(knowledge_base, facets, annotations)

    # --- Iterative Refinement Loop ---
    while not converged:
        # Sample a batch of schemas from the current frontier for expansion
        schema_batch = sample_schemas(frontier)
        new_facets = set()

        # Step 1: Facet Generation (Dimension Discovery)
        for schema in schema_batch:
            # Add new orthogonal dimensions or refine existing ones
            new_facets |= add_orthogonal_facets(schema, knowledge_base, annotations)
            new_facets |= revise_facets(schema, knowledge_base, annotations)

        # Step 2: Update State (Populate values for the new dimensions)
        new_annotations = annotate(knowledge_base, new_facets)

        # Step 3: Pareto Evaluation
        # Combine existing and new annotations to find the new frontier
        combined_annotations = {**annotations, **new_annotations}
        new_frontier = compute_frontier(knowledge_base, facets | new_facets, combined_annotations)

        # Check if the new frontier represents a Pareto improvement
        if is_pareto_improved(frontier, new_frontier):
            annotations.update(new_annotations)
            facets.update(new_facets)
            frontier = new_frontier
            
        if max_iterations_reached():
            break

    return facets, frontier