# Optional: Reduce the number of codes by merging close ones
if code_merge_threshold is not None:
    # append codes with context
    if embedding_context is not None:
        codes_with_context = [code + embedding_context for code in codes]
    else:
        codes_with_context = codes

    # get embeddings
    print("Computing embeddings for merging similar codes...")
    embeddings = embed(codes_with_context)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)  # normalize

    # compute embedding distance matrix
    cosine_sim = np.inner(embeddings, embeddings)
    cosine_dist = 1.0 - cosine_sim
    # plt.hist(np.reshape(cosine_dist,[-1]))
    # plt.show()

    # increase a distance threshold until the number of merged groups is under the desired threshold
    connection_matrix = np.where(cosine_dist < code_merge_threshold, 1, 0)
    n_components, labels = scipy.sparse.csgraph.connected_components(connection_matrix,
                                                                     directed=False,
                                                                     return_labels=True)
    # go through each group, find the code closest to the centroid and replace other codes with it
    print(f"Merging total {n_total_codes} codes into {n_components} merged codes...")
    merged_codes = []
    for group_idx in range(n_components):
        group = np.where(labels == group_idx)[0].tolist()

        if len(group) == 1:
            merged_codes.append(codes[group[0]])
        else:
            print("Merging code group:\n")
            for code_idx in group:
                print(codes[code_idx])
            print("\n")

            group_embeddings = embeddings[group]
            mean_group_embedding = np.mean(group_embeddings, axis=0)
            mean_group_embedding = mean_group_embedding / np.linalg.norm(mean_group_embedding)  # normalize
            similarity = np.inner(group_embeddings, mean_group_embedding.reshape([1, -1]))[:, 0]
            representative_code = codes[group[np.argmax(similarity)]]

            print("Representative code: ", representative_code, "\n")
            merged_codes.append(representative_code)
    codes = merged_codes
    exit()
