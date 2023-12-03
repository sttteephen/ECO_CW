import random

def integer_blx_crossover_list(parent1, parent2, alpha=0.5):

    # Iterate over each pair of genes in the parents
    for i in range(len(parent1)):
        # Calculate the range for each gene
        lower = min(parent1[i], parent2[i])
        upper = max(parent1[i], parent2[i])
        range_width = upper - lower

        # Extend the range by alpha
        lower -= alpha * range_width
        upper += alpha * range_width

        # Clip the values to the range [0, 9999]
        lower = max(int(round(lower)), 0)
        upper = min(int(round(upper)), 9999)

        # Generate offspring within the range
        parent1[i] = random.randint(lower, upper)
        parent2[i] = random.randint(lower, upper)
    
    return parent1, parent2