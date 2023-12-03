import random

def move_mutate(individual, row_move_range=10, indpb=0.5):
    # Generate the list of possible row moves
    row_moves = [i * 100 for i in range(-row_move_range, row_move_range + 1)]

    for i in range(len(individual)):
        if random.random() < indpb:
            row_move = random.choice(row_moves)
            column_move = random.randint(-row_move_range, row_move_range)
            
            # Calculate the current row and column
            current_row = individual[i] // 100
            current_col = individual[i] % 100

            # Apply column movement and handle edge cases
            new_col = max(0, min(current_col + column_move, 99))

            # Apply row movement and combine row and column to get new position
            new_row = (current_row * 100 + new_col) + row_move
            
            # Clip the new position to be within 0 and 9999
            individual[i] = max(0, min(new_row, 9999))

    return individual,