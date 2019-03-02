import itertools


assignments = []
rows = 'ABCDEFGHI'
cols = digits = '123456789'
square_rows = ('ABC', 'DEF', 'GHI')
square_cols = ('123', '456', '789')

def cross(A, B):
    "Cross product of elements in A and elements in B."
    return [r + c for r in A for c in B]

boxes = cross(rows, cols)
row_units = [cross(r, cols) for r in rows]
col_units = [cross(rows, c) for c in cols]
square_units = [cross(rs, cs) for rs in square_rows for cs in square_cols]
digonal_units = [[i + j for i, j in zip(rows, cols)], [i + j for i, j in zip(rows, cols[::-1])]]
unitlist = row_units + col_units + square_units + digonal_units
units = dict((s, [u for u in unitlist if s in u]) for s in boxes)
peers = dict((s, set(sum(units[s], [])) - set([s])) for s in boxes)

def assign_value(values, box, value):
    """
    Please use this function to update your values dictionary!
    Assigns a value to a given box. If it updates the board record it.
    """

    # Don't waste memory appending actions that don't actually change any values
    if values[box] == value:
        return values

    values[box] = value
    if len(value) == 1:
        assignments.append(values.copy())
    return values

def naked_twins(values):
    """Eliminate values using the naked twins strategy.
    Args:
        values(dict): a dictionary of the form {'box_name': '123456789', ...}

    Returns:
        the values dictionary with the naked twins eliminated from peers.
    """

    # Find all instances of naked twins
    # Eliminate the naked twins as possibilities for their peers


    for unit in unitlist:
        potential_twins = [box for box in unit if len(values[box]) == 2]
        naked_twins = [[box1, box2] for box1 in potential_twins \
                       for box2 in unit if (box1 != box2 and values[box1] == values[box2])]

        # Sorted inside list
        sorted_naked_twins = ([sorted(sublist) for sublist in naked_twins])
        # Removed duplicates
        naked_twins_no_dups = list(k for k, _ in itertools.groupby(sorted_naked_twins))
        for naked_twins_no_dup in naked_twins_no_dups:
            unit_intersect = set(unit) - set(naked_twins_no_dup)
            for box in unit_intersect:
                if len(values[box]) > 1:
                    for digit in values[naked_twins_no_dup[0]]:
                        values = assign_value(values, box, values[box].replace(digit, ''))
    return values


def grid_values(grid):
    """
    Convert grid into a dict of {square: char} with '123456789' for empties.
    Args:
        grid(string) - A grid in string form.
    Returns:
        A grid in dictionary form
            Keys: The boxes, e.g., 'A1'
            Values: The value in each box, e.g., '8'. If the box has no value, then the value will be '123456789'.
    """
    values = []
    for g in grid:
        if g == '.':
            values.append(digits)
        else:
            values.append(g)
    # assert len(values) == 81
    return dict(zip(boxes, values))

def display(values):
    """
    Display the values as a 2-D grid.
    Args:
        values(dict): The sudoku in dictionary form
    """
    width = 1 + max(len(values[box]) for box in values.keys())
    line = '+'.join(['-' * (width * 3)] * 3)
    for r in rows:
        print(''.join(values[r + c].center(width) + ('|' if c in '36' else '') for c in cols))
        if r in 'CF': print(line)
    print

def eliminate(values):
    sloved_boxes = [box for box in values.keys() if len(values[box]) == 1]
    for sloved_box in sloved_boxes:
        for peer in peers[sloved_box]:
            if values[sloved_box] in values[peer] and len(values[peer]) > 1:
                assign_value(values, peer, values[peer].replace(values[sloved_box], ''))
    return values

def only_choice(values):
    for unit in unitlist:
        for digit in digits:
            dplaces = [box for box in unit if digit in values[box]]
            if len(dplaces) == 1:
                assign_value(values, dplaces[0], digit)
    return values

def reduce_puzzle(values):
    stalled = False
    while not stalled:
        # Check how many boxes have a determined value
        solved_values_before = len([box for box in values.keys() if len(values[box]) == 1])

        # Your code here: Use the Eliminate Strategy
        values = eliminate(values)
        # Your code here: Use the Only Choice Strategy
        values = only_choice(values)
        # Naked Twins
        #values = naked_twins(values)
        # Check how many boxes have a determined value, to compare
        solved_values_after = len([box for box in values.keys() if len(values[box]) == 1])
        # If no new values were added, stop the loop.
        stalled = solved_values_before == solved_values_after
        # Sanity check, return False if there is a box with zero available values:
    return values

def is_solved(values):
    if all(len(values[s]) == 1 for s in boxes):
        return True
    else:
        return False

def is_solved_correctly(values):
    is_correct= True
    for unit in unitlist:
        unit_values = [values[box] for box in unit if len(values[box]) == 1]
        if(len(unit_values) == 9 and len(set(unit_values)) != 9):
            is_correct = False
            break
    return is_correct

def search(values):
    "Using depth-first search and propagation, create a search tree and solve the sudoku."
    # First, reduce the puzzle using the previous function
    values = reduce_puzzle(values)
    isSolved = is_solved(values)
    isCorrect = is_solved_correctly(values)

    if isSolved and isCorrect:
        return True, values

    if isCorrect == False:
        return False, values

    # Choose one of the unfilled squares with the fewest possibilities
    n, s = min((len(values[s]), s) for s in boxes if len(values[s]) > 1)
    # Now use recursion to solve each one of the resulting sudokus, and if one returns a value (not False), return that answer!
    for value in values[s]:
        new_sudoku = values.copy()
        new_sudoku[s] = value
        response, new_values = search(new_sudoku)
        if response == True:
            return True, new_values

    return False, values

def solve(grid):
    """
    Find the solution to a Sudoku grid.
    Args:
        grid(string): a string representing a sudoku grid.
            Example: '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'
    Returns:
        The dictionary representation of the final sudoku grid. False if no solution exists.
    """
    values = grid_values(grid)
    _,values = search(values)
    return values

if __name__ == '__main__':
    diag_sudoku_grid = '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'
    display(solve(diag_sudoku_grid))

    try:
        from visualize import visualize_assignments
        visualize_assignments(assignments)

    except SystemExit:
        pass
    except:
        print('We could not visualize your board due to a pygame issue. Not a problem! It is not a requirement.')
